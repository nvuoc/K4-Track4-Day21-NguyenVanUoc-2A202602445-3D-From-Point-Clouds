"""Topic A — original experiment prepared with Codex; see report AI disclosure.
Geometry conventions and data readers: starter/kitti_io.py and data/README.md.
No detector, no training, no external dataset downloads.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from starter.datasets import list_frames, load_frame
from starter.projection import velo_to_cam, cam_to_image, perturb_extrinsic, overlay_points, draw_box2d


def object_membership(camera_points, obj):
    """Inverse KITTI yaw; location is bottom centre. Keep GT membership fixed."""
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    local = (camera_points - obj.location) @ rotation
    h, w, l = obj.dimensions
    return (np.isfinite(local).all(axis=1) & (np.abs(local[:, 0]) <= l/2)
            & (local[:, 1] >= -h) & (local[:, 1] <= 0)
            & (np.abs(local[:, 2]) <= w/2))


def in_box(uv, bbox):
    x1, y1, x2, y2 = bbox
    return ((uv[:, 0] >= x1) & (uv[:, 0] <= x2)
            & (uv[:, 1] >= y1) & (uv[:, 1] <= y2))


def dense_projection(points, calib, shape):
    uv, depth, mask = cam_to_image(velo_to_cam(points[:, :3], calib), calib.P2, shape)
    dense = np.full((len(points), 2), np.nan)
    dense[mask] = uv
    return dense, uv, depth, mask


def save_overlay(fr, calib, output, title, bands=None):
    uv, depth, _ = cam_to_image(velo_to_cam(fr['points'][:, :3], calib), calib.P2, fr['image'].shape)
    if bands is not None:
        chosen = (depth >= bands[0]) & (depth < bands[1])
        uv, depth = uv[chosen], depth[chosen]
    vis = overlay_points(fr['image'], uv, depth, radius=1)
    for obj in fr['labels']:
        vis = draw_box2d(vis, obj.bbox, label=obj.type)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(vis, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    if not cv2.imwrite(str(output), vis):
        raise OSError(output)
    return vis


def main():
    ap = argparse.ArgumentParser(description='CPU calibration yaw QA, fixed GT-point cohorts, two real datasets')
    ap.add_argument('--data-roots', nargs='+', default=['data/kitti_mini', 'data/nuscenes_mini_subset'])
    ap.add_argument('--yaw-deg', nargs='+', type=float, default=[-3, -2, -1, -.5, 0, .5, 1, 2, 3])
    ap.add_argument('--out-dir', default='results')
    ap.add_argument('--min-object-points', type=int, default=5)
    args = ap.parse_args()
    if 0 not in args.yaw_deg:
        ap.error('--yaw-deg must include 0 as reference')
    out = Path(args.out_dir)
    figures = out / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    rows, objects, manifest = [], [], []
    best_failure = None
    for data_root in args.data_roots:
        dataset = Path(data_root).name
        for frame_id in list_frames(data_root):
            fr = load_frame(data_root, frame_id)
            points, shape = fr['points'], fr['image'].shape
            valid = np.isfinite(points[:, :3]).all(axis=1)
            base_cam = velo_to_cam(points[:, :3], fr['calib'])
            base_dense, _, _, base_mask = dense_projection(points, fr['calib'], shape)
            cohorts = []
            for i, obj in enumerate(fr['labels']):
                gt = object_membership(base_cam, obj) & base_mask
                # Matched reference points: inside GT 3D AND its own GT 2D box.
                cohort = gt & in_box(base_dense, obj.bbox)
                if cohort.sum() >= args.min_object_points:
                    cohorts.append((i, obj, cohort))
            denom = sum(int(cohort.sum()) for _, _, cohort in cohorts)
            manifest.append({'dataset':dataset, 'frame':frame_id, 'points':len(points),
                             'invalid_xyz':int((~valid).sum()), 'eligible_objects':len(cohorts),
                             'reference_object_points':denom,
                             'camera_lidar_gap_ms':(fr.get('timestamp_camera_us', 0)-fr.get('timestamp_lidar_us', 0))/1000})
            for yaw in args.yaw_deg:
                calib = perturb_extrinsic(fr['calib'], yaw_deg=yaw)
                dense, _, _, mask = dense_projection(points, calib, shape)
                common = mask & base_mask
                shift = np.linalg.norm(dense[common]-base_dense[common], axis=1)
                kept, gt_inside, gt_count = 0, 0, 0
                for i, obj, cohort in cohorts:
                    hits = int((cohort & mask & in_box(dense, obj.bbox)).sum())
                    n = int(cohort.sum())
                    kept += hits
                    gt = object_membership(base_cam, obj) & base_mask
                    gt_count += int(gt.sum())
                    gt_inside += int((gt & mask & in_box(dense, obj.bbox)).sum())
                    objects.append({'dataset':dataset,'frame':frame_id,'object':i,'class':obj.type,
                                    'range_m':float(np.linalg.norm(obj.location)), 'yaw_deg':yaw,
                                    'reference_points':n,'retained_points':hits,'retention_pct':100*hits/n})
                retention = 100*kept/denom if denom else np.nan
                fov = 100*mask.sum()/valid.sum() if valid.sum() else np.nan
                base_fov = 100*base_mask.sum()/valid.sum() if valid.sum() else np.nan
                rows.append({'dataset':dataset,'frame':frame_id,'yaw_deg':yaw,
                             'valid_points':int(valid.sum()),'inside_fov':int(mask.sum()),'fov_pct':fov,
                             'fov_change_pp':fov-base_fov,'reference_object_points':denom,
                             'retained_points':kept,'retention_pct':retention,
                             'gt_points':gt_count,'gt_inside_own_box':gt_inside,
                             'gt_inside_own_box_pct':100*gt_inside/gt_count if gt_count else np.nan,
                             'median_shift_px':float(np.median(shift)) if len(shift) else np.nan})
                # Global FOV alarm misses a >=20% object correspondence loss.
                if denom >= 50 and abs(fov-base_fov) < 1 and retention < 80:
                    score = 100-retention
                    if best_failure is None or score > best_failure['loss_pct']:
                        best_failure = {'dataset':dataset,'data_root':data_root,'frame':frame_id,
                                        'yaw_deg':yaw,'loss_pct':score,'retention_pct':retention,
                                        'fov_change_pp':fov-base_fov,'reference_object_points':denom}
            if dataset == 'kitti_mini' and frame_id == '000011':
                for lo, hi in [(0, 15), (15, 30), (30, 80)]:
                    save_overlay(fr, fr['calib'], figures/f'demo_depth_{lo}_{hi}m.png',
                                 f'KITTI {frame_id} baseline | depth {lo}-{hi} m', (lo,hi))
            if frame_id in ['000011', 'scene-0103_010', 'scene-1094_010']:
                save_overlay(fr, fr['calib'], figures/f'demo_{frame_id}.png', f'{dataset} {frame_id} baseline')
    df = pd.DataFrame(rows)
    df.to_csv(out/'yaw_perturb_sweep.csv', index=False, float_format='%.8f')
    pd.DataFrame(objects).to_csv(out/'object_retention.csv',index=False,float_format='%.8f')
    pd.DataFrame(manifest).to_csv(out/'frame_manifest.csv',index=False,float_format='%.8f')
    summary = df.groupby(['dataset','yaw_deg'],sort=True).agg(
        frames=('frame','count'),valid_points=('valid_points','sum'),inside_fov=('inside_fov','sum'),
        reference_points=('reference_object_points','sum'), retained_points=('retained_points','sum'),
        gt_points=('gt_points','sum'),gt_inside_own_box=('gt_inside_own_box','sum'),
        mean_shift_px=('median_shift_px','mean')).reset_index()
    summary['fov_pct'] = 100*summary.inside_fov/summary.valid_points
    summary['retention_pct'] = 100*summary.retained_points/summary.reference_points
    summary['gt_inside_own_box_pct'] = 100*summary.gt_inside_own_box/summary.gt_points
    summary.to_csv(out/'yaw_summary.csv',index=False,float_format='%.8f')
    fig, axes = plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for dataset, group in summary.groupby('dataset'):
        for ax, metric in zip(axes,['fov_pct','retention_pct','mean_shift_px']):
            ax.plot(group.yaw_deg,group[metric],'-o',label=dataset)
    for ax, title in zip(axes,['Points in camera FOV (%)','Fixed GT-point retention (%)','Mean per-frame median shift (px)']):
        ax.set(xlabel='LiDAR yaw drift (deg)',ylabel=title)
        ax.grid(alpha=.3)
        ax.legend()
    axes[1].axhline(80,color='red',linestyle='--',label='20% loss alert')
    axes[1].legend()
    fig.savefig(figures/'yaw_benchmark.png',dpi=150)
    plt.close(fig)
    if best_failure is None:
        raise RuntimeError('No FOV-metric failure found; inspect results rather than invent a case')
    failure = best_failure
    fr = load_frame(failure['data_root'], failure['frame'])
    baseline = save_overlay(fr,fr['calib'],figures/'failure_reference.png',
                            f"{failure['dataset']} {failure['frame']} | baseline yaw 0 deg")
    drift = save_overlay(fr,perturb_extrinsic(fr['calib'],yaw_deg=failure['yaw_deg']),
                         figures/'failure_drift.png',
                         f"Yaw {failure['yaw_deg']} deg | retention {failure['retention_pct']:.1f}% | FOV change {failure['fov_change_pp']:.2f} pp")
    cv2.imwrite(str(figures/'fail_01_fov_misses_drift.png'),np.vstack([baseline,drift]))
    (out/'failure_case.json').write_text(json.dumps(failure,indent=2),encoding='utf8')
    config = vars(args) | {'seed':2445,'random_operations':False,
                          'cohort':'GT 3D + baseline FOV + own baseline 2D bbox; >=min-object-points',
                          'aggregation':'point-weighted; duplicated memberships counted per object',
                          'ego_motion_compensation':True}
    (out/'experiment_config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    print(summary[['dataset','yaw_deg','fov_pct','retention_pct','gt_inside_own_box_pct']].to_string(index=False))
    print('Failure:',json.dumps(failure))

if __name__ == '__main__':
    main()
