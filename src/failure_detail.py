"""Zoom actual matched GT points in the measured failure; no synthesized evidence."""
import json
from pathlib import Path
import cv2
import numpy as np
from starter.datasets import load_frame
from starter.projection import velo_to_cam, perturb_extrinsic, draw_box2d
from src.calibration_qa import dense_projection, object_membership, in_box

def main():
    out=Path('results')
    failure=json.loads((out/'failure_case.json').read_text())
    fr=load_frame(failure['data_root'],failure['frame'])
    cam=velo_to_cam(fr['points'][:,:3],fr['calib'])
    base,_,_,mask=dense_projection(fr['points'],fr['calib'],fr['image'].shape)
    chosen=np.zeros(len(base),dtype=bool)
    boxes=[]
    for obj in fr['labels']:
        cohort=object_membership(cam,obj)&mask&in_box(base,obj.bbox)
        if cohort.sum()>=5:
            chosen|=cohort
            boxes.append(obj.bbox)
    boxes=np.array(boxes)
    h,w=fr['image'].shape[:2]
    x1=max(0,int(boxes[:,0].min())-90)
    y1=max(0,int(boxes[:,1].min())-50)
    x2=min(w,int(boxes[:,2].max())+90)
    y2=min(h,int(boxes[:,3].max())+50)
    panels=[]
    for yaw in [0,failure['yaw_deg']]:
        uv,_,_,valid=dense_projection(fr['points'],perturb_extrinsic(fr['calib'],yaw_deg=yaw),fr['image'].shape)
        vis=fr['image'].copy()
        for obj in fr['labels']:
            vis=draw_box2d(vis,obj.bbox,label=obj.type)
        for u,v in uv[chosen&valid]:
            cv2.circle(vis,(int(u),int(v)),2,(255,0,255),-1)
        zoom=vis[y1:y2,x1:x2]
        zoom=cv2.resize(zoom,(1100,round(1100*zoom.shape[0]/zoom.shape[1])))
        banner=np.zeros((45,1100,3),dtype=np.uint8)
        retention=100 if yaw==0 else failure['retention_pct']
        text=f"{failure['frame']} yaw {yaw:g} deg | tracked GT points (magenta) | retention {retention:.1f}%"
        cv2.putText(banner,text,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        panels.append(np.vstack([banner,zoom]))
    cv2.imwrite(str(out/'figures/fail_02_gt_points_zoom.png'),np.vstack(panels))
    print('Saved measured failure zoom')

if __name__=='__main__':
    main()
