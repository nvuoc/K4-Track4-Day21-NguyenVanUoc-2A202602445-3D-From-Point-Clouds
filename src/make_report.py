"""Generate report only from measured CSVs; prepared with Codex."""
import json
from pathlib import Path
import pandas as pd

def main():
    out = Path('results')
    summary = pd.read_csv(out/'yaw_summary.csv')
    frames = pd.read_csv(out/'frame_manifest.csv',dtype={'frame':str})
    failure = json.loads((out/'failure_case.json').read_text())
    table = '\n'.join(f"| {r.dataset} | {r.yaw_deg:g} | {r.fov_pct:.2f} | {r.gt_inside_own_box_pct:.2f} | {r.retention_pct:.2f} | {r.mean_shift_px:.2f} |" for r in summary.itertuples())
    max_retention = summary[summary.yaw_deg.abs()==3].retention_pct.max()
    supported = max_retention < 80
    claim = f"Trên 20 frame KITTI và 80 frame nuScenes, yaw ±3° làm mất hơn 20% tập điểm tham chiếu thuộc vật thể. Retention cao nhất ở bốn cấu hình này là {max_retention:.2f}%; claim {'được ủng hộ' if supported else 'bị bác bỏ'} bởi số liệu."
    frame_text = '; '.join(f"{name}: " + ', '.join(group.frame) for name,group in frames.groupby('dataset'))
    text = f'''# Báo cáo Day 6: Độ nhạy calibration LiDAR–camera với yaw drift

- **Họ tên:** Nguyễn Văn Ước
- **MSSV:** 2A202602445
- **Lớp:** VinUni AI20K — Track 4
- **Link repo:** https://github.com/nvuoc/K4-Track4-Day06-3D-From-Point-Clouds
- **Tên repo khi nộp theo đề:** NguyenVanUoc-2A202602445-Track4-Day21 (remote hiện tại chưa có tên này).
- **Topic:** A — LiDAR-camera projection QA (mức Good)
- **Dataset:** data/kitti_mini, data/nuscenes_mini_subset; synthetic dùng kiểm chứng hình học.
- **Các frame đã dùng:** {frame_text}. Synthetic kiểm thử: 000000.

## 1. Claim

{claim}
Giữ nguyên ảnh, point cloud, label và bù ego-motion; chỉ đổi yaw quanh z-up của LiDAR bằng Tr @ D.
Cấu hình có seed 2445; không có phép ngẫu nhiên nên số liệu hình học tái lập hoàn toàn.

## 2. Evidence

| Dataset | Yaw (°) | Trong FOV (%) | GT đúng box riêng (%) | Retention (%) | Dịch pixel trung bình* |
|---|---:|---:|---:|---:|---:|
{table}

*Dịch pixel: trung bình median từng frame trên điểm còn trong FOV ở cả baseline và drift.
GT đúng box riêng: điểm trong 3D GT và FOV baseline, sau perturb còn trong 2D box của cùng object; điểm ra khỏi ảnh tính là sai.
Retention: cố định các điểm GT khớp 2D box ở baseline, tối thiểu 5 điểm/object, rồi đo phần còn khớp. Baseline 100% là do định nghĩa, không chứng minh calibration tuyệt đối đúng.
Tổng hợp theo số điểm, không lấy trung bình % từng object; điểm thuộc nhiều box tính theo từng membership. FOV dùng mẫu số là mọi điểm XYZ hữu hạn.

![benchmark](../results/figures/yaw_benchmark.png)
![KITTI baseline](../results/figures/demo_000011.png)
![nuScenes ban ngày](../results/figures/demo_scene-0103_010.png)
![nuScenes ban đêm](../results/figures/demo_scene-1094_010.png)

Ba khoảng cách KITTI 000011: [0–15 m](../results/figures/demo_depth_0_15m.png), [15–30 m](../results/figures/demo_depth_15_30m.png), [30–80 m](../results/figures/demo_depth_30_80m.png).
Nguồn ảnh: KITTI Vision Benchmark Suite; nuScenes (Motional), dữ liệu sẵn trong repo.
CSV đầy đủ: yaw_perturb_sweep.csv (900 cấu hình frame/yaw), yaw_summary.csv, object_retention.csv. Frame, điểm lỗi và time gap ở frame_manifest.csv.
Hai dataset khác số beam (64/32), FOV, ảnh và phân bố object; nuScenes có chênh timestamp và bù ego-motion. Đây là các nguyên nhân có thể ảnh hưởng khác biệt; thí nghiệm chưa tách được đóng góp của từng yếu tố.

## 3. Failure case

![baseline trên, drift dưới; điểm tham chiếu màu tím](../results/figures/fail_02_gt_points_zoom.png)
Ảnh toàn cảnh: [baseline/drift](../results/figures/fail_01_fov_misses_drift.png).

{failure['dataset']} frame {failure['frame']}, yaw {failure['yaw_deg']:g}°: retention chỉ {failure['retention_pct']:.2f}% trên {failure['reference_object_points']} điểm tham chiếu, trong khi FOV chỉ đổi {failure['fov_change_pp']:+.3f} điểm phần trăm.
Lỗi **Geometry**: yaw extrinsic sai làm điểm lệch khỏi object. Lỗi **Metric**: cảnh báo chỉ dựa trên |ΔFOV| ≥ 1 điểm phần trăm sẽ bỏ sót case này, vì điểm lệch khỏi xe/người nhưng vẫn trong ảnh.
Loss ≥20% có thể cảnh báo case này, nhưng cần GT và baseline đúng; chưa phải bộ phát hiện drift tự động lúc vận hành.
Hạn chế: ngưỡng ≥5 điểm bỏ qua object quá thưa; box 2D có nền/che khuất, và box 2D nuScenes suy từ 3D nên không phải GT camera độc lập. Bù ego-motion chưa bù chuyển động object hay deskew toàn sweep.

## 4. Khuyến nghị nếu triển khai thật

Với ADAS, kiểm tra calibration sau va chạm/sửa gá sensor bằng object nhỏ như người/cyclist, kết hợp edge alignment và timestamp.
Ghi log invalid ratio, FOV ratio, số điểm/object, alignment score theo range/class, time gap và phiên bản calibration.
Projection dùng CPU, tránh chi phí GPU; giảm tần suất QA/số điểm giảm tải nhưng có thể bỏ qua object nhỏ và drift ngắn.
Ngưỡng 20% loss và 1 điểm phần trăm FOV là minh hoạ: phải hiệu chỉnh trên dữ liệu sạch, drift thật và false alarm trước triển khai. Không suy ra độ chính xác detector từ metric này.

## 5. Cách chạy lại

Từ gốc repo, Python ≥3.10, CPU là đủ:

    python -m pip install -r requirements.txt
    python tools/verify_data.py --data-root data/kitti_mini
    python tools/verify_data.py --data-root data/nuscenes_mini_subset
    python -m unittest src.test_geometry -v
    python -m starter.data_health --data-root data/synthetic
    python -m src.calibration_qa
    python -m src.failure_detail
    python -m src.make_report
    python tools/check_submission.py

Môi trường Windows đã chạy trong phiên này: .venv-cpu/Scripts/python.exe (Python 3.14). Dùng đường dẫn này thay python nếu PATH trỏ tới Python MSYS2; phiên bản thư viện ghi ở src/requirements-tested.txt.
Tham số: python -m src.calibration_qa --help. Colab: src/NguyenVanUoc_2A202602445_Colab.ipynb, chọn CPU, Run all.
Notebook lấy dữ liệu đề, ghi các file triển khai đi kèm, chạy test/benchmark/báo cáo; không cần đã push code mới. Chỉ kiểm tra notebook bằng môi trường local tương đương, chưa chạy dịch vụ Colab.
Tái lập: chạy benchmark hai lần và so SHA-256 các CSV hình học; log ở results/validation.txt.
Trước nộp: đổi tên fork đúng mẫu, push source/results/report, nộp URL và commit hash lên LMS. Các bước tài khoản/LMS chưa thực hiện trong phiên này.

## 6. Khai báo sử dụng AI

| Công cụ | Dùng cho việc gì | Kiểm chứng |
|---|---|---|
| OpenAI Codex | Đọc đề, chọn topic, viết projection, benchmark, tests, notebook, báo cáo và tạo artifacts từ code chạy thật | Agent chạy verify_data, kiểm thử điểm (10,0,0), NaN/Inf/FOV, box xoay; benchmark hai lần, đối chiếu CSV và mở ảnh. Học viên cần tự chạy notebook và đọc/giải thích code trước nộp. |

Không khai báo học viên đã tự kiểm chứng trong phiên này. Không dùng detector/checkpoint, không bịa số liệu; bảng sinh từ CSV.
'''
    Path('report/REPORT.md').write_text(text,encoding='utf8')
    print('Generated report/REPORT.md from measured CSVs')

if __name__ == '__main__':
    main()
