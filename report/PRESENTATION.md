# Gợi ý trình bày 3 phút — Nguyễn Văn Ước, 2A202602445

- **0:00–0:30:** Topic A, đo calibration drift bằng yaw. Dữ liệu: 20 KITTI + 80 nuScenes; CPU, không detector. Claim: ±3° làm mất hơn 20% điểm khớp vật thể.
- **0:30–1:15:** Công thức P2 × R0_rect × Tr × [x,y,z,1]. Chia thành phần thứ ba để ra pixel; lọc NaN/Inf, depth ≤0.1 m và ngoài ảnh. Test (10,0,0) cho depth gần 9.73 m, pixel gần (614,175).
- **1:15–2:00:** Mở yaw_benchmark.png. Có 9 mức, chỉ thay yaw. Retention = số điểm baseline thuộc cùng object vẫn khớp box 2D / số điểm baseline, mẫu số cố định. Ở ±3° KITTI còn 75–76%, nuScenes còn 73%; FOV gần như không đổi. Baseline 100% theo định nghĩa, không phải độ chính xác tuyệt đối.
- **2:00–2:35:** Mở fail_02_gt_points_zoom.png (điểm tham chiếu màu tím). KITTI 000001, −3°: 97 điểm tham chiếu đều ra khỏi box, FOV chỉ đổi 0.006 điểm phần trăm. Geometry làm lệch điểm; Metric chỉ đo FOV bỏ sót lỗi.
- **2:35–3:00:** ADAS nên log alignment theo object/range và timestamp, kiểm tra sau sửa gá sensor. Ngưỡng loss 20% là minh hoạ, còn cần GT; chưa tự kiểm tra calibration online. Giới hạn: point cloud thưa, occlusion, nuScenes 2D box suy từ 3D.

Câu hỏi có thể gặp:

1. **Vì sao không đếm mọi điểm trong bất kỳ box 2D nào?** Điểm nền/điểm thuộc xe khác có thể rơi vào box, tạo tín hiệu đúng giả. Bài cố định identity qua membership GT 3D ở baseline.
2. **Có khẳng định sensor 32 beam kém hơn 64 beam không?** Không; dataset, FOV, ảnh, object và timestamp cùng khác. Thí nghiệm không cô lập từng nguyên nhân.
3. **Yaw 1° có phát hiện được không?** Trung bình mất khoảng 5–7% điểm trong bộ này; ngưỡng minh hoạ 20% có thể bỏ sót. Vật nhỏ/xa có thể nhạy hơn trung bình toàn bộ.
4. **Chạy thật không có GT thì làm sao?** Cần nghiên cứu score edge/depth và kiểm chứng false alarm; retention hiện tại là metric offline.
5. **AI làm phần nào?** Codex hỗ trợ triển khai và kiểm tra, khai báo đầy đủ mục 6. Trước nộp, tự chạy notebook, đọc code và giải thích mẫu số/chuỗi transform.
