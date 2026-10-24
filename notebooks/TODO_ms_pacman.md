# TODO Ms. Pacman (làm tiếp ở nhà)

## 1. Reward âm theo thời gian (ý của bạn)

Mục tiêu: ép agent phải liên tục di chuyển và kiếm điểm, không đứng im hoặc đi vòng vòng.

Gợi ý thiết kế (chưa làm, chỉ để cân nhắc):
- Phạt nhỏ mỗi bước, ví dụ `-0.01` đến `-0.02` trên thang log hiện tại (chấm = +0.69, viên năng lượng = +1.79, ma = +3.0 đến +5.1). Phạt quá lớn thì agent chọn cách chết sớm cho đỡ bị trừ.
- Hoặc chỉ phạt khi "đứng bí": số bước liên tiếp không có reward dương vượt ngưỡng (ví dụ 50 đến 100 bước) thì mới trừ. Ít can thiệp hơn so với phạt đều mỗi bước.
- Chú ý tương tác với `EpisodicLife` (mất mạng = done): phạt theo thời gian cộng với việc chết cắt hết reward tương lai có thể làm agent liều lĩnh quá hoặc né ma kém. Nên so sánh có/không có phạt.
- Cài đặt: thêm vào `scale_reward` hoặc cộng thẳng vào `B_rew` / `buffer.add` trong `train_ppo` và `train_dqn`. Giữ điểm báo cáo là điểm thật, không bị phạt.

Cách kiểm chứng: so sánh 3 cấu hình (không phạt, phạt mỗi bước, phạt khi đứng bí) bằng biểu đồ điểm và biểu đồ ma ăn được trên mỗi viên năng lượng (cell mục 5 trong `ms_pacman.ipynb`). Mỗi cấu hình nên chạy 3 seed.

## Tình trạng hiện tại của notebook (để nhớ lại nhanh)
- Đã có: EpisodicLife, reward `log1p(r/10)`, 16 env song song chạy xen kẽ với GPU, bộ đếm viên năng lượng và ma, biểu đồ sự kiện.
- PPO khoảng 3.400 steps/s, DQN khoảng 600 đến 1.900 steps/s (phụ thuộc RAM trống; nên đóng bớt app khi chạy).
- Notebook đang để PPO 2M bước (bạn sửa), TOTAL_STEPS cho DQN là 10M.
