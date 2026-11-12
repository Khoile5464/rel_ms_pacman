# Nhật ký nghiên cứu và huấn luyện: PPO chơi Ms. Pacman (ALE/MsPacman-v5)

Môn REL301m · Cập nhật: 2026-11-08 · Phạm vi: toàn bộ lịch sử repo `rel_ms_pacman` (git từ 2026-10-12 đến 2026-11-06, cộng các run trong `notebooks/runs/`)

> **Quy ước đọc.** Mỗi con số đều lấy từ log huấn luyện (`*_log.npy`, `*_episodes.npy`), output còn lưu trong notebook, hoặc từ lần chấm điểm do tôi chạy lại ngày 2026-11-08 (ghi rõ ở từng bảng). Điều gì là **giả thuyết** thì ghi là giả thuyết. Điều gì **không có dữ liệu** thì ghi là không có. Mục 8 liệt kê các sự cố và điểm bất nhất đã phát hiện khi tổng hợp.

---

## 1. Tóm tắt

Dự án huấn luyện agent PPO chơi Ms. Pacman từ ảnh điểm ảnh. Chuỗi thí nghiệm đi theo một mạch chẩn đoán:

1. PPO cơ sở (CNN, ảnh xám) tăng nhanh rồi **đi ngang quanh 700 điểm** (random ≈ 200–280).
2. Quan sát bằng mắt cho thấy agent đứng im hoặc lặp một quy tắc. Giả thuyết: không có chi phí cho việc đứng im, và chết quá rẻ. Thêm `death_penalty` và `step_penalty`.
3. Phạt mạnh làm agent né ma nhưng không chủ động săn ma sau khi ăn viên năng lượng. Giả thuyết mới: ảnh xám làm mất thông tin màu của ma đang choáng. Thử đầu vào RGB, thử kiến trúc có nhớ (CNN+LSTM) và ViT.
4. Kết quả đo được **không ủng hộ rõ rệt** giả thuyết nào trong các giả thuyết trên ở ngân sách 1–5M bước và 1–2 seed: RGB không khác xám trong A/B 3M bước; phạt cố định cao (20/0.1) kém hơn phạt thấp; phạt tăng dần ngang phạt thấp; train dài hơn cải thiện điểm train (cao nhất: ViT 1181, LSTM 967 điểm trung bình 100 ván cuối ở giai đoạn C) nhưng điểm đánh giá nhiễu rất lớn.
5. Run cuối (CNN RGB khởi tạo từ model xám, lịch `death_penalty` 2.5 → 5 → 10) **không cải thiện** (điểm đánh giá 502 so với 828 của model xám nguồn, cùng seed). Log cho thấy run này dùng `step_penalty = 2`, gấp 40 lần các run khác (0.05); `v_loss` cỡ 300 thay vì cỡ 1. Đây là nghi vấn cấu hình chưa được xác nhận là chủ ý (mục 7.3 và 8).

Kết luận khoa học hiện tại: **chưa có bằng chứng** rằng bất kỳ can thiệp nào (RGB, penalty, LSTM, ViT) đưa agent vượt khỏi vùng ~700–1000 điểm một cách đáng tin cậy. Nguyên nhân plateau vẫn chưa được xác định.

---

## 2. Bối cảnh và mục tiêu

Mục tiêu học tập: cài đặt DQN và PPO từ đầu bằng PyTorch, huấn luyện trên Ms. Pacman, so sánh về điểm số, độ ổn định, hiệu quả mẫu (theo `Reports/progress_report.html`, 24/10). Không đặt mục tiêu đạt mức người chơi. Chỉ dùng đầu vào là ảnh, không dùng đặc trưng thiết kế tay.

Trạng thái theo từng thuật toán:

| Thuật toán | Trạng thái |
|---|---|
| PPO | Đã huấn luyện nhiều cấu hình (các mục dưới) |
| DQN | Đã cài đặt (`agents/dqn.py`), **chưa có kết quả trên Ms. Pacman** trong repo này. Báo cáo tiến độ 24/10 cũng ghi như vậy; thư mục `logs/DQN_1` tồn tại nhưng tôi chưa mở |

Các thành phần khác trong repo mà báo cáo này **không** tổng hợp (tôi không đọc kết quả của chúng): notebook làm quen CartPole, Acrobot, LunarLander, CarRacing; thư mục `REL_Ms_Pacman/` (DQN có giải thích, attention, giao diện game); `UC-Berkeley-AI-Pacman-Project/`.

---

## 3. Phương pháp chung

### 3.1 Môi trường và tiền xử lý

- `ALE/MsPacman-v5`, 9 hành động rời rạc, frame-skip 4 (1 bước agent = 4 frame), `repeat_action_probability = 0`.
- Quan sát: 4 khung xếp chồng, 84×84. Xám → tensor `(4, 84, 84)`; RGB → `(12, 84, 84)` (thứ tự kênh `[R,G,B]` của từng khung).
- `EpisodicLife`: mất một mạng thì báo `terminated=True` khi huấn luyện, game thật chỉ reset khi hết 3 mạng. Khi đánh giá, wrapper này tắt và chơi đủ một ván (tối đa 4500 bước).
- Điểm báo cáo luôn là **điểm thật của game** cộng dồn qua ván 3 mạng, không phải reward đã phạt.

### 3.2 Reward khi học

```
r_học = ln(1 + r_thật / 10) − death_penalty · [mất mạng] − step_penalty
```

Chấm 10 → 0.69; viên năng lượng 50 → 1.79; ăn ma 200/400/800/1600 → 3.04/3.71/4.39/5.08.

### 3.3 Thuật toán và siêu tham số (PPO)

| Tham số | Giá trị |
|---|---|
| Số env song song × bước/rollout | 32 × 512 = 16 384 mẫu/update (báo cáo đầu: 16 × 128) |
| Epoch / minibatch | 4 / 4 |
| γ, λ (GAE) | 0.99, 0.95 |
| Clip ε, c1 (value), c2 (entropy) | 0.1, 0.5, 0.01 |
| Cắt gradient | 0.5 |
| Learning rate | 2.5e-4 (train từ đầu); 1e-4 hoặc 5e-5 (train tiếp); giảm tuyến tính về 0 trong **mỗi lần gọi** `train_ppo` |
| Phần cứng | RTX 3070 Ti Laptop, torch 2.3.1+cu121, gymnasium 0.29.1 |

### 3.4 Kiến trúc

| Tên | Mô tả |
|---|---|
| `cnn` | Nature DQN: Conv(8×8,s4) → Conv(4×4,s2) → Conv(3×3,s1) → FC512; đầu actor 9 + critic 1 |
| `lstm` | Thân CNN như trên, thêm LSTM (hidden 512). Trạng thái ẩn reset khi mất mạng hoặc hết ván. PPO chia minibatch theo env để giữ nguyên chuỗi 512 bước |
| `vit` | ViT nhỏ: patch 12, dim 128, depth 4, 4 head, mlp ratio 2 |
| CNN RGB từ xám | Lớp conv đầu mở rộng 4 → 12 kênh: trọng số kênh R/G/B của mỗi khung = trọng số xám × (0.299, 0.587, 0.114). Đầu ra ban đầu xấp xỉ model xám, các lớp sau giữ nguyên (`models/convert.py`) |

### 3.5 Cách đánh giá và hạn chế của nó

- **Điểm train**: trung bình các ván hoàn thành trong lúc policy còn thay đổi, lấy mẫu từ policy. Dùng để xem xu hướng, không dùng làm điểm cuối.
- **Điểm đánh giá**: 20 ván, seed 1000–1019, policy lấy mẫu (`greedy=False`), tắt `EpisodicLife`.
- **Nhiễu đánh giá lớn.** Cùng một model, cùng seed 1000, cùng hàm `evaluate`, chấm ở các lần khác nhau cho kết quả khác nhau (mục 7.5: `vit_dp25_sp0.05_from_pre_s0` ra 748, 660, 499). Độ lệch chuẩn mỗi lần 200–500 điểm trên 20 ván, nên sai số chuẩn của trung bình ≈ 50–100 điểm. **Chênh lệch dưới ~150 điểm giữa hai model không thể coi là khác biệt.**
- Đa số cấu hình chỉ có **1 seed**; chỉ thí nghiệm penalty (mục 7.4) có 2 seed.

---

## 4. Dòng thời gian

| Ngày (git / mtime) | Sự kiện |
|---|---|
| 10-12 | Khởi tạo repo |
| 10-24 | Thư viện `pacman_rl` (PPO, DQN, env song song); tài liệu HTML; notebook thử nghiệm; báo cáo tiến độ |
| 10-26 | Phát hiện lỗi nối run; thêm `death_penalty`, cờ `died`, log theo mạng; biểu đồ theo mạng |
| 10-27/28 | Đầu vào RGB hoặc xám tùy chọn; `step_penalty`; vẽ thích ứng; tài liệu |
| ~10-29 | Chạy thí nghiệm penalty (mtime các file `pen_*`) |
| 11-06 | Thêm LSTM, ViT, chuyển model xám → RGB; pipeline CNN RGB; train 1M × 10 đoạn |
| 11-08 | Chấm điểm lại, viết báo cáo này |

(Ngày của các thí nghiệm ở giữa suy từ git và thời gian sửa file; không có nhật ký giờ chạy.)

---

## 5. Giai đoạn 0: PPO cơ sở và hiện tượng đi ngang

Cấu hình: CNN xám, 16 env, 2M bước, `death_penalty = 0`, lr 2.5e-4. Hai lần chạy (lần 2 khởi tạo từ trọng số lần 1, seed 1; không phải seed độc lập).

| Chỉ số | Lần 1 | Lần 2 |
|---|---|---|
| Số ván | 3158 | 3086 |
| Điểm TB toàn bộ | 690.7 | 720.3 |
| Điểm TB 100 ván cuối | 727.4 | 767.3 |
| Ván đơn tốt nhất | 3660 | 4250 |
| Đánh giá (10 ván, seed 1000) | 546 ± 195 (tối đa 910) | chưa chấm |

Random: 232 (5 ván, báo cáo tiến độ); 198 và 282 ở hai mẫu 5 ván khác trong các tài liệu khác. Ba con số khác nhau cho thấy mẫu 5 ván không đủ để định ra mức nền.

**Quan sát:** điểm tăng nhanh trong ~25 nghìn bước đầu rồi đi ngang quanh 700, kể cả khi gấp đôi số bước. Entropy giảm 2.20 → ~1.2. Số bước sống mỗi mạng gần như không đổi (205–220) suốt 2M bước. Ma ăn được ≈ 0.66 × số viên năng lượng.

---

## 6. Giai đoạn 1: Chẩn đoán hành vi và phạt

### 6.1 Quan sát và giả thuyết H1 (26/10)

Bằng mắt (`watch`, chưa đo): `greedy=True` → agent đi tới một điểm rồi đứng im; lấy mẫu → di chuyển lặp theo khuôn mẫu.

**H1:** không có chi phí cho việc đứng im, và `death_penalty = 2.5` (chỉ xuất hiện một lần) nhỏ hơn nhiều so với tổng reward một mạng (ước lượng thô 10–12; chưa đo trực tiếp). Hành vi "chờ sung" là tối ưu cục bộ.

### 6.2 Lỗi phương pháp phát hiện cùng lúc

Run `ppo_train_2610` thực chất là **3 lần train độc lập nối nhau** (entropy quay về ~2.2 ở update 160 và 798), vì cell gọi `train_ppo` không truyền `model=` nên mỗi lần bắt đầu từ mạng mới trong khi vẫn `save_run(append=True)`. Hệ quả: đường học có điểm reset giả. Đã sửa bằng cách truyền `model=` khi train tiếp. Kết luận từ các run cũ vì thế chỉ đáng tin ở mức từng đoạn riêng lẻ.

### 6.3 Can thiệp và quan sát định tính

| Can thiệp | Quan sát (tác giả, bằng mắt, chưa có số) |
|---|---|
| `death_penalty = 20` | Đi được nhiều bước/lượt hơn một chút; điểm TB lên "đầu 9" (chữ số hàng trăm); entropy giảm; agent né ma hơn nhưng **đứng im sau khi ăn viên năng lượng** |
| Thêm `step_penalty` | Agent di chuyển liên tục và điểm tăng, nhưng vẫn đứng yên sau khi ăn viên năng lượng |

### 6.4 Giả thuyết H2 và các thử nghiệm

**H2 (ghi trong `update_death_penalty.md`):** vì ảnh xám, agent không phân biệt được ma đang choáng; vì quan sát là một khung ngắn nên không có "trí nhớ" rằng vừa ăn viên năng lượng. Tăng `death_penalty` làm agent càng ngại tiến lại gần ma.

Kiểm tra trực tiếp trên ảnh (`gray_vs_power_pellet.ipynb`, một khung được thu sau khi ăn viên năng lượng):

| Đại lượng | Giá trị |
|---|---|
| Số pixel xanh dương (ma choáng) trong khung RGB | 116 |
| Độ xám của ma choáng (ITU-R 601) | 109 |
| Các mức xám phổ biến trong mê cung | 32, 109, 146, 167 |

Độ xám của ma choáng (109) **trùng với một mức xám phổ biến của mê cung (109)**. Đây là bằng chứng gián tiếp rằng màu xanh bị mất sau khi xám hóa, nhưng mới là **một khung hình**, chưa đo tỷ lệ pixel bị lẫn, và chưa chứng minh mạng thực sự không phân biệt được. Ghi chú trong notebook còn có dòng độ xám "nền (đen)" = 68, một giá trị lạ cho nền đen; vùng crop `[2:10, 2:10]` có thể không phải nền đen, chưa kiểm tra.

Hai hướng thử đã làm: đầu vào RGB (mục 6.5) và kiến trúc có nhớ (mục 7.1).

### 6.5 A/B xám vs RGB (3M bước, `death_penalty = 2.5`, cùng seed 0, train từ đầu)

Dữ liệu lấy từ output notebook `ab_gray_vs_rgb.ipynb`; các file run `ab_*` **không còn** trong `notebooks/runs/`, nên không tính lại được.

| Mốc bước | Xám: điểm TB 20 ván | RGB: điểm TB 20 ván |
|---|---|---|
| 0.82M | 610 | 741 |
| 1.64M | 790 | 826 |
| 2.46M | 860 | 662 |
| 3.00M | 674 | 772 |
| Entropy cuối | 1.725 | 1.644 |

Điểm tại mỗi mốc là trung bình trượt 20 ván nên dao động ±100–200. **Không thấy khác biệt có hệ thống** giữa xám và RGB. Kết luận hạn chế: ở ngân sách này, 1 seed, RGB không cải thiện điểm. Chưa trả lời được việc RGB có giúp nhận ra ma choáng hay không, vì không có phép đo hành vi quanh ma xanh cho các model này.

---

## 7. Giai đoạn 2: Thí nghiệm có kiểm soát và mở rộng

### 7.1 Penalty cố định so với tăng dần (1M bước, xám, train từ đầu, 2 seed)

| Nhánh | `death_penalty` / `step_penalty` |
|---|---|
| `const_low` | 2.5 / 0 |
| `const_high` | 20 / 0.1 từ đầu |
| `curriculum` | 2.5/0 → 10/0.05 → 20/0.1 tăng dần trong **một** run |

Điểm TB 100 ván cuối (điểm train):

| Nhánh | seed 0 | seed 1 | TB 2 seed |
|---|---|---|---|
| `const_low` | 703 | 598 | **651** |
| `const_high` | 464 | 533 | **499** |
| `curriculum` | 624 | 690 | **657** |

Entropy cuối: low 1.74/1.79; high 2.03/1.93; curriculum 1.67/1.65.

Nhánh RGB (1 seed): `const_low` 658, `const_high` 492; `curriculum` RGB **dừng sớm ở 127 120 bước** (219 ván) nên không có số liệu so sánh.

**Đọc kết quả.** Với 2 seed và chênh lệch giữa hai seed cùng nhánh khoảng 70–105 điểm, ta chỉ khẳng định được: phạt cao ngay từ đầu (`const_high`) **thấp hơn** hai nhánh còn lại ~150 điểm và có entropy cao hơn (học chậm hơn). `curriculum` và `const_low` **không phân biệt được**. Điều này phù hợp với phương án "chỉ cần phạt thấp" hơn là "cần học từ từ", nhưng chưa đủ số seed để loại trừ.

### 7.2 CNN+LSTM và ViT (2M bước, xám, `death_penalty = 2.5`)

| Run | Điểm train TB 100 ván cuối | Entropy cuối | Đánh giá 20 ván (TB ± std, tối đa) |
|---|---|---|---|
| `vit_dp25_s0` | 492 | 1.94 | 650 ± 288, 1130 (lần 1); 550 ± 226, 1270 (lần 2) |
| `lstm_dp25_s0` | 665 | 1.74 | 577 ± 207, 1080 (lần 1); 711 ± 524, 2480 (lần 2) |

Hai lần chấm cùng một model cho chênh 100–130 điểm, đúng bằng độ nhiễu đã nêu ở 3.5. Khi xem bằng mắt, tác giả ghi nhận ViT có điểm TB thấp hơn nhưng **chơi ổn định hơn** khi xem trực tiếp (nhận xét định tính, một ván).

### 7.3 Chuỗi train tiếp (ViT và LSTM, `death_penalty` 2.5, `step_penalty` 0.05, lr thấp)

Sơ đồ: **pre** (2M, sp 0) → **P1** (+1M, sp 0.05) → **A** (+5M) → nhánh **B** (+1M, `dp = 10`) hoặc **C** (+5M, `dp = 2.5`) → **D** (chỉ LSTM, +5M).

Điểm train TB 100 ván cuối (từ `*_episodes.npy`):

| Giai đoạn | ViT | LSTM |
|---|---|---|
| pre (2M, sp 0) | 492 | 665 |
| P1 (+1M, sp 0.05) | 655 | 729 |
| A (+5M) | 818 | 744 |
| B (dp = 10, +1M) | 858 | 732 |
| C (+5M từ A) | **1181** | **967** |
| D (+5M từ C, LSTM) | không có file episodes | ~900 theo log in |

Điểm đánh giá 20 ván (TB ± std, tối đa), các lần chấm khác nhau của cùng một notebook:

| Run | ViT | LSTM |
|---|---|---|
| P1 | 748 ± 316 (lần 1), 660 ± 323 (2), 499 ± 163 (3) | 803 ± 535 (1), 680 ± 253 (2), 752 ± 284 (3) |
| A | 896 ± 500 (1), 835 ± 294 (2) | 794 ± 410 (1), 849 ± 458 (2) |
| B (dp = 10) | 708 ± 228 (1), 744 ± 237 (2) | 710 ± 274 (1), 748 ± 309 (2) |
| C | 986 ± 823, tối đa **4290** | 862 ± 384, tối đa 2060 |

Quan sát:

1. Điểm train tăng đơn điệu theo tổng số bước ở cả hai kiến trúc (ViT 492 → 1181). Đây là tín hiệu **học thật nhưng chậm**, không còn là plateau cứng ở ~700 như baseline.
2. `death_penalty = 10` (nhánh B) có điểm đánh giá ~710–750, thấp hơn A (794–896). Hai lần chấm khác nhau không đảo ngược thứ tự, nhưng chênh lệch (~100–150) nằm sát ngưỡng nhiễu. Tác giả đã chọn đi tiếp từ A chứ không từ B.
3. Điểm đánh giá của C cao nhất (986 cho ViT) nhưng độ lệch chuẩn 823 và ván tối đa 4290 cho thấy **phân phối lệch đuôi nặng**: vài ván rất cao kéo trung bình. Trung vị không được ghi lại.
4. Khi xem bằng mắt `watch_probs` (greedy, một ván): ViT-C 1000 điểm (720 bước), LSTM-D 960 điểm (659 bước). Một ván mỗi model, không có giá trị thống kê.
5. Entropy LSTM ở giai đoạn D thấp hơn (≈1.34–1.53) so với ViT (≈1.7).

Không có phép so sánh có kiểm soát giữa ViT và LSTM về hiệu quả (khác lr, khác tốc độ, 1 seed); không rút ra kết luận về kiến trúc nào tốt hơn.

### 7.4 CNN RGB khởi tạo từ model xám, lịch `death_penalty` tăng dần

**Giả thuyết H3:** khởi tạo từ CNN xám đã học (`pen_const_low_s0`, 1M bước) và chỉ để lớp conv đầu học thêm các kênh màu, rồi tăng `death_penalty` dần (2.5 → 5 → 10), sẽ giúp mạng dùng màu để phân biệt ma choáng mà không phá những gì đã học.

Cấu hình: lr 1e-4; mỗi đoạn lưu model, log, episodes; có tiến độ để tiếp tục khi bị ngắt; `step_penalty = SP = 2` (xem dưới).

Điểm train theo từng khoảng 1M bước (tính từ `cnn_rgb_ramp_from_gray_s0_*`; tổng 7.65M bước, 11 840 ván):

| Khoảng | `death_penalty` | Số ván | Điểm TB | Trung vị | Phân vị 90 | Ma/ván | Viên NL/ván | Entropy |
|---|---|---|---|---|---|---|---|---|
| 0–1M | 2.5 | 1531 | 615 | 540 | 1040 | 0.66 | 1.11 | 1.22 |
| 1–2M | 2.5→5 | 1538 | 558 | 480 | 940 | 0.60 | 1.14 | 1.34 |
| 2–3M | 5 | 1536 | 523 | 460 | 860 | 0.44 | 0.98 | 1.29 |
| 3–4M | 5 | 1560 | 579 | 500 | 950 | 0.53 | 1.00 | 1.28 |
| 4–5M | 5→10 | 1541 | 572 | 500 | 970 | 0.65 | 1.11 | 1.30 |
| 5–6M | 10 | 1554 | 417 | 330 | 690 | 0.49 | 0.99 | 1.22 |
| 6–7M | 10 | 1553 | 497 | 430 | 878 | 0.56 | 1.09 | 1.29 |
| 7–7.65M | 10 | 1027 | 545 | 450 | 970 | 0.69 | 1.03 | 1.20 |

(Chuyển mức `death_penalty` thực tế xảy ra ở bước 1.07M và 4.07M; bảng gộp theo khoảng 1M nên các hàng có mũi tên là hàng chứa điểm chuyển.) Số mạng mất mỗi ván xấp xỉ 3.0 ở mọi khoảng, đúng như kỳ vọng với ván 3 mạng.

Điểm đánh giá (20 ván, seed 1000, policy lấy mẫu, chạy ngày 2026-11-08 trên CPU):

| Model | TB | Std | Trung vị | Tối đa |
|---|---|---|---|---|
| `pen_const_low_s0` (xám, nguồn, 1M bước) | **828** | 555 | 665 | 2240 |
| `rgb_pen_const_low_s0` (RGB train từ đầu, 1M bước) | 660 | 307 | 575 | 1660 |
| `cnn_rgb_ramp_from_gray_s0` (RGB từ xám, 7.65M bước) | **502** | 340 | 390 | 1600 |

**Kết quả: H3 không được ủng hộ.** Điểm không tăng theo `death_penalty` hay theo số bước, và model sau 7.65M bước không hơn model nguồn 1M bước (thấp hơn ~300 điểm, lớn hơn nhiều so với sai số chuẩn ước lượng ~75–120). Lưu ý chiều ngược lại không kết luận được: số ván chỉ 20, std lớn.

**Nghi vấn cấu hình (chưa xác nhận).** Log của run này ghi `step_penalty = 2`, trong khi mọi run khác dùng 0 hoặc 0.05–0.1. Với reward chấm 0.69 và mạng trung bình ≈ 215 bước, mức phạt 2/bước cho tổng phạt cỡ −430 mỗi mạng, lớn hơn mọi nguồn thưởng. Dấu vết khớp với điều này: `v_loss` trung bình 394 (10 update đầu) và 283 (10 update cuối) so với 0.8–1.4 ở các run khác. Đây là **suy luận từ log**, tôi chưa chạy đối chứng với `SP = 0.05`. Nếu `SP = 2` là lỗi nhập liệu (so với 0.05 ở các notebook trước), thì kết quả H3 ở trên bị nhiễu bởi cấu hình sai, không phải bằng chứng chống lại ý tưởng.

**Phụ lục 7.4a: lần train tiếp đoạn 11 (2026-11-08) suy biến.** Train tiếp từ model sau 10 đoạn với `CHUNK = 10M`, `death_penalty = 10`, `step_penalty = 2` (lr 1e-4). Log in (update 10 → 190, đến 3.1M bước):

| Update | Điểm TB 20 ván | Bước/lượt | Viên NL/lượt | Ma/lượt | Entropy |
|---|---|---|---|---|---|
| 10 | 581 | 204 | 0.24 | 0.20 | 1.06 |
| 50 | 262 | 175 | 0.11 | 0.06 | 1.04 |
| 80 | 318 | 161 | 0.07 | 0.06 | 0.24 |
| 120 | 212 | 131 | 0.01 | 0.02 | 0.07 |
| 190 | 224 | 128 | 0.00 | 0.00 | 0.22 |

Điểm 581 ở update 10 là điểm kế thừa của model cũ (entropy 1.06, không phải ~2.2 của mạng mới). Sau đó agent suy biến thành chết sớm, không ăn gì, điểm ngang mức random. Hành vi này nhất quán với `step_penalty = 2` (mỗi bước đều có reward âm lớn hơn mọi thưởng), nhưng là suy luận, chưa có đối chứng. Chưa giải thích được vì sao 10 đoạn trước (cùng `SP = 2`, đoạn 1M) không sụp hẳn. Phỏng đoán chưa kiểm chứng: đoạn 10M có lr giảm chậm hơn nhiều. Model và log của run này (7.65M bước) được sao lưu ở `notebooks/runs/backup_ramp_sp2_7.65M/` và không bị ghi đè. Notebook được sửa để chạy run mới `cnn_rgb_ramp_sp0.05_from_gray_s0` (`SP = 0.05`, 10 đoạn × 1M, `dp` 2.5 → 5 → 10) với cơ chế dừng khi điểm TB đoạn < 350.

**Chẩn đoán hành vi quanh ma xanh/ma thường (`ghost_diag`):** cell đã viết (dùng RAM ALE, byte 6–9/12–15 cho ma, 10/16 cho Pacman theo bảng AtariARI, tác giả đã ghi **chưa kiểm lại** với Ms. Pacman) nhưng output còn lưu chỉ từ bản chạy thử (2 ván), không đủ dữ liệu để kết luận. Chưa chạy trên run đầy đủ.

### 7.5 Nhiễu trong chấm điểm

Cùng model, cùng seed, các lần chấm khác nhau cho khác nhau (ví dụ `vit_dp25_sp0.05_from_pre_s0`: 748 → 660 → 499). Nguyên nhân khả dĩ: policy lấy mẫu ngẫu nhiên (không cố định seed của việc chọn hành động) và/hoặc nhiễu của môi trường tích lũy theo quỹ đạo. Chưa tìm ra nguyên nhân cụ thể. Hệ quả: các chênh lệch dưới ~150 điểm trong bảng ở 7.3 không nên được diễn giải.

---

## 8. Sự cố, bất nhất và hạn chế của dữ liệu

| # | Vấn đề | Tác động |
|---|---|---|
| 1 | Cell train không truyền `model=` ở giai đoạn đầu (mục 6.2) | Các run cũ (`ppo_train_2610`) là 3 lần train nối; đã sửa |
| 2 | **Pipeline RGB dùng `step_penalty = 2`** (mục 7.4), gấp 40 lần các run khác | Có thể là nguyên nhân run không học; chưa kiểm chứng |
| 3 | Lệch giữa tiến độ và log của run RGB: `progress.json` ghi 10 đoạn đã xong (cấu hình lúc đó: 10 × 1M = 10M bước), nhưng log chỉ có **7.65M bước / 465 update**. Cell in ra "xong 10/10, tổng 0.00h" khi chạy lại | Không rõ do một số đoạn bị dừng sớm, do log bị ghi đè, hay do ngắt giữa chừng. Số liệu mục 7.4 dựa trên log thực tế 7.65M, không phải 10M |
| 4 | Cell chấm điểm trong pipeline RGB lỗi `NameError: SMOKE` sau khi kernel crash | Điểm đánh giá của run RGB trong notebook không có; tôi chấm lại bên ngoài (bảng 7.4) |
| 5 | Run A/B xám–RGB (`ab_*`) không còn trong `runs/` | Số liệu 6.5 lấy từ output notebook, không kiểm lại được |
| 6 | Run `rgb_pen_curriculum_s0` dừng ở 127 120 bước | Thiếu nhánh so sánh |
| 7 | Chưa có `*_episodes.npy` cho `lstm_dp2.5_sp0.05_plus10M_from_C_s0` (giai đoạn D); `plus10M` chỉ có `.pt` | Chỉ có số từ output in, không tính lại điểm train được |
| 8 | Cấu hình hiện tại của `ppo_vit_lstm.ipynb` ghi `step_penalty=0.1`, nhưng output cũ in `bước 0.000` | Output là của lần chạy trước khi sửa cấu hình; số liệu 7.2 đúng với log (sp = 0) |
| 9 | Mức nền random: 232, 198, 282 tùy tài liệu | Mẫu 5 ván; không dùng làm mốc chặt |
| 10 | Chưa đo trực tiếp "đứng im" bằng chỉ số định lượng (số bước liên tiếp không có reward dương, số ô đã đi qua) | Các nhận xét về hành vi đứng im chỉ là quan sát bằng mắt |

---

## 9. Thảo luận

**Điều dữ liệu hỗ trợ:**

1. Tăng số bước huấn luyện (kết hợp `step_penalty = 0.05`, lr thấp, ViT hoặc LSTM) làm điểm train tăng từ ~500–660 lên ~970–1180 sau tổng ≈ 14M bước. Chưa có điều kiện đối chứng "cùng 14M bước nhưng không có `step_penalty`", nên không tách được đóng góp của từng yếu tố.
2. Phạt chết lớn ngay từ đầu (20) làm giảm điểm so với phạt nhỏ (2.5), và học chậm hơn.

**Điều dữ liệu không hỗ trợ (hoặc chưa đủ để kết luận):**

1. RGB không cải thiện ở A/B 3M bước (1 seed).
2. Tăng dần `death_penalty` không vượt phạt thấp cố định ở 1M bước (2 seed).
3. `death_penalty = 10` không cải thiện so với 2.5 (nhánh B).
4. Chưa có bằng chứng rằng agent học được "ma xanh nên đuổi". Tỷ lệ ma/viên năng lượng ~0.5–0.7 ở mọi run, đúng mức baseline.

**Giả thuyết còn mở (chưa kiểm chứng):**

- H2 (xám làm mất thông tin ma choáng) chỉ có bằng chứng một khung hình (mục 6.4). Phép kiểm trực tiếp chưa làm: so sánh hành vi quanh ma xanh giữa model xám và RGB (đoạn `ghost_diag` đã có, chưa chạy đủ).
- Nguyên nhân plateau không cần là thông tin đầu vào: có thể là ngân sách bước (2–14M là nhỏ so với các nghiên cứu Atari), thiết kế reward thang log, hoặc khám phá (entropy còn ~1.2–1.9).

**Mối đe dọa với tính hợp lệ:**

- Đa số kết luận dựa trên 1 seed; thí nghiệm 2 seed có chênh lệch giữa seed cùng cỡ với chênh lệch giữa nhánh.
- Nhiễu đánh giá 100–200 điểm (mục 3.5, 7.5).
- Nhiều biến đổi cùng lúc giữa các giai đoạn (lr, `step_penalty`, kiến trúc, số bước), nên khó quy nguyên nhân.
- Lịch lr răng cưa khi train theo đoạn (mỗi lần gọi `train_ppo` giảm lr tuyến tính về 0).

---

## 10. Việc cần làm tiếp (đề xuất theo ưu tiên)

1. Xác nhận `SP` của pipeline RGB: chạy lại ngắn (vài M bước) với `step_penalty = 0.05` và xem `v_loss` về cỡ 1, điểm train có tăng không.
2. Cố định cách chấm điểm: cố định seed cho việc lấy mẫu hành động, đánh giá ≥ 50 ván, ghi cả trung vị.
3. Chạy `ghost_diag` đầy đủ (≥ 30 ván) trên model xám và model RGB; kiểm lại byte RAM với Ms. Pacman trước.
4. Đồng bộ lại `progress.json` với log (mục 8, #3) trước khi tiếp tục train, để số bước ghi nhận là số bước thực.
5. Chạy thêm seed (≥ 3) cho các so sánh đã có (xám vs RGB, penalty cố định vs tăng dần).
6. Hoàn tất DQN trên Ms. Pacman để đạt mục tiêu so sánh thuật toán của môn.

---

## Phụ lục A: Vị trí dữ liệu

| Nội dung | Vị trí |
|---|---|
| Log huấn luyện, episodes, model | `notebooks/runs/<tên>_log.npy`, `_episodes.npy`, `<tên>.pt` |
| Tiến độ pipeline RGB | `notebooks/runs/cnn_rgb_ramp_from_gray_s0_progress.json` |
| Thí nghiệm penalty | `notebooks/penalty_curriculum.ipynb` |
| A/B xám vs RGB | `notebooks/ab_gray_vs_rgb.ipynb` |
| Ảnh xám vs ma choáng | `notebooks/gray_vs_power_pellet.ipynb` |
| ViT/LSTM | `notebooks/ppo_vit_lstm.ipynb`, `ppo_finetune_step_penalty.ipynb`, `ppo_extend_then_dp10.ipynb` |
| Pipeline RGB từ xám | `notebooks/ppo_cnn_rgb_pipeline.ipynb` |
| Ghi chú thiết kế/phân tích cũ | `Reports/update_death_penalty.md`, `docs/enhance_penalty_capsule.md`, `Reports/progress_report.html` |
| Thư viện | `pacman_rl/src/pacman_rl/` |

## Phụ lục B: Cách số liệu được tính

- *Điểm train TB 100 ván cuối*: trung bình cột điểm của 100 dòng cuối trong `*_episodes.npy` (mỗi dòng là một ván hoàn chỉnh 3 mạng, cột 0 là bước env, cột 1 là điểm thật).
- *Bảng theo khoảng 1M bước (7.4)*: lọc `*_episodes.npy` theo cột bước; ma, viên năng lượng và số mạng mất lấy từ các cột tích lũy `ghosts`, `pellets`, `deaths` trong `*_log.npy` (nội suy tuyến tính tại ranh giới khoảng) chia cho số ván.
- *Điểm đánh giá ngày 2026-11-08*: `evaluate(ppo_policy(model, "cpu"), n_episodes=20, seed=1000)`, môi trường conda `rel_pacman`, một lần chạy duy nhất cho mỗi model. Theo mục 7.5, kết quả của một lần chạy khác có thể lệch trăm điểm.
- Các số đánh giá khác trích nguyên văn từ output đã lưu trong notebook tương ứng.
