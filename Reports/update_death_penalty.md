# Update: Setup mức phạt cho agent khi chết

Ngày: 2026-10-26 · Thuật toán: PPO · Env: ALE/MsPacman-v5

## 1. Quan sát

Khi chạy thử hai lần, agent có hành vi sau (quan sát bằng mắt khi `watch`, chưa có số liệu đo):

- Với `greedy=True`: agent di chuyển tới **một vị trí cố định rồi đứng im** ở đó.
- Bỏ `greedy` (lấy mẫu từ policy): agent có di chuyển "ngẫu nhiên", nhưng vẫn **lặp lại theo một quy tắc** như thể có khuôn mẫu.

## 2. Nhận định

Agent đang rơi vào hành vi "há miệng chờ sung": nó tìm được một điểm đứng chờ, nhận được lượng điểm nhỏ mà không phải mạo hiểm, nên không có áp lực phải đi kiếm điểm hay đuổi ma.

Nguyên nhân gợi ý: **không có chi phí nào cho việc đứng im**. Reward học hiện chỉ có phần thưởng dương (chấm, viên năng lượng, ma), nên không di chuyển không bị trừ gì. Hiện chỉ có `death_penalty = 2.5`, và nó chỉ xuất hiện đúng một lần khi chết.

Nhận định này **chưa được kiểm chứng**. Số liệu bên dưới phù hợp với nó nhưng không chứng minh nó.

## 3. Số liệu hỗ trợ

Phân tích `notebooks/runs/ppo_train_2610` và `ppo_test_2410` (xem mục 5 về việc file `2610` gồm 3 lần train nối nhau). Trong cả bốn lần train:

| Chỉ số | Giá trị |
|---|---|
| Bước sống mỗi mạng (1 bước = 4 frame) | **205 đến 220**, không tăng suốt 2M bước |
| Viên năng lượng ăn được mỗi mạng | ~0.3 |
| Ma ăn được mỗi mạng | ~0.2 |
| Điểm TB (20 ván) | tăng nhanh tới ~300k bước (≈450 → ≈730), sau đó đi ngang ở 650 đến 750 |
| Random (từ notebook) | 282 |

Số bước sống mỗi mạng gần như không đổi dù điểm đã tăng ở giai đoạn đầu. Điều này phù hợp với việc agent chết vào lúc ma tới nơi, không phải do agent chơi tốt hơn hay tệ hơn.

## 4. Kế hoạch

1. **Tăng `death_penalty`** đủ lớn để một mạng có tổng reward học **âm** (agent phải thấy chết là lỗ ròng).
2. **Thêm phạt liên tục** để ép di chuyển, theo hướng đã ghi ở `notebooks/TODO_ms_pacman.md` mục 1: phạt nhỏ mỗi bước, hoặc phạt khi "đứng bí" (N bước liên tiếp không có reward dương). Việc này chưa cài đặt; cách làm ở `pacman_rl/docs/extending.md` (wrapper `StuckPenalty`).
3. **So sánh có kiểm soát**: mỗi cấu hình train với ít nhất 3 seed, dùng cùng số bước, so các chỉ số theo mạng (xem mục 6).

### Chọn độ lớn của mức phạt chết

Reward học là `log1p(r / 10)`: chấm ≈ 0.69, viên năng lượng ≈ 1.79, ma ≈ 3.0 đến 5.1.

Ước lượng thô từ số liệu mục 3: điểm trung bình ~650 cho 3 mạng, tức ~217 điểm mỗi mạng, phần lớn đến từ chấm nhỏ (10 điểm). Tổng reward học mỗi mạng khoảng **10 đến 12**. Con số này là ước lượng, chưa đo trực tiếp.

| `death_penalty` | Ý nghĩa |
|---|---|
| 2.5 (hiện tại) | Lớn hơn một viên năng lượng, nhưng nhỏ hơn nhiều so với tổng reward của một mạng. Chết vẫn có lãi ròng. |
| ≳ 10 đến 12 | Tổng reward một mạng gần 0 hoặc âm. Mức bắt đầu thử để ép "chết là lỗ". |
| rất lớn (hàng chục trở lên) | Rủi ro: value loss lớn, agent né ma quá mức hoặc không dám ăn. Chỉ nên thử sau khi mức trung bình đã chạy. |

Cần lưu ý: PPO dùng `gamma = 0.99` nên mức phạt xuất hiện ở cuối mạng bị chiết khấu khi nhìn từ đầu mạng, nên ngưỡng "tổng âm" thực tế còn cao hơn ước lượng trên một chút. Bắt đầu thử một dãy giá trị (ví dụ 5, 10, 15) thay vì chọn một số.

## 5. Phát hiện kèm theo: các run cũ train từ mô hình mới

Khi phân tích, phát hiện `ppo_train_2610` **không phải một run liên tục**:

- File gồm **3 lần train độc lập** nối lại (entropy quay về 2.2 ở update 160 và 798). Lần 1 dừng sớm ở 328k bước, lần 2 dài 1.31M bước, lần 3 dài 530k bước.
- Nguyên nhân: cell notebook gọi `train_ppo(cfg, ...)` mà **không truyền `model=ppo.model`**, nên mỗi lần chạy đều bắt đầu từ mạng mới, dù trước đó đã `load_run` mô hình cũ. Sau đó `save_run(..., append=True)` nối chúng vào cùng tên.
- Hệ quả: đường học gộp có các điểm reset giả (ví dụ điểm tụt ở ~1.3M bước), và khó kết luận về hiệu quả của `death_penalty = 2.5`. Dù vậy, với từng lần train riêng lẻ, kết quả vẫn đi ngang như ở mục 3.
- Cách sửa: train tiếp từ mô hình cũ thì dùng `train_ppo(cfg, model=ppo.model, ...)`; train lại từ đầu thì đặt tên run mới mỗi lần, đừng `append=True`.

## 6. Cách đánh giá từ giờ

Dòng log của `train_ppo` và biểu đồ `plot_training` đã đo theo từng cửa sổ log (không cộng dồn):

- **bước/mạng**: agent sống lâu hơn không. Thay đổi của số này là dấu hiệu rõ nhất khi đổi mức phạt.
- **viên năng lượng/mạng, ma/mạng**: sống lâu hơn do ăn nhiều hơn hay chỉ né giỏi hơn.
- **điểm TB 20 ván**: thước đo chung, nhiễu lớn.

Hành vi "đứng im" cần được kiểm tra bằng mắt (`watch`), vì không có chỉ số nào trong log đo trực tiếp việc agent có đứng yên hay không. Nếu cần chỉ số định lượng, có thể thêm sau: số bước liên tiếp không có reward dương, hoặc số ô khác nhau mà agent đã đi qua trong một mạng.

## 7. Việc cần làm tiếp

- [ ] Sửa cell notebook: truyền `model=ppo.model`, đặt tên run mới.
- [ ] Thử `death_penalty` trong khoảng 5 đến 15 trên cùng số bước, so các chỉ số ở mục 6.
    - Author Note: 
        - Đã cài đặt `death_penalty=20`. Có khả quan, model đi được nhiều bước/lượt hơn một chút so với trước đó. Nhưng tốc độ hội tụ vẫn không đáng kể. Tuy nhiên, điểm trung bình đạt được đã lên đến đầu 9.
        - Entropy có xu hướng giảm: tốt.
    - Note: death_penalty đang làm cho con Agent chủ động tránh con ma hơn, nhưng lại sinh ra một vấn đề là khi nó ăn viên capsule, agent được học rằng "KHÔNG ĐƯỢC CHẾT" nên nó quyết định đứng im mẹ luôn.
- [ ] Cài đặt phạt liên tục (phạt mỗi bước hoặc phạt đứng bí).
- [ ] Xem lại bằng `watch` xem agent còn đứng im không.

- [] Cài đặt reward_fn để nó trừ điểm liên tục, ép agent phải di chuyển
    - Author Note: 
        - Thật mừng là mô hình đã học được cách di chuyển liên tục. Điểm đạt cũng được nhiều hơn trước.
        - Tuy nhiên, nó ăn viên năng lượng xong cái nó đứng yên luôn. Có thể là vì: An toàn, không có động lực đi ăn ma để kiếm được nhiều điểm hơn. Có thể điều chỉnh policy rằng khi ăn viên capsule thì điểm giảm theo thời gian sẽ nhanh hơn chăng?
        - Trước hết thì tôi sẽ thử train với mức phạt steps cao rồi train trên 10_000_000 steps thử trước. Lấy mô hình steps_penalty=0.02 để train tiếp.
        - Không, tôi sẽ thử kiểm tra lại đầu vào cho Agent, vì ảnh đang bị làm grayscale_obs, nên khả năng ma chưa phân biệt được chính xác khi nào thì con ma bị choáng.
        - Có nghĩa là nếu dùng gray scale thì thực sự trời mới biết được là con ma bị biến đổi chưa đúng ko? Tại về cơ bản, trong setup policy env với cơ chế dense, con agent không nhận thức được việc nó mới ăn viên capsule, ý là, nó ko có tính tuần tự ấy, cho nên là nó ko có thông tin gì về việc là con pacman nó vừa ăn viên capsule nên là nó phải đi ăn con ma. 
        - Nếu như giờ tôi cứ cố gắng tăng death_penalty lên thì con pacman của tôi nó càng không đi dí mấy con ma nữa, lí do entropy cao cũng là vì vậy, khi ở gần con ma, con agent sẽ phân vân không biết có nên ăn thử con ma này ko, vì nó chưa hiểu trạng thái. 
        - Nhưng tôi vẫn sẽ thử dùng RGB scale trước mới tính coi là có cần sequence network ko. Nhưng có lẽ trong tương lai tôi cũng sẽ thử sử dụng sequence network thôi.
    - Author Note ngày 28:
        - Không khả quan lắm, train gần 20M bước rồi, nhưng nó cứ dao động, gần như là ko học được gì.
        - Thử A/B testing để kiểm chứng.