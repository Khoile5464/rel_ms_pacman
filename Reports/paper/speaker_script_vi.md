# Script thuyết trình: Pixels to Power Pellets

Thời lượng mục tiêu: khoảng 12 phút (15 slide, mỗi slide ~45 giây). Mốc thời gian trong ngoặc là mốc tích lũy gợi ý. Điền tên `[Author]`. Số liệu khớp với `report.tex` bản mình viết; nếu bạn đã đổi số trên Overleaf thì sửa lại ở đây.

---

## 1. Tiêu đề [0:00]
"Chào mọi người. Đề tài của tụi em là *Pixels to Power Pellets*. Tụi em huấn luyện một agent học tăng cường chơi Ms. Pac-Man chỉ từ điểm ảnh, và đặt một câu hỏi: khi agent bị kẹt thì **vì sao** nó kẹt, và điều gì thực sự giúp nó thoát ra? Nhóm gồm Lê Minh Khôi, Phạm Quang Huy và Ngô Quang Huy, dưới sự hướng dẫn của thầy NamLG."

## 2. Vấn đề [0:30]
"Tụi em bắt đầu với agent PPO chuẩn dùng CNN. Nó học rất nhanh: chỉ sau khoảng 25 nghìn bước điểm đã tăng vọt. Nhưng sau đó nó đứng yên quanh mức 700 điểm. Chơi ngẫu nhiên được khoảng 274, nên 700 tốt hơn ngẫu nhiên, nhưng gấp đôi thời gian train từ 1 lên 2 triệu bước thì điểm không đổi.
Khi xem agent chơi, nó hoặc đứng im, hoặc lặp lại một mẫu hành vi, và sau khi ăn viên năng lượng nó không đuổi ma. Đó là vấn đề, vì điểm lớn nằm ở việc ăn ma.
Nói trước kết quả: agent tốt nhất của tụi em đạt 1125. Phần còn lại của bài là chuyện để đạt được nó, và những gì **không** hiệu quả."

## 3. Agent thấy gì và làm gì [1:15]
"Thiết lập nhanh. Tụi em dùng Ms. Pac-Man phiên bản Atari trong Gymnasium: 9 hành động, mỗi bước của agent là 4 khung hình của emulator. Agent nhìn 4 khung gần nhất, resize về 84×84, ở dạng xám hoặc RGB.
Điểm: dot 10, viên năng lượng 50, ma từ 200 đến 1600 khi ăn liên tiếp. Chú ý là phần thưởng lớn đòi hỏi một *chuỗi* quyết định: ăn viên năng lượng, nhận ra ma đã chuyển xanh, rồi đuổi trước khi hiệu ứng hết.
Khi huấn luyện, mất một mạng thì tính là kết thúc episode; khi chấm điểm luôn chơi đủ ván 3 mạng, và luôn báo cáo điểm thật của game, không phải reward đã phạt."

## 4. Ba giả thuyết [2:00]
"Tụi em biến các nghi ngờ thành ba giả thuyết, mỗi cái có một thí nghiệm rẻ để kiểm tra.
**H1:** đứng im không tốn gì. Chết thì rẻ còn chờ thì miễn phí. Nên tụi em thêm phạt khi chết và phạt mỗi bước.
**H2:** mất thông tin màu. Ở ảnh xám, ma xanh có thể trông giống mê cung. Nên tụi em thử đầu vào RGB.
**H3:** thiếu trí nhớ. Bốn khung hình không nhớ được việc 'vừa ăn viên năng lượng'. Nên tụi em thử LSTM và Vision Transformer.
PPO, LSTM và ViT do tụi em tự cài bằng PyTorch; chỉ emulator là của Gymnasium."

## 5. Phương pháp: reward và PPO [2:45]
"Reward khi học là reward game nén theo log, trừ phạt chết khi mất mạng, trừ phạt cố định mỗi bước. Thang log giữ cho sự kiện lớn có giá trị lớn hơn dot mà không lấn át.
Một điểm quan trọng: các penalty này **không** phải potential-based shaping, nên chúng có thể làm thay đổi chính sách tối ưu. Điều này sẽ quan trọng ở phần sau.
Thiết lập PPO là chuẩn: 32 môi trường song song, 512 bước mỗi rollout, gamma 0.99, clip 0.1. Tất cả chạy trên một GPU laptop."

## 6. Bốn policy [3:30]
"Tụi em so sánh bốn mạng, cùng đầu actor và critic.
Baseline là CNN kiểu Nature-DQN. LSTM thêm một lớp hồi quy 512 đơn vị; tụi em reset trạng thái khi mất mạng và chia minibatch theo môi trường để giữ nguyên chuỗi.
ViT cắt ảnh thành patch 12×12, được 49 token cộng một token CLS, với bốn lớp transformer.
Cái thứ tư là của tụi em: để thử màu mà không bỏ đi những gì agent xám đã học, tụi em mở rộng lớp conv đầu từ 4 lên 12 kênh bằng trọng số độ sáng, nên mạng RGB lúc đầu hành xử gần giống mạng xám."

## 7. Kiểm tra H2: ảnh xám có làm mất ma không? [4:15]
"Đây là bằng chứng cho H2 trên một khung thật. Bên trái là RGB sau khi ăn viên năng lượng: ma màu xanh, 116 pixel xanh. Bên phải là cùng khung ở dạng xám. Độ xám của ma choáng là 109, đúng bằng một mức xám phổ biến của mê cung. Nên có thể thông tin bị mất.
Nhưng, và đây là điểm bất ngờ, RGB **không** làm điểm tốt hơn. Cả CNN xám và RGB đều ăn khoảng 0,59 ma mỗi ván. Đây chỉ là kiểm tra trên một khung hình, nên chỉ mang tính gợi ý, chưa phải bằng chứng chắc."

## 8. Kết quả H1: penalty [5:00]
"Giờ đến H1, 2 seed, 1 triệu bước. Phạt nhỏ cố định đạt trung bình 651. Phạt lớn ngay từ đầu, 20 và 0,1, chỉ đạt 499: thấp hơn khoảng 150 điểm và entropy còn cao, tức học chậm hơn. Curriculum tăng dần penalty đạt 657, bằng phạt nhỏ nếu xét nhiễu.
Vậy chỉ riêng penalty không chữa được việc đứng im, và phạt càng lớn không có nghĩa là tốt hơn."

## 9. H2 và H3 ở 1M bước [5:45]
"Với cùng ngân sách 1 triệu bước, xám và RGB gần như không phân biệt được: 653 và 665, sai số chuẩn khoảng 25. LSTM nhỉnh hơn một chút còn ViT thấp hơn một chút. Mỗi loại chỉ có 1 seed, nên chênh lệch dưới khoảng 2 sai số chuẩn là chưa kết luận được. Ở ngân sách này, cả màu sắc lẫn kiến trúc đều không phá được điểm nghẽn."

## 10. Train tiếp dài hạn [6:30]
"Thứ phá được điểm nghẽn là sự kiên nhẫn. Tụi em train LSTM và ViT theo từng giai đoạn, tổng 13 triệu bước, với phạt mỗi bước nhỏ 0,05 và learning rate thấp hơn. Khác với CNN, chúng tiếp tục cải thiện.
Trong 300 ván cuối, ViT trung bình 1125 với ván cao nhất 4720, LSTM 900, còn các CNN khoảng 650. Lưu ý: các run này **không** cùng điều kiện với CNN 1 triệu bước, nên đây không phải so sánh kiến trúc sạch."

## 11. Hành vi thay đổi thế nào? [7:15]
"Có phải chỉ do sống lâu hơn không? Không. Sau khoảng 8 triệu bước, ViT ăn khoảng 1,06 ma mỗi ván, tăng từ 0,6, và 1,58 viên năng lượng mỗi ván, tăng từ 1,0. LSTM vẫn ở 0,6. Vì ma đáng 200 đến 1600 điểm, việc số ma ăn được gần gấp đôi giải thích phần lớn mức tăng điểm. Vậy agent đã bắt đầu biết dùng viên năng lượng, nhưng chỉ sau hơn mười triệu bước."

## 12. Thất bại: sụp đổ [8:00]
"Giờ đến thất bại. Khi tụi em train tiếp ViT tốt nhất với penalty lớn hơn, phạt chết 5 và phạt mỗi bước lên tới 1, điểm huấn luyện tụt từ khoảng 1140 xuống 490 và không hồi phục. Run RGB với phạt mỗi bước 2 cũng hỏng theo cách tương tự.
Nhìn vào value loss: khoảng 1 đến 2 khi mọi thứ ổn, 15 khi phạt 0,5, gần 400 khi phạt 2.
Cách giải thích của tụi em, **chưa** được kiểm chứng bằng run đối chứng: khi mỗi bước đều bị trừ điểm và chết thì kết thúc episode, chết sớm có lợi bất cứ khi nào phạt chết nhỏ hơn tổng phạt bước mà agent kỳ vọng phải trả cho phần đời còn lại. Bài học thực tế: value loss tăng gấp 10 lần là cảnh báo sớm rẻ tiền rằng thang reward đang sai."

## 13. Đánh giá có nhiễu [9:00]
"Một lưu ý quan trọng cho mọi con số trong bài. Cùng một model xám được 828 điểm ở một lần chấm 20 ván và 608 ở lần khác. Cùng một checkpoint ViT, cùng seed, được 748, 660 rồi 499. Sai số chuẩn của trung bình 20 ván là 50 đến 100, nên chênh lệch dưới khoảng 150 điểm là không diễn giải được.
Vì vậy tụi em dựa vào hàng nghìn ván huấn luyện, gọi kết quả 1 seed là xu hướng, và coi 'ViT hơn LSTM' là một giả thuyết, vì hai bên khác learning rate, thời điểm chuyển giai đoạn và seed."

## 14. Kết luận [9:45]
"Điều tụi em khá chắc: train tiếp dài với learning rate thấp và phạt bước nhỏ đã đưa ViT lên 1125, chủ yếu nhờ ăn ma nhiều gấp khoảng hai lần; penalty lớn làm hại và có thể làm sụp đổ quá trình học; còn RGB hay curriculum penalty đều không cho cải thiện đo được.
Hạn chế: phần lớn chỉ có 1 seed, các giai đoạn bị lẫn nhiều thay đổi cùng lúc, và đánh giá chỉ 20 ván. Hướng tiếp theo: chạy ít nhất 3 seed với từ 50 ván đánh giá trở lên, so ViT và LSTM cùng ngân sách, chạy lại 13 triệu bước không có phạt bước, và thử thêm exploration bonus."

## 15. Đạo đức [10:45]
"Ba rủi ro. Một là reward bị thiết kế sai: chính cách tụi em 'chữa' việc đứng im đã khiến việc bỏ cuộc thành tối ưu; tụi em sẽ ghi lại các chỉ số hành vi mong muốn, chạy thử ngắn trước, và có người xem xét mỗi khi đổi reward. Hai là chi phí tính toán và bằng chứng: 23 run, khoảng 84 triệu bước; tụi em sẽ báo cáo seed và độ nhiễu, và chạy thử nhiều seed trước. Ba là agent chỉ dùng điểm ảnh có thể bị biến thành bot chơi game, nên tụi em chỉ chia sẻ code huấn luyện cho emulator và nêu rõ phạm vi sử dụng.
Em xin cảm ơn, và sẵn sàng nhận câu hỏi."

---

## Câu hỏi có thể gặp (trả lời ngắn)
- **Vì sao PPO mà không DQN?** PPO song song hóa tốt trên một GPU; tụi em có cài DQN nhưng chưa có kết quả trên Ms. Pac-Man nên không đưa ra kết luận nào về nó.
- **Vì sao RGB không giúp?** Có thể thông tin vẫn lấy được từ ảnh xám (ví dụ qua chuyển động của ma), hoặc agent chưa khai thác được ở ngân sách này; tụi em chưa chạy phép kiểm tra hành vi trực tiếp quanh ma choáng.
- **ViT có thật sự hơn LSTM không?** Chưa chứng minh được: khác learning rate, khác giai đoạn và chỉ có 1 seed.
- **Phạt mỗi bước có phải yếu tố giúp ích không?** Chưa biết: không có run 13 triệu bước không dùng phạt, nên không tách được khỏi tác dụng của việc train lâu hơn.
- **Đếm ma như thế nào?** Từ giá trị reward chính xác (200/400/800/1600); trái cây 200 điểm ở màn 2 có thể bị đếm nhầm là ma.
