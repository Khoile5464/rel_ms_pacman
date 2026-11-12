# rel_ms_pacman

Huấn luyện agent học tăng cường (PPO, DQN) chơi **Ms. Pacman (Atari)** từ ảnh điểm ảnh. Đồ án môn **REL301m** (FPT University).

Dự án so sánh các hướng cải thiện PPO trên `ALE/MsPacman-v5`: reward shaping (`death_penalty`, `step_penalty`), đầu vào xám so với RGB, và kiến trúc mạng (CNN, CNN+LSTM, ViT). Kết quả, kể cả kết quả âm, được ghi lại đầy đủ trong báo cáo.

## Kết quả chính

- PPO cơ sở (CNN, ảnh xám) tăng nhanh rồi đi ngang quanh **~700 điểm** (random ≈ 200–280).
- Điểm train cao nhất ở giai đoạn C: ViT 1181, CNN+LSTM 967 (trung bình 100 ván cuối).
- RGB không khác xám trong A/B 3M bước. Phạt cố định cao (20/0.1) kém phạt thấp. Phạt tăng dần ngang phạt thấp.
- Chưa có bằng chứng rằng can thiệp nào đưa agent vượt khỏi vùng ~700–1000 điểm một cách đáng tin cậy. Ngân sách chỉ 1–5M bước và 1–2 seed, nên điểm đánh giá nhiễu rất lớn.
- DQN đã được cài đặt nhưng chưa có kết quả trên Ms. Pacman.

Chi tiết và các sự cố cấu hình: [Reports/nhat_ky_nghien_cuu.md](Reports/nhat_ky_nghien_cuu.md).

## Cấu trúc repo

| Thư mục | Nội dung |
|---|---|
| [pacman_rl/](pacman_rl/) | Thư viện chính: env/wrapper, mạng (CNN, LSTM, ViT), PPO, DQN, đánh giá, vẽ biểu đồ, CLI. Có [README riêng](pacman_rl/README.md) và tài liệu HTML đầy đủ ([pacman_rl/docs/html/index.html](pacman_rl/docs/html/index.html)). |
| [notebooks/](notebooks/) | Notebook thí nghiệm: `ms_pacman`, `ab_gray_vs_rgb`, `ppo_vit_lstm`, `penalty_curriculum`, `compare_best_models`, `demo_vit_best`... và các bài làm quen (CartPole, Acrobot, LunarLander, CarRacing). |
| [Reports/](Reports/) | Nhật ký nghiên cứu, báo cáo LaTeX ([report.tex](Reports/paper/report.tex), [report.pdf](Reports/paper/report.pdf)), slide, kịch bản thuyết trình, hình vẽ. |
| [docs/](docs/) | Ghi chú kỹ thuật (ví dụ `enhance_penalty_capsule.md`). |
| [REL_Ms_Pacman/](REL_Ms_Pacman/) | Bản DQN có giải thích, attention và giao diện game. |
| [UC-Berkeley-AI-Pacman-Project/](UC-Berkeley-AI-Pacman-Project/) | Bài tập Pacman của UC Berkeley (search, multi-agent). |

## Cài đặt

Yêu cầu: Python 3.10, GPU NVIDIA khuyến nghị (đã kiểm chứng trên Windows, torch 2.3.1+cu121).

```bash
# 1. Tạo môi trường (conda hoặc venv đều được)
conda create -n rel_pacman python=3.10
conda activate rel_pacman

# 2. Cài torch bản CUDA TRƯỚC, để pip không kéo bản CPU đè lên
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Cài thư viện (chạy từ thư mục gốc repo)
pip install -e pacman_rl
```

Nếu muốn cài toàn bộ phụ thuộc của các phần khác (notebook, Streamlit...), dùng `pip install -r requirements-venv.txt`. Giữ `numpy<2`, vì torch 2.3 được build cho numpy 1.x.

## Bắt đầu nhanh

Dòng lệnh:

```bash
python -m pacman_rl train ppo --steps 2000000     # kết quả lưu vào runs/
python -m pacman_rl eval  ppo --episodes 10
python -m pacman_rl play  ppo --greedy            # xem agent chơi
python -m pacman_rl plot  ppo dqn --out curves.png
```

Trong notebook:

```python
from pacman_rl import PPOConfig, train_ppo, save_run, ppo_policy, watch, get_device

device = get_device()
result = train_ppo(PPOConfig(total_steps=2_000_000), save_path="ms_pacman_ppo.pt", device=device)
save_run(result, ".", "ms_pacman_ppo")
watch(ppo_policy(result.model, device), episodes=2)
```

## Phương pháp tóm tắt

- **Môi trường:** `ALE/MsPacman-v5`, 9 hành động, frame-skip 4, 4 khung xếp chồng 84×84 (xám `(4,84,84)` hoặc RGB `(12,84,84)`). `EpisodicLife` khi huấn luyện.
- **Reward khi học:** `r = ln(1 + r_thật/10) − death_penalty·[mất mạng] − step_penalty`. Điểm báo cáo luôn là điểm thật của game.
- **PPO:** 32 env song song × 512 bước/rollout, 4 epoch, 4 minibatch.
- **Mạng:** `ActorCriticCNN` (Nature DQN), `ActorCritic_LSTM_CNN`, `ActorCriticViT`, và `QNetwork` cho DQN.

## Test

```bash
pip install pytest
pytest pacman_rl
```

## Tài liệu

- Báo cáo: [Reports/paper/report.pdf](Reports/paper/report.pdf)
- Nhật ký nghiên cứu: [Reports/nhat_ky_nghien_cuu.md](Reports/nhat_ky_nghien_cuu.md)
- Tài liệu thư viện: [pacman_rl/docs/html/index.html](pacman_rl/docs/html/index.html)
- Hướng dẫn RL: `rl_learning_guide.html`

## Tác giả

Khoi Le (Khoile5464), môn REL301m, FPT University.
