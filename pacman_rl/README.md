# pacman_rl

PPO và DQN chơi **Ms. Pacman (Atari)**. Code được tách từ `notebooks/ms_pacman.ipynb` thành thư viện, hành vi giữ nguyên (cùng tiền xử lý, mạng, siêu tham số, định dạng file lưu).

```
pacman_rl/
├── pyproject.toml
├── README.md               ← bạn đang đọc
├── docs/
│   ├── html/               TÀI LIỆU ĐẦY ĐỦ (mở docs/html/index.html bằng trình duyệt)
│   ├── architecture.md     cấu trúc module, luồng dữ liệu, lý do thiết kế
│   ├── usage.md            dùng trong notebook và dòng lệnh, API từng hàm
│   └── extending.md        thêm reward shaping, thuật toán, chỉ số log
├── src/pacman_rl/
│   ├── envs/               tạo env, wrapper, env song song, reward (không import torch)
│   ├── models/             CNN Nature DQN, ActorCriticCNN (PPO), QNetwork (DQN)
│   ├── agents/             train_ppo, train_dqn, replay buffer, config
│   ├── evaluation.py       chấm điểm, xem agent chơi, policy
│   ├── runs.py             lưu / nạp kết quả train
│   ├── plotting.py         vẽ đường học, sự kiện, đánh giá
│   ├── cli.py              python -m pacman_rl ...
│   └── utils.py            device, seed, trung bình trượt
└── tests/                  test nhanh, không cần emulator
```

## Cài đặt

Dùng env conda `rel_pacman` (Python 3.10, torch 2.3.1+cu121, gymnasium 0.29.1, numpy 1.26). Chạy từ thư mục `rel_ms_pacman/`:

```bash
conda activate rel_pacman
pip install -e pacman_rl --no-deps     # -e: sửa code trong src/ là có hiệu lực ngay, không cần cài lại
```

Env mới thì bỏ `--no-deps`. Torch phải cài riêng bản CUDA **trước**:
`pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121`

## Bắt đầu nhanh

Trong notebook:

```python
from pacman_rl import PPOConfig, train_ppo, save_run, load_run, ppo_policy, watch, get_device
from pacman_rl.plotting import plot_training

device = get_device()
result = train_ppo(PPOConfig(total_steps=2_000_000), save_path="ms_pacman_ppo.pt", device=device)
save_run(result, ".", "ms_pacman_ppo")                 # .pt + _episodes.npy + _log.npy

result = load_run(".", "ms_pacman_ppo", "ppo", device)  # sau khi restart kernel
plot_training({"PPO": result})
watch(ppo_policy(result.model, device), episodes=2)
```

Dòng lệnh:

```bash
python -m pacman_rl train ppo --steps 2000000              # lưu vào runs/
python -m pacman_rl eval  ppo --episodes 10
python -m pacman_rl play  ppo --greedy
python -m pacman_rl plot  ppo dqn --out curves.png
```

Tài liệu đầy đủ (cài đặt, setup huấn luyện, ý nghĩa từng tham số, hướng dẫn đóng góp thuật toán mới): mở [docs/html/index.html](docs/html/index.html) bằng trình duyệt. Các file `.md` trong `docs/` là bản ngắn cũ.

## Test

```bash
pip install pytest
pytest pacman_rl
```
