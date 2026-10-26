# Hướng dẫn sử dụng

Mọi ví dụ giả sử đã `pip install -e pacman_rl` vào env `rel_pacman` (xem README).

## 1. Trong notebook

### Train

```python
from pacman_rl import PPOConfig, DQNConfig, train_ppo, train_dqn, save_run, get_device

device = get_device()                       # cuda nếu có, bật TF32 + cuDNN benchmark

ppo = train_ppo(PPOConfig(total_steps=2_000_000), save_path="ms_pacman_ppo.pt", device=device)
save_run(ppo, ".", "ms_pacman_ppo")

dqn = train_dqn(DQNConfig(total_steps=10_000_000), save_path="ms_pacman_dqn.pt", device=device)
save_run(dqn, ".", "ms_pacman_dqn")
```

- `save_path`: checkpoint `.pt` được ghi giữa chừng (PPO mỗi `save_every` update, DQN mỗi `save_every` bước) và khi kết thúc.
- Muốn xem trong lúc train: `render=True` (env số 0 mở cửa sổ, chạy tuần tự nên chậm) hoặc `watch_every=500_000` (cứ 500k bước mở 1 ván).
- Bấm Stop (Ctrl+C) giữa chừng: hàm vẫn lưu `.pt` và trả về phần đã train.
- Train tiếp từ model có sẵn: `train_ppo(cfg, model=result.model)`. Lưu ý optimizer và lịch lr bắt đầu lại.
- `save_run(result, dir, name, append=True)`: nếu run `name` đã tồn tại thì episodes / log được **nối vào sau** run cũ (bước env cộng offset để đường cong liền mạch, `pellets` / `ghosts` / `deaths` cộng tiếp; run cũ chưa có `deaths` thì đệm NaN), `.pt` ghi đè bằng model mới; chưa tồn tại thì tạo mới. `append=False` để ghi đè hoàn toàn. Đừng gọi hai lần với cùng một `result`, sẽ bị nối trùng.

Đổi siêu tham số:

```python
cfg = PPOConfig(total_steps=5_000_000, lr=1e-4, c2=0.02, seed=1)
```

### Nạp lại sau khi restart kernel

```python
from pacman_rl import load_run, load_model

ppo = load_run(".", "ms_pacman_ppo", "ppo", device)    # TrainResult(model, log, episodes)
model, log, episodes = ppo                             # unpack như tuple cũng được
only_model = load_model("ms_pacman_ppo.pt", "ppo", device)
```

Run cũ của notebook nằm ở `notebooks/` cùng tên file, nên nạp thẳng được.

### Vẽ

```python
from pacman_rl import random_baseline
from pacman_rl.plotting import plot_training, plot_scores, plot_metric, plot_events, plot_eval

runs = {"PPO": ppo, "DQN": dqn}                  # tên -> TrainResult; một hay nhiều run đều được
plot_training(runs, random_scores=random_baseline(5), save_path="curves.png")
plot_events(runs)                                # viên năng lượng / ma ăn được mỗi 100k bước
plot_metric(runs, "q_mean")                      # chỉ số bất kỳ trong log; run không có thì bỏ qua
```

Các hàm `plot_*` nhỏ nhận `ax=` (hoặc `axes=`) để ghép vào figure của bạn.

### Chấm điểm và xem chơi

```python
from pacman_rl import evaluate, watch, ppo_policy, dqn_policy
from pacman_rl.plotting import plot_eval

act_ppo = ppo_policy(ppo.model, device)              # greedy=True: luôn chọn action tốt nhất
act_dqn = dqn_policy(dqn.model, device, eps=0.05)

plot_eval({"Random": random_baseline(5),
           "PPO": evaluate(act_ppo, n_episodes=10),  # seed cố định 1000..1009, công bằng giữa các phương pháp
           "DQN": evaluate(act_dqn, n_episodes=10)})

watch(act_ppo, episodes=2)                           # mở cửa sổ game, cần màn hình
```

Policy chỉ là hàm `obs (4,84,84) uint8 -> int`, nên tự viết policy khác (luật cứng, mạng khác...) rồi đưa vào `evaluate` / `watch` được.

## 2. Dòng lệnh

Chạy được ở bất kỳ thư mục nào khi env `rel_pacman` đang bật. File lưu mặc định vào `runs/` của thư mục hiện tại.

```bash
python -m pacman_rl train ppo --steps 2000000
python -m pacman_rl train dqn --steps 10000000 --seed 1 --name dqn_seed1
python -m pacman_rl train ppo --set lr=1e-4 n_envs=8 c2=0.02   # ghi đè bất kỳ trường nào của config
python -m pacman_rl train ppo --watch-every 500000             # thỉnh thoảng mở 1 ván để xem

python -m pacman_rl eval ppo --episodes 10                     # mean / std / max trên seed cố định
python -m pacman_rl eval dqn --name dqn_seed1 --eps 0.01
python -m pacman_rl play ppo --greedy --episodes 3             # mở cửa sổ game

python -m pacman_rl plot ppo dqn --out curves.png
python -m pacman_rl plot ppo:ms_pacman_ppo --dir ../notebooks  # ALGO:NAME, nạp từ thư mục khác
```

Thêm `-h` sau mỗi lệnh để xem đủ tùy chọn. Lệnh `pacman-rl` (cài kèm vào `Scripts/` của env) tương đương `python -m pacman_rl`.

## 3. Bảng siêu tham số mặc định

**PPO** (`PPOConfig`)

| Trường | Mặc định | Ý nghĩa |
|---|---|---|
| `total_steps` | 2,000,000 | tổng bước env (1 bước = 4 frame) |
| `n_envs` | 16 | env song song, mỗi env một process |
| `n_steps` | 128 | bước mỗi env trong một rollout, batch = 2048 |
| `epochs`, `n_minibatches` | 4, 4 | minibatch 512 |
| `lr` | 2.5e-4 | giảm tuyến tính về 0 |
| `gamma`, `lam` | 0.99, 0.95 | GAE |
| `clip_eps` | 0.1 | |
| `c1`, `c2` | 0.5, 0.01 | value loss, entropy bonus |
| `max_grad` | 0.5 | clip grad norm |
| `log_every`, `save_every` | 50, 200 | tính theo **update** |
| `reward_fn` | `log_reward` | xem extending.md |

**DQN** (`DQNConfig`)

| Trường | Mặc định | Ý nghĩa |
|---|---|---|
| `total_steps` | 10,000,000 | |
| `n_envs` | 16 | |
| `buffer_size` | 200,000 | khoảng 1.4 GB RAM |
| `batch_size`, `updates_per_iter` | 128, 1 | |
| `lr`, `gamma` | 1e-4, 0.99 | |
| `learning_starts` | 20,000 | bước thu thập trước khi học |
| `target_update` | 10,000 | bước env |
| `eps_start`, `eps_end`, `eps_frac` | 1.0, 0.05, 0.05 | giảm trong 5% đầu |
| `max_grad` | 10 | |
| `log_every`, `save_every` | 100k, 500k | tính theo **bước env** |

## 4. Nội dung `log`

| Khoá | PPO | DQN |
|---|---|---|
| `step` | bước env sau mỗi update | bước env mỗi `log_every` |
| `pellets`, `ghosts` | tổng tích luỹ viên năng lượng / ma đã ăn | như PPO |
| `deaths` | tổng tích luỹ số mạng đã mất | như PPO |
| `ent`, `kl`, `clipfrac`, `v_loss`, `pg_loss` | của minibatch cuối mỗi update | |
| `loss`, `q_mean`, `eps` | | trung bình từ lần log trước |
