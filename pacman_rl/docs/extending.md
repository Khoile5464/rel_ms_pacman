# Mở rộng

## Đổi reward khi học (reward shaping)

`PPOConfig.reward_fn` / `DQNConfig.reward_fn` nhận **mảng reward thật** của một nhóm env (độ dài `n_envs / 2`) và trả về reward dùng để học. Điểm báo cáo, số viên năng lượng và số ma vẫn tính từ reward thật, không bị ảnh hưởng.

Ví dụ phạt nhỏ mỗi bước (mục 1 trong `notebooks/TODO_ms_pacman.md`):

```python
from functools import partial
from pacman_rl import PPOConfig, train_ppo
from pacman_rl.envs import log_reward

def log_reward_with_step_penalty(r, penalty=0.01):
    return log_reward(r) - penalty

cfg = PPOConfig(reward_fn=partial(log_reward_with_step_penalty, penalty=0.02))
train_ppo(cfg, save_path="ppo_penalty.pt")
```

Nên đặt hàm ở cấp module (hoặc dùng `partial`), không dùng lambda, để sau này lưu config dễ hơn.

**Phạt khi "đứng bí"** (không có reward dương N bước liên tiếp) cần nhớ trạng thái theo từng env, nên hàm thuần như trên chưa đủ. Có hai cách:

1. **Wrapper env** (khuyên dùng): viết `StuckPenalty(gym.Wrapper)` trong `envs/wrappers.py`, đếm số bước không có reward dương và trả thêm `info["shaping"]`. Bọc nó trong `make_env` khi `episodic_life=True` (chỉ lúc học). Đánh giá không bị ảnh hưởng vì `evaluation.py` gọi `make_env()` không có `episodic_life`. Sau đó cho `PipelinedEnvs.recv` trả thêm phần shaping để train cộng vào.
2. **Lớp có trạng thái**: `reward_fn` là object có `__call__(reward)` và tự giữ bộ đếm. Chú ý mỗi lần gọi chỉ nhận một nhóm env (gọi xen kẽ nhóm 0, nhóm 1), và cần biết env nào vừa `done` để reset bộ đếm. Cách này rối hơn cách 1.

So sánh cấu hình: train mỗi cấu hình với `--seed 0/1/2` và `--name` khác nhau, rồi vẽ chung:

```bash
python -m pacman_rl plot ppo:base_s0 ppo:penalty_s0 --out compare.png    # nhãn = tên run
```

Trong notebook: `plot_training({"base": r1, "penalty": r2})`.

## Thêm một chỉ số vào log

Trong `train_ppo`, khoá của `log` và danh sách giá trị được `zip` theo thứ tự:

```python
log = {"step": [], "ent": [], ..., "ghosts": []}
...
for k, v in zip(log, [global_step, entropy.item(), ..., n_ghost]):
```

Thêm khoá mới vào cả hai chỗ, **cùng vị trí**. Sau đó `plot_metric(runs, "<khoá>")` vẽ được ngay.

## Thêm một thuật toán (vd Double DQN, A2C)

1. `agents/<tên>.py`: một dataclass config và hàm `train_<tên>(cfg=None, *, model=None, save_path=None, device=None, render=False, watch_every=0, watch_episodes=1) -> TrainResult`. Dùng `PipelinedEnvs`, ghi `log` với ít nhất `step`, `pellets`, `ghosts` để vẽ được bằng `plotting`.
2. Nếu cần mạng mới: thêm vào `models/networks.py` và vào `MODEL_CLASSES` trong `models/__init__.py` để `load_run` / `load_model` nạp được.
3. Đăng ký vào `ALGOS` trong `agents/__init__.py` và thêm tên vào `choices` trong `cli.py` để chạy được từ dòng lệnh.
4. Nếu cách chọn action khác: thêm `<tên>_policy` vào `evaluation.py`.

Double DQN chỉ khác một dòng tính đích, nên cũng có thể thêm `double: bool = False` vào `DQNConfig` thay vì tạo file mới:

```python
a2 = q(s2).argmax(1, keepdim=True)
y = r + cfg.gamma * (1 - d) * q_target(s2).gather(1, a2).squeeze(1)
```

## Quy ước

- `envs/` không import torch (xem architecture.md).
- Hàm train in log bằng `print`, không dùng tqdm, để chạy trong notebook hay terminal đều đọc được.
- Comment và docstring bằng tiếng Việt, theo phong cách notebook gốc.
- Thêm test vào `tests/` cho phần không cần emulator (reward, buffer, mạng, lưu/nạp).
