# Tăng step_penalty khi Pacman đang ở trạng thái power-up (ăn viên năng lượng)

Mục tiêu: bình thường phạt mỗi bước `step_low` (ví dụ 0.02). Từ lúc ăn viên năng lượng cho đến hết thời gian ma xanh thì phạt `step_high` (ví dụ 0.1-0.3), để agent tranh thủ săn ma trong cửa sổ có hạn.

## 1. Vì sao `reward_fn` hiện tại không đủ

- `reward_fn` chỉ nhận reward thật của **một bước** và không có bộ nhớ, nên không biết env đang ở power-up hay không.
- Ở `train_ppo` (`ppo.py`, dòng `r = cfg.reward_fn(reward) - cfg.death_penalty * died`) nó được gọi **theo từng nhóm env** (`envs.slices[g]`, mỗi nhóm `envs.gs = n_envs // 2` env, 2 nhóm xen kẽ), không phải cả `n_envs`.
- Vì vậy cần một đối tượng có state **theo từng env**, và biết nhóm nào đang được gọi.

## 2. Thiết kế

Giữ nguyên `reward_fn`. Thêm một trường mới trong `PPOConfig`, `penalty_fn`, trả về mảng penalty theo từng env, và `train_ppo` trừ nó vào reward.

```
r = reward_fn(reward) - death_penalty * died - penalty_fn(reward, died, sl)
```

Chữ ký: `penalty_fn(reward, died, sl) -> np.ndarray (gs,)`, trong đó:

| Tham số  | Ý nghĩa                                                         |
| -------- | --------------------------------------------------------------- |
| `reward` | reward thật (chưa scale) của nhóm hiện tại, shape `(gs,)`       |
| `died`   | cờ mất mạng của nhóm hiện tại, shape `(gs,)`                    |
| `sl`     | `envs.slices[g]`, để chọn đúng phần state của nhóm              |

Object có state nằm bên trong `penalty_fn`, `train_ppo` không cần biết state chứa gì.

## 3. Cài đặt

### 3.1 Thêm class vào `pacman_rl/src/pacman_rl/envs/rewards.py`

```python
class PowerAwarePenalty:
    """Phạt mỗi bước: step_low bình thường, step_high khi đang power-up.
    Power-up bắt đầu khi reward thật == PELLET_REWARD và kéo dài power_steps bước agent."""

    def __init__(self, n_envs, step_low=0.02, step_high=0.2, power_steps=90):
        self.step_low, self.step_high, self.power_steps = step_low, step_high, power_steps
        self.left = np.zeros(n_envs, dtype=np.int32)       # số bước power-up còn lại của từng env

    def reset(self):
        self.left[:] = 0

    def __call__(self, reward, died, sl):
        left = self.left[sl]                               # view, sửa tại chỗ là sửa self.left
        reward = np.asarray(reward)
        left[reward == PELLET_REWARD] = self.power_steps   # ăn viên năng lượng: nạp lại bộ đếm
        left[np.asarray(died) > 0] = 0                     # mất mạng: hết power-up
        penalty = np.where(left > 0, self.step_high, self.step_low).astype(np.float32)
        left[left > 0] -= 1
        return penalty
```

Lưu ý `self.left[sl]` với `sl` là `slice` trả về **view**, nên các phép gán/trừ trên `left` cập nhật thẳng vào `self.left`. Nếu đổi sang index bằng mảng thì phải gán ngược lại.

Export trong `pacman_rl/src/pacman_rl/envs/__init__.py` (và `__init__.py` của package nếu muốn import trực tiếp từ `pacman_rl`).

### 3.2 Sửa `PPOConfig` và `train_ppo` trong `agents/ppo.py`

```python
@dataclass
class PPOConfig:
    ...
    reward_fn: Callable = log_reward
    death_penalty: float = 0.0
    penalty_fn: Callable | None = None   # (reward, died, sl) -> penalty theo env, trừ vào reward
```

Trong `train_ppo`, trước vòng lặp:

```python
    if cfg.penalty_fn is not None and hasattr(cfg.penalty_fn, "reset"):
        cfg.penalty_fn.reset()           # tránh mang state của lần train trước sang
```

Trong pha 1, thay dòng tính `r`:

```python
    r = cfg.reward_fn(reward) - cfg.death_penalty * died
    if cfg.penalty_fn is not None:
        r = r - cfg.penalty_fn(reward, died, sl)
```

`sl = envs.slices[g]` đã có sẵn ở đầu vòng lặp nhóm.

### 3.3 Dùng trong notebook

```python
from pacman_rl.envs import log_reward, PowerAwarePenalty

n_envs = 32
cfg = PPOConfig(
    total_steps=2_000_000,
    n_envs=n_envs,
    n_steps=256,
    n_minibatches=4,
    death_penalty=20,
    reward_fn=log_reward,                                  # không còn partial step_penalty nữa
    penalty_fn=PowerAwarePenalty(n_envs, step_low=0.02, step_high=0.2, power_steps=90),
)
```

Lưu ý: `step_penalty` cũ được chuyển hết sang `penalty_fn`, đừng dùng cả hai kẻo bị phạt gấp đôi.

## 4. Chọn tham số

- **`power_steps`**: env dùng `frame_skip=4` (xem `factory.py`), game chạy 60 fps nên 1 bước agent bằng 4 frame, tức 15 bước/giây. Thời gian ma xanh trong Ms. Pacman ở màn 1 cỡ vài giây và ngắn dần theo màn, nên bắt đầu thử khoảng 60-100 bước. Đây là **ước lượng**, chưa đo trong ALE. Cách kiểm chứng: cho agent chơi, ghi lại số bước từ lúc ăn viên năng lượng đến lúc ma hết xanh (xem bằng `render=True`), rồi chỉnh.
- **`step_high`**: reward ma sau `log_reward` là 3.0-5.1, nên 0.02 gần như vô hình. Thử 0.1-0.3. Quá lớn thì agent lao vào ma bất chấp và chết nhiều. Theo dõi `deaths` và `ghosts` trong log để chỉnh.
- **`step_low`**: giữ như cũ (0.02).

## 5. Hạn chế

- Phát hiện power-up qua `reward == 50` là suy đoán từ reward, giống `count_events`. Hiếm khi trùng với sự kiện khác trong cùng 4 frame thì bị bỏ sót.
- Ăn ma không kết thúc power-up, chỉ hết giờ mới hết, nên bộ đếm không reset khi ăn ma.
- Mất mạng reset bộ đếm qua `died`. Cần kiểm tra `died` cũng được báo khi mất mạng cuối cùng (hết game); nếu không thì bộ đếm sẽ còn dư sang ván mới trong tối đa `power_steps` bước, sai số nhỏ.
- Nếu cần chính xác tuyệt đối, phải đọc RAM ALE để lấy bộ đếm power-up thật, phức tạp hơn nhiều.

## 6. Kiểm tra nhanh

```python
import numpy as np
from pacman_rl.envs import PowerAwarePenalty

p = PowerAwarePenalty(n_envs=4, power_steps=3)
sl = slice(0, 2)
print(p(np.array([0, 50]), np.zeros(2), sl))   # [0.02 0.2 ]  env 1 vào power-up
print(p(np.array([0, 0]),  np.zeros(2), sl))   # [0.02 0.2 ]
print(p(np.array([0, 0]),  np.zeros(2), sl))   # [0.02 0.2 ]
print(p(np.array([0, 0]),  np.zeros(2), sl))   # [0.02 0.02] hết power-up
print(p.left)                                   # [0 0 0 0], nhóm slice(2, 4) không bị ảnh hưởng
```
