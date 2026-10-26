# Kiến trúc

## Các module phụ thuộc nhau thế nào

Mũi tên `A → B` nghĩa là A import B. Không có vòng import.

```
                 cli.py
                   │
      ┌────────────┼──────────────┬──────────────┐
      ▼            ▼              ▼              ▼
  plotting.py    runs.py      agents/        evaluation.py
      │            │   ┌────────┤ ppo.py ────────┤
      │            │   │        │ dqn.py         │
      │            ▼   ▼        │ buffers.py     │
      │          models/        │ common.py      │
      │            │            ▼                ▼
      └──────► utils.py ◄──── envs/ ◄────────────┘
                              (factory, wrappers, vector, rewards)
                              KHÔNG import torch
```

| Module | Trách nhiệm | Nội dung chính |
|---|---|---|
| `envs/factory.py` | tạo một env đã tiền xử lý | `make_env`, `ENV_ID`, `N_ACTIONS`, `OBS_SHAPE` |
| `envs/wrappers.py` | đổi tín hiệu kết thúc khi học, đánh dấu mất mạng | `EpisodicLife` |
| `envs/vector.py` | chạy nhiều env song song, xen kẽ với GPU | `PipelinedEnvs`, `pop_game_scores`, `pop_deaths` |
| `envs/rewards.py` | biến đổi reward, đếm sự kiện | `log_reward`, `count_events` |
| `models/networks.py` | mạng neural | `make_cnn`, `ActorCriticCNN`, `QNetwork` |
| `agents/ppo.py`, `dqn.py` | vòng lặp train | `PPOConfig` / `train_ppo`, `DQNConfig` / `train_dqn` |
| `agents/buffers.py` | replay buffer của DQN | `VecFrameReplayBuffer` |
| `agents/common.py` | kiểu kết quả chung | `TrainResult(model, log, episodes)` |
| `evaluation.py` | chấm điểm, xem chơi | `evaluate`, `watch`, `ppo_policy`, `dqn_policy`, `random_baseline` |
| `runs.py` | lưu / nạp lên đĩa | `save_run`, `load_run`, `load_model` |
| `plotting.py` | biểu đồ | `plot_training`, `plot_scores`, `plot_metric`, `plot_events`, `plot_eval` |
| `cli.py` | dòng lệnh | `train`, `eval`, `play`, `plot` |

## Luồng dữ liệu khi train

```
 16 process con (AsyncVectorEnv), chia 2 nhóm x 8
 ┌───────────────────────────────────────────────────────┐
 │ gym.make(ALE/MsPacman-v5)                             │
 │   → AtariPreprocessing (skip 4, xám, 84x84, noop 30)  │
 │   → EpisodicLife (mất mạng = done, ghi game_score,        │
 │                    life_lost)                          │
 │   → FrameStack(4)            → obs uint8 (4,84,84)    │
 └───────────────┬───────────────────────▲───────────────┘
   obs, reward thật, done, game_score, died │ action
                 ▼                       │
 PipelinedEnvs.recv(g)            PipelinedEnvs.send(g)
                 │                       ▲
                 ├─ count_events(reward thật) → log["pellets"], log["ghosts"]
                 ├─ died                     → log["deaths"]
                 ├─ game_score               → episodes [(bước, điểm thật)]
                 └─ cfg.reward_fn(reward) - cfg.death_penalty * died
                                             → reward dùng để học
                                  │
     PPO: B_rew → GAE → clipped loss       DQN: replay buffer → Bellman / Huber
                                  │
                         model (GPU) chọn action cho nhóm kế tiếp
```

Trong lúc CPU chạy emulator cho nhóm 0, GPU tính action cho nhóm 1, và ngược lại. Với DQN thì bước cập nhật mạng cũng chạy chồng lên lúc emulator đang chạy.

## Ba loại điểm, đừng lẫn

| | Ở đâu | Dùng để |
|---|---|---|
| reward thật (0, 10, 50, 200...) | `reward` trả về từ `envs.recv` | đếm sự kiện, báo cáo |
| reward học (`log1p(r/10)`, mặc định, trừ `death_penalty` khi mất mạng) | `cfg.reward_fn(reward) - cfg.death_penalty * died` | gradient |
| điểm cả ván 3 mạng | `info["game_score"]` → `episodes` | đường học, so sánh |

Đánh giá (`evaluation.py`) luôn chạy game đầy đủ 3 mạng, không `EpisodicLife`, và cộng reward thật.

## Lý do thiết kế và đánh đổi

**Config là dataclass, tham số chạy là keyword riêng.** `PPOConfig` / `DQNConfig` chỉ chứa siêu tham số, tức những gì quyết định kết quả và nên ghi vào báo cáo. Các thứ như `model`, `save_path`, `device`, `render`, `watch_every` là tham số của `train_*` vì chúng không đổi kết quả. Đổi lại, gọi hàm dài hơn bản notebook một chút: `train_ppo(PPOConfig(total_steps=...))` thay vì `train_ppo(total_steps)`.

**`TrainResult` là NamedTuple.** Viết `model, log, eps = train_ppo(...)` như notebook cũ vẫn chạy, mà đọc `result.episodes` cũng rõ nghĩa.

**`import pacman_rl` không kéo torch.** `__init__.py` nạp lười (PEP 562). Mỗi process con của `AsyncVectorEnv` chỉ cần `pacman_rl.envs`, nên 16 process không phải nạp torch, khởi động nhanh hơn và đỡ hàng GB RAM. Vì vậy **đừng import torch trong `envs/`**. `PipelinedEnvs` dùng `functools.partial(make_env, ...)` thay cho lambda để hàm tạo env pickle được theo tham chiếu.

**Định dạng file lưu giữ như notebook** (`{name}.pt`, `{name}_episodes.npy`, `{name}_log.npy`), nên `load_run("notebooks", "ms_pacman_ppo", "ppo")` nạp được kết quả đã train trước khi có lib. Nhược điểm: không lưu config và optimizer, nên chưa resume train chính xác được (xem mục bên dưới).

**Ctrl+C / nút Stop khi train** không làm mất kết quả: vòng lặp bắt `KeyboardInterrupt`, đóng env, lưu `.pt` rồi trả về `TrainResult` của phần đã chạy.

**Code train giữ nguyên từng dòng** so với notebook (chỉ đổi biến toàn cục thành `cfg.*` và `device`). Muốn tối ưu thì đổi sau, khi đã có lib để so trước và sau.

## Nên xem lại khi project lớn lên

- **Lưu config cùng run** (vd `{name}_config.json`) để biết run nào dùng siêu tham số gì khi chạy nhiều cấu hình / seed (TODO reward phạt theo thời gian cần 3 cấu hình x 3 seed).
- **Checkpoint đầy đủ** (model + optimizer + bước hiện tại + lịch lr) để train tiếp đúng chỗ sau khi máy treo. Hiện giờ `model=` chỉ nạp trọng số, optimizer và lịch lr bắt đầu lại từ đầu.
- **Reward có trạng thái** (phạt khi "đứng bí" N bước): `reward_fn` hiện là hàm thuần, chưa đủ. Xem [extending.md](extending.md).
- **Logger chung** (TensorBoard hoặc CSV) thay cho dict, nếu cần theo dõi run dài từ xa.
