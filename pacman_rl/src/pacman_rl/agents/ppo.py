"""PPO (clipped objective + GAE) cho Ms. Pacman, siêu tham số theo thiết lập chuẩn cho Atari."""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..envs import PipelinedEnvs, count_events, log_reward
from ..evaluation import ppo_policy, watch
from ..models import PPO_ARCHS
from ..utils import get_device, set_seed
from .common import TrainResult


@dataclass
class PPOConfig:
    total_steps: int = 2_000_000
    n_envs: int = 16              # số env song song (mỗi env một process)
    # số bước mỗi env trong một rollout (batch = n_envs x n_steps)
    n_steps: int = 128
    epochs: int = 4               # số lần lặp trên mỗi rollout
    n_minibatches: int = 4
    lr: float = 2.5e-4            # giảm tuyến tính về 0
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.1
    c1: float = 0.5               # trọng số value loss
    c2: float = 0.01              # trọng số entropy bonus
    max_grad: float = 0.5
    seed: int = 0
    log_every: int = 50           # in log mỗi bao nhiêu update
    # lưu checkpoint mỗi bao nhiêu update (khi có save_path)
    save_every: int = 200
    # reward thật (mảng n_envs) -> reward dùng để học
    reward_fn: Callable = log_reward
    death_penalty: float = 0.0      # trừ điểm khi mất mạng (mất mạng = done)
    step_penalty: float = 0.0       # trừ mỗi bước env (đừng dùng cùng lúc với step_penalty trong reward_fn, sẽ bị phạt hai lần)
    # tuỳ chọn: hàm tiến độ (0 -> 1) -> (death_penalty, step_penalty), tính lại mỗi update.
    # None = dùng hai hằng số ở trên. Xem piecewise_penalty để dựng lịch trình tăng dần.
    penalty_schedule: Callable | None = None
    arch: str = "cnn"                  # "cnn" (Nature DQN), "vit" (Vision Transformer nhỏ) hoặc "lstm" (CNN + LSTM)
    rgb_scale: bool = False             # True: đầu vào RGB (12 kênh), False: xám (4 kênh)


def piecewise_penalty(points):
    """Dựng penalty_schedule nội suy tuyến tính giữa các mốc (tiến độ, death_penalty, step_penalty).
    Tiến độ = update / tổng update, trong [0, 1]; ngoài khoảng mốc thì giữ giá trị của mốc đầu / cuối.
    Ví dụ: piecewise_penalty([(0, 2.5, 0.0), (0.3, 2.5, 0.0), (0.6, 20, 0.1)])
    giữ 2.5 / 0 tới 30% đầu, tăng dần lên 20 / 0.1 ở 60%, rồi giữ nguyên."""
    pts = sorted(points)
    xs = [p[0] for p in pts]
    def schedule(progress):
        return tuple(float(np.interp(progress, xs, [p[i] for p in pts])) for i in (1, 2))
    return schedule


def train_ppo(cfg=None, *, model=None, save_path=None, device=None, render=False,
              watch_every=0, watch_episodes=1):
    """Train PPO, trả về TrainResult(model, log, episodes).

    model      : mạng có sẵn để train tiếp (mặc định tạo mới).
    save_path  : file .pt lưu state_dict giữa chừng và lúc kết thúc.
    render     : env số 0 mở cửa sổ game, chạy tuần tự (chậm) để quan sát agent.
    watch_every: cứ mỗi bấy nhiêu bước env thì mở cửa sổ cho agent chơi watch_episodes ván.
    """
    cfg = cfg or PPOConfig()
    device = device or get_device()
    set_seed(cfg.seed)
    if save_path:
        save_path = Path(save_path)
        # tạo thư mục từ đầu, tránh train xong mới lỗi khi lưu
        save_path.parent.mkdir(parents=True, exist_ok=True)
    envs = PipelinedEnvs(cfg.n_envs, render=render, rgb_scale=cfg.rgb_scale)
    obs = torch.as_tensor(envs.reset(seed=cfg.seed), device=device)
    obs_shape = tuple(obs.shape[1:])

    model = (model or PPO_ARCHS[cfg.arch](envs.n_actions, in_channels=obs_shape[0])).to(device)
    assert model.in_channels == obs_shape[0], \
        f"model nhận {model.in_channels} kênh nhưng env (rgb_scale={cfg.rgb_scale}) trả {obs_shape[0]} kênh"
    rec = getattr(model, "recurrent", False)             # mạng có LSTM: truyền trạng thái ẩn, minibatch theo env
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    n_envs, n_steps = cfg.n_envs, cfg.n_steps
    batch = n_envs * n_steps
    mb_size = batch // cfg.n_minibatches
    n_updates = cfg.total_steps // batch
    B_obs = torch.zeros((n_steps, n_envs, *obs_shape),
                        dtype=torch.uint8, device=device)
    B_act = torch.zeros((n_steps, n_envs), dtype=torch.long, device=device)
    B_logp = torch.zeros((n_steps, n_envs), device=device)
    B_rew = torch.zeros((n_steps, n_envs), device=device)
    B_done = torch.zeros((n_steps, n_envs), device=device)
    B_val = torch.zeros((n_steps, n_envs), device=device)
    if rec:
        state = model.initial_state(n_envs, device)       # (h, c) hiện tại của từng env, giữ qua các rollout
        prev_done = torch.zeros(n_envs, device=device)    # bước trước có kết thúc episode không (để reset state)
        B_pd = torch.zeros((n_steps, n_envs), device=device)

    ep_log = []
    log = {"step": [], "ent": [], "kl": [], "clipfrac": [],
           "v_loss": [], "pg_loss": [], "pellets": [], "ghosts": [], "deaths": [],
           "death_penalty": [], "step_penalty": []}
    # n_pellet / n_ghost / n_death: tổng số viên năng lượng / ma đã ăn / mạng đã mất
    global_step, t0, n_pellet, n_ghost, n_death = 0, time.time(), 0, 0, 0
    last = {"step": 0, "death": 0, "pellet": 0, "ghost": 0}              # mốc của lần in trước, để in số liệu riêng của cửa sổ log gần nhất
    print(
        f"PPO: {n_updates} updates x {batch} mẫu = {n_updates * batch:,} bước, {n_envs} env song song")

    def launch(t, g):
        sl = envs.slices[g]
        if rec:
            B_pd[t, sl] = prev_done[sl]
            a, logp, value, (h, c) = model.act(obs[sl], (state[0][sl], state[1][sl]), prev_done[sl])
            state[0][sl], state[1][sl] = h, c
        else:
            a, logp, value = model.act(obs[sl])
        B_obs[t, sl], B_act[t, sl], B_logp[t,
                                           sl], B_val[t, sl] = obs[sl], a, logp, value
        envs.send(g, a.cpu().numpy())

    try:
        for update in range(n_updates):
            for g in optimizer.param_groups:                   # giảm lr tuyến tính
                g["lr"] = cfg.lr * (1.0 - update / n_updates)
            death_pen, step_pen = (cfg.penalty_schedule(update / n_updates) if cfg.penalty_schedule
                                   else (cfg.death_penalty, cfg.step_penalty))

            if rec:                                            # state đầu rollout, để tính lại chuỗi lúc update
                h0, c0 = state[0].clone(), state[1].clone()
            # ---- Pha 1: thu thập rollout (2 nhóm env xen kẽ: GPU chọn action nhóm này khi CPU đang chạy nhóm kia) ----
            for g in range(envs.G):
                launch(0, g)
            for t in range(n_steps):
                for g in range(envs.G):
                    sl = envs.slices[g]
                    next_obs, reward, done, scores, died = envs.recv(g)
                    r = cfg.reward_fn(reward) - step_pen - death_pen * died
                    global_step += envs.gs
                    # điểm thật của ván vừa hết 3 mạng
                    ep_log.extend((global_step, s) for s in scores)
                    p, gh = count_events(reward)
                    n_pellet += p
                    n_ghost += gh
                    n_death += int(died.sum())
                    B_rew[t, sl] = torch.as_tensor(
                        r, device=device)
                    B_done[t, sl] = torch.as_tensor(
                        done, dtype=torch.float32, device=device)
                    if rec:
                        prev_done[sl] = B_done[t, sl]
                    obs[sl] = torch.as_tensor(next_obs, device=device)
                    if t + 1 < n_steps:
                        launch(t + 1, g)

            # ---- Pha 2: GAE ----
            with torch.no_grad():
                last_value = model.step(obs, state, prev_done)[1] if rec else model(obs)[1]
            adv = torch.zeros_like(B_rew)
            gae = torch.zeros(n_envs, device=device)
            for t in reversed(range(n_steps)):
                next_value = last_value if t == n_steps - 1 else B_val[t + 1]
                nonterm = 1.0 - B_done[t]
                delta = B_rew[t] + cfg.gamma * next_value * nonterm - B_val[t]
                gae = delta + cfg.gamma * cfg.lam * nonterm * gae
                adv[t] = gae
            returns = adv + B_val

            # ---- Pha 3: cập nhật PPO ----
            f_obs = B_obs.reshape(batch, *obs_shape)
            f_act, f_logp = B_act.reshape(-1), B_logp.reshape(-1)
            f_adv, f_ret = adv.reshape(-1), returns.reshape(-1)
            def minibatches():
                """Sinh (dist, value, act, logp_cũ, adv, ret). Mạng thường: xáo trộn mẫu. Mạng LSTM: chia theo env, giữ nguyên chuỗi thời gian."""
                if not rec:
                    perm = torch.randperm(batch, device=device)
                    for i in range(0, batch, mb_size):
                        idx = perm[i:i + mb_size]
                        dist, value = model(f_obs[idx])
                        yield dist, value, f_act[idx], f_logp[idx], f_adv[idx], f_ret[idx]
                else:
                    per = max(n_envs // cfg.n_minibatches, 1)
                    perm = torch.randperm(n_envs, device=device)
                    for i in range(0, n_envs, per):
                        e = perm[i:i + per]
                        dist, value = model.forward_seq(B_obs[:, e], (h0[e], c0[e]), B_pd[:, e])
                        yield (dist, value, B_act[:, e].reshape(-1), B_logp[:, e].reshape(-1),
                               adv[:, e].reshape(-1), returns[:, e].reshape(-1))

            for _ in range(cfg.epochs):
                for dist, value, m_act, m_logp, a, m_ret in minibatches():
                    logp = dist.log_prob(m_act)
                    ratio = torch.exp(logp - m_logp)
                    a = (a - a.mean()) / (a.std() + 1e-8)

                    pg_loss = -torch.min(ratio * a, torch.clamp(ratio,
                                         1 - cfg.clip_eps, 1 + cfg.clip_eps) * a).mean()
                    v_loss = ((value - m_ret) ** 2).mean()
                    entropy = dist.entropy().mean()
                    loss = pg_loss + cfg.c1 * v_loss - cfg.c2 * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad)
                    optimizer.step()

            with torch.no_grad():                              # chỉ số của minibatch cuối, đủ để theo dõi xu hướng
                kl = (m_logp - logp).mean().item()
                cf = ((ratio - 1).abs() > cfg.clip_eps).float().mean().item()
            for k, v in zip(log, [global_step, entropy.item(), kl, cf, v_loss.item(), pg_loss.item(), n_pellet, n_ghost, n_death, death_pen, step_pen]):
                log[k].append(v)

            if watch_every and global_step // watch_every > (global_step - batch) // watch_every:
                print(f"[watch] step {global_step:,}")
                watch(ppo_policy(model, device), watch_episodes)

            # lưu giữa chừng, phòng treo máy khi chạy lâu
            if save_path and (update + 1) % cfg.save_every == 0:
                torch.save(model.state_dict(), save_path)

            if (update + 1) % cfg.log_every == 0 or update == n_updates - 1:
                recent = [r for _, r in ep_log[-20:]]
                # mỗi lượt chơi (1 mạng) kết thúc bằng chết, nên số lượt = số lần chết trong cửa sổ
                lives = n_death - last["death"]
                d = max(lives, 1)                                  # chưa ai chết trong cửa sổ thì chia 1, số liệu vẫn là tổng của cả cửa sổ
                window = (f"{lives} lượt | {(global_step - last['step']) / d:.0f} bước/lượt | "
                          f"{(n_pellet - last['pellet']) / d:.2f} viên NL/lượt | {(n_ghost - last['ghost']) / d:.2f} ma/lượt")
                last.update(step=global_step, death=n_death, pellet=n_pellet, ghost=n_ghost)
                print(f"Update {update + 1}/{n_updates} | step {global_step:,} | "
                      f"avg score(20) {np.mean(recent) if recent else float('nan'):.0f} | "
                      f"{window} | phạt chết {death_pen:.1f} bước {step_pen:.3f} | entropy {entropy.item():.3f} | KL {kl:.4f} | clipfrac {cf:.3f} | "
                      f"{global_step / (time.time() - t0):.0f} steps/s")
    except KeyboardInterrupt:                                  # bấm Stop/Ctrl+C: vẫn lưu và trả về phần đã train
        print(f"Dừng sớm ở step {global_step:,}")
    finally:
        envs.close()
    if save_path:
        torch.save(model.state_dict(), save_path)
    return TrainResult(model, log, ep_log)
