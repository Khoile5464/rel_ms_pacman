"""PPO (clipped objective + GAE) cho Ms. Pacman, siêu tham số theo thiết lập chuẩn cho Atari."""
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..envs import PipelinedEnvs, count_events, log_reward
from ..evaluation import ppo_policy, watch
from ..models import ActorCriticCNN
from ..utils import get_device, set_seed
from .common import TrainResult


@dataclass
class PPOConfig:
    total_steps: int = 2_000_000
    n_envs: int = 16              # số env song song (mỗi env một process)
    n_steps: int = 128            # số bước mỗi env trong một rollout (batch = n_envs x n_steps)
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
    save_every: int = 200         # lưu checkpoint mỗi bao nhiêu update (khi có save_path)
    reward_fn: Callable = log_reward   # reward thật (mảng n_envs) -> reward dùng để học


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
    envs = PipelinedEnvs(cfg.n_envs, render=render)
    obs = torch.as_tensor(envs.reset(seed=cfg.seed), device=device)

    model = (model or ActorCriticCNN(envs.n_actions)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    n_envs, n_steps = cfg.n_envs, cfg.n_steps
    batch = n_envs * n_steps
    mb_size = batch // cfg.n_minibatches
    n_updates = cfg.total_steps // batch
    B_obs = torch.zeros((n_steps, n_envs, 4, 84, 84), dtype=torch.uint8, device=device)
    B_act = torch.zeros((n_steps, n_envs), dtype=torch.long, device=device)
    B_logp = torch.zeros((n_steps, n_envs), device=device)
    B_rew = torch.zeros((n_steps, n_envs), device=device)
    B_done = torch.zeros((n_steps, n_envs), device=device)
    B_val = torch.zeros((n_steps, n_envs), device=device)

    ep_log = []
    log = {"step": [], "ent": [], "kl": [], "clipfrac": [], "v_loss": [], "pg_loss": [], "pellets": [], "ghosts": []}
    global_step, t0, n_pellet, n_ghost = 0, time.time(), 0, 0      # n_pellet / n_ghost: tổng số viên năng lượng / ma đã ăn
    print(f"PPO: {n_updates} updates x {batch} mẫu = {n_updates * batch:,} bước, {n_envs} env song song")

    def launch(t, g):
        sl = envs.slices[g]
        a, logp, value = model.act(obs[sl])
        B_obs[t, sl], B_act[t, sl], B_logp[t, sl], B_val[t, sl] = obs[sl], a, logp, value
        envs.send(g, a.cpu().numpy())

    try:
        for update in range(n_updates):
            for g in optimizer.param_groups:                   # giảm lr tuyến tính
                g["lr"] = cfg.lr * (1.0 - update / n_updates)

            # ---- Pha 1: thu thập rollout (2 nhóm env xen kẽ: GPU chọn action nhóm này khi CPU đang chạy nhóm kia) ----
            for g in range(envs.G):
                launch(0, g)
            for t in range(n_steps):
                for g in range(envs.G):
                    sl = envs.slices[g]
                    next_obs, reward, done, scores = envs.recv(g)
                    global_step += envs.gs
                    ep_log.extend((global_step, s) for s in scores)    # điểm thật của ván vừa hết 3 mạng
                    p, gh = count_events(reward); n_pellet += p; n_ghost += gh
                    B_rew[t, sl] = torch.as_tensor(cfg.reward_fn(reward), device=device)
                    B_done[t, sl] = torch.as_tensor(done, dtype=torch.float32, device=device)
                    obs[sl] = torch.as_tensor(next_obs, device=device)
                    if t + 1 < n_steps:
                        launch(t + 1, g)

            # ---- Pha 2: GAE ----
            with torch.no_grad():
                last_value = model(obs)[1]
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
            f_obs = B_obs.reshape(batch, 4, 84, 84)
            f_act, f_logp = B_act.reshape(-1), B_logp.reshape(-1)
            f_adv, f_ret = adv.reshape(-1), returns.reshape(-1)
            for _ in range(cfg.epochs):
                perm = torch.randperm(batch, device=device)
                for i in range(0, batch, mb_size):
                    idx = perm[i:i + mb_size]
                    dist, value = model(f_obs[idx])
                    logp = dist.log_prob(f_act[idx])
                    ratio = torch.exp(logp - f_logp[idx])
                    a = f_adv[idx]; a = (a - a.mean()) / (a.std() + 1e-8)

                    pg_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * a).mean()
                    v_loss = ((value - f_ret[idx]) ** 2).mean()
                    entropy = dist.entropy().mean()
                    loss = pg_loss + cfg.c1 * v_loss - cfg.c2 * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad)
                    optimizer.step()

            with torch.no_grad():                              # chỉ số của minibatch cuối, đủ để theo dõi xu hướng
                kl = (f_logp[idx] - logp).mean().item()
                cf = ((ratio - 1).abs() > cfg.clip_eps).float().mean().item()
            for k, v in zip(log, [global_step, entropy.item(), kl, cf, v_loss.item(), pg_loss.item(), n_pellet, n_ghost]):
                log[k].append(v)

            if watch_every and global_step // watch_every > (global_step - batch) // watch_every:
                print(f"[watch] step {global_step:,}")
                watch(ppo_policy(model, device), watch_episodes)

            if save_path and (update + 1) % cfg.save_every == 0:   # lưu giữa chừng, phòng treo máy khi chạy lâu
                torch.save(model.state_dict(), save_path)

            if (update + 1) % cfg.log_every == 0 or update == n_updates - 1:
                recent = [r for _, r in ep_log[-20:]]
                print(f"Update {update + 1}/{n_updates} | step {global_step:,} | "
                      f"avg score(20) {np.mean(recent) if recent else float('nan'):.0f} | "
                      f"viên NL {n_pellet:,} | ma {n_ghost:,} | entropy {entropy.item():.3f} | KL {kl:.4f} | clipfrac {cf:.3f} | "
                      f"{global_step / (time.time() - t0):.0f} steps/s")
    except KeyboardInterrupt:                                  # bấm Stop/Ctrl+C: vẫn lưu và trả về phần đã train
        print(f"Dừng sớm ở step {global_step:,}")
    finally:
        envs.close()
    if save_path:
        torch.save(model.state_dict(), save_path)
    return TrainResult(model, log, ep_log)
