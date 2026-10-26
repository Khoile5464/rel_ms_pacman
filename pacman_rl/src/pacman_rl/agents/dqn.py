"""DQN gốc (replay buffer + target network, không Double/Dueling/PER) cho Ms. Pacman."""
import copy
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..envs import PipelinedEnvs, count_events, log_reward
from ..evaluation import dqn_policy, watch
from ..models import QNetwork
from ..utils import get_device, set_seed
from .buffers import VecFrameReplayBuffer
from .common import TrainResult


@dataclass
class DQNConfig:
    total_steps: int = 10_000_000
    n_envs: int = 16              # mỗi vòng lặp thu n_envs chuyển tiếp
    buffer_size: int = 200_000
    batch_size: int = 128
    updates_per_iter: int = 1     # 128 mẫu / 16 chuyển tiếp = tỷ lệ tái sử dụng 8 như DQN gốc (batch 32 mỗi 4 bước)
    lr: float = 1e-4
    gamma: float = 0.99
    learning_starts: int = 20_000 # số bước thu thập trước khi học
    target_update: int = 10_000   # bước env
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_frac: float = 0.05        # epsilon giảm tuyến tính trong eps_frac đầu tổng số bước
    max_grad: float = 10.0
    seed: int = 0
    log_every: int = 100_000      # bước env
    save_every: int = 500_000     # bước env
    reward_fn: Callable = log_reward   # reward thật (mảng n_envs) -> reward dùng để học
    death_penalty: float = 0.0         # trừ điểm khi mất mạng (mất mạng = done)


def train_dqn(cfg=None, *, model=None, save_path=None, device=None, render=False,
              watch_every=0, watch_episodes=1):
    """Train DQN, trả về TrainResult(model, log, episodes). Tham số giống train_ppo."""
    cfg = cfg or DQNConfig()
    device = device or get_device()
    set_seed(cfg.seed)
    envs = PipelinedEnvs(cfg.n_envs, render=render)
    obs = envs.reset(seed=cfg.seed)
    n_envs, n_actions = cfg.n_envs, envs.n_actions

    q = (model or QNetwork(n_actions)).to(device)
    q_target = copy.deepcopy(q).eval()
    optimizer = optim.Adam(q.parameters(), lr=cfg.lr)
    buffer = VecFrameReplayBuffer(cfg.buffer_size, n_envs, device)

    eps_steps = int(cfg.eps_frac * cfg.total_steps)
    ep_log, first = [], np.ones(n_envs, dtype=bool)                # first[e]: obs hiện tại của env e là frame đầu episode
    pend = [None] * envs.G                                         # (obs, action, is_first) đã gửi đi, chờ kết quả
    trans = [None] * envs.G                                        # chuyển tiếp đã hoàn tất của vòng này
    log = {"step": [], "loss": [], "q_mean": [], "eps": [], "pellets": [], "ghosts": [], "deaths": []}
    n_pellet = n_ghost = n_death = 0                               # tổng số viên năng lượng / ma đã ăn / mạng đã mất
    last_log_step, last_log_death = 0, 0                           # mốc của lần in trước, để tính số bước mỗi mạng trong cửa sổ gần nhất
    loss_acc, q_acc, n_acc = torch.zeros((), device=device), torch.zeros((), device=device), 0   # cộng dồn trên GPU, đỡ sync
    t0 = time.time()
    n_iters = cfg.total_steps // n_envs
    next_target, next_log, next_save, next_watch = cfg.target_update, cfg.log_every, cfg.save_every, watch_every
    step = 0
    print(f"DQN: {n_iters * n_envs:,} bước, {n_envs} env song song, buffer {cfg.buffer_size:,}, "
          f"epsilon giảm trong {eps_steps:,} bước đầu")

    def launch(g, eps):
        """Chọn action epsilon-greedy cho nhóm g rồi gửi đi, emulator chạy trong khi GPU làm việc khác."""
        sl = envs.slices[g]
        with torch.no_grad():
            greedy = q(torch.as_tensor(obs[sl], device=device)).argmax(1).cpu().numpy()
        action = np.where(np.random.random(envs.gs) < eps, np.random.randint(0, n_actions, size=envs.gs), greedy)
        pend[g] = (obs[sl].copy(), action, first[sl].copy())
        envs.send(g, action)

    try:
        for g in range(envs.G):
            launch(g, cfg.eps_start)
        for it in range(1, n_iters + 1):
            step = it * n_envs
            eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * step / max(eps_steps, 1))
            for g in range(envs.G):                            # nhận kết quả nhóm g rồi gửi ngay action kế tiếp
                sl = envs.slices[g]
                next_obs, reward, done, scores, died = envs.recv(g)
                r = cfg.reward_fn(reward) - cfg.death_penalty * died
                ep_log.extend((step, s) for s in scores)       # điểm thật của ván vừa hết 3 mạng
                p, gh = count_events(reward); n_pellet += p; n_ghost += gh
                n_death += int(died.sum())
                p_obs, p_act, p_first = pend[g]
                trans[g] = (p_obs, p_act, r, done.astype(np.float32), p_first)
                obs[sl], first[sl] = next_obs, done
                launch(g, eps)
            buffer.add(*[np.concatenate(x) for x in zip(*trans)])  # ghi cả n_envs env của bước này cùng lúc

            # bước cập nhật mạng chạy trong lúc emulator đang chạy bước kế tiếp
            if step >= cfg.learning_starts:
                for _ in range(cfg.updates_per_iter):
                    s, a, r, s2, d = buffer.sample(cfg.batch_size)
                    with torch.no_grad():
                        y = r + cfg.gamma * (1 - d) * q_target(s2).max(1)[0]
                    q_sa = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                    loss = nn.functional.smooth_l1_loss(q_sa, y)
                    optimizer.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), cfg.max_grad)
                    optimizer.step()
                    loss_acc += loss.detach(); q_acc += q_sa.detach().mean(); n_acc += 1

            if step >= next_target:
                q_target.load_state_dict(q.state_dict()); next_target += cfg.target_update

            if watch_every and step >= next_watch:
                next_watch += watch_every
                print(f"[watch] step {step:,}")
                watch(dqn_policy(q, device), watch_episodes)

            if save_path and step >= next_save:                # lưu giữa chừng, phòng treo máy khi chạy lâu
                torch.save(q.state_dict(), save_path); next_save += cfg.save_every

            if step >= next_log:
                next_log += cfg.log_every
                recent = [r for _, r in ep_log[-20:]]
                log["step"].append(step); log["eps"].append(eps); log["pellets"].append(n_pellet); log["ghosts"].append(n_ghost); log["deaths"].append(n_death)
                d_death = n_death - last_log_death
                steps_per_life = (step - last_log_step) / d_death if d_death else float("inf")
                last_log_step, last_log_death = step, n_death
                log["loss"].append(loss_acc.item() / n_acc if n_acc else float("nan"))
                log["q_mean"].append(q_acc.item() / n_acc if n_acc else float("nan"))
                loss_acc.zero_(); q_acc.zero_(); n_acc = 0
                print(f"Step {step:,}/{cfg.total_steps:,} | avg score(20) {np.mean(recent) if recent else float('nan'):.0f} | "
                      f"viên NL {n_pellet:,} | ma {n_ghost:,} | chết {n_death:,} ({steps_per_life:.0f} bước/mạng) | eps {eps:.2f} | loss {log['loss'][-1]:.4f} | "
                      f"Q {log['q_mean'][-1]:.2f} | {step / (time.time() - t0):.0f} steps/s")

        for g in range(envs.G):                                # nhận nốt các bước đang chạy dở trước khi đóng env
            envs.recv(g)
    except KeyboardInterrupt:                                  # bấm Stop/Ctrl+C: vẫn lưu và trả về phần đã train
        print(f"Dừng sớm ở step {step:,}")
    finally:
        envs.close()
    if save_path:
        torch.save(q.state_dict(), save_path)
    return TrainResult(q, log, ep_log)
