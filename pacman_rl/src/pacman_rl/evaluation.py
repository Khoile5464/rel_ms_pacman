"""Chấm điểm và xem agent chơi. Luôn chạy game đầy đủ 3 mạng và dùng điểm thật (không scale).

Một "policy" ở đây là hàm act_fn(obs: np.ndarray (4,84,84) hoặc (12,84,84) uint8) -> int.
"""
import random

import numpy as np
import torch

from .envs import make_env


def _is_rgb(model):
    """Mạng 12 kênh đầu vào = RGB, 4 kênh = xám; evaluate/watch dựa vào đó để dựng đúng env."""
    return model.cnn[0].in_channels == 12


def ppo_policy(model, device, greedy=False):
    """greedy=False: lấy mẫu từ policy (giống lúc train). greedy=True: luôn chọn action xác suất cao nhất."""
    def act(obs):
        with torch.no_grad():
            dist, _ = model(torch.as_tensor(obs, device=device).unsqueeze(0))
            return dist.probs.argmax(1).item() if greedy else dist.sample().item()
    act.rgb_scale = _is_rgb(model)
    return act


def dqn_policy(model, device, eps=0.05):
    """epsilon-greedy với eps nhỏ (tránh kẹt do greedy hoàn toàn)."""
    n_actions = model.head.out_features
    def act(obs):
        if random.random() < eps:
            return random.randrange(n_actions)
        with torch.no_grad():
            return model(torch.as_tensor(obs, device=device).unsqueeze(0)).argmax(1).item()
    act.rgb_scale = _is_rgb(model)
    return act


def random_policy(n_actions):
    return lambda obs: random.randrange(n_actions)


def run_episodes(act_fn, n_episodes=10, max_steps=4500, seed=None, render_mode=None, verbose=False, rgb_scale=None):
    """Chạy act_fn trên n_episodes ván, trả về danh sách điểm thật.
    seed=None: mỗi ván ngẫu nhiên; seed=k: ván thứ i dùng seed k+i (so sánh công bằng giữa các phương pháp).
    rgb_scale=None: lấy theo act_fn (policy từ ppo_policy / dqn_policy tự biết model dùng xám hay RGB)."""
    if rgb_scale is None:
        rgb_scale = getattr(act_fn, "rgb_scale", False)
    env = make_env(render_mode=render_mode, rgb_scale=rgb_scale); scores = []
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=None if seed is None else seed + ep)
            score, done, steps = 0.0, False, 0
            while not done and steps < max_steps:
                obs, r, te, tr, _ = env.step(act_fn(np.asarray(obs)))
                score += r; done = te or tr; steps += 1
            scores.append(score)
            if verbose:
                print(f"[play] ván {ep + 1}: {score:.0f} điểm ({steps} bước)")
    finally:
        env.close()
    return scores


def evaluate(act_fn, n_episodes=10, max_steps=4500, seed=1000):
    """Chấm điểm không mở cửa sổ, seed cố định."""
    return run_episodes(act_fn, n_episodes, max_steps, seed=seed)


def watch(act_fn, episodes=2, max_steps=4500):
    """Mở cửa sổ game (cần màn hình) để xem act_fn chơi, in điểm từng ván."""
    return run_episodes(act_fn, episodes, max_steps, render_mode="human", verbose=True)


def random_baseline(n_episodes=5, n_actions=9):
    """Điểm của agent ngẫu nhiên (để biết "không học gì" đạt bao nhiêu)."""
    return run_episodes(random_policy(n_actions), n_episodes, max_steps=10**9, seed=0)
