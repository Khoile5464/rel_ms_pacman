"""Chấm điểm và xem agent chơi. Luôn chạy game đầy đủ 3 mạng và dùng điểm thật (không scale).

Một "policy" ở đây là hàm act_fn(obs: np.ndarray (4,84,84) hoặc (12,84,84) uint8) -> int.
"""
import random

import numpy as np
import torch

from .envs import make_env


def _is_rgb(model):
    """Mạng 12 kênh đầu vào = RGB, 4 kênh = xám; evaluate/watch dựa vào đó để dựng đúng env."""
    return model.in_channels == 12


def ppo_policy(model, device, greedy=False):
    """greedy=False: lấy mẫu từ policy (giống lúc train). greedy=True: luôn chọn action xác suất cao nhất."""
    rec = getattr(model, "recurrent", False)
    mem = {}

    def reset():                                   # đầu mỗi ván: xoá trí nhớ LSTM
        mem["state"] = model.initial_state(1, device) if rec else None
    reset()

    def act(obs):
        with torch.no_grad():
            x = torch.as_tensor(obs, device=device).unsqueeze(0)
            if rec:
                dist, _, mem["state"] = model.step(x, mem["state"], torch.zeros(1, device=device))
            else:
                dist, _ = model(x)
            return dist.probs.argmax(1).item() if greedy else dist.sample().item()
    act.reset = reset
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
            getattr(act_fn, "reset", lambda: None)()
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


def watch_probs(model, device, greedy=True, eps=0.0, scale=3, max_steps=4500, delay_ms=30, seed=None):
    """Xem model PPO chơi kèm bảng xác suất 9 action ngay trên màn hình (cần màn hình + opencv).

    Khung game bên trái, bên phải là thanh xác suất của từng action; thanh xanh = action được chọn.
    greedy=True chọn action xác suất cao nhất, False lấy mẫu từ policy. eps>0: thêm xác suất eps chọn action
    ngẫu nhiên (tránh kẹt do greedy hoàn toàn). Phím q để thoát. Trả về điểm thật của ván.
    Mạng LSTM bị xoá trí nhớ khi mất mạng (giống lúc train, mất mạng = done)."""
    import cv2

    rec = getattr(model, "recurrent", False)
    env = make_env(render_mode="rgb_array", rgb_scale=_is_rgb(model))
    names = env.unwrapped.get_action_meanings()
    obs, info = env.reset(seed=seed)
    state = model.initial_state(1, device) if rec else None
    lives, score = info.get("lives"), 0.0
    try:
        for t in range(max_steps):
            with torch.no_grad():
                x = torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0)
                if rec:
                    dist, _, state = model.step(x, state, torch.zeros(1, device=device))
                else:
                    dist, _ = model(x)
            p = dist.probs[0].cpu().numpy()
            a = int(p.argmax()) if greedy else int(dist.sample().item())
            if eps and random.random() < eps:
                a = random.randrange(len(p))

            game = cv2.resize(env.render()[:, :, ::-1], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            panel = np.zeros((game.shape[0], 300, 3), np.uint8)
            for i, (nm, pi) in enumerate(zip(names, p)):
                y = 30 + i * 32
                color = (0, 200, 0) if i == a else (150, 150, 150)
                cv2.rectangle(panel, (110, y - 14), (110 + int(pi * 170), y + 6), color, -1)
                cv2.putText(panel, nm, (5, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(panel, f"{pi:.2f}", (110 + int(pi * 170) + 4, y + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(panel, f"score {score:.0f}  step {t}", (5, game.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("Ms. Pacman - xac suat action", np.hstack([game, panel]))
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                break

            obs, r, te, tr, info = env.step(a)
            score += r
            if rec and info.get("lives", lives) < lives:
                state = model.initial_state(1, device)
            lives = info.get("lives", lives)
            if te or tr:
                break
    finally:
        env.close()
        cv2.destroyAllWindows(); cv2.waitKey(1)
    print(f"[watch_probs] {score:.0f} điểm ({t + 1} bước)")
    return score


def random_baseline(n_episodes=5, n_actions=9):
    """Điểm của agent ngẫu nhiên (để biết "không học gì" đạt bao nhiêu)."""
    return run_episodes(random_policy(n_actions), n_episodes, max_steps=10**9, seed=0)
