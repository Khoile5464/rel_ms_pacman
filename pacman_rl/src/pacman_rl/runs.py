"""Lưu / nạp kết quả một lần train. Một "run" tên `name` trong thư mục `directory` gồm 3 file:

    {name}.pt             state_dict của model
    {name}_episodes.npy   mảng (N, 2): (bước env, điểm thật) của từng ván
    {name}_log.npy        dict log (pickle)

Trùng tên file với notebook cũ (ms_pacman_ppo.pt, ms_pacman_ppo_episodes.npy, ...) nên nạp lại được kết quả đã có.
"""
from pathlib import Path

import numpy as np
import torch

from .agents.common import TrainResult
from .envs import N_ACTIONS
from .models import MODEL_CLASSES


def run_paths(directory, name):
    d = Path(directory)
    return d / f"{name}.pt", d / f"{name}_episodes.npy", d / f"{name}_log.npy"


def save_run(result, directory, name):
    """Lưu TrainResult (hoặc tuple (model, log, episodes)), trả về 3 đường dẫn đã ghi."""
    model, log, episodes = result
    Path(directory).mkdir(parents=True, exist_ok=True)
    p_model, p_eps, p_log = run_paths(directory, name)
    torch.save(model.state_dict(), p_model)
    np.save(p_eps, np.array(episodes).reshape(-1, 2))
    np.save(p_log, log, allow_pickle=True)
    return p_model, p_eps, p_log


def load_model(path, algo, device="cpu", n_actions=N_ACTIONS):
    """Nạp riêng model (algo = "ppo" hoặc "dqn"), để ở chế độ eval."""
    model = MODEL_CLASSES[algo](n_actions).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()


def load_run(directory, name, algo, device="cpu"):
    """Nạp lại TrainResult. File log / episodes thiếu thì trả về rỗng (chỉ có checkpoint .pt)."""
    p_model, p_eps, p_log = run_paths(directory, name)
    model = load_model(p_model, algo, device)
    episodes = [tuple(x) for x in np.load(p_eps)] if p_eps.exists() else []
    log = np.load(p_log, allow_pickle=True).item() if p_log.exists() else {}
    return TrainResult(model, log, episodes)
