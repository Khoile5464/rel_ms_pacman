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


def _merge(old_log, old_eps, log, episodes):
    """Nối lần train mới vào sau lần cũ: bước env của lần mới được cộng thêm offset (= bước cuối của lần cũ),
    các chỉ số tích luỹ (pellets, ghosts) cộng tiếp từ giá trị cuối của lần cũ."""
    old_steps = old_log.get("step") or []
    offset = max(old_steps[-1] if old_steps else 0, old_eps[-1][0] if len(old_eps) else 0)
    merged = {}
    for k in set(old_log) | set(log):
        a, b = list(old_log.get(k, [])), list(log.get(k, []))
        if k == "deaths" and not a:                       # run cũ chưa ghi số lần chết: đệm NaN cho khớp độ dài với step
            a = [float("nan")] * len(old_steps)
        if k == "step":
            b = [x + offset for x in b]
        elif k in ("pellets", "ghosts", "deaths") and a:
            base = a[-1] if a[-1] == a[-1] else 0         # NaN (run cũ không có) thì cộng tiếp từ 0
            b = [x + base for x in b]
        merged[k] = a + b
    eps = np.array(episodes, dtype=float).reshape(-1, 2)
    eps[:, 0] += offset
    return merged, np.concatenate([np.asarray(old_eps, dtype=float).reshape(-1, 2), eps])


def save_run(result, directory, name, append=True):
    """Lưu TrainResult (hoặc tuple (model, log, episodes)), trả về 3 đường dẫn đã ghi.

    Nếu run `name` đã tồn tại trong `directory` và append=True thì episodes / log của lần train này được NỐI vào
    sau kết quả cũ (bước env tiếp nối từ bước cuối của lần cũ); model luôn ghi đè bằng bản mới nhất.
    Run chưa tồn tại thì tạo mới. Đừng gọi save_run hai lần với cùng một result, sẽ bị nối trùng.
    append=False: ghi đè hoàn toàn."""
    model, log, episodes = result
    Path(directory).mkdir(parents=True, exist_ok=True)
    p_model, p_eps, p_log = run_paths(directory, name)
    eps = np.array(episodes).reshape(-1, 2)
    if append and (p_eps.exists() or p_log.exists()):
        old_eps = np.load(p_eps) if p_eps.exists() else np.empty((0, 2))
        old_log = np.load(p_log, allow_pickle=True).item() if p_log.exists() else {}
        log, eps = _merge(old_log, old_eps, log, episodes)
    torch.save(model.state_dict(), p_model)
    np.save(p_eps, eps)
    np.save(p_log, log, allow_pickle=True)
    return p_model, p_eps, p_log


def load_model(path, algo, device="cpu", n_actions=N_ACTIONS):
    """Nạp riêng model (algo = "ppo" hoặc "dqn"), để ở chế độ eval."""
    state = torch.load(path, map_location=device)
    in_channels = state["cnn.0.weight"].shape[1]       # 4 (xám) hoặc 12 (RGB), đọc thẳng từ checkpoint
    model = MODEL_CLASSES[algo](n_actions, in_channels=in_channels).to(device)
    model.load_state_dict(state)
    return model.eval()


def load_run(directory, name, algo, device="cpu"):
    """Nạp lại TrainResult. File log / episodes thiếu thì trả về rỗng (chỉ có checkpoint .pt)."""
    p_model, p_eps, p_log = run_paths(directory, name)
    model = load_model(p_model, algo, device)
    episodes = [tuple(x) for x in np.load(p_eps)] if p_eps.exists() else []
    log = np.load(p_log, allow_pickle=True).item() if p_log.exists() else {}
    return TrainResult(model, log, episodes)


def list_runs(directory):
    """Tên tất cả run trong thư mục (mỗi file {name}.pt là một run), sắp theo tên. Thư mục chưa có thì trả về []."""
    d = Path(directory)
    return sorted(p.stem for p in d.glob("*.pt")) if d.is_dir() else []


def load_runs(directory, names=None, algo=None, device="cpu"):
    """Nạp nhiều run một lượt -> dict tên -> TrainResult, đưa thẳng vào plot_*.

    names: list tên session, hoặc dict {tên: algo}; None thì lấy tất cả run trong `directory` (xem list_runs).
    algo không cho thì lấy `algo` chung, nếu cũng None thì đoán từ tên
    ("dqn" có trong tên -> dqn, ngược lại ppo). Chỉ nạp model qua load_run nên cần đủ file .pt."""
    if names is None:
        names = list_runs(directory)
    algos = names if isinstance(names, dict) else {}
    out = {}
    for name in names:
        a = algos.get(name) or algo or ("dqn" if "dqn" in name.lower() else "ppo")
        out[name] = load_run(directory, name, a, device)
    return out
