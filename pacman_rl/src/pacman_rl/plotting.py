"""Vẽ đường học và kết quả đánh giá. Mọi hàm nhận `runs`: dict tên -> TrainResult (hoặc tuple (model, log, episodes)),
nên vẽ một hay nhiều thuật toán / seed đều cùng một cách."""
import matplotlib.pyplot as plt
import numpy as np

from .utils import moving_avg

COLORS = {"PPO": "tab:blue", "DQN": "tab:orange"}


def _color(name, i):
    return COLORS.get(name, f"C{i}")


def score_curve(episodes, w=20):
    """(bước, điểm trung bình trượt w ván) từ danh sách (bước, điểm)."""
    steps, scores = zip(*episodes)
    return (np.array(steps)[w - 1:] if len(scores) >= w else np.array(steps)), moving_avg(scores, w)


def events_per_bin(log, total=None, bin_size=100_000):
    """Từ tổng tích luỹ log["pellets"], log["ghosts"] -> số sự kiện trong mỗi khoảng bin_size bước."""
    total = total or log["step"][-1]
    edges = np.arange(0, total + bin_size, bin_size)
    out = {}
    for key in ("pellets", "ghosts"):
        cum = np.interp(edges, np.r_[0, log["step"]], np.r_[0, log[key]])
        out[key] = np.diff(cum)
    return edges[1:], out


def plot_scores(runs, random_scores=None, w=20, ax=None):
    """Điểm thật từng ván theo số bước env, kèm trung bình trượt w ván."""
    ax = ax or plt.gca()
    for i, (name, (_, _, eps)) in enumerate(runs.items()):
        if not eps:
            continue
        st, sc = zip(*eps)
        ax.scatter(st, sc, s=4, alpha=0.15, color=_color(name, i))
        x, y = score_curve(eps, w)
        ax.plot(x, y, color=_color(name, i), label=f"{name} (TB trượt {w} ván)")
    if random_scores is not None:
        ax.axhline(np.mean(random_scores), ls="--", color="gray", label="Random")
    ax.set(title="Điểm theo số bước tương tác", xlabel="Bước env", ylabel="Điểm Ms. Pacman")
    ax.legend()
    return ax


def plot_metric(runs, key, title=None, ax=None):
    """Một chỉ số trong log theo số bước (vd "ent", "kl" của PPO; "loss", "q_mean", "eps" của DQN).
    Run nào không có chỉ số đó thì bỏ qua."""
    ax = ax or plt.gca()
    for i, (name, (_, log, _)) in enumerate(runs.items()):
        if key in log:
            ax.plot(log["step"], log[key], color=_color(name, i), label=name)
    ax.set(title=title or key, xlabel="Bước env")
    ax.legend()
    return ax


def plot_events(runs, total=None, bin_size=100_000, axes=None):
    """3 biểu đồ: viên năng lượng / ma ăn được trên mỗi bin_size bước, và tỷ lệ ma trên viên năng lượng."""
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    for i, (name, (_, log, _)) in enumerate(runs.items()):
        if not log.get("step"):
            continue
        x, ev = events_per_bin(log, total, bin_size)
        c = _color(name, i)
        axes[0].plot(x, ev["pellets"], color=c, label=name)
        axes[1].plot(x, ev["ghosts"], color=c, label=name)
        axes[2].plot(x, ev["ghosts"] / np.maximum(ev["pellets"], 1), color=c, label=name)
    k = f"{bin_size // 1000}k"
    axes[0].set(title=f"Viên năng lượng ăn được / {k} bước", xlabel="Bước env")
    axes[1].set(title=f"Ma ăn được / {k} bước", xlabel="Bước env")
    axes[2].set(title="Ma ăn được trên mỗi viên năng lượng", xlabel="Bước env")
    for a in axes:
        a.legend()
    return axes


def plot_training(runs, random_scores=None, save_path=None, show=True):
    """Tổng hợp: điểm, entropy (PPO), sự kiện ăn được. Trả về Figure."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    plot_scores(runs, random_scores, ax=ax[0])
    plot_metric(runs, "ent", "Entropy của policy (PPO)", ax=ax[1])
    for i, (name, (_, log, _)) in enumerate(runs.items()):
        if log.get("step"):
            x, ev = events_per_bin(log)
            ax[2].plot(x, ev["ghosts"] / np.maximum(ev["pellets"], 1), color=_color(name, i), label=name)
    ax[2].set(title="Ma ăn được trên mỗi viên năng lượng", xlabel="Bước env"); ax[2].legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_eval(results, save_path=None, show=True):
    """Boxplot điểm đánh giá cuối. results: dict tên -> list điểm. Đồng thời in bảng mean/std/max."""
    print(f"{'Method':8s} {'mean':>8s} {'std':>8s} {'max':>8s}")
    for k, v in results.items():
        print(f"{k:8s} {np.mean(v):8.0f} {np.std(v):8.0f} {np.max(v):8.0f}")
    fig = plt.figure(figsize=(6, 4))
    plt.boxplot(list(results.values()), labels=list(results.keys()))
    plt.ylabel("Điểm"); plt.title("Đánh giá cuối")
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig
