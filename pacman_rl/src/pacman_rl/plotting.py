"""Vẽ đường học và kết quả đánh giá. Mọi hàm nhận `runs`: dict tên -> TrainResult (hoặc tuple (model, log, episodes)),
nên vẽ một hay nhiều thuật toán / seed đều cùng một cách."""
import matplotlib.pyplot as plt
import numpy as np

from .utils import moving_avg

COLORS = {"PPO": "tab:blue", "DQN": "tab:orange"}


MANY = 4  # từ số run này trở lên: bỏ chấm điểm thô, đường mảnh hơn, gộp chú thích thành một


def _color(name, i, n=10):
    if name in COLORS:
        return COLORS[name]
    return plt.get_cmap("tab10" if n <= 10 else "tab20")(i % (10 if n <= 10 else 20))


def _lw(n):
    return 1.6 if n < MANY else 1.1


def shared_legend(fig, axes, ncol=5):
    """Một chú thích chung dưới cùng figure thay cho chú thích lặp lại trên từng ô; trả về mép dưới (cho tight_layout)."""
    seen = {}
    for a in np.ravel(axes):
        leg = a.get_legend()
        if leg:
            leg.remove()
        for h, l in zip(*a.get_legend_handles_labels()):
            seen.setdefault(l, h)
    if not seen:
        return 0
    ncol = min(ncol, len(seen))
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=ncol, frameon=False, fontsize=9)
    return 0.035 * -(-len(seen) // ncol) + 0.02


def score_curve(episodes, w=20):
    """(bước, điểm trung bình trượt w ván) từ danh sách (bước, điểm)."""
    steps, scores = zip(*episodes)
    return (np.array(steps)[w - 1:] if len(scores) >= w else np.array(steps)), moving_avg(scores, w)


def events_per_bin(log, total=None, bin_size=100_000):
    """Từ tổng tích luỹ log["pellets"], log["ghosts"] (và "deaths" nếu có) -> số sự kiện trong mỗi khoảng bin_size bước.
    Thêm out["steps"]: số bước env thật sự nằm trong mỗi bin (bin cuối có thể ngắn hơn bin_size)."""
    total = total or log["step"][-1]
    edges = np.arange(0, total + bin_size, bin_size)
    out = {}
    for key in ("pellets", "ghosts", "deaths"):
        if key in log:
            cum = np.interp(edges, np.r_[0, log["step"]], np.r_[0, log[key]])
            out[key] = np.diff(cum)
    out["steps"] = np.diff(np.minimum(edges, log["step"][-1]))
    return edges[1:], out


def per_life(ev, key):
    """Số `key` ("steps", "pellets", "ghosts") trung bình mỗi lượt chơi (một mạng) trong từng bin; bin không ai chết -> NaN."""
    d = ev["deaths"]
    return np.divide(ev[key], d, out=np.full_like(d, np.nan, dtype=float), where=d > 0)


def plot_scores(runs, random_scores=None, w=20, ax=None, scatter=None):
    """Trung bình trượt w ván của điểm theo số bước env; scatter=None -> chỉ vẽ điểm thô từng ván khi ít run (< MANY)."""
    ax = ax or plt.gca()
    n = len(runs)
    scatter = n < MANY if scatter is None else scatter
    for i, (name, (_, _, eps)) in enumerate(runs.items()):
        if not eps:
            continue
        c = _color(name, i, n)
        if scatter:
            st, sc = zip(*eps)
            ax.scatter(st, sc, s=4, alpha=0.15, color=c)
        x, y = score_curve(eps, w)
        ax.plot(x, y, color=c, lw=_lw(n), label=name)
    if random_scores is not None:
        ax.axhline(np.mean(random_scores), ls="--", color="gray", label="Random")
    ax.set(title=f"Điểm (TB trượt {w} ván)", xlabel="Bước env", ylabel="Điểm Ms. Pacman")
    ax.legend()
    return ax


def plot_ranking(runs, last=50, ax=None):
    """Thanh ngang xếp hạng các run theo điểm TB `last` ván cuối (kèm std) - gọn khi có nhiều run."""
    ax = ax or plt.gca()
    n = len(runs)
    rows = [(name, np.mean([e[1] for e in eps[-last:]]), np.std([e[1] for e in eps[-last:]]), i)
            for i, (name, (_, _, eps)) in enumerate(runs.items()) if eps]
    rows.sort(key=lambda r: r[1])
    ax.barh([r[0] for r in rows], [r[1] for r in rows], xerr=[r[2] for r in rows],
            color=[_color(r[0], r[3], n) for r in rows], error_kw=dict(lw=0.8, capsize=2, alpha=0.6))
    ax.set(title=f"Điểm TB {last} ván cuối", xlabel="Điểm Ms. Pacman")
    return ax


def plot_metric(runs, key, title=None, ax=None):
    """Một chỉ số trong log theo số bước (vd "ent", "kl" của PPO; "loss", "q_mean", "eps" của DQN).
    Run nào không có chỉ số đó thì bỏ qua."""
    ax = ax or plt.gca()
    for i, (name, (_, log, _)) in enumerate(runs.items()):
        if key in log:
            ax.plot(log["step"], log[key], color=_color(name, i, len(runs)), lw=_lw(len(runs)), label=name)
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
        c, lw = _color(name, i, len(runs)), _lw(len(runs))
        axes[0].plot(x, ev["pellets"], color=c, lw=lw, label=name)
        axes[1].plot(x, ev["ghosts"], color=c, lw=lw, label=name)
        axes[2].plot(x, ev["ghosts"] / np.maximum(ev["pellets"], 1), color=c, lw=lw, label=name)
    k = f"{bin_size // 1000}k"
    axes[0].set(title=f"Viên năng lượng ăn được / {k} bước", xlabel="Bước env")
    axes[1].set(title=f"Ma ăn được / {k} bước", xlabel="Bước env")
    axes[2].set(title="Ma ăn được trên mỗi viên năng lượng", xlabel="Bước env")
    for a in axes:
        a.legend()
    return axes


def plot_training(runs, random_scores=None, save_path=None, show=True, bin_size=100_000):
    """Tổng hợp 2x3: điểm, entropy (PPO), bước sống / lượt, viên năng lượng / lượt, ma / lượt, ma trên mỗi viên.
    "Lượt" = một mạng (kết thúc khi chết); các biểu đồ theo lượt tính trong từng bin `bin_size` bước, giống dòng log của
    train_ppo. Run không có log["deaths"] (lưu từ bản cũ) thì bỏ qua ba biểu đồ theo lượt. Trả về Figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ax = axes.ravel()
    plot_scores(runs, random_scores, ax=ax[0])
    plot_metric(runs, "ent", "Entropy của policy (PPO)", ax=ax[1])
    for i, (name, (_, log, _)) in enumerate(runs.items()):
        if not log.get("step"):
            continue
        x, ev = events_per_bin(log, bin_size=bin_size)
        c, lw = _color(name, i, len(runs)), _lw(len(runs))
        if "deaths" in ev:
            for a, key in zip(ax[2:5], ("steps", "pellets", "ghosts")):
                a.plot(x, per_life(ev, key), color=c, lw=lw, label=name)
        ax[5].plot(x, ev["ghosts"] / np.maximum(ev["pellets"], 1), color=c, lw=lw, label=name)
    for a, title in zip(ax[2:], ("Bước sống mỗi lượt (mỗi mạng)", "Viên năng lượng ăn được mỗi lượt",
                                 "Ma ăn được mỗi lượt", "Ma ăn được trên mỗi viên năng lượng")):
        a.set(title=title, xlabel="Bước env")
    bottom = shared_legend(fig, ax)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def compare(names=None, directory="runs", random_scores=None, metrics=("ent",), algo=None, device="cpu",
            save_path=None, show=True, bin_size=100_000):
    """So sánh nhiều session: list tên (hoặc dict {tên: algo}); names=None thì lấy tất cả run trong `directory`. Nạp các run rồi vẽ plot_training,
    plot_events và từng chỉ số trong `metrics` (run không có chỉ số thì bỏ qua). Trả về dict tên -> TrainResult."""
    from .runs import load_runs
    runs = load_runs(directory, names, algo, device)
    plot_training(runs, random_scores, save_path, False, bin_size)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(runs) + 1.5))
    plot_ranking(runs, ax=ax)
    fig.tight_layout()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    plot_events(runs, bin_size=bin_size, axes=axes)
    fig.tight_layout(rect=(0, shared_legend(fig, axes), 1, 1))
    for key in metrics:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        plot_metric(runs, key, ax=ax)
        fig.tight_layout(rect=(0, shared_legend(fig, ax), 1, 1))
    if show:
        plt.show()
    return runs


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
