"""Dòng lệnh: python -m pacman_rl <train|eval|play|plot> ...  (xem docs/usage.md)."""
import argparse
import dataclasses
import sys
from pathlib import Path


def _default_name(algo):
    return f"ms_pacman_{algo}"


def _parse_overrides(cfg_cls, pairs):
    """["lr=1e-4", "n_envs=8"] -> {"lr": 1e-4, "n_envs": 8}, ép kiểu theo giá trị mặc định của config."""
    fields = {f.name: f for f in dataclasses.fields(cfg_cls)}
    out = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        if key not in fields or key == "reward_fn":
            raise SystemExit(f"Không có tham số '{key}'. Hợp lệ: {', '.join(k for k in fields if k != 'reward_fn')}")
        default = fields[key].default
        out[key] = type(default)(float(value)) if isinstance(default, int) else type(default)(value)
    return out


def cmd_train(args):
    from .agents import ALGOS
    from .runs import save_run
    from .utils import describe_device, get_device

    cfg_cls, train = ALGOS[args.algo]
    overrides = _parse_overrides(cfg_cls, args.set)
    if args.steps:
        overrides["total_steps"] = args.steps
    overrides["seed"] = args.seed
    cfg = cfg_cls(**overrides)
    device = get_device()
    name = args.name or _default_name(args.algo)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("device:", describe_device(device), "| config:", {k: v for k, v in dataclasses.asdict(cfg).items() if k != "reward_fn"})

    result = train(cfg, save_path=out / f"{name}.pt", device=device, render=args.render, watch_every=args.watch_every)
    for p in save_run(result, out, name):
        print("Đã lưu:", p.resolve())


def _load(args):
    from .runs import load_model
    from .utils import get_device
    device = get_device()
    path = Path(args.dir) / f"{args.name or _default_name(args.algo)}.pt"
    return load_model(path, args.algo, device), device


def _policy(args, model, device):
    from .evaluation import dqn_policy, ppo_policy
    return ppo_policy(model, device, greedy=args.greedy) if args.algo == "ppo" else dqn_policy(model, device, eps=args.eps)


def cmd_eval(args):
    import numpy as np
    from .evaluation import evaluate
    model, device = _load(args)
    scores = evaluate(_policy(args, model, device), n_episodes=args.episodes, seed=args.seed)
    print(f"{args.algo.upper()} trên {args.episodes} ván: mean {np.mean(scores):.0f} | std {np.std(scores):.0f} | "
          f"max {np.max(scores):.0f}")
    print("Điểm:", [int(s) for s in scores])


def cmd_play(args):
    from .evaluation import watch
    model, device = _load(args)
    watch(_policy(args, model, device), episodes=args.episodes)


def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")                                      # chỉ lưu file, không mở cửa sổ
    from .plotting import plot_training
    from .runs import load_run
    runs = {}
    for spec in args.runs:                                     # "ppo:ms_pacman_ppo" hoặc "ppo"
        algo, _, name = spec.partition(":")
        runs[name or algo.upper()] = load_run(args.dir, name or _default_name(algo), algo)   # nhãn: tên run nếu có
    plot_training(runs, save_path=args.out, show=False)
    print("Đã lưu:", Path(args.out).resolve())


def build_parser():
    p = argparse.ArgumentParser(prog="pacman_rl", description="PPO vs DQN trên Ms. Pacman (Atari)")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train một thuật toán rồi lưu run")
    t.add_argument("algo", choices=["ppo", "dqn"])
    t.add_argument("--steps", type=int, help="tổng số bước env (mặc định theo config)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="runs", help="thư mục lưu (mặc định: runs/)")
    t.add_argument("--name", help="tên run (mặc định: ms_pacman_<algo>)")
    t.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="ghi đè config, vd --set lr=1e-4 n_envs=8")
    t.add_argument("--render", action="store_true", help="mở cửa sổ env số 0 (chậm)")
    t.add_argument("--watch-every", type=int, default=0, help="cứ mỗi N bước env cho agent chơi 1 ván để xem")
    t.set_defaults(func=cmd_train)

    for cmd, func, help_ in [("eval", cmd_eval, "chấm điểm model đã lưu"), ("play", cmd_play, "xem model đã lưu chơi")]:
        s = sub.add_parser(cmd, help=help_)
        s.add_argument("algo", choices=["ppo", "dqn"])
        s.add_argument("--dir", default="runs")
        s.add_argument("--name")
        s.add_argument("--episodes", type=int, default=10 if cmd == "eval" else 2)
        s.add_argument("--greedy", action="store_true", help="PPO: chọn action xác suất cao nhất thay vì lấy mẫu")
        s.add_argument("--eps", type=float, default=0.05, help="DQN: epsilon khi chơi")
        if cmd == "eval":
            s.add_argument("--seed", type=int, default=1000)
        s.set_defaults(func=func)

    pl = sub.add_parser("plot", help="vẽ đường học từ các run đã lưu ra file PNG")
    pl.add_argument("runs", nargs="+", metavar="ALGO[:NAME]", help="vd: ppo dqn  hoặc  ppo:ms_pacman_ppo_seed1")
    pl.add_argument("--dir", default="runs")
    pl.add_argument("--out", default="curves.png")
    pl.set_defaults(func=cmd_plot)
    return p


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):                    # console Windows mặc định cp1252, không in được tiếng Việt
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    args.func(args)
