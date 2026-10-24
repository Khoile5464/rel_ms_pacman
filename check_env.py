#!/usr/bin/env python
"""
check_env.py - kiem tra moi truong truoc khi chay du an.

Kiem tra hai phan cua repo:
  berkeley : UC-Berkeley-AI-Pacman-Project/reinforcement  (chi can thu vien chuan + tkinter)
  rl       : REL_Ms_Pacman                                 (DQN tren Atari, can torch, SB3, ale-py...)

Cach dung (chay bang chinh Python ban dinh dung de chay du an):
  python check_env.py              # kiem tra ca hai
  python check_env.py --berkeley   # chi Berkeley
  python check_env.py --rl         # chi repo nhom truong
  python check_env.py --no-smoke   # bo qua cac bai thu chay thuc te (nhanh hon)

Ma thoat: 0 = on, 1 = co muc FAIL. Script chi DOC, khong cai gi va khong sua file nao.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BERKELEY_DIR = ROOT / "UC-Berkeley-AI-Pacman-Project" / "reinforcement"
RL_DIR = ROOT / "REL_Ms_Pacman"

try:  # tranh loi ma hoa tren console Windows (cp1252)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

results = {"OK": 0, "WARN": 0, "FAIL": 0}


def report(level: str, title: str, detail: str = "") -> None:
    results[level] += 1
    print(f"  [{level:<4}] {title}")
    for line in detail.splitlines():
        print(f"         {line}")


def header(text: str) -> None:
    print(f"\n=== {text} ===")


def vtuple(version: str) -> tuple:
    nums = [int(n) for n in re.findall(r"\d+", version)[:3]]
    return tuple(nums + [0] * (3 - len(nums)))


def installed_version(dist: str):
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


# --------------------------------------------------------------------------
# Python
# --------------------------------------------------------------------------
def check_python(want_berkeley: bool, want_rl: bool) -> None:
    header("Python")
    v = sys.version_info
    print(f"  {sys.executable}")
    print(f"  Python {v.major}.{v.minor}.{v.micro}")
    if v < (3, 8):
        report("FAIL", "Python qua cu", "Can Python >= 3.8.")
        return
    report("OK", f"Python {v.major}.{v.minor}")
    if want_berkeley and v >= (3, 12):
        report(
            "WARN",
            "Autograder cua Berkeley khong chay tren Python >= 3.12",
            "autograder.py import 'imp' (bi go o 3.12), grading.py import 'cgi' (bi go o 3.13).\n"
            "pacman.py / gridworld.py van chay binh thuong; chi lenh autograder can Python 3.10/3.11.\n"
            "Windows: py -3.10 autograder.py --no-graphics",
        )
    if want_rl and v >= (3, 13):
        report(
            "WARN",
            "requirements.txt cua nhom truong ghim torch<2.4 -> chua co wheel cho Python 3.13",
            "Dung Python 3.10-3.12 de cai torch/ale-py theo requirements.",
        )


# --------------------------------------------------------------------------
# Berkeley
# --------------------------------------------------------------------------
def check_berkeley(smoke: bool) -> None:
    header("Berkeley Pacman (reinforcement/)")
    if not BERKELEY_DIR.is_dir():
        report("WARN", "Khong thay thu muc reinforcement/", str(BERKELEY_DIR))
        return

    try:
        import tkinter  # noqa: F401

        report("OK", "tkinter (can cho cua so Pacman/Gridworld/Crawler)")
    except Exception as exc:
        report(
            "FAIL",
            "tkinter khong import duoc -> khong mo duoc UI",
            f"{type(exc).__name__}: {exc}\n"
            "Windows: cai lai Python va tick 'tcl/tk and IDLE'. Ubuntu: sudo apt install python3-tk",
        )

    for name in ("optparse", "pickle", "random", "threading"):
        try:
            __import__(name)
        except Exception as exc:  # gan nhu khong xay ra
            report("FAIL", f"thu vien chuan '{name}'", str(exc))

    needed = [
        "pacman.py", "game.py", "util.py", "qlearningAgents.py", "valueIterationAgents.py",
        "featureExtractors.py", "learningAgents.py", "gridworld.py", "autograder.py",
    ]
    missing = [f for f in needed if not (BERKELEY_DIR / f).is_file()]
    if missing:
        report("FAIL", "Thieu file framework", ", ".join(missing))
    else:
        report("OK", "Du file framework / file sinh vien")
    if not (BERKELEY_DIR / "layouts" / "smallClassic.lay").is_file():
        report("FAIL", "Thieu layouts/smallClassic.lay")

    if smoke and not missing:
        _run(
            [sys.executable, "pacman.py", "-p", "ApproximateQAgent", "-a", "extractor=SimpleExtractor",
             "-x", "2", "-n", "3", "-l", "smallGrid", "-q"],
            cwd=BERKELEY_DIR,
            ok_title="Chay thu Pacman 3 van (khong do hoa): ApproximateQAgent chay duoc",
            fail_title="Chay thu pacman.py bi loi",
            timeout=120,
        )


def _run(cmd, cwd, ok_title, fail_title, timeout):
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        report("FAIL", fail_title, f"Qua {timeout}s.")
        return
    except Exception as exc:
        report("FAIL", fail_title, f"{type(exc).__name__}: {exc}")
        return
    if proc.returncode == 0:
        report("OK", ok_title)
    else:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-6:])
        report("FAIL", fail_title, tail)


# --------------------------------------------------------------------------
# Repo nhom truong (DQN)
# --------------------------------------------------------------------------
# (ten distribution pip, spec trong requirements.txt, bat buoc?, ghi chu)
# spec: ("==", "x.y.z") hoac (">=,<", "min", "max") hoac (">=", "min")
RL_PACKAGES = [
    ("stable-baselines3", ("==", "2.1.0"), True, "DQN"),
    ("gymnasium", ("==", "0.29.1"), True, "API moi truong"),
    ("ale-py", (">=,<", "0.9.0", "0.11.0"), True, "Atari emulator"),
    ("torch", (">=,<", "2.2.0", "2.4.0"), True, "mang no-ron"),
    ("numpy", (">=", "1.26.0"), True, ""),
    ("opencv-python", ("==", "4.8.1.78"), False, "Atari wrapper can cv2 de resize"),
    ("matplotlib", ("==", "3.8.2"), False, ""),
    ("pandas", ("==", "2.1.3"), False, ""),
    ("Pillow", (">=", "10.1.0"), False, ""),
    ("tqdm", ("==", "4.66.1"), False, "progress_bar=True khi train"),
    ("tensorboard", ("==", "2.14.1"), False, "tensorboard_log khi train"),
    ("streamlit", ("==", "1.28.1"), True, "dashboard"),
    ("plotly", ("==", "5.18.0"), False, "dashboard"),
    # Duoc import trong src/ nhung KHONG co trong requirements.txt:
    ("pygame", (">=", "2.0"), True, "tab Play Game / Pacman2.0  [thieu trong requirements.txt]"),
    ("psutil", (">=", "5.0"), True, "src/pacman_game_integration.py  [thieu trong requirements.txt]"),
]


def spec_ok(version: str, spec: tuple) -> bool:
    v = vtuple(version)
    if spec[0] == "==":
        return v == vtuple(spec[1])
    if spec[0] == ">=":
        return v >= vtuple(spec[1])
    if spec[0] == ">=,<":
        return vtuple(spec[1]) <= v < vtuple(spec[2])
    return True


def spec_text(spec: tuple) -> str:
    if spec[0] == ">=,<":
        return f">={spec[1]},<{spec[2]}"
    return f"{spec[0]}{spec[1]}"


# (ten module de import, ten distribution pip)
IMPORT_TARGETS = [
    ("torch", "torch"), ("gymnasium", "gymnasium"), ("ale_py", "ale-py"),
    ("stable_baselines3", "stable-baselines3"), ("cv2", "opencv-python"),
    ("streamlit", "streamlit"), ("pygame", "pygame"), ("psutil", "psutil"),
]

_IMPORT_PROBE = (
    "import importlib, json, sys\n"
    "out = {}\n"
    "for m in sys.argv[1:]:\n"
    "    try:\n"
    "        importlib.import_module(m); out[m] = None\n"
    "    except BaseException as e:\n"
    "        msg = str(e).strip().splitlines()[0] if str(e).strip() else ''\n"
    "        out[m] = (type(e).__name__ + ': ' + msg)[:300]\n"
    "print(json.dumps(out))\n"
)


def check_imports(versions: dict) -> dict:
    """Import thu tung goi da cai trong 1 tien trinh con. Tra ve {module: True/False}."""
    import json

    mods = [m for m, dist in IMPORT_TARGETS if versions.get(dist)]
    if not mods:
        return {}
    try:
        proc = subprocess.run([sys.executable, "-c", _IMPORT_PROBE, *mods], capture_output=True,
                              text=True, timeout=300, cwd=str(ROOT))
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        report("WARN", "Khong chay duoc bai thu import", f"{type(exc).__name__}: {exc}")
        return {}

    ok = {m: err is None for m, err in data.items()}
    bad = {m: err for m, err in data.items() if err}
    for m, err in bad.items():
        hint = ""
        if m == "gymnasium":
            hint = (f"\nDang cai gymnasium {versions.get('gymnasium')}, requirements.txt ghi ==0.29.1; "
                    "ban gymnasium moi co the khong hop voi phien ban Python/typing_extensions dang dung.")
        report("FAIL", f"CAI ROI NHUNG KHONG IMPORT DUOC '{m}'", err + hint)
    if not bad:
        report("OK", "Import thu thanh cong: " + ", ".join(data))
    return ok


def check_rl(smoke: bool) -> None:
    header("Repo nhom truong (REL_Ms_Pacman: DQN tren Atari)")
    if not RL_DIR.is_dir():
        report("WARN", "Khong thay thu muc REL_Ms_Pacman/", str(RL_DIR))
        return

    versions = {}
    to_install = []
    for dist, spec, required, note in RL_PACKAGES:
        ver = installed_version(dist)
        versions[dist] = ver
        suffix = f"  ({note})" if note else ""
        if ver is None:
            report("FAIL" if required else "WARN", f"{dist}: CHUA CAI, can {spec_text(spec)}{suffix}")
            to_install.append(f"{dist}{spec_text(spec) if spec[0] != '>=,<' else spec_text(spec)}")
        elif spec_ok(ver, spec):
            report("OK", f"{dist} {ver}{suffix}")
        else:
            report("WARN", f"{dist} {ver}: requirements.txt ghi {spec_text(spec)}{suffix}")

    # Quy tac tuong thich da biet (lay tu chu thich trong requirements.txt)
    g, a = versions.get("gymnasium"), versions.get("ale-py")
    if g and a and vtuple(a) >= (0, 11, 0) and vtuple(g) < (1, 0, 0):
        report("FAIL", "ale-py >= 0.11 can gymnasium >= 1.0", f"Dang co gymnasium {g}, ale-py {a}.")
    if versions.get("numpy") and vtuple(versions["numpy"]) >= (2, 0, 0) and versions.get("pandas"):
        report("WARN", "numpy >= 2.0: requirements.txt duoc viet cho numpy 1.x (pyarrow<17, pandas 2.1)")

    # GPU
    if versions.get("torch"):
        try:
            import torch

            if torch.cuda.is_available():
                report("OK", f"CUDA san sang: {torch.cuda.get_device_name(0)}")
            else:
                report(
                    "WARN",
                    "Khong co CUDA -> train se chay CPU (rat cham, 2M buoc ~ nhieu ngay)",
                    "config.py dat device='cuda'; tren may khong GPU can doi sang 'cpu' hoac 'auto'.",
                )
        except Exception as exc:
            report("FAIL", "import torch loi", f"{type(exc).__name__}: {exc}")

    # gym.register_envs chi co tu gymnasium >= 1.0, ma src/environment.py goi no
    if versions.get("gymnasium") and vtuple(versions["gymnasium"]) < (1, 0, 0):
        report(
            "WARN",
            "gymnasium < 1.0 khong co gym.register_envs()",
            "src/environment.py goi gym.register_envs(ale_py) -> se loi AttributeError.\n"
            "train.py / dashboard.py dung make_atari_env nen khong bi anh huong; chi 'python src/environment.py' bi.",
        )

    # Co goi (metadata) khong co nghia la import duoc -> thu import that.
    import_ok = {}
    if smoke:
        import_ok = check_imports(versions)

    # Thu tao moi truong Atari that (can ROM)
    if smoke and import_ok.get("gymnasium") and import_ok.get("ale_py"):
        code = (
            "import gymnasium as gym, ale_py\n"
            "env = gym.make('MsPacmanNoFrameskip-v4')\n"
            "obs, _ = env.reset(seed=0)\n"
            "env.step(env.action_space.sample())\n"
            "print(obs.shape, env.action_space)\n"
        )
        _run([sys.executable, "-c", code], cwd=ROOT,
             ok_title="Tao duoc moi truong MsPacmanNoFrameskip-v4 (ROM co san)",
             fail_title="Khong tao duoc MsPacmanNoFrameskip-v4",
             timeout=120)

    if smoke and import_ok.get("stable_baselines3") and import_ok.get("torch") and import_ok.get("gymnasium"):
        code = (
            "import numpy as np\n"
            "from gymnasium import spaces\n"
            "from stable_baselines3.dqn.policies import CnnPolicy\n"
            "p = CnnPolicy(spaces.Box(0,255,(4,84,84),np.uint8), spaces.Discrete(9), lambda _: 1e-4)\n"
            "print(sum(x.numel() for x in p.q_net.parameters()))\n"
        )
        _run([sys.executable, "-c", code], cwd=ROOT,
             ok_title="SB3 dung duoc CnnPolicy 4x84x84 -> 9 action (khop kien truc cua nhom truong)",
             fail_title="SB3 CnnPolicy loi",
             timeout=120)

    if to_install:
        print("\n  Goi y cai cac goi con thieu:")
        print(f"    {sys.executable} -m pip install -r {RL_DIR / 'requirements.txt'}")
        print(f"    {sys.executable} -m pip install pygame psutil")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Kiem tra moi truong cho du an Pacman RL")
    ap.add_argument("--berkeley", action="store_true", help="chi kiem tra phan Berkeley")
    ap.add_argument("--rl", action="store_true", help="chi kiem tra repo nhom truong")
    ap.add_argument("--no-smoke", action="store_true", help="bo qua cac bai thu chay thuc te")
    args = ap.parse_args()

    both = not (args.berkeley or args.rl)
    want_b, want_r = both or args.berkeley, both or args.rl
    smoke = not args.no_smoke

    print("check_env.py - kiem tra moi truong (chi doc, khong cai/sua gi)")
    check_python(want_b, want_r)
    if want_b:
        check_berkeley(smoke)
    if want_r:
        check_rl(smoke)

    header("Tong ket")
    print(f"  OK: {results['OK']}   WARN: {results['WARN']}   FAIL: {results['FAIL']}")
    if results["FAIL"]:
        print("  -> Co muc FAIL: sua truoc khi chay.")
        return 1
    if results["WARN"]:
        print("  -> Chay duoc, nhung doc cac muc WARN o tren.")
    else:
        print("  -> Moi truong san sang.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
