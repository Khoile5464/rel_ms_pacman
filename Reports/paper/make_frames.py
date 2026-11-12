"""Captures Ms. Pac-Man frames (RGB, gray, 84x84) for the slides. Run from repo root."""
import numpy as np, gymnasium as gym, ale_py
from PIL import Image
gym.register_envs(ale_py)
env = gym.make("ALE/MsPacman-v5", frameskip=4, repeat_action_probability=0.0)
frames = {}
for seed in range(40):
  env.reset(seed=seed); rng = np.random.default_rng(seed)
  for t in range(3000):
    o, r, term, trunc, _ = env.step(int(rng.integers(9)))
    if r == 50 and "pre" not in frames: frames["pre"] = o.copy(); eat_t = t
    if "pre" in frames and "fright" not in frames and t > eat_t + 8: frames["fright"] = o.copy()
    if t == 60 and "early" not in frames: frames["early"] = o.copy()
    if term or trunc or "fright" in frames: break
  if "fright" in frames: break
for k, f in frames.items():
    Image.fromarray(f).resize((f.shape[1]*3, f.shape[0]*3), Image.NEAREST).save(f"Reports/paper/figures/frame_{k}_rgb.png")
if "fright" in frames:
    f = frames["fright"]; g = np.array(Image.fromarray(f).convert("L"))
    Image.fromarray(g).resize((g.shape[1]*3, g.shape[0]*3), Image.NEAREST).save("Reports/paper/figures/frame_fright_gray.png")
    Image.fromarray(f).convert("L").resize((84, 84), Image.BILINEAR).resize((336, 336), Image.NEAREST).save("Reports/paper/figures/frame_obs84_gray.png")
print(list(frames), t)
