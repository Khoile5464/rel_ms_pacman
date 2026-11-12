"""Regenerates the English figures and the numbers quoted in report.tex from notebooks/runs/*.npy.
Run from the repo root:  python Reports/paper/make_figures.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path("notebooks/runs"); OUT = Path("Reports/paper/figures"); OUT.mkdir(exist_ok=True, parents=True)
CUM = ("pellets", "ghosts", "deaths")
RANDOM = 274
PICKS = {
    "CNN (gray)": ["pen_const_low_s0"],
    "CNN (RGB)": ["rgb_pen_const_low_s0"],
    "CNN+LSTM": ["lstm_dp25_s0", "lstm_dp25_sp0.05_from_pre_s0", "lstm_dp25_sp0.05_ext5M_s0", "lstm_dp2_sp0.05_plus5M_from_A_s0"],
    "ViT": ["vit_dp25_s0", "vit_dp25_sp0.05_from_pre_s0", "vit_dp25_sp0.05_ext5M_s0", "vit_dp2_sp0.05_plus5M_from_A_s0"],
}
COL = {"CNN (gray)": "tab:gray", "CNN (RGB)": "tab:green", "CNN+LSTM": "tab:blue", "ViT": "tab:red"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

def load_stage(n):
    ep = np.load(RUN / f"{n}_episodes.npy").astype(float)
    log = {k: np.asarray(v, float) for k, v in np.load(RUN / f"{n}_log.npy", allow_pickle=True).item().items()}
    return ep, log

def lineage(names):
    eps, logs, bounds, off, base = [], {}, [], 0.0, {k: 0.0 for k in CUM}
    for n in names:
        ep, log = load_stage(n)
        eps.append(np.c_[ep[:, 0] + off, ep[:, 1]])
        for k, v in log.items():
            v = v + off if k == "step" else (v + base[k] if k in CUM else v)
            logs.setdefault(k, []).append(v)
        for k in CUM: base[k] += log[k][-1]
        off += max(log["step"][-1], ep[-1, 0]); bounds.append(off)
    return dict(ep=np.vstack(eps), log={k: np.concatenate(v) for k, v in logs.items()}, bounds=bounds[:-1], total=off)

def curve(ep, w):
    return ep[:, 0] / 1e6, pd.Series(ep[:, 1]).rolling(w, min_periods=max(10, w // 5)).mean().to_numpy()

def per_game(L, b=500_000):
    s, tot, ep = L["log"]["step"], L["total"], L["ep"]
    edges = np.arange(0, tot, b); edges = np.r_[edges, tot] if edges[-1] < tot else edges
    g = np.histogram(ep[:, 0], bins=edges)[0].astype(float); safe = np.where(g > 0, g, np.nan)
    o = {"x": (edges[:-1] + edges[1:]) / 2e6, "steps": np.diff(edges) / safe}
    for k in ("ghosts", "pellets"):
        o[k] = np.diff(np.interp(edges, np.r_[0, s], np.r_[0, L["log"][k]])) / safe
    ok = g >= 30
    return {k: v[ok] for k, v in o.items()}

lin = {k: lineage(v) for k, v in PICKS.items()}
FULL_VIT = lin["ViT"]                      # 16.65M steps incl. the later high-penalty sessions (collapse)
def cut(L, T):
    m = L["ep"][:, 0] <= T; ms = L["log"]["step"] <= T
    return dict(ep=L["ep"][m], log={k: v[ms] for k, v in L["log"].items()}, bounds=[b for b in L["bounds"] if b < T], total=T)
lin["ViT"] = cut(FULL_VIT, 13e6)          # peak of the ViT lineage (before sessions 5-8 raised the penalties)

# Fig 1: learning curves
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for lab, L in lin.items():
    x, y = curve(L["ep"], 200)
    ax.plot(x, y, color=COL[lab], lw=1.5, label=f"{lab} ({L['total']/1e6:.0f}M steps)")
    for b in L["bounds"]:
        i = min(np.searchsorted(L["ep"][:, 0], b), len(y) - 1)
        ax.plot(b / 1e6, y[i], marker="v", color=COL[lab], ms=5, mec="white")
ax.axhline(RANDOM, color="k", ls="--", lw=0.8, label=f"random ({RANDOM})")
ax.set_xlabel("environment steps (millions)"); ax.set_ylabel("game score, 200-episode moving avg.")
ax.legend(frameon=False, loc="upper left", fontsize=7.5); fig.tight_layout()
fig.savefig(OUT / "fig_learning_curves.pdf"); fig.savefig(OUT / "fig_learning_curves.png", dpi=200); plt.close(fig)

# Fig 2: behaviour metrics
fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.4), sharex=True)
for lab, L in lin.items():
    c = COL[lab]; s = L["log"]
    ent = pd.Series(s["ent"]).rolling(10, min_periods=1).mean().to_numpy()
    pg = per_game(L)
    axes[0, 0].plot(s["step"] / 1e6, ent, color=c, lw=1.2, label=lab)
    axes[0, 1].plot(pg["x"], pg["steps"], color=c, marker="o", ms=2, lw=1.2)
    axes[1, 0].plot(pg["x"], pg["ghosts"], color=c, marker="o", ms=2, lw=1.2)
    axes[1, 1].plot(pg["x"], pg["pellets"], color=c, marker="o", ms=2, lw=1.2)
    for ax in axes.ravel():
        for b in L["bounds"]: ax.axvline(b / 1e6, color=c, ls=":", lw=0.6, alpha=0.6)
axes[0, 0].axhline(np.log(9), color="k", ls="--", lw=0.7)
for ax, t in zip(axes.ravel(), ["policy entropy (nats)", "agent steps per game", "ghosts eaten per game", "power pellets per game"]):
    ax.set_title(t, fontsize=9)
for ax in axes[1]: ax.set_xlabel("environment steps (millions)")
axes[0, 0].legend(frameon=False, loc="lower left", fontsize=7); fig.tight_layout()
fig.savefig(OUT / "fig_behaviour.pdf"); fig.savefig(OUT / "fig_behaviour.png", dpi=200); plt.close(fig)

# Fig 3: first 1M steps + final-300 box
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), gridspec_kw={"width_ratios": [1.5, 1]})
vals = {}
for lab, L in lin.items():
    m = L["ep"][:, 0] <= 1_000_000
    x, y = curve(L["ep"][m], 100); axes[0].plot(x, y, color=COL[lab], lw=1.5, label=lab)
    vals[lab] = L["ep"][m][-200:, 1]
axes[0].axhline(RANDOM, color="k", ls="--", lw=0.8)
axes[0].set_xlabel("environment steps (millions)"); axes[0].set_ylabel("game score, 100-ep. avg."); axes[0].legend(frameon=False, fontsize=7)
bp = axes[1].boxplot(list(vals.values()), showmeans=True, patch_artist=True, widths=0.6)
axes[1].set_xticklabels(["gray", "RGB", "LSTM", "ViT"])
for p, lab in zip(bp["boxes"], vals): p.set_facecolor(COL[lab]); p.set_alpha(0.45)
axes[1].set_title("last 200 games of first 1M", fontsize=8); fig.tight_layout()
fig.savefig(OUT / "fig_first1M.pdf"); fig.savefig(OUT / "fig_first1M.png", dpi=200); plt.close(fig)

# ---- numbers ----
print("== first 1M, last 200 games: mean +- s.e.")
for k, v in vals.items(): print(f"{k:12s} {v.mean():.0f} +- {v.std()/np.sqrt(len(v)):.0f}")
print("== final lineage summary (last 300 games)")
for lab, L in lin.items():
    last = L["ep"][-300:, 1]; lo = max(L["total"] - 1e6, 0); s = L["log"]["step"]
    n_g = max(int((L["ep"][:, 0] >= lo).sum()), 1)
    ev = {k: (np.interp(L["total"], np.r_[0, s], np.r_[0, L["log"][k]]) - np.interp(lo, np.r_[0, s], np.r_[0, L["log"][k]])) / n_g for k in ("ghosts", "pellets")}
    print(f"{lab:12s} steps {L['total']/1e6:.2f}M games {len(L['ep'])} mean {last.mean():.0f} med {np.median(last):.0f} p90 {np.percentile(last,90):.0f} max {L['ep'][:,1].max():.0f} ent {L['log']['ent'][-10:].mean():.2f} ghosts/g {ev['ghosts']:.2f} pellets/g {ev['pellets']:.2f}")
print("== evaluation jsons (20 games, seed 1000)")
seen = {}
for f in sorted(RUN.glob("*_eval.json")):
    if f.name.startswith("smoke"): continue
    for k, v in json.load(open(f)).items():
        seen.setdefault(k, v)
for k, v in seen.items():
    v = np.array(v); print(f"{k:38s} mean {v.mean():.0f} std {v.std():.0f} med {np.median(v):.0f} max {v.max():.0f} se {v.std()/np.sqrt(len(v)):.0f}")
print("== RGB-ramp variants: training score by 1M bin")
for n in ["cnn_rgb_ramp_sp0.05_from_gray_s0", "cnn_rgb_ramp_sp0.3_from_gray_s0", "cnn_rgb_ramp_sp0.5_from_gray_s0", "cnn_rgb_ramp_from_gray_s0"]:
    ep, log = load_stage(n)
    tot = ep[-1, 0]; print(n, f"{tot/1e6:.2f}M steps, {len(ep)} games, first100 {ep[:100,1].mean():.0f}, last100 {ep[-100:,1].mean():.0f}, all {ep[:,1].mean():.0f}, vloss_first10 {log['v_loss'][:10].mean() if 'v_loss' in log else float('nan'):.1f} last10 {log['v_loss'][-10:].mean() if 'v_loss' in log else float('nan'):.1f}")
print("log keys:", list(load_stage("pen_const_low_s0")[1].keys()))

# Fig 4: ViT collapse after raising step penalty (own-step offset 8.0M => session boundaries from sessions.json)
sess = json.load(open(RUN / "vit_dp2_sp0.05_plus5M_from_A_s0_sessions.json", encoding="utf8"))
key = lambda d, k: [v for kk, v in d.items() if kk.startswith(k)][0]
fig, ax = plt.subplots(figsize=(6.4, 2.8))
x, y = curve(FULL_VIT["ep"], 200); ax.plot(x, y, color="tab:red", lw=1.5)
ax.axvspan(8 + 6.57, 8 + 7.06, color="orange", alpha=0.15); ax.axvspan(8 + 7.06, 8 + 8.66, color="red", alpha=0.10)
ax.text(8 + 6.5, 1330, "dp 5; sp 0.3 | 1.0 → 0.5", fontsize=7, va="top", ha="right")
ax.axhline(RANDOM, color="k", ls="--", lw=0.8)
ax.set_xlabel("environment steps (millions)"); ax.set_ylabel("game score, 200-ep. avg."); fig.tight_layout()
fig.savefig(OUT / "fig_vit_collapse.pdf"); fig.savefig(OUT / "fig_vit_collapse.png", dpi=200); plt.close(fig)
print("== ViT peak-13M summary")
L = lin["ViT"]; last = L["ep"][-300:, 1]
print(f"steps {L['total']/1e6:.1f}M games {len(L['ep'])} mean {last.mean():.0f} med {np.median(last):.0f} p90 {np.percentile(last,90):.0f} max {L['ep'][:,1].max():.0f} ent {L['log']['ent'][-10:].mean():.2f}")
lo = 12e6; s_ = L["log"]["step"]; n_g = int((L["ep"][:, 0] >= lo).sum())
print({k: (np.interp(13e6, np.r_[0, s_], np.r_[0, L["log"][k]]) - np.interp(lo, np.r_[0, s_], np.r_[0, L["log"][k]])) / n_g for k in ("ghosts", "pellets")})
f=FULL_VIT; print("full ViT last 300:", f["ep"][-300:,1].mean())
print("== last-300 std / s.e.")
for lab, L in lin.items():
    v = L["ep"][-300:, 1]; print(lab, f"{v.mean():.0f} std {v.std():.0f} se {v.std()/np.sqrt(300):.0f}")
