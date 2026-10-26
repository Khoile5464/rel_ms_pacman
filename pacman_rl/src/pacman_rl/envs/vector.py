"""Env song song chạy xen kẽ với GPU."""
from functools import partial

import gymnasium as gym
import numpy as np

from .factory import make_env


def pop_game_scores(infos):
    """Lấy điểm thật của các ván vừa kết thúc hẳn (hết 3 mạng) từ info của vector env."""
    out = []
    for fi in infos.get("final_info", []):
        if fi is not None and "game_score" in fi:
            out.append(fi["game_score"])
    return out

def pop_deaths(infos, n):
    died = np.zeros(n, dtype=np.float32)
    for i, fi in enumerate(infos.get("final_info", [])):
        if fi is not None and fi.get("life_lost"):
            died[i] = 1.0
    return died

class PipelinedEnvs:
    """Env song song dùng khi học (mất mạng = done), chia thành n_groups nhóm chạy xen kẽ.
    Trong lúc emulator (CPU) đang chạy nhóm này thì GPU tính action cho nhóm kia, nên CPU và GPU không chờ nhau.
    render=False: mỗi env một process riêng (AsyncVectorEnv) để dùng hết các nhân CPU.
    render=True : chạy tuần tự trong 1 process và mở cửa sổ game ở env số 0 (chậm, chỉ để xem)."""

    def __init__(self, n_envs, n_groups=2, render=False):
        assert n_envs % n_groups == 0
        self.G, self.gs, self.n = n_groups, n_envs // n_groups, n_envs
        cls = gym.vector.SyncVectorEnv if render else gym.vector.AsyncVectorEnv
        # partial thay cho lambda: pickle được và process con chỉ import pacman_rl.envs (không kéo torch theo)
        fn = lambda human: partial(make_env, "human" if human else None, episodic_life=True)
        self.envs = [cls([fn(render and g == 0 and i == 0) for i in range(self.gs)]) for g in range(n_groups)]
        self.slices = [slice(g * self.gs, (g + 1) * self.gs) for g in range(n_groups)]
        self.n_actions = self.envs[0].single_action_space.n

    def reset(self, seed=0) -> np.ndarray:
        return np.concatenate([e.reset(seed=seed + g * self.gs)[0] for g, e in enumerate(self.envs)])

    def send(self, g, actions):
        self.envs[g].step_async(actions)

    def recv(self, g) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, np.ndarray]:
        """Trả về (obs, reward thật, done, danh sách điểm thật của các ván vừa hết 3 mạng)."""
        obs, reward, term, trunc, infos = self.envs[g].step_wait()
        return obs, reward, np.logical_or(term, trunc), pop_game_scores(infos), pop_deaths(infos, self.gs)

    def close(self):
        for e in self.envs:
            e.close()
