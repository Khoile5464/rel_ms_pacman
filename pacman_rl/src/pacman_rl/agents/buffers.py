"""Replay buffer tiết kiệm bộ nhớ cho DQN."""
import numpy as np
import torch


class VecFrameReplayBuffer:
    """Replay buffer cho n_envs env song song, chỉ lưu frame mới nhất của mỗi bước và dựng lại stack 4 frame khi lấy mẫu.
    Mỗi env là một "cột" riêng nên stack không bao giờ trộn frame của hai env khác nhau.
    Đầu episode thì frame đầu được lặp lại (giống FrameStack). 200k mẫu tốn khoảng 1.4 GB RAM."""

    def __init__(self, capacity, n_envs, device="cpu"):
        self.n, self.T, self.device = n_envs, capacity // n_envs, device
        self.frames = np.zeros((self.T, n_envs, 84, 84), dtype=np.uint8)
        self.act = np.zeros((self.T, n_envs), dtype=np.int64)
        self.rew = np.zeros((self.T, n_envs), dtype=np.float32)
        self.done = np.zeros((self.T, n_envs), dtype=np.float32)
        self.ep_start = np.zeros((self.T, n_envs), dtype=np.int64)   # thời điểm tuyệt đối của frame đầu episode
        self.t = 0                                                   # số bước thời gian đã ghi
        self.cur_start = np.zeros(n_envs, dtype=np.int64)
        self.offs = np.arange(-3, 1)

    def add(self, obs, action, reward, done, is_first):    # mỗi tham số có độ dài n_envs, obs: (n_envs,4,84,84)
        i = self.t % self.T
        self.cur_start[is_first] = self.t
        self.frames[i] = obs[:, -1]
        self.act[i], self.rew[i], self.done[i], self.ep_start[i] = action, reward, done, self.cur_start
        self.t += 1

    def __len__(self):
        return min(self.t, self.T) * self.n

    def _stack(self, P, E):
        S = self.ep_start[P % self.T, E]
        idx = np.maximum(P[:, None] + self.offs[None, :], S[:, None]) % self.T
        return self.frames[idx, E[:, None]]                         # (B,4,84,84)

    def sample(self, batch_size):
        """Trả về tensor (s, a, r, s', done) trên self.device."""
        lo = max(0, self.t - self.T + 3)                            # frame cũ hơn đã bị ghi đè
        hi = self.t - 2                                             # cần có sẵn P+1
        P = np.random.randint(lo, hi + 1, size=batch_size)
        E = np.random.randint(0, self.n, size=batch_size)
        i = P % self.T
        to_t = lambda a: torch.as_tensor(a, device=self.device)
        return (to_t(self._stack(P, E)), to_t(self.act[i, E]), to_t(self.rew[i, E]),
                to_t(self._stack(P + 1, E)), to_t(self.done[i, E]))
