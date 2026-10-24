"""Biến đổi reward khi học và đếm sự kiện trong game (theo reward THẬT, chưa scale)."""
import numpy as np

PELLET_REWARD = 50                         # viên năng lượng
GHOST_REWARDS = (200, 400, 800, 1600)      # ăn ma liên tiếp trong một viên năng lượng


def log_reward(r):
    """Reward thang log dùng khi học: chấm 10 -> 0.69, viên năng lượng 50 -> 1.79, ma 200..1600 -> 3.0..5.1."""
    return np.log1p(np.asarray(r, dtype=np.float32) / 10.0)


def count_events(reward):
    """Đếm theo reward thật của một bước: (số viên năng lượng, số con ma ăn được).
    Chỉ đếm khi reward khớp đúng; hiếm khi hai sự kiện rơi vào cùng 4 frame thì bị bỏ sót.
    Trái cây 200 điểm (màn 2 trở đi) có thể bị tính nhầm là ma, sai số nhỏ."""
    reward = np.asarray(reward)
    return int((reward == PELLET_REWARD).sum()), int(np.isin(reward, GHOST_REWARDS).sum())
