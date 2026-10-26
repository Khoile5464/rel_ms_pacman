"""Kiểu dữ liệu dùng chung giữa các thuật toán."""
from typing import NamedTuple

import torch.nn as nn


class TrainResult(NamedTuple):
    """Kết quả train. Unpack được như tuple: model, log, episodes = train_ppo(...)

    log      : dict tên chỉ số -> list, cùng độ dài với log["step"] (bước env lúc ghi).
               Luôn có "step", "pellets", "ghosts" (tổng tích luỹ số viên năng lượng / ma đã ăn).
               "deaths" (tổng tích luỹ số mạng đã mất) cũng được ghi.
    episodes : list (bước env, điểm thật của cả ván 3 mạng) cho mỗi ván đã kết thúc trong lúc train.
    """
    model: nn.Module
    log: dict
    episodes: list
