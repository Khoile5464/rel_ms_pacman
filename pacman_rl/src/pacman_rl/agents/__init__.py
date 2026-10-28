from .buffers import VecFrameReplayBuffer
from .common import TrainResult
from .dqn import DQNConfig, train_dqn
from .ppo import PPOConfig, piecewise_penalty, train_ppo

# tên thuật toán -> (lớp config, hàm train), dùng cho CLI
ALGOS = {"ppo": (PPOConfig, train_ppo), "dqn": (DQNConfig, train_dqn)}

__all__ = ["TrainResult", "PPOConfig", "piecewise_penalty", "train_ppo", "DQNConfig", "train_dqn", "VecFrameReplayBuffer", "ALGOS"]
