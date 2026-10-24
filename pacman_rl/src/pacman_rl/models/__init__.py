from .networks import ActorCriticCNN, QNetwork, make_cnn

# tên thuật toán -> lớp mạng, dùng khi nạp lại model từ file
MODEL_CLASSES = {"ppo": ActorCriticCNN, "dqn": QNetwork}

__all__ = ["ActorCriticCNN", "QNetwork", "make_cnn", "MODEL_CLASSES"]
