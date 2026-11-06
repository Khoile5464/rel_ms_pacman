from .networks import ActorCriticCNN, ActorCriticViT, ActorCritic_LSTM_CNN, QNetwork, make_cnn

# tên thuật toán -> lớp mạng, dùng khi nạp lại model từ file
MODEL_CLASSES = {"ppo": ActorCriticCNN, "dqn": QNetwork}
PPO_ARCHS = {"cnn": ActorCriticCNN, "vit": ActorCriticViT, "lstm": ActorCritic_LSTM_CNN}

__all__ = ["ActorCriticCNN", "ActorCriticViT", "ActorCritic_LSTM_CNN", "QNetwork", "make_cnn", "MODEL_CLASSES", "PPO_ARCHS"]
