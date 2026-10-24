"""Mạng neural: thân CNN Nature DQN (Mnih 2015) dùng chung, đầu actor-critic cho PPO, đầu Q cho DQN.

input (4,84,84) uint8 -> chia 255
Conv 32 8x8 s4 + ReLU -> (32,20,20)
Conv 64 4x4 s2 + ReLU -> (64, 9, 9)
Conv 64 3x3 s1 + ReLU -> (64, 7, 7)
Flatten (3136) -> Linear 512 + ReLU
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def make_cnn():
    return nn.Sequential(
        nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
    )


class ActorCriticCNN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.cnn = make_cnn()
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)
        # khởi tạo trực giao (chuẩn của PPO): actor nhỏ để policy ban đầu gần đều
        for m in self.cnn:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01); nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, 1.0); nn.init.zeros_(self.critic.bias)

    def forward(self, x):                       # x: uint8 (B,4,84,84)
        h = self.cnn(x.float() / 255.0)
        return Categorical(logits=self.actor(h)), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, x):
        """Chọn action khi thu rollout, trả về (action, log_prob, value). Lấy mẫu bằng Gumbel-max,
        cùng phân phối với Categorical nhưng ít op hơn nên nhanh hơn khi gọi từng bước nhỏ."""
        h = self.cnn(x.float() / 255.0)
        logits = self.actor(h)
        gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp_(1e-10, 1 - 1e-7)))
        a = (logits + gumbel).argmax(-1)
        logp = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(1)
        return a, logp, self.critic(h).squeeze(-1)


class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.cnn = make_cnn()
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):                       # x: uint8 (B,4,84,84)
        return self.head(self.cnn(x.float() / 255.0))
