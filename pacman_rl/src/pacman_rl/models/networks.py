"""Mạng neural: thân CNN Nature DQN (Mnih 2015) dùng chung, đầu actor-critic cho PPO, đầu Q cho DQN.

input (4,84,84) uint8 (xám) hoặc (12,84,84) (4 frame RGB gộp kênh) -> chia 255
Conv 32 8x8 s4 + ReLU -> (32,20,20)
Conv 64 4x4 s2 + ReLU -> (64, 9, 9)
Conv 64 3x3 s1 + ReLU -> (64, 7, 7)
Flatten (3136) -> Linear 512 + ReLU
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def make_cnn(in_channels=4):
    """in_channels: 4 (xám) hoặc 12 (RGB)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
    )


class ActorCriticCNN(nn.Module):
    def __init__(self, n_actions, in_channels=4):
        super().__init__()
        self.cnn = make_cnn(in_channels)
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)
        # khởi tạo trực giao (chuẩn của PPO): actor nhỏ để policy ban đầu gần đều
        for m in self.cnn:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01); nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, 1.0); nn.init.zeros_(self.critic.bias)

    @property
    def in_channels(self):
        return self.cnn[0].in_channels

    def features(self, x):                      # uint8 -> vector đặc trưng (B, 512)
        return self.cnn(x.float() / 255.0)

    def forward(self, x):                       # x: uint8 (B,12,84,84)
        h = self.features(x)
        return Categorical(logits=self.actor(h)), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, x):
        """Chọn action khi thu rollout, trả về (action, log_prob, value). Lấy mẫu bằng Gumbel-max,
        cùng phân phối với Categorical nhưng ít op hơn nên nhanh hơn khi gọi từng bước nhỏ."""
        h = self.features(x)
        logits = self.actor(h)
        gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp_(1e-10, 1 - 1e-7)))
        a = (logits + gumbel).argmax(-1)
        logp = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(1)
        return a, logp, self.critic(h).squeeze(-1)


class ActorCritic_LSTM_CNN(ActorCriticCNN):
    """CNN (Nature DQN) -> LSTM(512) -> actor/critic. Có trí nhớ qua các bước nên PPO phải truyền trạng thái ẩn.
    state = (h, c), mỗi tensor (B, 512). prev_done (B,) = 1 nếu bước trước kết thúc episode -> reset state về 0."""
    recurrent = True

    def __init__(self, n_actions, in_channels=4, hidden=512):
        super().__init__(n_actions, in_channels)
        self.hidden = hidden
        self.lstm = nn.LSTM(512, hidden)
        for name, w in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.zeros_(w)
            else:
                nn.init.orthogonal_(w, 1.0)

    def initial_state(self, n, device):
        z = torch.zeros(n, self.hidden, device=device)
        return z, z.clone()

    def _cell(self, f, state, prev_done):       # f (B,512) -> z (B,hidden), state mới
        m = (1.0 - prev_done)[:, None]
        h, c = state[0] * m, state[1] * m
        out, (h, c) = self.lstm(f[None], (h[None].contiguous(), c[None].contiguous()))
        return out[0], (h[0], c[0])

    def step(self, x, state, prev_done):
        z, state = self._cell(self.features(x), state, prev_done)
        return Categorical(logits=self.actor(z)), self.critic(z).squeeze(-1), state

    @torch.no_grad()
    def act(self, x, state, prev_done):
        z, state = self._cell(self.features(x), state, prev_done)
        logits = self.actor(z)
        gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp_(1e-10, 1 - 1e-7)))
        a = (logits + gumbel).argmax(-1)
        logp = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(1)
        return a, logp, self.critic(z).squeeze(-1), state

    def forward_seq(self, x, state, prev_done):
        """x (T,E,C,84,84), prev_done (T,E) -> (Categorical trên T*E, value (T*E,)), thứ tự t-major."""
        T, E = x.shape[:2]
        f = self.features(x.reshape(T * E, *x.shape[2:])).reshape(T, E, -1)
        zs = []
        for t in range(T):
            z, state = self._cell(f[t], state, prev_done[t])
            zs.append(z)
        z = torch.stack(zs).reshape(T * E, -1)
        return Categorical(logits=self.actor(z)), self.critic(z).squeeze(-1)

    def forward(self, x):                       # không trạng thái (state 0): chỉ để kiểm tra shape / debug
        z = torch.zeros(len(x), device=x.device)
        dist, v, _ = self.step(x, self.initial_state(len(x), x.device), z)
        return dist, v


class ActorCriticViT(ActorCriticCNN):
    """Vision Transformer nhỏ cho PPO: ảnh 84x84 chia patch 12x12 -> 7x7 = 49 token (+ CLS), pre-norm encoder.
    Mặc định ~0.8M tham số (dim 128, 4 lớp, 4 head), chạy được trên GPU phổ thông. Đầu actor/critic như CNN."""

    def __init__(self, n_actions, in_channels=4, img_size=84, patch=12, dim=128, depth=4, heads=4, mlp_ratio=2):
        nn.Module.__init__(self)
        assert img_size % patch == 0, "img_size phải chia hết cho patch"
        n_tok = (img_size // patch) ** 2
        self._in_channels = in_channels
        self.patch_embed = nn.Conv2d(in_channels, dim, patch, stride=patch)   # patchify + chiếu tuyến tính
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n_tok + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * mlp_ratio, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.actor = nn.Linear(dim, n_actions)
        self.critic = nn.Linear(dim, 1)
        nn.init.orthogonal_(self.actor.weight, 0.01); nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, 1.0); nn.init.zeros_(self.critic.bias)

    @property
    def in_channels(self):
        return self._in_channels

    def features(self, x):
        t = self.patch_embed(x.float() / 255.0).flatten(2).transpose(1, 2)    # (B, 49, dim)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        return self.norm(self.encoder(t)[:, 0])                               # token CLS


class QNetwork(nn.Module):
    def __init__(self, n_actions, in_channels=4):
        super().__init__()
        self.cnn = make_cnn(in_channels)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):                       # x: uint8 (B,12,84,84)
        return self.head(self.cnn(x.float() / 255.0))

