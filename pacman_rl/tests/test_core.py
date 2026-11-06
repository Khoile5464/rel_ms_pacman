"""Test nhanh, không cần chạy emulator: reward, mạng, replay buffer, lưu/nạp run. Chạy: pytest"""
import numpy as np
import torch

from pacman_rl.agents import TrainResult, VecFrameReplayBuffer
from pacman_rl.envs import count_events, log_reward
from pacman_rl.models import ActorCriticCNN, QNetwork
from pacman_rl.runs import load_run, save_run


def test_log_reward_and_events():
    assert np.allclose(log_reward([0, 10, 50]), [0.0, np.log(2), np.log(6)])
    assert count_events(np.array([0, 50, 200, 1600, 10, 50])) == (2, 2)


def test_network_shapes():
    x = torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8)
    dist, v = ActorCriticCNN(9)(x)
    assert dist.probs.shape == (3, 9) and v.shape == (3,)
    a, logp, v2 = ActorCriticCNN(9).act(x)
    assert a.shape == logp.shape == v2.shape == (3,)
    assert QNetwork(9)(x).shape == (3, 9)


def test_replay_buffer_stacks_within_episode():
    """Frame ghi ở bước t của env e có giá trị 10*e + t; đầu episode thì frame đầu được lặp lại."""
    n_envs, buf = 2, VecFrameReplayBuffer(capacity=40, n_envs=2)
    first = np.ones(n_envs, dtype=bool)
    for t in range(6):
        obs = np.zeros((n_envs, 4, 84, 84), dtype=np.uint8)
        obs[:, -1] = (10 * np.arange(n_envs) + t)[:, None, None]
        done = np.array([t == 2, False])                   # env 0 hết episode sau bước 2
        buf.add(obs, np.zeros(n_envs), np.zeros(n_envs), done.astype(np.float32), first)
        first = done
    stack = buf._stack(np.array([1, 4, 4]), np.array([0, 0, 1]))[:, :, 0, 0]
    assert stack[0].tolist() == [0, 0, 0, 1]           # đầu episode: lặp frame 0
    assert stack[1].tolist() == [3, 3, 3, 4]           # episode mới của env 0 bắt đầu ở bước 3
    assert stack[2].tolist() == [11, 12, 13, 14]       # env 1 không bị trộn frame env 0


def test_save_load_run_roundtrip(tmp_path):
    model = ActorCriticCNN(9)
    log = {"step": [2048], "ent": [2.1], "pellets": [1], "ghosts": [0]}
    save_run(TrainResult(model, log, [(100, 250.0), (200, 300.0)]), tmp_path, "run")
    m2, log2, eps2 = load_run(tmp_path, "run", "ppo")
    assert eps2 == [(100, 250.0), (200, 300.0)] and log2 == log
    for a, b in zip(model.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)


def test_vit_shapes_and_reload(tmp_path):
    from pacman_rl.models import ActorCriticViT
    from pacman_rl.runs import load_model
    x = torch.randint(0, 256, (2, 12, 84, 84), dtype=torch.uint8)
    m = ActorCriticViT(9, in_channels=12).eval()
    dist, v = m(x)
    assert dist.probs.shape == (2, 9) and v.shape == (2,) and m.in_channels == 12
    torch.save(m.state_dict(), tmp_path / "vit.pt")
    m2 = load_model(tmp_path / "vit.pt", "ppo")
    assert torch.allclose(m2(x)[1], v, atol=1e-5)


def test_lstm_shapes_and_reload(tmp_path):
    from pacman_rl.models import ActorCritic_LSTM_CNN
    from pacman_rl.runs import load_model
    m = ActorCritic_LSTM_CNN(9, in_channels=4).eval()
    x = torch.randint(0, 256, (5, 2, 4, 84, 84), dtype=torch.uint8)       # (T, E, C, H, W)
    pd = torch.zeros(5, 2)
    dist, v = m.forward_seq(x, m.initial_state(2, "cpu"), pd)
    assert dist.probs.shape == (10, 9) and v.shape == (10,)
    # forward_seq phải khớp từng bước step() (cùng state, cùng reset)
    state, vs = m.initial_state(2, "cpu"), []
    for t in range(5):
        _, vt, state = m.step(x[t], state, pd[t]); vs.append(vt)
    assert torch.allclose(torch.stack(vs).reshape(-1), v, atol=1e-4)
    torch.save(m.state_dict(), tmp_path / "l.pt")
    assert load_model(tmp_path / "l.pt", "ppo").hidden == 512
