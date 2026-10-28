"""Tạo env Ms. Pacman với pipeline tiền xử lý chuẩn của DeepMind (ảnh xám hoặc RGB)."""
import gymnasium as gym
import numpy as np

from .wrappers import EpisodicLife

ENV_ID = "ALE/MsPacman-v5"
N_ACTIONS = 9            # minimal action set: NOOP + 8 hướng
N_STACK = 4              # số frame xếp chồng


def obs_shape(rgb_scale=False):
    """(4, 84, 84) khi xám, (12, 84, 84) khi RGB (4 frame x 3 kênh màu gộp thành kênh)."""
    return (N_STACK * (3 if rgb_scale else 1), 84, 84)


OBS_SHAPE = obs_shape(False)  # mặc định (xám), giữ lại cho code cũ


class StackToChannels(gym.ObservationWrapper):
    """FrameStack trả (4, 84, 84, 3); gộp thành (12, 84, 84) để Conv2d đọc 12 kênh."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 255, obs_shape(True), np.uint8)

    def observation(self, obs):
        return np.asarray(obs).transpose(0, 3, 1, 2).reshape(*obs_shape(True))


def make_env(render_mode=None, episodic_life=False, rgb_scale=False):
    """Env một bản: frame_skip=4, xám hoặc RGB (rgb_scale), resize 84x84, noop reset (tối đa 30), stack 4 frame.
    episodic_life=True chỉ dùng khi học (mất mạng = done)."""
    import ale_py  # noqa: F401  để process con cũng đăng ký được env ALE
    env = gym.make(ENV_ID, frameskip=1, repeat_action_probability=0.0,
                   full_action_space=False, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84,
                                          terminal_on_life_loss=False, grayscale_obs=not rgb_scale,
                                          scale_obs=False)
    if episodic_life:
        env = EpisodicLife(env)
    env = gym.wrappers.FrameStack(env, N_STACK)
    return StackToChannels(env) if rgb_scale else env
