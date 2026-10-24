"""Tạo env Ms. Pacman với pipeline tiền xử lý chuẩn của DeepMind."""
import gymnasium as gym

from .wrappers import EpisodicLife

ENV_ID = "ALE/MsPacman-v5"
N_ACTIONS = 9            # minimal action set: NOOP + 8 hướng
OBS_SHAPE = (4, 84, 84)  # 4 frame xám 84x84, uint8


def make_env(render_mode=None, episodic_life=False):
    """Env một bản: frame_skip=4, grayscale, resize 84x84, noop reset (tối đa 30), stack 4 frame.
    episodic_life=True chỉ dùng khi học (mất mạng = done)."""
    import ale_py  # noqa: F401  để process con cũng đăng ký được env ALE
    env = gym.make(ENV_ID, frameskip=1, repeat_action_probability=0.0,
                   full_action_space=False, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84,
                                          terminal_on_life_loss=False, grayscale_obs=True,
                                          scale_obs=False)
    if episodic_life:
        env = EpisodicLife(env)
    return gym.wrappers.FrameStack(env, 4)
