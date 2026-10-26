"""Wrapper dùng khi học."""
import gymnasium as gym


class EpisodicLife(gym.Wrapper):
    """Mất một mạng thì báo terminated=True để agent thấy cái chết có giá.
    Game thật chỉ reset khi hết mạng (lúc đó điểm cả ván nằm trong info["game_score"])."""

    def __init__(self, env):
        super().__init__(env)
        self.lives, self.real_done, self.score = 0, True, 0.0

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.score += r
        self.real_done = term or trunc
        lives = self.env.unwrapped.ale.lives()
        died = lives < self.lives
        if 0 < lives < self.lives:
            term = True                                    # mất mạng nhưng game chưa hết
        self.lives = lives
        info["life_lost"] = died
        if self.real_done:
            info["game_score"] = self.score
        return obs, r, term, trunc, info

    def reset(self, **kwargs):
        if self.real_done:
            obs, info = self.env.reset(**kwargs); self.score = 0.0
        else:                                              # chưa hết game: đi tiếp 1 bước NOOP thay vì reset thật
            obs, r, term, trunc, info = self.env.step(0)
            self.score += r
            if term or trunc:
                obs, info = self.env.reset(**kwargs); self.score = 0.0
        self.real_done = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info
