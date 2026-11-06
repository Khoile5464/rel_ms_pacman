"""pacman_rl: PPO và DQN trên Ms. Pacman (Atari), tách ra từ notebooks/ms_pacman.ipynb.

    from pacman_rl import PPOConfig, train_ppo, save_run
    result = train_ppo(PPOConfig(total_steps=2_000_000))
    save_run(result, "runs", "ms_pacman_ppo")

Các tên dưới đây được nạp lười (lazy): `import pacman_rl` không kéo torch theo, để 16 process con
của env (chỉ cần pacman_rl.envs) khởi động nhanh và tốn ít RAM.
"""
import importlib

__version__ = "0.1.0"

_EXPORTS = {
    "envs": ["ENV_ID", "N_ACTIONS", "make_env", "PipelinedEnvs", "EpisodicLife", "log_reward", "count_events"],
    "models": ["ActorCriticCNN", "QNetwork"],
    "agents": ["PPOConfig", "piecewise_penalty", "train_ppo", "DQNConfig", "train_dqn", "TrainResult"],
    "evaluation": ["evaluate", "watch", "watch_probs", "run_episodes", "random_baseline", "ppo_policy", "dqn_policy", "watch_probs"],
    "runs": ["save_run", "load_run", "load_runs", "load_model"],
    "utils": ["get_device", "describe_device", "set_seed", "moving_avg"],
}
_WHERE = {name: mod for mod, names in _EXPORTS.items() for name in names}
__all__ = sorted(_WHERE)


def __getattr__(name):
    if name in _WHERE:
        return getattr(importlib.import_module(f".{_WHERE[name]}", __name__), name)
    raise AttributeError(f"module 'pacman_rl' has no attribute '{name}'")


def __dir__():
    return __all__ + ["__version__"]
