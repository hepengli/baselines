import os
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import gym
from baselines.bench import Monitor
from baselines.common import retro_wrappers
from baselines.common.wrappers import ClipActionsWrapper

def make_safety_env(env_id, seed, logger_dir=None, train=True, reward_scale=1.0):
    """
    Create a wrapped, monitored gym.Env for safety.
    """
    env = gym.make(env_id, **{"train":train})
    env = Monitor(env, logger_dir, allow_early_resets=True, info_keywords=tuple("s"))
    env.seed(seed)
    if isinstance(env.action_space, gym.spaces.Box):
        env = ClipActionsWrapper(env)
    if reward_scale != 1.0:
        from baselines.common.retro_wrappers import RewardScaler
        env = RewardScaler(env, reward_scale)
    return env
