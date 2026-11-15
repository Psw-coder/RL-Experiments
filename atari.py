import runtime  # configure storage first
import gymnasium as gym
import ale_py
import numpy as np

gym.register_envs(ale_py)


class EpisodicLife(gym.Wrapper):
    """End learning episodes on life loss, reset emulator only at real game over."""
    def __init__(self, env):
        super().__init__(env)
        self.real_done = True
        self.lives = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.real_done = terminated or truncated
        lives = self.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.real_done or kwargs.get('seed') is not None:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.unwrapped.ale.lives()
        return obs, info


class FireReset(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for action in (1, 2):
            obs, _, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                obs, info = self.env.reset()
        return obs, info


class SignReward(gym.RewardWrapper):
    def reward(self, reward):
        return float(np.sign(reward))


def make_env(training=True):
    # Match gym 0.10.8 BreakoutNoFrameskip-v4 (the authors' environment):
    # sticky actions p=0.25, minimal action set, frame skip applied below.
    env = gym.make('ALE/Breakout-v5', frameskip=1,
                   repeat_action_probability=0.25, full_action_space=False)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4,
        screen_size=84, terminal_on_life_loss=False, grayscale_obs=True,
        scale_obs=False)
    if training:
        env = EpisodicLife(env)
    env = FireReset(env)
    if training:
        env = SignReward(env)
    return gym.wrappers.FrameStackObservation(env, stack_size=4)
