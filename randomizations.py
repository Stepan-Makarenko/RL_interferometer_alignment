import gym
import numpy as np


RADIUS = 0.714


def shift(state):
    start = np.random.randint(0, len(state))
    result = []
    for i in range(start, start + len(state)):
        result.append(state[i % len(state)])
    return np.asarray(result)


class BeamRadiusRandomizer(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs, rew, done, info
    
    def reset(self, **kwargs):
        r = RADIUS * np.random.uniform(0.8, 1.2)
        self.env.set_radius(r)
        return self.env.reset(**kwargs)


class ChannelShifter(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        return shift(self.env.reset(**kwargs))

    def step(self, actions):
        n_back = np.random.randint(0, 8)
        self.env.set_backward_frames(n_back)
        obs, rew, done, info = self.env.step(actions)
        return shift(obs), rew, done, info


class BrightnessRandomizer(gym.Wrapper):
    def __init__(self, e):
        super().__init__(e)

    def randomize(self, obs):
        obs = obs * np.random.uniform(0.7, 1.3)
        obs = np.minimum(obs, 255)
        return obs.astype(np.uint8)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.randomize(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return self.randomize(obs), rew, done, info

    
class ActionNoiseRandomizer(gym.Wrapper):
    def __init__(self, e):
        super().__init__(e)

    def randomize(self, act):
        noise = np.random.randn(len(act)) * act * 0.06
        return act + noise

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        action = self.randomize(action)
        obs, rew, done, info = self.env.step(action)
        return obs, rew, done, info


def make_env(name='interf-v1', randomizations=('Radius', 'Brightness', 'ChannelShifter', 'CameraPosition'), seed=None):
    env = gym.make(name)
    env.set_radius(RADIUS)
    env.set_max_steps(100)
    if 'Radius' in randomizations:
        env = BeamRadiusRandomizer(env)
    if 'Brightness' in randomizations:
        env = BrightnessRandomizer(env)
    if 'ChannelShifter' in randomizations:
        env = ChannelShifter(env)
    if 'ActionNoise' in randomizations:
        env = ActionNoiseRandomizer(env)
    env.seed(seed)
    return env
