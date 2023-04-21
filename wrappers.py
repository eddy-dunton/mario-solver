#From: https://github.com/openai/large-scale-curiosity/blob/master/wrappers.py
# And https://github.com/facebookresearch/torchbeast/blob/main/torchbeast/atari_wrappers.py
# And https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/wrappers.py



import itertools
from collections import deque
from copy import copy

import cv2
import gym
import numpy as np

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))

def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        acc_info = {}
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            acc_info.update(info)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, acc_info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    """
        Downsamples image to 84x84
        Greyscales image

        Returns numpy array
        """

    def __init__(self, env = None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low = 0, high = 255, shape = (84, 84, 1), dtype = np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation = cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ExtraTimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps > self._max_episode_steps:
            done = True
        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()


class AddRandomStateToInfo(gym.Wrapper):
    def __init__(self, env):
        """Adds the random state to the info field on the first step after reset
        """
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        ob, r, d, info = self.env.step(action)
        if self.random_state_copy is not None:
            info['random_state'] = self.random_state_copy
            self.random_state_copy = None
        return ob, r, d, info

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.random_state_copy = copy(self.unwrapped.np_random)
        return self.env.reset(**kwargs)


class MontezumaInfoWrapper(gym.Wrapper):
    ram_map = {
        "room": dict(
            index=3,
        ),
        "x": dict(
            index=42,
        ),
        "y": dict(
            index=43,
        ),
    }

    def __init__(self, env):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.visited = set()
        self.visited_rooms = set()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        ram_state = unwrap(self.env).ale.getRAM()
        for name, properties in MontezumaInfoWrapper.ram_map.items():
            info[name] = ram_state[properties['index']]
        pos = (info['x'], info['y'], info['room'])
        self.visited.add(pos)
        self.visited_rooms.add(info["room"])
        if done:
            info['mz_episode'] = dict(pos_count=len(self.visited),
                                      visited_rooms=copy(self.visited_rooms))
            self.visited.clear()
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()


class MarioXReward(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.current_level = [0, 0]
        self.visited_levels = set()
        self.visited_levels.add(tuple(self.current_level))
        self.current_max_x = 0.

    def reset(self):
        ob = self.env.reset()
        self.current_level = [0, 0]
        self.visited_levels = set()
        self.visited_levels.add(tuple(self.current_level))
        self.current_max_x = 0.
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        levellow, levelhigh, xscrollHi, xscrollLo = \
            info["levelLo"], info["levelHi"], info["xscrollHi"], info["xscrollLo"]
        currentx = xscrollHi * 256 + xscrollLo
        new_level = [levellow, levelhigh]
        if new_level != self.current_level:
            self.current_level = new_level
            self.current_max_x = 0.
            reward = 0.
            self.visited_levels.add(tuple(self.current_level))
        else:
            if currentx > self.current_max_x:
                delta = currentx - self.current_max_x
                self.current_max_x = currentx
                reward = delta
            else:
                reward = 0.
        if done:
            info["levels"] = copy(self.visited_levels)
            info["retro_episode"] = dict(levels=copy(self.visited_levels))
        return ob, reward, done, info


class LimitedDiscreteActions(gym.ActionWrapper):
    KNOWN_BUTTONS = {"A", "B"}
    KNOWN_SHOULDERS = {"L", "R"}

    '''
    Reproduces the action space from curiosity paper.
    '''

    def __init__(self, env, all_buttons, whitelist=KNOWN_BUTTONS | KNOWN_SHOULDERS):
        gym.ActionWrapper.__init__(self, env)

        self._num_buttons = len(all_buttons)
        button_keys = {i for i in range(len(all_buttons)) if all_buttons[i] in whitelist & self.KNOWN_BUTTONS}
        buttons = [(), *zip(button_keys), *itertools.combinations(button_keys, 2)]
        shoulder_keys = {i for i in range(len(all_buttons)) if all_buttons[i] in whitelist & self.KNOWN_SHOULDERS}
        shoulders = [(), *zip(shoulder_keys), *itertools.permutations(shoulder_keys, 2)]
        arrows = [(), (4,), (5,), (6,), (7,)]  # (), up, down, left, right
        acts = []
        acts += arrows
        acts += buttons[1:]
        acts += [a + b for a in arrows[-2:] for b in buttons[1:]]
        self._actions = acts
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        mask = np.zeros(self._num_buttons)
        for i in self._actions[a]:
            mask[i] = 1
        return mask


class FrameSkip(gym.Wrapper):
    def __init__(self, env, n):
        gym.Wrapper.__init__(self, env)
        self.n = n

    def step(self, action):
        done = False
        totrew = 0
        for _ in range(self.n):
            ob, rew, done, info = self.env.step(action)
            totrew += rew
            if done: break
        return ob, totrew, done, info



class OneChannel(gym.ObservationWrapper):
    def __init__(self, env, crop=True):
        self.crop = crop
        super(OneChannel, self).__init__(env)
        assert env.observation_space.dtype == np.uint8
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return obs[:, :, 2:3]


class RetroALEActions(gym.ActionWrapper):
    def __init__(self, env, all_buttons, n_players=1):
        gym.ActionWrapper.__init__(self, env)
        self.n_players = n_players
        self._num_buttons = len(all_buttons)
        bs = [-1, 0, 4, 5, 6, 7]
        actions = []

        def update_actions(old_actions, offset=0):
            actions = []
            for b in old_actions:
                for button in bs:
                    action = []
                    action.extend(b)
                    if button != -1:
                        action.append(button + offset)
                    actions.append(action)
            return actions

        current_actions = [[]]
        for i in range(self.n_players):
            current_actions = update_actions(current_actions, i * self._num_buttons)
        self._actions = current_actions
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        mask = np.zeros(self._num_buttons * self.n_players)
        for i in self._actions[a]:
            mask[i] = 1
        return mask


class NoReward(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        return ob, 0.0, done, info