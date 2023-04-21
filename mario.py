import random

import gym.wrappers
import gym_super_mario_bros, gym_super_mario_bros.actions
import nes_py.wrappers
import torch
import numpy as np
import wrappers

torch.autograd.set_detect_anomaly(True)

#TODO:
# Implement mini-batch training
# GPU accelerate
from nes_py.wrappers import JoypadSpace


class DQN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        print(input_size)

        # Remove args?
        super(DQN, self).__init__()
        self.conv = torch.nn.Sequential(
            # TODO play with kernel sizes?
            torch.nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )

        # Get output size by running zeros through network, a little hacky, but easy
        conv_out_size = int(np.prod(self.conv(torch.zeros(1, *input_size)).size()))

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, output_size)
        )

    def forward(self, x):
        # Flatten may not be correct:
        # Others used: .view(x.size()[0], -1), but I think it'll have the same effect
        return self.fc(self.conv(x).flatten())


def create_env():
    # Use rectangle ROM
    env = gym_super_mario_bros.make('SuperMarioBros-v3')
    # env = wrappers.MarioXReward(env) # Remove?
    env = wrappers.ProcessFrame84(env)
    env = wrappers.MaxAndSkipEnv(env)
    env = wrappers.ImageToPyTorch(env)
    env = wrappers.BufferWrapper(env, 4)
    env = wrappers.ScaledFloatFrame(env)
    # return JoypadSpace(env, gym_super_mario_bros.actions.SIMPLE_MOVEMENT)
    return JoypadSpace(env, gym_super_mario_bros.actions.RIGHT_ONLY)

def get_action(state):
    # Check for exploration
    if (random.random() > 0.2):
        return np.argmax(models[0].forward(torch.FloatTensor(state)).detach().numpy())
    else:
        return env.action_space.sample()


def calc_loss(state, action, next_state, reward, done):
    # Check that action index is correct here, not sure
    current_q = [model.forward(torch.FloatTensor(state))[action] for model in models]
    next_q = torch.min(*[torch.max(model.forward(torch.FloatTensor(next_state))) for model in models])
    #Flatten next q?
    # move lambda somewhere better
    expected_q = next_q * 0.99 * (1 - done) + reward
    # print(expected_q)

    return [torch.nn.functional.mse_loss(q, expected_q) for q in current_q]

env = create_env()

models = [
    DQN(env.observation_space.shape, env.action_space.n),
    DQN(env.observation_space.shape, env.action_space.n)
]

opts = [torch.optim.Adam(model.parameters()) for model in models]

for episode in range(10000):
    episode_reward = 0
    state = env.reset()

    # Max steps, may need tweaking
    for step in range(1000):
        action = get_action(state)

        # print(action)

        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        loss = calc_loss(state, action, next_state, reward, done)
        #print(loss)

        # Not sure why but it has to be done this way
        opts[0].zero_grad()
        opts[1].zero_grad()

        loss[0].backward(retain_graph=True)
        loss[1].backward()

        opts[0].step()
        opts[1].step()

        state = next_state

        if done:
            #Finish episode
            break

        # env.render()

    print(f"Episode: {episode}, reward: {episode_reward}")

env.close()