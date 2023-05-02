import random

import gym_super_mario_bros
import gym_super_mario_bros.actions
import numpy as np
import torch
import nes_py.wrappers

import wrappers

"""
Originally written by Eddy Dunton, eed34@bath.ac.uk

Based on:

Clipped DDQN (2018)
https://github.com/cyoon1729/deep-Q-networks/blob/cb3f1551bc927fedf7166d7b0b3834aaff07d32e/doubleDQN/clipped_ddqn.py
https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/

Current state:
Working, however there is some strange behaviour with the training, training does work negatively (ex. trying to get the
 lowest possible score), however, only seems to provide average results when training positive, training is also very
 slow right now, I would recommened implementing batches and GPU support first else you'll wait forever to get and 
 results

TODO:
 - Implement mini-batch
 - Implement GPU acceleration
 - Chase down poor learning behaviour
"""


# Deep Q Network, NN module responsible for the actual network
# There are 2 different formats of this network
# One which uses a CNN feature extractor and then a fc classifier (used here)
# and a fc only network
#
# I've seen 2 different forms of the FC section of the network (the 2 examples use a different implementation)
# I couldn't find a meaningful performance difference between them
class DQN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_size[0], 32, kernel_size = 8, stride = 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
            torch.nn.ReLU()
        )

        # Get output size of conv by running zeros through network, a little hacky, but works
        conv_out_size = int(np.prod(self.conv(torch.zeros(1, *input_size)).size()))

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, output_size)
        )

    # Forward pass method, pass values through conv first, then flatten results and pass through fc
    def forward(self, x):
        # Flatten may not be correct:
        # Others used: .view(x.size()[0], -1), but I think it'll have the same effect
        # I think this will probably need changing when you change to mini-batch
        return self.fc(self.conv(x).flatten())


# Creates the actual gym, adds a load of wrappers to make things easier to process
def create_env():
    # Uses the rectangles rom, a significantly simplified version of the ROM
    env = gym_super_mario_bros.make('SuperMarioBros-v3')
    # Wrapper for multiple level support, not currently use
    # env = wrappers.MarioXReward(env) # Remove?
    # Returns only every fourth frame
    env = wrappers.MaxAndSkipEnv(env)
    # Converts frame down to 84x84
    env = wrappers.ProcessFrame84(env)
    # Converts the image to PyTorch friendly format
    env = wrappers.ImageToPyTorch(env)
    # Repeats each input for 4 frames
    env = wrappers.BufferWrapper(env, 4)
    # Scale pixel values from 255-0 ints to 1-0 floats (which makes it easier for PyTorch)
    env = wrappers.ScaledFloatFrame(env)
    # 2 different control schemes (I've found right only trains quicker)
    # return JoypadSpace(env, gym_super_mario_bros.actions.SIMPLE_MOVEMENT)
    return nes_py.wrappers.JoypadSpace(env, gym_super_mario_bros.actions.RIGHT_ONLY)


# Gets an action from the policy
def get_action(state):
    # Check for exploration
    if random.random() > 0.2:
        # Only get action from local model
        # State needs to be converted into the correct format (a FloatTensor), which is then discarded and turned into
        # a regular numpy array, the index of the max value is returned from that
        return np.argmax(models[0].forward(torch.FloatTensor(state)).detach().numpy())
    else:
        # Random exploration
        return env.action_space.sample()


# Calculates loss
# I'm not sure if the training problem is in here or somewhere else?
def calc_loss(state, action, next_state, reward, done):
    # Get the q value for the current action
    current_q = [model.forward(torch.FloatTensor(state))[action] for model in models]
    # Get q values for next actions
    next_q = torch.min(*[torch.max(model.forward(torch.FloatTensor(next_state))) for model in models])
    # move lambda somewhere better?
    # Expected reward given the expected q value
    expected_q = next_q * 0.99 * (1 - done) + reward
    # print(expected_q)

    # Return distance between expected and real
    return [torch.nn.functional.mse_loss(q, expected_q) for q in current_q]


# Create environment
env = create_env()

# Create local and target network
models = [
    DQN(env.observation_space.shape, env.action_space.n),
    DQN(env.observation_space.shape, env.action_space.n)
]

# Create optimizers, Adam is most popular here
opts = [torch.optim.Adam(model.parameters()) for model in models]

# Run 10000 episodes (rarely reached)
for episode in range(10000):
    episode_reward = 0
    state = env.reset()

    # Max steps, may need tweaking
    for step in range(1000):
        action = get_action(state)

        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        loss = calc_loss(state, action, next_state, reward, done)

        # Reset the gradients of the optimizers
        opts[0].zero_grad()
        opts[1].zero_grad()

        # Reverse loss (so that it can be used for gradient descent)
        loss[0].backward(retain_graph = True)
        loss[1].backward()

        # Take a single gradient descent step
        opts[0].step()
        opts[1].step()

        # Move to next state
        state = next_state

        # Stop episode if episode is finished
        if done:
            break

        env.render()

    print(f"Episode: {episode}, reward: {episode_reward}")

env.close()
