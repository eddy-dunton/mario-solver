import random

import gym_super_mario_bros, gym_super_mario_bros.actions
import nes_py.wrappers
import torch
import numpy as np

#TODO:
# Implement mini-batch training
# GPU accelerate

class DQN(torch.nn.module):
    def __init__(self, input_size, output_size):
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )

        # Get output size by running zeros through network, a little hacky, but easy
        conv_out_size = int(np.prod(self.conv(torch.zeros(1, input_size))))

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size)
        )

        def forward(self, x):
            # Flatten may not be correct:
            # Others used: .view(x.size()[0], -1), but I think it'll have the same effect
            return self.fc(self.conv(x).flatten())

def get_action(state):
    # Check for exploration
    if (random.random() > 0.2):
        # Detach requied?
        return np.argmax(models[0].forward(torch.FloatTensor(state)))
    else:
        return env.action_space.sample()

def update(state, action, next_state, reward, done):

def loss(state, action, next_state, reward, done):
    

#Create env
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = nes_py.wrappers.JoypadSpace(env, gym_super_mario_bros.actions.SIMPLE_MOVEMENT)

# Create networks
#Action space may need to be converted to an int
models = [
    DQN(env.observation_space.shape, env.action_space.n),
    DQN(env.observation_space.shape, env.action_space.n)
]

opt = [torch.optim.Adam(model.parameters()) for model in models]

rewards = []
for episode in range(10000):
    episode_reward = 0
    state = env.reset()

    # Max steps, may need tweaking
    for step in range(5000):
        action = get_action(state)

        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        update(state, action, next_state, reward, done)

        state = next_state

        env.render()

env.close()