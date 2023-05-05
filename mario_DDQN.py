import random

import gym_super_mario_bros
import gym_super_mario_bros.actions
import torch
import numpy as np
import nes_py.wrappers
import wrappers
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""
Originally written by Eddy Dunton, eed34@bath.ac.uk
Additional code by Harry Patient , hp610@bath.ac.uk, Ishaan Singh iks33@bath.ac.uk and Daniel Forman (df477@bath.ac.uk)

Based on:

Clipped DDQN (2018)
https://github.com/cyoon1729/deep-Q-networks/blob/cb3f1551bc927fedf7166d7b0b3834aaff07d32e/doubleDQN/clipped_ddqn.py
https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/



Current state:
Working, however there is some strange behaviour with the training, training does work negatively (ex. trying to get the
 lowest possible score), however, only seems to provide average results when training positive, train   ing is also very
 slow right now, I would recommened implementing batches and GPU support first else you'll wait forever to get and 
 results

TODO:
 - GPU acceleration
"""

#GPU Acceleration
#NEED TO INSTALL A PYTORCH VERSION THAT SUPPORTS YOUR CUDA VERSION ELSE THIS WONT WORK
#CAN VERIFY WITH THIS: IF TRUE YOU'RE OK ELSE THIS WONT RUN 
print(torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#ADDED- tweakable parameters
epochs = 2000 #total number of epochs
epsilonstart = 1 
decayrate = 0.9992
burnin = 150
epsilon_min = 0.15
batch_size = 32 #number of samples in mini batch for loss calculation
D_size = 10000 #number of steps in a replay memory
steps = 2000
gamma = 0.9 #learning rate
C = 1000 #how often you update weights on Q' from Q


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
        self.fc = self.fc.to(device)
        self.conv = self.conv.to(device)

    # Forward pass method, pass values through conv first, then flatten results and pass through fc
    def forward(self, x):
        # Flatten may not be correct:
        # Others used: .view(x.size()[0], -1), but I think it'll have the same effect
        # I think this will probably need changing when you change to mini-batch
        return self.fc(self.conv(x).flatten())

# Creates the actual gym, adds a load of wrappers to make things easier to process
def create_env():
    # Uses the rectangles rom, a significantly simplified version of the ROM
    env = gym_super_mario_bros.make('SuperMarioBros-3-2-v3')
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
    # return JoypadSpace(env, gym_super_mario_bros.actions.RIGHT_ONLY)
    return nes_py.wrappers.JoypadSpace(env, gym_super_mario_bros.actions.SIMPLE_MOVEMENT)


# Gets an action from the policy -epsilon greedy
def get_action(state, epsilon):
    # Check for exploration
    if (random.random() > epsilon):
        # Only get action from local model
        # State needs to be converted into the correct format (a FloatTensor), which is then discarded and turned into
        # a regular numpy array, the index of the max value is returned from that
        return np.argmax(models[0].forward(torch.FloatTensor(state).to(device)).cpu().detach().numpy())
    else:
        # Random exploration
        return env.action_space.sample()
    
# Calculates loss
# I'm not sure if the training problem is in here or somewhere else?

#THIS SHOULD NOW BE REDUNDANT, LEFT IN CASE BATCH LOSS DOESN'T WORK
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

#New DDQN Batch loss function with Experience 
#https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c -- Implentation inspiration from here
def batch_loss(batch): 
    y = []
    Q = []
    for transition in batch:
        state,action,reward,next_state, done = transition
        qi = models[0].forward(torch.cuda.FloatTensor(state).to(device))[action]
        Q.append(qi)
        if done:
            yi = reward
        else:
            k = torch.argmax(models[0].forward(torch.cuda.FloatTensor(state).to(device)))
            yi = reward + (1-done)* gamma* models[1].forward(torch.cuda.FloatTensor(next_state).to(device))[k]
        y.append(yi)
    Q_t = torch.tensor(Q).to(device)
    y_t = torch.tensor(y).to(device)

    return Q_t,y_t

# Create environment
env = create_env()

# Create local and target network
models = [
    DQN(env.observation_space.shape, env.action_space.n),
    DQN(env.observation_space.shape, env.action_space.n)
]

# Create optimizers, Adam is most popular here
opt = torch.optim.AdamW(models[0].parameters(),lr=0.01)

#Running the algorithm 

D = [] #experience replay

env_rand=create_env()

#Trains a random agent that does not learn (Just for results comparison)
def train_random():
    total_rewards= []
    for i in range(epochs):
        obs = env_rand.reset()
        random_episode_reward=0

        for step in range(steps):
            action = env_rand.action_space.sample()
            obs, reward, done, info = env_rand.step(action)
            random_episode_reward+= reward
            if done:
                break
        total_rewards.append(random_episode_reward)
        
    env_rand.close()    
    return  total_rewards 

total_episode_reward=[]
# Run 10000 episodes (rarely reached)

for episode in range(epochs):
    
    episode_reward = 0
    state = env.reset() #essentially initalised agent 
    eps = epsilonstart*(decayrate**(episode)) + 0.2
    eps = max(eps,epsilon_min)


    # Max steps, may need tweaking
    for step in range(steps):
        action = get_action(state, eps)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        #IMPLEMENTATION OF EXPERIENCE BUFFER (MINI BATCH)
        memory = (state,action,reward,next_state,done)
        D.append(memory)

        if len(D) >= batch_size+1:
            batch = random.sample(D,batch_size)
            Q_t,y_t = batch_loss(batch)
            loss = torch.nn.MSELoss()(Q_t,y_t)
            loss = torch.autograd.Variable(loss, requires_grad = True)
            # Reset the gradients of the optimizers
            opt.zero_grad()
            # Reverse loss (so that it can be used for gradient descent)
            loss.backward(retain_graph = True)
            # Take a single gradient descent step
            opt.step()
            # Move to next state
            if step % C ==0 and episode > burnin:
                models[1].load_state_dict(models[0].state_dict()) # every C iterations update weights of Q' to be Q
            #remove first element so rolling memory
            D.pop(0)
            state = next_state
            # Stop episode if episode is finished
            if done:
                break
        env.render()

        
    total_episode_reward.append(episode_reward)    
    print(f"Episode: {episode}, reward: {episode_reward}")
    
env.close()    
    
random_rewards = train_random()
  
""" Rest of the code is just to plot the graphs"""    

iters = list(range(epochs))    

x = iters
y1 = total_episode_reward 
y2 = random_rewards
    
plt.plot(x, y1, color='r', label='Our Agent')
plt.plot(x, y2, color='g', label='Random Agent')
  
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Rewards per Episode by the DDQN agent vs An Agent that doesn not learn anything")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()


