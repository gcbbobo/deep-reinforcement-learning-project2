#######################################################
# reference : 
#1. Udacity Deep Reinforcement Learning Nanodegree: pong_utils.py
#2. https://github.com/jknthn/reacher-ppo/blob/master/agent.py

#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# Modified by Jeremi Kaczmarczyk (jeremi.kaczmarczyk@gmail.com) 2018 
# For Udacity Deep Reinforcement Learning Nanodegree

#######################################################

import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib import animation
from IPython.display import display
import random as rand

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# collect trajectories for a parallelized parallelEnv object
def collect_trajectories(env, policy, tmax=200, nrand=5):
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]     # reset the environment
    
    # number of parallel instances
    n=len(env_info.agents)

    #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    log_prob_list=[]
    action_list=[]

    states = env_info.vector_observations                  # get the current state (for each agent)
    states = torch.from_numpy(states).float().to(device)
    
    for t in range(tmax):
        
        #print('states before',type(states))
        #print('states before:',states)
        action,log_probs = policy.forward(states)
        #print('succeed index:',t)
        
        # To use np and the environment as follows, tranfer everything into numpy
        action = action.cpu().detach().numpy()
        log_probs = log_probs.cpu().detach().numpy()
        action = np.clip(action, -1, 1)                  # all actions between -1 and 1
        
        env_info = env.step(action)[brain_name]           # send all actions to tne environment

        # remove [0] for multiple agents
        next_state = env_info.vector_observations         # get next state (for each agent)
        reward = env_info.rewards                         # get reward (for each agent)
        done = env_info.local_done                        # see if episode finished
        
        # store the result in form of numpy & list
        state_list.append(states)
        reward_list.append(reward)
        log_prob_list.append(log_probs)
        action_list.append(action)
        
        states = next_state
        
        #print('states middle',type(states))
        #print('states middle:',states)
        
        # return states from list to be tensor
        #states = torch.FloatTensor(states)
        states = torch.from_numpy(states).float().to(device)
        #print('states after',type(states))
        #print('states after:',states)
        
        if done:
            break

    # stop if any of the trajectories is done
    # we want all the lists to be retangular
    '''
    if is_done.any():
        break
    '''

    # return pi_theta, states, actions, rewards, probability
    return log_prob_list, state_list, \
        action_list, reward_list


# clipped surrogate function
# similar as -policy_loss for REINFORCE, but for PPO
def clipped_surrogate(policy, old_probs, states, actions, rewards,
                      discount=0.995,
                      epsilon=0.1, beta=0.01):

    discount = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount[:,np.newaxis]
    
    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_log_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)
    states = torch.tensor(states, dtype=torch.float, device=device)
    
    # convert states to policy (or probability)
    __,new_log_probs = policy.forward(states)
    
    # ratio for clipping
    log_ratio = new_log_probs-old_log_probs
    ratio = torch.exp(log_ratio)

    # clipped function
    clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(torch.exp(new_log_probs)*old_log_probs+ \
        (1.0-torch.exp(new_log_probs))*old_log_probs)

    
    # this returns an average of all the entries of the tensor
    # effective computing L_sur^clip / T
    # averaged over time-step and number of trajectories
    # this is desirable because we have normalized our rewards
    return torch.mean(clipped_surrogate + beta*entropy)

import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Policy(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        self.std = nn.Parameter(torch.ones(1, action_size))
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""

        if state.dim()==1:
            state = torch.unsqueeze(state,0)
            
        x = state
        #x = self.bn0(state)
        
        x = self.fc1(x)
        #x = self.bn1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        #x = self.bn2(x)
        x = F.relu(x)
        
        # here x is already the clean action
        x = F.tanh(self.fc3(x))
        
        # adding noise to be real action and for Gradient Ascent
        dist = torch.distributions.Normal(x, self.std)
        
        act = dist.sample()
        
        log_prob = dist.log_prob(act)
        
        # for multiple parallel agents
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        
        return act,log_prob
    
