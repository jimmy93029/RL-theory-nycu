# Spring 2024, 535514 Reinforcement Learning
# HW2: REINFORCE and baseline

import os
import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_3")
        
class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the GAE parameters, shared layer(s), the action layer(s), and the value layer(s))
            2. Random weight initialization of each layer
    """
    def __init__(self):
        super(Policy, self).__init__()
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        self.double()
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_size),
            nn.ReLU()
        )

        # Actor layers
        if self.discrete:
            self.actor = nn.Sequential(
                nn.Linear(self.hidden_size, self.action_dim),
                nn.Softmax(dim=-1)
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(self.hidden_size, self.action_dim)
            )

        # Critic layers
        self.critic = nn.Sequential(  
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

        # Initialize weights
        self.apply(self.init_weights)
        ########## END OF YOUR CODE ##########
        
        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.1)

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########
        # Shared layers
        features = self.shared_layers(state)

        # Actor
        action_prob = self.actor(features)

        # Critic
        state_value = self.critic(features)

        ########## END OF YOUR CODE ##########

        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """

        ########## YOUR CODE HERE (3~5 lines) ##########
        state = torch.FloatTensor(state).unsqueeze(0)
        action_prob, state_value = self.forward(state)

        if self.discrete:
            distribution = torch.distributions.Categorical(action_prob)
            action = distribution.sample()
        else:
            action = action_prob

        ########## END OF YOUR CODE ##########

        # save to action buffer
        self.saved_actions.append(SavedAction(distribution.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self, gamma=0.999):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """
        
        # Initialize the lists and variables
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 
        returns = []

        ########## YOUR CODE HERE (8-15 lines) ##########
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # use value function as baseline
        for (log_prob, _), advantage in zip(self.saved_actions, returns):
            policy_losses.append(-advantage * log_prob)

        loss = torch.stack(policy_losses).sum()
        ########## END OF YOUR CODE ##########
        
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]


class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps          # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, done):
        """
            Implement Generalized Advantage Estimation (GAE) for your value prediction
            TODO (1): Pass correct corresponding inputs (rewards, values, and done) into the function arguments
            TODO (2): Calculate the Generalized Advantage Estimation and return the obtained value
        """
        ########## YOUR CODE HERE (8-15 lines) ##########
        advantages = []
        advantage = 0
        next_value = 0

        # gen_advantage(t) = sum((r * lambda)^(l) * td_error(t+l)) from l = 0 to infinity
        for r, v in zip(reversed(rewards), reversed(values)):
            td_error = r + self.gamma * next_value - v
            advantage = td_error + (self.gamma * self.lambda_) * advantage
            next_value = v
            advantages.insert(0, advantage)

        advantages = torch.tensor(advantages)
        return advantages
        ########## END OF YOUR CODE ##########


def train(lr=0.01, lambda_=0.95):
    """
        Train the model using SGD (via backpropagation)
        TODO (1): In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode

        TODO (2): In each episode, 
        1. record all the value you aim to visualize on tensorboard (lr, reward, length, ...)
    """
    
    # Instantiate the policy model and the optimizer
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    # scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    
    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0

        # Uncomment the following line to use learning rate scheduler
        # scheduler.step()
        
        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process
        
        ########## YOUR CODE HERE (10-15 lines) ##########
        while t < 9999:
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)

            # saved reward
            model.rewards.append(reward)
            ep_reward += reward

            t += 1
            if done:
                break

        # After finishing an episode, collect values and calculate loss
        values = torch.cat([sa.value for sa in model.saved_actions])
        rewards = torch.FloatTensor(model.rewards)
        if t < 9999:
            # If the episode ended due to the time limit, estimate the value of the last state
            with torch.no_grad():
                _, last_value = model.forward(torch.FloatTensor(state).unsqueeze(0))
                values = torch.cat([values, last_value])
        else:
            # If the episode ended due to the maximum steps, just set the last value to 0
            values = torch.cat([values, torch.zeros(1)])

        gae = GAE(gamma=0.99, lambda_=lambda_, num_steps=None)  # Set num_steps to None for full batch
        advantages = gae(model.rewards, values, done)

        # Calculate loss
        log_probs_and_discount = []
        gammas = 1
        for (log_prob, _), gen_advantage in zip(model.saved_actions, advantages):
            log_probs_and_discount.append(gammas * log_prob * gen_advantage)

        policy_loss = -torch.cat(log_probs_and_discount).sum()
        value_loss = F.mse_loss(values[:-1], rewards.unsqueeze(1))
        loss = policy_loss + value_loss

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # clear memory of the current trajectory
        model.clear_memory()
        ########## END OF YOUR CODE ##########
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        # Try to use Tensorboard to record the behavior of your implementation
        ########## YOUR CODE HERE (4-5 lines) ##########
        writer.add_scalar("Reward", ep_reward, i_episode)
        writer.add_scalar("length", t, i_episode)
        writer.add_scalar("ewma reward", ewma_reward, i_episode)
        writer.add_scalar("learning rate", lr, i_episode)
        ########## END OF YOUR CODE ##########

        # check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
        if ewma_reward > 120:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), f'./preTrained/LunarLander_gae_{lr}_lambda_{lambda_}.pth')
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


def test(name, n_episodes=10):
    """
        Test the learned model (no change needed)
    """     
    model = Policy()
    
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True
    max_episode_len = 10000
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10  
    lr = 0.015
    lambda_ = 0.98
    env = gym.make('LunarLander-v2')
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  
    train(lr, lambda_)
    test(f'LunarLander_gae_{lr}_lambda_{lambda_}.pth')
