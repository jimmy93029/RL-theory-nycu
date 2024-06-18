import gym
import numpy as np
import os
import time
import random
from collections import namedtuple
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_cheetah")


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh()
        )

    def forward(self, inputs):
        return self.layers(inputs)


class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.layers = nn.Sequential(
            nn.Linear(num_inputs + num_outputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, inputs, actions):
        x = torch.cat([inputs, actions], 1)
        return self.layers(x)


class DDPG(object):
    def __init__(self, num_inputs, action_space, gamma=0.995, tau=0.0005, hidden_size=128, lr_a=1e-4, lr_c=1e-3):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, action_noise=None):
        self.actor.eval()
        action = self.actor((state))
        action = action.data

        if action_noise is not None:
            action += torch.Tensor(action_noise.noise())

        return action.clamp(-1, 1)

    def update_parameters(self, batch):
        state_batch = torch.cat([transition.state for transition in batch])
        action_batch = torch.cat([transition.action for transition in batch])
        reward_batch = torch.cat([transition.reward for transition in batch])
        mask_batch = torch.cat([transition.mask for transition in batch])
        next_state_batch = torch.cat([transition.next_state for transition in batch])

        next_action_batch = self.actor_target(next_state_batch)
        Q1 = self.critic_target(next_state_batch, next_action_batch)
        target_Q = reward_batch.unsqueeze(1) + (self.gamma * Q1 * mask_batch.unsqueeze(1))
        current_Q = self.critic(state_batch, action_batch)

        critic_loss = F.smooth_l1_loss(current_Q, target_Q.detach())  # why detach

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        local_time = time.localtime()
        timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
        if not os.path.exists('preTrained_cheetah/'):
            os.makedirs('preTrained_cheetah/')

        if actor_path is None:
            actor_path = "preTrained_cheetah/ddpg_actor_{}_{}_{}".format(env_name, timestamp, suffix)
        if critic_path is None:
            critic_path = "preTrained_cheetah/ddpg_critic_{}_{}_{}".format(env_name, timestamp, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))


def train(env, env_name):
    num_episodes = 200
    gamma = 0.995
    tau = 0.002
    hidden_size = 128
    noise_scale = 0.1
    replay_size = 100000
    batch_size = 128
    updates_per_step = 1
    print_freq = 2
    ewma_reward = 0
    rewards = []
    ewma_reward_history = []
    total_numsteps = 0
    updates = 0

    agent = DDPG(env.observation_space.shape[0], env.action_space, gamma, tau, hidden_size)
    ounoise = OUNoise(env.action_space.shape[0])
    memory = ReplayMemory(replay_size)

    for i_episode in range(num_episodes):

        ounoise.scale = noise_scale
        ounoise.reset()

        state = torch.Tensor([env.reset()])

        episode_reward = 0
        while True:
            action = agent.select_action(state, ounoise)
            next_state, reward, done, _ = env.step(action.numpy()[0])

            next_state = torch.Tensor([next_state])
            mask = torch.Tensor([not done])
            reward = torch.Tensor([reward])

            memory.push(state, action, mask, next_state, reward)

            state = next_state
            episode_reward += reward[0]
            total_numsteps += 1

            if len(memory) > batch_size:
                for _ in range(updates_per_step):
                    batch = memory.sample(batch_size)
                    agent.update_parameters(batch)
                    updates += 1

            if done:
                break

        rewards.append(episode_reward)
        t = 0
        if i_episode % print_freq == 0:
            state = torch.Tensor([env.reset()])
            episode_reward = 0
            while True:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action.numpy()[0])
                # env.render()

                episode_reward += reward
                next_state = torch.Tensor([next_state])
                state = next_state

                t += 1
                if done:
                    break

            rewards.append(episode_reward)
            # update EWMA reward and log the results
            ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
            ewma_reward_history.append(ewma_reward)
            print("Episode: {}, length: {}, reward: {:.2f}, ewma reward: {:.2f}".format(i_episode, t, rewards[-1], ewma_reward))

        writer.add_scalar('reward', rewards[-1], i_episode)
        writer.add_scalar('ep_reward', ewma_reward, i_episode)
        writer.add_scalar('episode_length', t, i_episode)

    agent.save_model(env_name, '.pth')


if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10
    env = gym.make('HalfCheetah-v2')
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    env_name = "HalfCheetah-v2"
    train(env, env_name)
