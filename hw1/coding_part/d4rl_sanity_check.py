#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 535514 Reinforcement Learning, HW1

The sanity check suggested by D4RL official repo
Source: https://github.com/Farama-Foundation/D4RL
"""

import gym
import d4rl # Import required to register environments, you may need to also import the submodule

# Create the environment
env = gym.make('halfcheetah-random-v2')

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()
print(dataset['observations']) # An N x dim_observation Numpy array of observations

# Alternatively, use d4rl.qlearning_dataset which
# also adds next_observations.
dataset = d4rl.qlearning_dataset(env)