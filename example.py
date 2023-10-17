import gymnasium as gym
from ReacherGA import Individual
import torch
import numpy as np

env = gym.make("Reacher-v4", render_mode="human")
observation, info = env.reset(seed=42)

hammerhaj = Individual(neuronAmount=4)

for _ in range(300):
    
   action = env.action_space.sample()  # our policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated:
      observation, info = env.reset()
   elif truncated:
      observation, info = env.reset()
env.close()
