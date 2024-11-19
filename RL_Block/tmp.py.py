# %%
import gymnasium as gym
import minigrid
from minigrid.envs import GoToObjectEnv
import pygame
import numpy as np
import torch
import os

# %%
env = GoToObjectEnv(
    size=20, 
    max_steps=200, 
    render_mode="rgb_array",
    # agent_view_size=3,
    # see_through_walls=False,
)

# %%
os.environ['SDL_VIDEO_WINDOW_POS'] = '500,0'

# %%
obs, _ = env.reset()
# print(obs['image'])
# print(obs['direction'])
print(obs['mission'])

# %%
pygame.init()
window_size = (env.render().shape[1], env.render().shape[0])  # Get size from the env frame
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Minigrid Environment")

# %%
import random

for _ in range(200):  # Take 20 random actions
    # while action:= int(input("What action should the agent take: ")) not in []:
    action = random.randint(0, 2)
    # action = env.action_space.sample()  # Sample a random action
    obs, reward, terminated, truncated, info = env.step(action)  # Take the action
    frame = env.render()
    frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1,0,2)))
    screen.blit(frame_surface, (0,0))
    pygame.display.flip()
    
    # print("Action:", action)
    # print("Observation:", obs)
    # print("Reward:", reward)
    # print("Terminated:", terminated)

    if terminated or truncated:
        obs = env.reset()  # Reset if the episode is over

# %%
env.close()
pygame.quit()

# %%
from agent import Agent

# %%
agent = Agent(
    input_space_size=20,
    output_space_size=7,
    lstm_hidden_size=128, 
    lstm_num_layers=2,
)

# %%
dummy_data = np.random.rand(3, 2, 2, 5)
dummy_data.shape

# %%
t = torch.tensor(dummy_data, dtype=torch.float32).flatten(1)
t.size()

# %%
agent.train()


