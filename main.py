from gymnasium.wrappers import FlattenObservation

from dqn import DQN
from snakeenv import SnakeEnv

from tqdm import tqdm
import torch.nn as nn
import torch
import math
import random

from collections import deque

GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 2500
LEARNING_RATE = 0.0003
TARGET_NET_UPDATE_RATE = 0.005
BATCH_SIZE = 512
REPLAY_MEMORY_SIZE = 10000
NUM_EPISODES = 8000

env = FlattenObservation(SnakeEnv(render_mode=None, size_x=5, size_y=5))
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print(f"Device: {device}")

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

n_actions = env.action_space.n
observation, info = env.reset()
n_observations = len(observation)

print(f"Number of observations: {n_observations}")
print(f"Number of actions: {n_actions}")

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

print(f"Policy net: {policy_net}")
print(f"Target net: {target_net}")

optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
reply_memory = deque([], maxlen=REPLAY_MEMORY_SIZE)

steps_done = 0
epsilon = EPSILON_START

def get_action(observation, exploit_only=False):
    if (not exploit_only) and (random.random() < epsilon):
        return env.action_space.sample()
    else:
        with torch.no_grad():
            return policy_net(observation).argmax(dim=1, keepdim=False).item()

def process_observation(observation):
    return torch.as_tensor(observation, device=device, dtype=torch.float32).unsqueeze(0)

def optimize_model():
    if len(reply_memory) < BATCH_SIZE:
        return

    transitions = random.sample(reply_memory, BATCH_SIZE)
    batch = list(zip(*transitions))

    # observations and next_observations were stored as tensors; stack them
    observations = torch.stack(batch[0])  # shape: (B, 1, n_obs)
    observations = observations.squeeze(1)  # shape: (B, n_obs)

    next_observations = torch.stack(batch[4]).squeeze(1)  # shape: (B, n_obs)

    # actions stored as scalars; make shape (B, 1)
    actions = torch.tensor(batch[1], dtype=torch.long, device=device).unsqueeze(1)  # (B,1)

    # rewards as float
    rewards = torch.tensor(batch[2], dtype=torch.float32, device=device)  # (B,)

    # dones as float mask: 1.0 if done else 0.0
    dones = torch.tensor(batch[3], dtype=torch.float32, device=device)  # (B,)

    # Q(s,a) for actions taken
    q_values = policy_net(observations).gather(1, actions).squeeze(1)  # (B,)

    with torch.no_grad():
        # Use target network for the bootstrap
        next_state_values = target_net(next_observations).max(dim=1)[0]  # (B,)
        targets = rewards + GAMMA * (1.0 - dones) * next_state_values  # (B,)

    criterion = nn.SmoothL1Loss()
    loss = criterion(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    if steps_done % 500 == 0:
        print(f"Loss: {loss.item()}")

steps_in_episode_batch = 0
for episode in tqdm(range(NUM_EPISODES)):
    observation, info = env.reset()
    observation = process_observation(observation)
    done = False
    while not done:
        action = get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        steps_in_episode_batch += 1
        done = terminated or truncated
        next_observation = process_observation(next_observation)

        reward = float(reward)

        reply_memory.append((
            observation,  # tensor (1, n_obs)
            action,  # scalar int
            reward,  # float
            done,  # bool
            next_observation  # tensor (1, n_obs)
        ))

        optimize_model()

        observation = next_observation

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TARGET_NET_UPDATE_RATE \
                                         + target_net_state_dict[key] * (1 - TARGET_NET_UPDATE_RATE)
        target_net.load_state_dict(target_net_state_dict)

        steps_done += 1
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                  math.exp(-1. * steps_done / EPSILON_DECAY)

    if episode % 100 == 0:
        print(f"Episode: {episode} Average steps in episode: {steps_in_episode_batch / 100}")
        steps_in_episode_batch = 0

env.close()

torch.save(target_net.state_dict(), "model.pth")
# env = FlattenObservation(SnakeEnv(render_mode="human", size_x=10, size_y=10))
# observation, info = env.reset()
# done = False
#
# while not done:
#     action = get_action(process_observation(observation), exploit_only=True)
#     next_observation, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated
#     observation = next_observation
# env.close()