from datetime import datetime
from gymnasium.wrappers import FlattenObservation

from snakeenv import SnakeEnv

from tqdm import tqdm
import torch.nn as nn
import torch
import random

from collections import deque

class DQN(nn.Module):

    def __init__(self, num_observations, num_actions):
        super(DQN, self).__init__()

        self.layer1 = nn.Linear(num_observations, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class SnakeAgent:

    def __init__(
            self,
            device,
            game_size_x=5,
            game_size_y=5,
            num_episodes=10000,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_end_episode = 7000,
            learning_rate=0.0003,
            target_net_update_rate=0.005,
            batch_size=512,
            replay_memory_size=10000
    ):
        self.device = device
        self._game_size_x = game_size_x
        self._game_size_y = game_size_y
        self._num_episodes = num_episodes
        self._gamma = gamma
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_end_episode = epsilon_end_episode
        self._learning_rate = learning_rate
        self._target_net_update_rate = target_net_update_rate
        self._batch_size = batch_size
        self._replay_memory_size = replay_memory_size

    def train(self, save_path="model.pth"):
        print(f"Starting training at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; Device: {self.device}")

        env, policy_net, target_net, optimizer, reply_memory = self._initialize_training()

        steps_in_episode_batch = 0
        rewards_in_episode_batch = 0
        for episode in tqdm(range(self._num_episodes)):
            observation, info = env.reset()
            observation = self._process_observation(observation)
            done = False
            while not done:
                epsilon = self._epsilon_start - \
                          (episode / self._epsilon_end_episode) * (self._epsilon_start - self._epsilon_end)
                action = self._get_action(env, policy_net, observation, epsilon)
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_observation = self._process_observation(next_observation)
                reward = float(reward)

                reply_memory.append((
                    observation,  # tensor (1, n_obs)
                    action,  # scalar int
                    reward,  # float
                    done,  # bool
                    next_observation  # tensor (1, n_obs)
                ))

                self._optimize_model(policy_net, target_net, optimizer, reply_memory)

                observation = next_observation

                self._soft_update_net(policy_net, target_net)

                steps_in_episode_batch += 1
                rewards_in_episode_batch += reward

            if episode % 100 == 0:
                print(f"Episode: {episode} || Average steps: {steps_in_episode_batch / 100} || Average reward: {rewards_in_episode_batch / 100}")
                steps_in_episode_batch = 0
                rewards_in_episode_batch = 0

        env.close()

        print(f"Training finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
        print(f"Saving model to {save_path}")
        torch.save(target_net.state_dict(), save_path)

    def play_once(self, model_path="model.pth"):
        env = self._wrap_env(SnakeEnv(render_mode="human", size_x=self._game_size_x, size_y=self._game_size_y))
        n_actions = env.action_space.n
        observation, info = env.reset()
        n_observations = len(observation)
        net = DQN(n_observations, n_actions).to(self.device)
        net.load_state_dict(torch.load(model_path))

        done = False
        while not done:
            observation = self._process_observation(observation)
            action = self._get_action(env, net, observation, 0)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            observation = next_observation
        env.close()

    def _initialize_training(self):
        env = self._wrap_env(SnakeEnv(render_mode=None, size_x=self._game_size_x, size_y=self._game_size_y))

        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        n_actions = env.action_space.n
        observation, info = env.reset()
        n_observations = len(observation)

        print(f"Number of observations: {n_observations}")
        print(f"Number of actions: {n_actions}")

        policy_net = DQN(n_observations, n_actions).to(self.device)
        target_net = DQN(n_observations, n_actions).to(self.device)
        target_net.load_state_dict(policy_net.state_dict())

        print(f"Policy net: {policy_net}")
        print(f"Target net: {target_net}")

        optimizer = torch.optim.Adam(policy_net.parameters(), lr=self._learning_rate)
        reply_memory = deque([], maxlen=self._replay_memory_size)

        return env, policy_net, target_net, optimizer, reply_memory

    def _wrap_env(self, env):
        return FlattenObservation(env)

    def _get_action(self, env, net, observation, epsilon):
        if random.random() < epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                return net(observation).argmax(dim=1, keepdim=False).item()

    def _process_observation(self, observation):
        return torch.as_tensor(observation, device=self.device, dtype=torch.float32).unsqueeze(0)

    def _optimize_model(self, policy_net, target_net, optimizer, reply_memory):
        if len(reply_memory) < self._batch_size:
            return

        transitions = random.sample(reply_memory, self._batch_size)
        batch = list(zip(*transitions))

        # observations and next_observations were stored as tensors; stack them
        observations = torch.stack(batch[0])  # shape: (B, 1, n_obs)
        observations = observations.squeeze(1)  # shape: (B, n_obs)

        next_observations = torch.stack(batch[4])  # shape: (B, 1, n_obs)
        next_observations = next_observations.squeeze(1)  # shape: (B, n_obs)

        # actions stored as scalars; make shape (B, 1)
        actions = torch.tensor(batch[1], dtype=torch.long, device=self.device).unsqueeze(1)  # (B,1)

        # rewards as float
        rewards = torch.tensor(batch[2], dtype=torch.float32, device=self.device)  # (B,)

        # dones as float mask: 1.0 if done else 0.0
        dones = torch.tensor(batch[3], dtype=torch.float32, device=self.device)  # (B,)

        # Q(s,a) for actions taken
        q_values = policy_net(observations).gather(1, actions).squeeze(1)  # (B,)

        with torch.no_grad():
            next_state_values = target_net(next_observations).max(dim=1)[0]  # (B,)
            targets = rewards + self._gamma * (1.0 - dones) * next_state_values  # (B,)

        criterion = nn.SmoothL1Loss()
        loss = criterion(q_values, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    def _soft_update_net(self, policy_net, target_net):
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self._target_net_update_rate \
                                         + target_net_state_dict[key] * (1 - self._target_net_update_rate)
        target_net.load_state_dict(target_net_state_dict)