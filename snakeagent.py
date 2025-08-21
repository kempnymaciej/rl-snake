import torch
import torch.nn as nn
import math
import random

from collections import namedtuple, deque

Transition = namedtuple('Transition', ('observation', 'action', 'next_observation', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity, seed=None):
        self.memory = deque([], maxlen=capacity)
        self.random = random.Random(seed)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return self.random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, num_observations, num_actions):
        super(DQN, self).__init__()

        self.layer1 = nn.Linear(num_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class SnakeAgent:

    def __init__(
            self,
            env,
            batch_size,
            gamma,
            epsilon_start,
            epsilon_end,
            epsilon_decay,
            target_net_update_rate,
            optimizer_learning_rate
    ):
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_net_update_rate = target_net_update_rate

        self.epsilon = epsilon_start
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        n_actions = self.env.action_space.n
        observation, info = env.reset()
        n_observations = len(observation)

        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=optimizer_learning_rate)
        self.reply_memory = ReplayMemory(10000)

        self.steps_done = 0

    def get_action(self, observation, exploit_only=False):
        if (not exploit_only) and (random.random() < self.epsilon):
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                observation = self._process_observation(observation)
                q = self.policy_net(observation)  # shape: (1, num_actions)
                action = q.argmax(dim=1, keepdim=True).to(torch.long)  # shape: (1, 1)
                return action

    def update(
            self,
            observation,
            action,
            reward,
            terminated,
            next_observation
    ):
        observation = self._process_observation(observation)
        next_observation = self._process_observation(next_observation)
        reward = torch.tensor([reward], device=self.device)

        self.reply_memory.push(
            observation,
            action,
            next_observation,
            reward,
        )
        self._optimize_model()

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.target_net_update_rate + \
                target_net_state_dict[key] * (1 - self.target_net_update_rate)
        self.target_net.load_state_dict(target_net_state_dict)

        self.epsilon = self.epsilon_end + \
                       (self.epsilon_start - self.epsilon_end) * math.exp(-1 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1


    def _optimize_model(self):

        if len(self.reply_memory) < self.batch_size:
            return

        transitions = self.reply_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_observation)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_observation if s is not None])
        state_batch = torch.cat(batch.observation)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = torch.nn.SmoothL1Loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def _process_observation(self, observation):
        observation = torch.as_tensor(observation, device=self.device, dtype=torch.float)
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)  # shape: (1, num_features)
        return observation
