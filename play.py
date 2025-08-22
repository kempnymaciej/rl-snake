import random
import torch

from gymnasium.wrappers import FlattenObservation

from dqn import DQN
from snakeenv import SnakeEnv

env = FlattenObservation(SnakeEnv(render_mode="human", size_x=10, size_y=10))
n_actions = env.action_space.n
observation, info = env.reset()
n_observations = len(observation)
policy_net = DQN(n_observations, n_actions)
policy_net.load_state_dict(torch.load("model.pth"))
device = torch.device("cpu")

observation, info = env.reset()
done = False

def get_action(observation, exploit_only=False):
    if (not exploit_only) and (random.random() < 0):
        return env.action_space.sample()
    else:
        with torch.no_grad():
            return policy_net(observation).argmax(dim=1, keepdim=False).item()

def process_observation(observation):
    return torch.as_tensor(observation, device=device, dtype=torch.float32).unsqueeze(0)


while not done:
    action = get_action(process_observation(observation), exploit_only=True)
    next_observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    observation = next_observation
env.close()