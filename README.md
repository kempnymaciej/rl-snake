# Reinforcement Learning Snake 🐍
This project is a learning and exploration exercise in applying reinforcement learning to the classic Snake game.
Different branches may contain different approaches, experiments, or ideas for solving the problem - so expect some variety!

## Features
### Custom Snake Environment
- Built as a **Gymnasium environment** (OpenAI)
- **Reward structure:**
  - `+1` for eating food 
  - `-1` for dying
- **Actions available:**
  - None (move forward)
  - Turn left 
  - Turn right
- **Observation space includes:**
  - `collisions`: 2D boolean array (snake shape marked as ones, empty spaces as zeros)
  - Snake head coordinates (x, y)
  - Food coordinates (x, y)
  - Snake direction (x, y)

### Learning Agent
- **DQN-based**
- Enhancements:
  - Memory replay 
  - Soft update of the target network
- The agent can:
  - Learn to play 
  - Play in real time after training 

## Technology used
- Python 3.12
- Gymnasium 1.2.0
- PyTorch 2.8.0+cu129
- Jupyter Notebook

## How to run
- Clone the repository
- Install the required packages
- Open `notebook.ipynb`
