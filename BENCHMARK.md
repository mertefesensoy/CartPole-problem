# 🏆 AI Learning Benchmark: CartPole DQN

**Date:** December 18, 2025
**Project:** CartPole-v1 Reinforcement Learning

## 📖 Overview
This document serves as a benchmark record for my journey into Reinforcement Learning (RL). The goal was to train an intelligent agent to balance a pole on a cart using a Deep Q-Network (DQN).

## 🧪 Experiment Setup

### 1. The Environment (CartPole-v1)
We modified the standard `gymnasium` environment to be more "forgiving" but strictly penalized:
- **Physics Change**: The pole is allowed to fall completely (90 degrees / $\pi/2$ radians) before the episode resets.
- **Goal**: Keep the pole upright for as long as possible (Max default: 500 steps).

### 2. The Agent (Brain)
- **Architecture**: Neural Network (TensorFlow Keras)
    - Input Layer: 4 States (Cart Position, Cart Velocity, Pole Angle, Pole Velocity)
    - Hidden Layer 1: 64 Neurons (ReLU activation) - *Upgraded to Pro*
    - Hidden Layer 2: 64 Neurons (ReLU activation) - *Upgraded to Pro*
    - Output Layer: 2 Actions (Left, Right)
- **Optimizer**: Adam
- **Learning Rate**: 0.001

### 3. Hyperparameters (The Tuning)
| Parameter | Value | Reason |
| :--- | :--- | :--- |
| **Memory Size** | 10,000 steps | Allows the agent to remember history to avoid sticking to local minima. |
| **Batch Size** | 64 | Learns from more diverse examples at once. |
| **Gamma** | 0.99 | Prioritizes long-term future rewards. |
| **Epsilon Decay** | 0.999 | Extremely slow decay to force thorough exploration. |
| **Min Epsilon** | 0.05 | Always keeps 5% randomness. |

### 4. Reward Shaping (The "Secret Sauce")
Instead of the default "+1 for every frame", we implemented a custom reward function:
1.  **Verticality Bonus**: `Reward = 1.0 - (Angle / 90 degrees)`.
    - Perfect vertical = 1.0 points.
    - Leaning = ~0.5 points.
    - Falling = ~0.0 points.
2.  **Death Penalty**: If the pole hits the ground, the agent receives **-100 points**.

## 📊 Results & Observations
- **Initial Phase (Epsilon ~1.0 - 0.7)**:
  - The agent began exploring the environment, learning basic balance.

- **Progress Phase (Epsilon ~0.66 - 0.53)**:
  - Episode 415 (Epsilon 0.66): Score **295**. Significant improvement.
  - Episode 627 (Epsilon 0.53): Score **445**. Near mastery, consistent balancing.

- **Mastery Phase (Epsilon < 0.35)**:
  - **Solved**: At Episode 1079 (Epsilon 0.34), the agent achieved the perfect score of **500**.
  - The policy is now robust.

**Verification Log:**
```text
Episode: 415, Score: 295, Epsilon: 0.66
Episode: 627, Score: 445, Epsilon: 0.53
Episode: 1079, Score: 500, Epsilon: 0.34
```

## 🚀 Reproduction
1.  Activate environment: `.venv\Scripts\activate`
2.  Run the trainer: `python 01_CartPole_DQN/02_interactive_cartpole.py`
