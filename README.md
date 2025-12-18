# TensorFlow CartPole Project

This project uses **Reinforcement Learning** to teach an AI how to balance a pole on a cart.

## Setup

1.  **Activate the Environment**:
    Open your terminal in this folder and run:
    ```powershell
    .venv\Scripts\activate
    ```

2.  **Run the Agent**:
    ```powershell
    python 01_cartpole_agent.py
    ```

## What will happen?

1.  **Training**: You will see logs like `Episode: 1/50, Score: 12`.
    - **Score** is how many frames the pole stayed up (Max 500).
    - At first, the score will be low (random actions).
    - As the episodes progress, the score should increase as the AI learns the physics.

2.  **Result**: 
    - At the end, a file named `cartpole_run.gif` will be created.
    - Open this GIF to watch your trained AI in action!

## Interactive Mode (New!)

I have added a visual control panel where you can watch the learning live.

1.  **Run the Interactive App**:
    ```powershell
    .venv\Scripts\python 02_interactive_cartpole.py
    ```

2.  **Controls**:
    - **Start Learning**: The AI begins practicing. The game speeds up to train faster.
    - **Stop Learning**: Pause the training to inspect the current score.
    - **Stats**: Watch the "Episode" counter and see "Epsilon" go down (meaning the AI is becoming more confident).

## Key Concepts
- **Agent**: The AI player.
- **Environment**: The game world (physics of the pole).
- **Action**: Move Left or Move Right.
- **Reward**: +1 point for every moment the pole is upright. -10 points if it drops.
- **Q-Network**: The brain that predicts which action gives the most future reward.
