import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
import imageio
import os

# ==========================================
# 1. SETTINGS & HYPERPARAMETERS
# ==========================================
ENV_NAME = "CartPole-v1"
EPISODES = 50           # Total games to play (increase this for better results, e.g., 200)
MAX_STEPS = 500          # Max steps per game (to prevent infinite loops)
BATCH_SIZE = 32          # How many memories to train on at once
GAMMA = 0.95             # Discount factor (how much we care about future rewards)
EPSILON = 1.0            # Exploration rate (1.0 = 100% random actions)
EPSILON_MIN = 0.01       # Minimum exploration rate
EPSILON_DECAY = 0.995    # How fast we stop exploring randomly
LEARNING_RATE = 0.001    # How fast the network learns

# ==========================================
# 2. THE BRAIN (NEURAL NETWORK)
# ==========================================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) # Memory to store past game experiences
        self.epsilon = EPSILON
        
        # Build the Neural Network
        self.model = Sequential([
            Dense(24, input_dim=state_size, activation='relu'), # Hidden Layer 1
            Dense(24, activation='relu'),                       # Hidden Layer 2
            Dense(action_size, activation='linear')             # Output Layer (Left or Right)
        ])
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))

    def act(self, state):
        # Exploration: Decide randomly based on epsilon
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation: Ask the Neural Network for the best move
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # Returns action 0 or 1

    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        # Learn from past experiences (Training step)
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        
        states = np.array([i[0] for i in minibatch])
        states = np.squeeze(states)
        
        next_states = np.array([i[3] for i in minibatch])
        next_states = np.squeeze(next_states)
        
        targets = self.model.predict_on_batch(states)
        next_q_values = self.model.predict_on_batch(next_states)
        
        # Convert to numpy to allow modification
        if hasattr(targets, 'numpy'):
            targets = targets.numpy()
        if hasattr(next_q_values, 'numpy'):
            next_q_values = next_q_values.numpy()
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + GAMMA * np.amax(next_q_values[i])
            
            targets[i][action] = target
        
        # Train the model
        self.model.fit(states, targets, epochs=1, verbose=0)

        # Decrease exploration rate
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

# ==========================================
# 3. MAIN TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    # Create the environment
    env = gym.make(ENV_NAME, render_mode='rgb_array')
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    
    agent = DQNAgent(state_size, action_size)
    print(f"Starting training on {ENV_NAME}...")

    # Loop for each episode (game)
    for e in range(EPISODES):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        
        total_score = 0
        
        for time in range(MAX_STEPS):
            # 1. Agent picks an action
            action = agent.act(state)
            
            # 2. Environment reacts
            next_state, reward, done, truncated, _ = env.step(action)
            reward = reward if not done else -10 # Penalize falling
            next_state = np.reshape(next_state, [1, state_size])
            
            # 3. Save to memory
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_score += 1
            
            if done or truncated:
                print(f"Episode: {e+1}/{EPISODES}, Score: {total_score}, Epsilon: {agent.epsilon:.2f}")
                break
                
            # 4. Learn from memory!
            agent.replay()
            
    # ==========================================
    # 4. SAVE REPLAY AS GIF
    # ==========================================
    print("\nTraining finished! Recording a video of the trained agent...")
    frames = []
    
    # Run one final game with no exploration (pure exploitation)
    agent.epsilon = 0 
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    
    for i in range(MAX_STEPS):
        frames.append(env.render()) # Capture the frame
        action = agent.act(state)
        state, _, done, truncated, _ = env.step(action)
        state = np.reshape(state, [1, state_size])
        if done or truncated:
            break
            
    output_path = "cartpole_run.gif"
    imageio.mimsave(output_path, frames, fps=30)
    print(f"Video saved to {output_path}. Open it to see your AI in action!")
    
    env.close()
