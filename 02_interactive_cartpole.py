import os
import warnings

# 1. Suppress TensorFlow oneDNN logs (Must be done before importing tensorflow)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TF info/warnings

# 2. Suppress Warnings (pkg_resources, np.object, etc)
# Use broader filters to ensure we catch them
warnings.simplefilter("ignore", category=UserWarning) 
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="pkg_resources")
warnings.filterwarnings("ignore", message=".*pkg_resources.*") 
warnings.filterwarnings("ignore", message=".*np.object.*")

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
import pygame
import sys
import math

# ==========================================
# 1. SETTINGS & HYPERPARAMETERS
# ==========================================
ENV_NAME = "CartPole-v1"
GAMMA = 0.99             # Higher focus on long-term survival
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999    # Slow decay for deeper learning
LEARNING_RATE = 0.001
BATCH_SIZE = 64

# UI Settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER_COLOR = (100, 149, 237)
TEXT_COLOR = (255, 255, 255)
BG_COLOR = (30, 30, 30)

# ==========================================
# 2. THE BRAIN (NEURAL NETWORK)
# ==========================================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = EPSILON_START
        
        # Main Model (Learns every step)
        self.model = self._build_model()
        
        # Target Model (Stabilizes predictions)
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Increased capacity (64 neurons) for Pro-Level learning
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'), 
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
        
    def act(self, state, force_greedy=False):
        if not force_greedy and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        states = np.array([i[0] for i in minibatch]).squeeze()
        next_states = np.array([i[3] for i in minibatch]).squeeze()
        
        # Predict Q-values for starting states using the MAIN model
        targets = self.model.predict_on_batch(states)
        
        # Predict future Q-values using the TARGET model (Stable!)
        next_q_values = self.target_model.predict_on_batch(next_states)
        
        # Convert to numpy to allow modification
        if hasattr(targets, 'numpy'): targets = targets.numpy()
        if hasattr(next_q_values, 'numpy'): next_q_values = next_q_values.numpy()
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + GAMMA * np.amax(next_q_values[i])
            targets[i][action] = target
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

# ==========================================
# 3. GUI HELPERS
# ==========================================
class Button:
    def __init__(self, x, y, w, h, text, callback):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.is_hovered = False

    def draw(self, screen, font):
        color = BUTTON_HOVER_COLOR if self.is_hovered else BUTTON_COLOR
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        
        text_surf = font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered and event.button == 1:
                self.callback()

# ==========================================
# 4. MAIN APP (SINGLE AGENT)
# ==========================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("CartPole AI: Single Agent Pro")
    font = pygame.font.SysFont("Arial", 20)
    clock = pygame.time.Clock()

    # AI Setup
    env = gym.make(ENV_NAME, render_mode='rgb_array')
    env.unwrapped.theta_threshold_radians = math.pi / 2
    
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    agent = DQNAgent(state_size, action_size)

    # State Variables
    is_training = False
    is_turbo = False
    episode_count = 0
    current_score = 0
    best_score = 0
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    
    def start_training_normal():
        nonlocal is_training, is_turbo
        is_training = True
        is_turbo = False
        
    def start_training_turbo():
        nonlocal is_training, is_turbo
        is_training = True
        is_turbo = True
        
    def stop_training():
        nonlocal is_training, is_turbo
        is_training = False
        is_turbo = False

    # Buttons
    btn_start = Button(25, 350, 200, 50, "Start (1x)", start_training_normal)
    btn_turbo = Button(25, 410, 200, 50, "Turbo (25x)", start_training_turbo)
    btn_stop = Button(25, 470, 200, 50, "Stop", stop_training)
    buttons = [btn_start, btn_turbo, btn_stop]

    # Layout Dimensions
    SIDEBAR_WIDTH = 250
    GAME_WIDTH = SCREEN_WIDTH - SIDEBAR_WIDTH

    running = True
    while running:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            for btn in buttons:
                btn.handle_event(event)

        # AI Logic
        if is_training:
            # Determine speed based on mode
            # User request: 25x speed, but more FPS
            steps_per_frame = 25 if is_turbo else 1
            
            for _ in range(steps_per_frame): 
                action = agent.act(state)
                next_state, reward, done, truncated, _ = env.step(action)
                
                # --- CUSTOM REWARD SYSTEM ---
                if done:
                    reward = -100
                else:
                    angle = next_state[2]
                    reward = 1.0 - (abs(angle) / (math.pi / 2)) 
                
                next_state = np.reshape(next_state, [1, state_size])
                
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                current_score += 1
                
                # Track Best Score
                if current_score > best_score:
                    best_score = current_score
                
                if done or truncated:
                    print(f"Episode: {episode_count}, Score: {current_score}, Epsilon: {agent.epsilon:.2f}")
                    episode_count += 1
                    current_score = 0
                    state, _ = env.reset()
                    state = np.reshape(state, [1, state_size])
                    
                    # Update Target Network for PRO stability
                    agent.update_target_model()
                    
                    agent.replay()
                
        # Rendering
        screen.fill(BG_COLOR)
        
        # 1. Draw Sidebar Background
        sidebar_rect = pygame.Rect(0, 0, SIDEBAR_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(screen, (40, 40, 40), sidebar_rect)
        pygame.draw.line(screen, (100, 100, 100), (SIDEBAR_WIDTH, 0), (SIDEBAR_WIDTH, SCREEN_HEIGHT), 2)

        # 2. Draw Environment (Centered in remaining space)
        frame = env.render()
        if frame is not None:
            frame = np.transpose(frame, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            
            # Center in the Game Area (Right side)
            game_center_x = SIDEBAR_WIDTH + (GAME_WIDTH // 2)
            x_pos = game_center_x - (surf.get_width() // 2)
            y_pos = (SCREEN_HEIGHT - surf.get_height()) // 2
            
            screen.blit(surf, (x_pos, y_pos))

        # 3. UI Stats (Inside Sidebar)
        status_text = "PAUSED"
        if is_training:
            status_text = "TURBO (25x)" if is_turbo else "NORMAL (1x)"
            
        stats = [
            f"Episode: {episode_count}",
            f"Best Score: {best_score}",
            f"Score: {current_score}",
            f"Epsilon: {agent.epsilon:.2f}",
            f"Status: {status_text}"
        ]
        
        # Title
        title_surf = font.render("CartPole Pro", True, (100, 200, 255))
        screen.blit(title_surf, (20, 20))

        # Stats List
        for i, stat in enumerate(stats):
            text = font.render(stat, True, TEXT_COLOR)
            screen.blit(text, (20, 80 + i * 40))

        # 4. Draw Buttons
        for btn in buttons:
            btn.draw(screen, font)

        pygame.display.flip()
        
        # Increase FPS to 120 for smoother visuals
        clock.tick(120)

    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
