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
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05       # Never stop exploring completely (5% random)
EPSILON_DECAY = 0.999    # Learn slower to avoid getting stuck
LEARNING_RATE = 0.001
BATCH_SIZE = 64          # Learn from more examples at once

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
        self.memory = deque(maxlen=10000) # Big memory: Remember 10,000 steps
        self.epsilon = EPSILON_START
        
        self.model = Sequential([
            Dense(32, input_dim=state_size, activation='relu'), # Bigger brain
            Dense(32, activation='relu'),
            Dense(action_size, activation='linear')
        ])
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))

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
        
        targets = self.model.predict_on_batch(states)
        next_q_values = self.model.predict_on_batch(next_states)
        
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
# 4. MAIN APP
# ==========================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Interactive AI Trainer: CartPole")
    font = pygame.font.SysFont("Arial", 20)
    clock = pygame.time.Clock()

    # AI Setup
    env = gym.make(ENV_NAME, render_mode='rgb_array')
    
    # CUSTOMIZATION: Allow the pole to fall all the way (90 degrees) before resetting
    # Default is roughly 12 degrees.
    env.unwrapped.theta_threshold_radians = math.pi / 2
    
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    agent = DQNAgent(state_size, action_size)

    # State Variables
    is_training = False
    episode_count = 0
    current_score = 0
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    
    def start_training():
        nonlocal is_training
        is_training = True
        
    def stop_training():
        nonlocal is_training
        is_training = False

    # Buttons
    btn_start = Button(50, 500, 150, 50, "Start Learning", start_training)
    btn_stop = Button(220, 500, 150, 50, "Stop Learning", stop_training)
    buttons = [btn_start, btn_stop]

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
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # --- CUSTOM REWARD SYSTEM ---
            # 1. Big penalty for falling
            if done:
                reward = -100
            else:
                # 2. Reward depends on how straight it is
                # Angle is in next_state[2]. 0 is perfect.
                angle = next_state[2]
                reward = 1.0 - (abs(angle) / (math.pi / 2)) 
                # Explanation: 
                # If angle is 0 (perfect), reward is 1.0
                # If angle is 90 (fallen), reward is near 0
            
            next_state = np.reshape(next_state, [1, state_size])
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            current_score += 1
            
            if done or truncated:
                # Update stats only on failure to keep it readable? 
                # Actually real-time score is better.
                print(f"Episode: {episode_count}, Score: {current_score}, Epsilon: {agent.epsilon:.2f}")
                episode_count += 1
                current_score = 0
                state, _ = env.reset()
                state = np.reshape(state, [1, state_size])
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
        stats = [
            f"Episode: {episode_count}",
            f"Score: {current_score}",
            f"Epsilon: {agent.epsilon:.2f}",
            f"Status: {'LEARNING' if is_training else 'PAUSED'}"
        ]
        
        # Title
        title_surf = font.render("CartPole AI", True, (100, 200, 255))
        screen.blit(title_surf, (20, 20))

        # Stats List
        for i, stat in enumerate(stats):
            text = font.render(stat, True, TEXT_COLOR)
            screen.blit(text, (20, 80 + i * 40))

        # 4. Draw Buttons (Sidebar Bottom)
        # Reposition buttons to fit sidebar
        btn_start.rect.topleft = (25, 400)
        btn_stop.rect.topleft = (25, 470)
        
        for btn in buttons:
            btn.draw(screen, font)

        pygame.display.flip()
        
        # Limit FPS to 60 for visibility
        clock.tick(60)

    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
