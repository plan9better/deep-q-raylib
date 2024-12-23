import select
import socket
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Dict

# Constants
HOST = "127.0.0.1"  # localhost
PORT = 65432
ACTIONS = ["LEFT", "RIGHT", "THRUST", "SHOOT", "LEFT_THRUST", 
           "RIGHT_THRUST", "LEFT_SHOOT", "RIGHT_SHOOT", "NONE"]

class StateProcessor:
    def __init__(self, max_meteors=10, max_shots=15):
        self.max_meteors = max_meteors
        self.max_shots = max_shots
        
    def process_state(self, game_state):
        # Extract player features
        player_features = [
            game_state['player']['x'] / 800.0,
            game_state['player']['y'] / 600.0,
            np.sin(np.radians(game_state['player']['rotation'])),
            np.cos(np.radians(game_state['player']['rotation'])),
            game_state['shotsCount'] / 10.0
        ]
        
        # Process meteors
        meteor_features = []
        for meteor in game_state['meteors'][:self.max_meteors]:
            type_value = {'small': 0.33, 'medium': 0.66, 'big': 1.0}[meteor['type']]
            meteor_features.extend([
                meteor['x'] / 800.0,
                meteor['y'] / 600.0,
                meteor['speed_x'] / 4.0,
                meteor['speed_y'] / 4.0,
                type_value
            ])
        
        # Pad meteor features
        padding_length = (self.max_meteors * 5) - len(meteor_features)
        meteor_features.extend([0.0] * padding_length)
        
        # Process shots
        shot_features = []
        for shot in game_state['shots'][:self.max_shots]:
            shot_features.extend([
                shot['x'] / 800.0,
                shot['y'] / 600.0,
                np.sin(np.radians(shot['rotation'])),
                np.cos(np.radians(shot['rotation']))
            ])
            
        # Pad shot features
        padding_length = (self.max_shots * 4) - len(shot_features)
        shot_features.extend([0.0] * padding_length)
        
        return np.array(player_features + meteor_features + shot_features, dtype=np.float32)

class DQNNetwork(nn.Module):
    def __init__(self, input_size):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(ACTIONS))
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_processor, learning_rate=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999995,
                 memory_size=10000, batch_size=32):
        self.state_processor = state_processor
        self.action_size = len(ACTIONS)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        sample_state = np.zeros(5 + (state_processor.max_meteors * 5) + (state_processor.max_shots * 4))
        self.input_size = len(sample_state)
        
        self.model = DQNNetwork(self.input_size)
        self.target_model = DQNNetwork(self.input_size)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        processed_state = self.state_processor.process_state(state)
        processed_next_state = self.state_processor.process_state(next_state)
        self.memory.append((processed_state, action, reward, processed_next_state, done))
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        processed_state = self.state_processor.process_state(state)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(processed_state).unsqueeze(0)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(np.vstack(states))
        next_states = torch.FloatTensor(np.vstack(next_states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

    def save(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

def train_agent():
    # Initialize agent
    state_processor = StateProcessor()
    agent = DQNAgent(state_processor)
    agent.load("dqn_model_episode_1149.pth")
    
    episode = 0
    total_steps = 0
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        print(f"Server listening on {HOST}:{PORT}")
        conn, _ = server_socket.accept()
        
        with conn:
            while True:  # Training loop
                print(f"Episode {episode} started, Epsilon: {agent.epsilon}")
                
                try:
                    steps = 0
                    total_reward = 0
                    previous_state = None
                    previous_action = None
                    done = False
                    
                    while True:
                        data = conn.recv(4096)
                        if not data:
                            exit()
                            
                        current_state = json.loads(data)
                        
                        if previous_state is not None:
                            reward = current_state['reward']
                            done = steps >= 5000 or current_state['game_over'] != 0
                            
                            # If game is over, we might want to add a negative reward
                            # if current_state['game_over'] == 1:
                            #     reward += 100  # Optional: penalize death
                            
                            agent.remember(previous_state, previous_action, reward, 
                                        current_state, done)
                            loss = agent.replay()
                            
                            total_reward += reward
                            
                            # if loss is not None:
                            #     print(f"Step {steps}, Loss: {loss:.4f}, Reward: {reward}, "
                            #             f"Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")
                        
                        # Get next action
                        action_idx = agent.act(current_state)
                        action = ACTIONS[action_idx]
                        
                        # Send action to game
                        if done:
                            conn.sendall("RESET".encode())
                            break
                        else:
                            conn.sendall(action.encode())
                        
                        # Store state and action for next iteration
                        previous_state = current_state
                        previous_action = action_idx
                        
                        steps += 1
                        total_steps += 1
                        
                        # Update target network periodically
                        if total_steps % 100000 == 0:
                            agent.update_target_network()
                            agent.save(f"dqn_model_episode_{episode}.pth")
                    
                    print(f"Episode {episode} finished after {steps} steps. "
                            f"Total reward: {total_reward}")
                    episode += 1
                        
                except Exception as e:
                    print(f"Error during episode {episode}: {e}")
                    continue

if __name__ == "__main__":
    train_agent()
