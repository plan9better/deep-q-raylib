from time import sleep
import random
import json
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        # Create main and target networks
        self.model = DQNNetwork(state_size, action_size)
        self.target_model = DQNNetwork(state_size, action_size)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)
            return q_values.argmax().item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

import socket

HOST = "127.0.0.1"  # Localhost
PORT = 12345        # Arbitrary non-privileged port

ACTIONS = {
    0: "FORWARD",
    1: "BACKWARD",
    2: "LEFT",
    3: "RIGHT",
    4: "SHOOT",
    5: "LSHOOT",
    6: "RSHOOT",
    7: "FSHOOT",
    8: "BSHOOT",
    9: "NOOP"
}

def start_server() -> Dict:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        print(f"Server listening on {HOST}:{PORT}")
        
        conn, addr = server_socket.accept()
        obj = Dict
        counter = 0;
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(4096)
                if data == None:
                    break
                obj = json.loads(data)
                if(counter == 1000):
                    conn.sendall("RESET".encode())
                    counter = 0
                else:
                    conn.sendall(ACTIONS[random.randint(0, 9)].encode())
                    counter += 1

def train_agent(env, agent, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            loss = agent.replay()
            
            # Update target network periodically (e.g., every 100 steps)
            if episode % 100 == 0:
                agent.update_target_network()
        
        print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

if __name__ == "__main__":
    data = start_server()
    print(data)

