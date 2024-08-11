import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random

env = gym.make('Blackjack-v1')


# creating the DuelingQ network
class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)  # Additional layer
        self.fc4 = nn.Linear(128, 128)  # Additional layer


        self.fc_value = nn.Linear(128, 1)
        self.fc_adv = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))  # Additional layer
        x = torch.relu(self.fc4(x))  # Additional layer

        value = self.fc_value(x)
        adv = self.fc_adv(x)

        q_values = value + (adv - adv.mean(dim=1, keepdim=True))
        return q_values


# creating the replay buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, error, experience):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return experiences, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority


# Hyperparameters
state_size = 3
action_size = env.action_space.n
lr = 0.0001
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.999
num_episodes = 300000
batch_size = 64
memory_size = 20000
target_update_freq = 10000
alpha = 0.6
beta_start = 0.4
beta_frames = 100000

# Creating the Q-networks and target networks
qnetwork = DuelingQNetwork(state_size, action_size)
target_qnetwork = DuelingQNetwork(state_size, action_size)
optimizer = optim.Adam(qnetwork.parameters(), lr=lr)
criterion = nn.MSELoss()

memory = PrioritizedReplayBuffer(memory_size, alpha)
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


# Choosing an action using an epsilon-greedy policy
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = qnetwork(state)
        return np.argmax(q_values.numpy())


# Updating the Q-network
def update_qnetwork(beta):
    if len(memory.buffer) < batch_size:
        return
    experiences, indices, weights = memory.sample(batch_size, beta)
    batch = Experience(*zip(*experiences))

    states = torch.FloatTensor(np.array(batch.state))
    actions = torch.LongTensor(batch.action).unsqueeze(1)
    rewards = torch.FloatTensor(batch.reward)
    next_states = torch.FloatTensor(np.array(batch.next_state))
    dones = torch.FloatTensor(batch.done)
    weights = torch.FloatTensor(weights)

    q_values = qnetwork(states).gather(1, actions)
    with torch.no_grad():
        best_next_actions = qnetwork(next_states).argmax(1).unsqueeze(1)
        next_q_values = target_qnetwork(next_states).gather(1, best_next_actions).squeeze(1)
    targets = rewards + gamma * next_q_values * (1 - dones)

    loss = (q_values.squeeze() - targets).pow(2) * weights
    prios = loss + 1e-5
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    memory.update_priorities(indices, prios.detach().numpy())


# Training the agent
epsilon = epsilon_start
beta = beta_start
for episode in range(num_episodes):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    done = False

    while not done:
        action = choose_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        experience = Experience(state, action, reward, next_state, done)
        state = next_state

        q_values = qnetwork(torch.FloatTensor(state).unsqueeze(0))
        next_q_values = target_qnetwork(torch.FloatTensor(next_state).unsqueeze(0))
        target = reward + gamma * next_q_values.max() * (1 - done)
        error = abs(q_values.max() - target).item()

        memory.add(error, experience)
        update_qnetwork(beta)

    if epsilon > epsilon_end:
        epsilon *= epsilon_decay

    beta = min(1.0, beta_start + episode * (1.0 - beta_start) / beta_frames)

    if episode % target_update_freq == 0:
        target_qnetwork.load_state_dict(qnetwork.state_dict())

print("Training finished.\n")


# Agent evaluation
def evaluate_agent(env, qnetwork, num_episodes=1000):
    total_reward = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = qnetwork(state_tensor)
            action = np.argmax(q_values.numpy())
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            total_reward += reward
            state = next_state
    return total_reward / num_episodes


average_reward = evaluate_agent(env, qnetwork)
print(f"Average reward over 1000 episodes: {average_reward}")