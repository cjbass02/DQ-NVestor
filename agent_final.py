# dqn agent
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

class MultiStockQNetwork(nn.Module):
    """DQN net with per stock heads."""
    def __init__(self, state_dim, n_stocks, actions_per_stock):
        super(MultiStockQNetwork, self).__init__()
        self.n_stocks = n_stocks
        self.actions_per_stock = actions_per_stock
        # shared layers
        hidden_size = 128
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # per stock output heads
        self.heads = nn.ModuleList([nn.Linear(hidden_size, actions_per_stock) for _ in range(n_stocks)])
    
    def forward(self, state):
        # state
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
    # each head give a q
        # return list of tensors
        q_values = [head(x) for head in self.heads]  # length n_stocks
        return q_values

class ReplayBuffer:
    def __init__(self, capacity=int(1e5)):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        # save transition
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        # wrap around when full
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, n_stocks, actions_per_stock, 
                 lr=1e-3, gamma=0.99, batch_size=64, buffer_capacity=100000, 
                 epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995, target_update_freq=100):
        self.state_dim = state_dim
        self.n_stocks = n_stocks
        self.actions_per_stock = actions_per_stock
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        # main + target nets
        self.q_network = MultiStockQNetwork(state_dim, n_stocks, actions_per_stock)
        self.target_network = MultiStockQNetwork(state_dim, n_stocks, actions_per_stock)
        self.target_network.load_state_dict(self.q_network.state_dict())  # initialize target with same weights
        self.target_network.eval()
        # optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        # replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.train_steps = 0
    
    def select_action(self, state):
        # state
        if np.random.rand() < self.epsilon:
            # explor random per stock
            action = [random.randrange(self.actions_per_stock) for _ in range(self.n_stocks)]
            return np.array(action, dtype=int)
        else:
            # exploit argmax per stock
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            self.q_network.train()
            # argmax per stock
            q_values = [q.detach().cpu().numpy().squeeze() for q in q_values]
            action = [int(np.argmax(q_values[i])) for i in range(self.n_stocks)]
            return np.array(action, dtype=int)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Save one transition."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update_target_network(self):
        """Copy weights to target net."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_step(self):
        """One training step."""
        # train only when buffer ready
        if len(self.replay_buffer) < self.batch_size:
            return
        # grab a batch
        batch = self.replay_buffer.sample(self.batch_size)
        # split it up
        states = torch.FloatTensor([b[0] for b in batch])        # (batch, state_dim)
        actions = np.array([b[1] for b in batch])                # (batch, n_stocks)
        rewards = torch.FloatTensor([b[2] for b in batch])       # (batch,)
        next_states = torch.FloatTensor([b[3] for b in batch])   # (batch, state_dim)
        dones = torch.FloatTensor([float(b[4]) for b in batch])  # (batch,)
        
        # Qs for chosen actions
        # forward() gives per-stock Qs
        all_q = self.q_network(states)
        # pick Qs for actions
        q_values = []
        for stock_idx in range(self.n_stocks):
            # actions for this stock
            stock_actions = torch.LongTensor(actions[:, stock_idx])
            # grab chosen Q
            q_val_stock = all_q[stock_idx].gather(1, stock_actions.view(-1,1)).squeeze(1)
            q_values.append(q_val_stock)
        q_values = torch.stack(q_values, dim=1)
        
        # target Qs from target net
        all_q_next = self.target_network(next_states)
        # max next Q per stock
        max_q_next = []
        for stock_idx in range(self.n_stocks):
            # max over actions
            max_q_next_stock, _ = torch.max(all_q_next[stock_idx], dim=1)
            max_q_next.append(max_q_next_stock)
        max_q_next = torch.stack(max_q_next, dim=1) 
        
        reward_expand = rewards.view(-1, 1).expand(-1, self.n_stocks)
        done_expand = dones.view(-1, 1).expand(-1, self.n_stocks)
        target_q = reward_expand + self.gamma * max_q_next * (1 - done_expand)
        
        # loss over all stocks
        # MSE
        loss = nn.MSELoss()(q_values, target_q.detach())
        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # sync target net sometimes
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.update_target_network()
        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min
