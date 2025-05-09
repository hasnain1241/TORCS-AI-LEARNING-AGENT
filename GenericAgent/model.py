import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# Determines if we use GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNetwork(nn.Module):
    """
    Actor network for DDPG - maps states to actions
    """
    def __init__(self, state_dim, action_dim, hidden1=400, hidden2=300):
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)
        
    def forward(self, state):
        """
        Forward pass through the network
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Use tanh to bound output between -1 and 1
        action = torch.tanh(self.fc3(x))
        
        return action

class CriticNetwork(nn.Module):
    """
    Critic network for DDPG - maps (state, action) pairs to Q-values
    """
    def __init__(self, state_dim, action_dim, hidden1=400, hidden2=300):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1 + action_dim, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)
        
    def forward(self, state, action):
        """
        Forward pass through the network
        """
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        
        return q_value

class DDPGAgent:
    """
    DDPG Agent with Actor-Critic architecture
    """
    def __init__(self, state_dim, action_dim, actor_learning_rate=1e-4, 
                 critic_learning_rate=1e-3, gamma=0.99, tau=0.001, 
                 hidden1=400, hidden2=300, batch_size=64, buffer_size=1000000):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # discount factor
        self.tau = tau      # soft update parameter
        self.batch_size = batch_size
        
        # Actor network (current and target)
        self.actor = ActorNetwork(state_dim, action_dim, hidden1, hidden2).to(device)
        self.actor_target = ActorNetwork(state_dim, action_dim, hidden1, hidden2).to(device)
        
        # Critic network (current and target)
        self.critic = CriticNetwork(state_dim, action_dim, hidden1, hidden2).to(device)
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden1, hidden2).to(device)
        
        # Initialize target networks with same weights
        self.hard_update(self.actor, self.actor_target)
        self.hard_update(self.critic, self.critic_target)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        
        # Create directory for model checkpoints if it doesn't exist
        self.checkpoint_dir = 'models'
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        # Training stats
        self.train_iter = 0
    
    def get_action(self, state, add_noise=True, noise_scale=0.1):
        """
        Get action from actor with optional noise for exploration
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.actor.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        self.actor.train()  # Back to training mode
        
        if add_noise:
            # Add Ornstein-Uhlenbeck noise for exploration
            action += noise_scale * np.random.standard_normal(self.action_dim)
            # Clip action to valid range
            action = np.clip(action, -1.0, 1.0)
            
        return action
    
    def update(self, batch):
        """
        Update actor and critic networks using batch of experiences
        """
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        # Update critic
        next_actions = self.actor_target(next_states)
        next_q_values = self.critic_target(next_states, next_actions)
        
        # Compute target Q value
        target_q = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Current Q estimate
        current_q = self.critic(states, actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q, target_q.detach())
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update of target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        
        self.train_iter += 1
        
        return critic_loss.item(), actor_loss.item()
    
    def hard_update(self, source, target):
        """
        Hard update: target = source
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)
    
    def soft_update(self, source, target):
        """
        Soft update: target = tau*source + (1-tau)*target
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )
    
    def save_models(self, episode):
        """
        Save actor and critic models
        """
        torch.save(
            self.actor.state_dict(),
            f'{self.checkpoint_dir}/actor_episode_{episode}.pt'
        )
        torch.save(
            self.critic.state_dict(),
            f'{self.checkpoint_dir}/critic_episode_{episode}.pt'
        )
        
        # Save latest version as well
        torch.save(
            self.actor.state_dict(),
            f'{self.checkpoint_dir}/actor_latest.pt'
        )
        torch.save(
            self.critic.state_dict(),
            f'{self.checkpoint_dir}/critic_latest.pt'
        )
        
        print(f"Models saved at episode {episode}")
    
    def load_models(self, episode=None):
        """
        Load actor and critic models
        If episode is None, load latest model
        """
        if episode is None:
            # Load latest models
            actor_path = f'{self.checkpoint_dir}/actor_latest.pt'
            critic_path = f'{self.checkpoint_dir}/critic_latest.pt'
        else:
            # Load specific episode models
            actor_path = f'{self.checkpoint_dir}/actor_episode_{episode}.pt'
            critic_path = f'{self.checkpoint_dir}/critic_episode_{episode}.pt'
        
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=device))
            self.critic.load_state_dict(torch.load(critic_path, map_location=device))
            self.hard_update(self.actor, self.actor_target)
            self.hard_update(self.critic, self.critic_target)
            print(f"Models loaded successfully{'.' if episode is None else f' from episode {episode}.'}")
            return True
        else:
            print(f"No saved models found{'.' if episode is None else f' for episode {episode}.'}")
            return False