import numpy as np
import os
import pickle
import datetime
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
import random
from collections import deque

class ReplayBuffer:
    """Experience replay buffer for storing and sampling experiences"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        # Random sample of experiences
        experiences = random.sample(self.buffer, batch_size)
        
        # Convert to batch of tensors
        states = torch.tensor(np.array([e[0] for e in experiences]), dtype=torch.float32)
        actions = torch.tensor(np.array([e[1] for e in experiences]), dtype=torch.float32)
        rewards = torch.tensor(np.array([e[2] for e in experiences]), dtype=torch.float32)
        next_states = torch.tensor(np.array([e[3] for e in experiences]), dtype=torch.float32)
        dones = torch.tensor(np.array([e[4] for e in experiences]), dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class ActorNetwork(nn.Module):
    """Neural network for determining actions in continuous action space"""
    
    def __init__(self, state_size, action_size, hidden_size=64):
        super(ActorNetwork, self).__init__()
        
        # Neural network layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        # Output layer activations
        # - Steering: tanh to get range [-1, 1]
        # - Acceleration: sigmoid to get range [0, 1]
        # - Brake: sigmoid to get range [0, 1]
        # - Gear: not directly predicted, handled separately
    
    def forward(self, state):
        """Forward pass through the network"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Different activations for different actions
        # Output format: [steering, acceleration, brake]
        x = self.fc3(x)
        
        # Steering uses tanh activation for [-1, 1]
        steering = torch.tanh(x[:, 0]).unsqueeze(1)
        
        # Acceleration and brake use sigmoid activation for [0, 1]
        accel_brake = torch.sigmoid(x[:, 1:])
        
        # Combine outputs
        return torch.cat([steering, accel_brake], dim=1)


class CriticNetwork(nn.Module):
    """Neural network for estimating Q-values"""
    
    def __init__(self, state_size, action_size, hidden_size=64):
        super(CriticNetwork, self).__init__()
        
        # Neural network layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, state, action):
        """Forward pass through the network"""
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPGAgent:
    """Deep Deterministic Policy Gradient agent for continuous control in racing"""
    
    def __init__(self, state_size, action_size, hidden_size=64, lr_actor=1e-4, 
                 lr_critic=1e-3, gamma=0.99, tau=1e-3, batch_size=64,
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01):
        """Initialize the DDPG agent"""
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount factor
        self.tau = tau  # soft update parameter
        self.batch_size = batch_size
        
        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Create actor network and target
        self.actor = ActorNetwork(state_size, action_size, hidden_size)
        self.actor_target = ActorNetwork(state_size, action_size, hidden_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Create critic network and target
        self.critic = CriticNetwork(state_size, action_size, hidden_size)
        self.critic_target = CriticNetwork(state_size, action_size, hidden_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Set target weights equal to model weights initially
        self._update_target_networks(tau=1.0)
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training metrics
        self.loss_history = []
        self.reward_history = []
        self.total_steps = 0
        
        # For saving models
        self.best_reward = float('-inf')
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)
    
    def get_action(self, state, add_noise=True):
        """Get action from actor network with optional exploration noise"""
        # Convert state to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        # Set network to evaluation mode
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze().numpy()
        self.actor.train()
        
        # Add exploration noise if training
        if add_noise and self.epsilon > self.epsilon_min:
            # Add noise scaled by epsilon
            noise = np.random.normal(0, self.epsilon, size=self.action_size)
            action += noise
            
            # Ensure actions are within valid ranges
            # Steering: [-1, 1]
            # Acceleration: [0, 1]
            # Brake: [0, 1]
            action[0] = np.clip(action[0], -1.0, 1.0)  # steering
            action[1] = np.clip(action[1], 0.0, 1.0)   # acceleration
            action[2] = np.clip(action[2], 0.0, 1.0)   # brake
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1
    
    def train(self):
        """Train the agent from sampled experiences"""
        # Only train if we have enough experiences
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_value = rewards + (1 - dones) * self.gamma * target_q
        
        # Current Q-values
        current_q = self.critic(states, actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q, target_value)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._update_target_networks(self.tau)
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Store losses
        self.loss_history.append((critic_loss.item(), actor_loss.item()))
        
        return critic_loss.item(), actor_loss.item()
    
    def _update_target_networks(self, tau):
        """Soft update of target network parameters"""
        def update_params(target, source, tau):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1.0 - tau) * target_param.data
                )
        
        update_params(self.actor_target, self.actor, tau)
        update_params(self.critic_target, self.critic, tau)
    
    def save_model(self, episode, avg_reward, filepath=None):
        """Save the model"""
        if filepath is None:
            # Create a filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.models_dir, f'model_{timestamp}_ep{episode}_r{avg_reward:.1f}.pt')
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'loss_history': self.loss_history,
            'reward_history': self.reward_history
        }, filepath)
        
        print(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath):
        """Load a saved model"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            self.epsilon = checkpoint['epsilon']
            self.total_steps = checkpoint['total_steps']
            self.loss_history = checkpoint['loss_history']
            self.reward_history = checkpoint['reward_history']
            
            print(f"Model loaded from {filepath}")
            return True
        else:
            print(f"No model found at {filepath}")
            return False
    
    def save_checkpoint(self, episode, avg_reward):
        """Save a checkpoint if performance improves"""
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            filepath = os.path.join(self.models_dir, 'best_model.pt')
            self.save_model(episode, avg_reward, filepath)
            return True
        return False


class StateProcessor:
    """Process raw state information from TORCS into suitable inputs for the RL agent"""
    
    def __init__(self, num_track_sensors=19, num_opponent_sensors=36, history_len=5):
        """Initialize the state processor"""
        self.num_track_sensors = num_track_sensors
        self.num_opponent_sensors = num_opponent_sensors
        self.history_len = history_len
        
        # Initialize history buffers
        self.speed_history = deque(maxlen=history_len)
        self.angle_history = deque(maxlen=history_len)
        self.trackPos_history = deque(maxlen=history_len)
        self.steering_history = deque(maxlen=history_len)
        
        # Fill histories with zeros
        for _ in range(history_len):
            self.speed_history.append(0.0)
            self.angle_history.append(0.0)
            self.trackPos_history.append(0.0)
            self.steering_history.append(0.0)
    
    def get_state_dim(self):
        """Return the dimension of the processed state vector"""
        # Core features
        basic_features = 10  # speed, angle, trackPos, rpm, damage, 4 wheel speeds, z
        
        # Track and opponent sensors (possibly downsampled)
        track_features = self.num_track_sensors
        opponent_features = min(10, self.num_opponent_sensors)  # We'll downsample opponents
        
        # Historical features
        history_features = 3 * (self.history_len - 1)  # speed, angle, trackPos histories
        
        # Total features
        return basic_features + track_features + opponent_features + history_features
    
    def process_state(self, car_state):
        """Convert car state to neural network input"""
        if not car_state:
            # Return zero vector if no state is provided
            return np.zeros(self.get_state_dim())
        
        # Extract basic features
        speed_x = car_state.getSpeedX() if car_state.getSpeedX() is not None else 0.0
        speed_y = car_state.getSpeedY() if car_state.getSpeedY() is not None else 0.0
        speed_z = car_state.getSpeedZ() if car_state.getSpeedZ() is not None else 0.0
        total_speed = np.sqrt(speed_x**2 + speed_y**2 + speed_z**2)
        
        angle = car_state.getAngle() if car_state.getAngle() is not None else 0.0
        track_pos = car_state.getTrackPos() if car_state.getTrackPos() is not None else 0.0
        rpm = car_state.getRpm() / 10000.0 if car_state.getRpm() is not None else 0.0  # Normalize RPM
        damage = car_state.getDamage() / 10000.0 if car_state.getDamage() is not None else 0.0  # Normalize damage
        z = car_state.getZ() / 10.0 if car_state.getZ() is not None else 0.0  # Normalize height
        
        # Get wheel spin velocities
        wheel_speeds = car_state.getWheelSpinVel() if car_state.getWheelSpinVel() else [0.0, 0.0, 0.0, 0.0]
        wheel_speeds = [w / 100.0 for w in wheel_speeds]  # Normalize wheel speeds
        
        # Process track sensors
        track_sensors = car_state.getTrack() if car_state.getTrack() else [0.0] * self.num_track_sensors
        track_sensors = [min(1.0, s / 200.0) for s in track_sensors]  # Normalize to [0,1] with max at 200m
        
        # Process opponent sensors (downsample to 10)
        opponents = car_state.getOpponents() if car_state.getOpponents() else [200.0] * self.num_opponent_sensors
        # Take a subset of opponent sensors (e.g., every 36/10 = ~4 sensors)
        step = max(1, self.num_opponent_sensors // 10)
        opponent_sample = [opponents[i] for i in range(0, self.num_opponent_sensors, step)][:10]
        opponent_sample = [min(1.0, o / 200.0) for o in opponent_sample]  # Normalize to [0,1]
        
        # Update histories
        self.speed_history.append(total_speed / 300.0)  # Normalize to [0,1] with max at 300 m/s
        self.angle_history.append(angle / 3.14159)  # Normalize angle to [-1,1]
        self.trackPos_history.append(track_pos)  # Already in [-1,1]
        
        # Calculate historical features (deltas)
        speed_history = list(self.speed_history)[:-1]  # All but current
        angle_history = list(self.angle_history)[:-1]  # All but current
        trackPos_history = list(self.trackPos_history)[:-1]  # All but current
        
        # Combine all features
        state = [
            speed_x / 300.0,  # Normalize to approximately [-1,1]
            speed_y / 300.0,
            total_speed / 300.0,
            angle / 3.14159,  # Normalize angle to [-1,1]
            track_pos,  # Already in [-1,1]
            rpm,
            damage,
            z
        ]
        
        # Add wheel speeds
        state.extend(wheel_speeds)
        
        # Add track sensors
        state.extend(track_sensors)
        
        # Add opponent samples
        state.extend(opponent_sample)
        
        # Add historical features
        state.extend(speed_history)
        state.extend(angle_history)
        state.extend(trackPos_history)
        
        return np.array(state, dtype=np.float32)
    
    def process_action(self, action):
        """Convert neural network output to car control values"""
        # Neural network outputs: [steering, acceleration, brake]
        steering = float(action[0])  # Already in [-1, 1]
        accel = float(action[1])     # Already in [0, 1]
        brake = float(action[2])     # Already in [0, 1]
        
        return steering, accel, brake
    
    def calculate_reward(self, car_state, previous_state=None, action=None):
        """Calculate reward based on car state and action"""
        if car_state is None:
            return -10.0  # Large negative reward for invalid state
        
        # Extract relevant state information
        speed_x = car_state.getSpeedX() if car_state.getSpeedX() is not None else 0.0
        track_pos = car_state.getTrackPos() if car_state.getTrackPos() is not None else 0.0
        angle = car_state.getAngle() if car_state.getAngle() is not None else 0.0
        damage = car_state.getDamage() if car_state.getDamage() is not None else 0.0
        track = car_state.getTrack() if car_state.getTrack() else None
        
        # Previous damage for comparison
        prev_damage = 0.0
        if previous_state and hasattr(previous_state, 'getDamage'):
            prev_damage = previous_state.getDamage() if previous_state.getDamage() is not None else 0.0
        
        # Basic reward components
        
        # 1. Speed reward - encourage high speed in the forward direction
        speed_reward = speed_x / 300.0  # Normalize to approximately [0, 1]
        
        # 2. Track position penalty - encourage staying near center of track
        position_penalty = -abs(track_pos) * 2.0  # Higher penalty for being off-center
        
        # 3. Angle penalty - encourage car to face forward
        angle_penalty = -abs(angle) * 2.0
        
        # 4. Damage penalty - high penalty for taking damage
        damage_penalty = -max(0, (damage - prev_damage) / 100.0) * 10.0
        
        # 5. Off-track penalty
        off_track_penalty = 0.0
        if abs(track_pos) >= 1.0:
            off_track_penalty = -10.0
        
        # 6. Track sensor penalty - discourage getting too close to track edges
        track_penalty = 0.0
        if track:
            # Minimum distance from any track edge
            min_track_dist = min([s for s in track if s > 0.0], default=100.0)
            if min_track_dist < 5.0:
                track_penalty = -((5.0 - min_track_dist) / 5.0) * 2.0
        
        # 7. Progress reward - encourage making forward progress
        progress_reward = 0.0
        if previous_state and hasattr(previous_state, 'getDistRaced'):
            prev_dist = previous_state.getDistRaced() if previous_state.getDistRaced() is not None else 0.0
            curr_dist = car_state.getDistRaced() if car_state.getDistRaced() is not None else 0.0
            progress_reward = (curr_dist - prev_dist) * 10.0  # Scale up progress reward
        
        # Combine rewards with weights
        total_reward = (
            speed_reward * 1.0 +           # Weight for speed
            position_penalty * 0.5 +       # Weight for track position
            angle_penalty * 0.2 +          # Weight for angle
            damage_penalty * 2.0 +         # Weight for damage
            off_track_penalty +            # Off-track is a hard penalty
            track_penalty * 0.3 +          # Weight for track sensors
            progress_reward * 2.0          # Weight for progress
        )
        
        return total_reward


# Create a simple helper class to manage the ML pipeline
class RacingAI:
    """Main class for managing the TORCS racing AI"""
    
    def __init__(self, load_model_path=None):
        """Initialize the racing AI"""
        # Create state processor
        self.state_processor = StateProcessor()
        
        # Get state dimension
        state_dim = self.state_processor.get_state_dim()
        
        # Define action dimension (steering, acceleration, brake)
        action_dim = 3
        
        # Create DDPG agent
        self.agent = DDPGAgent(
            state_size=state_dim,
            action_size=action_dim,
            hidden_size=128,
            lr_actor=1e-4,
            lr_critic=1e-3,
            gamma=0.99,
            tau=1e-3,
            batch_size=64,
            epsilon=1.0,
            epsilon_decay=0.9995,
            epsilon_min=0.01
        )
        
        # Load a pre-trained model if provided
        if load_model_path:
            self.agent.load_model(load_model_path)
        
        # Training parameters
        self.episode_count = 0
        self.step_count = 0
        self.prev_state = None
        self.prev_action = None
        self.episode_reward = 0.0
        self.training_mode = True  # Set to False for deployment/evaluation
        
        # Create directories for saving results
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def get_action(self, car_state):
        """Get action from agent based on current state"""
        # Process state for the agent
        state = self.state_processor.process_state(car_state)
        
        # Get action from agent (with noise in training mode)
        action = self.agent.get_action(state, add_noise=self.training_mode)
        
        # Process action for car control
        steering, accel, brake = self.state_processor.process_action(action)
        
        # Store state and action for later learning
        self.prev_state = car_state
        self.prev_action = action
        
        return steering, accel, brake
    
    def learn(self, car_state, done=False):
        """Learn from experience"""
        if not self.training_mode or self.prev_state is None:
            return
        
        # Process current state
        current_state = self.state_processor.process_state(car_state)
        
        # Process previous state
        prev_state_vec = self.state_processor.process_state(self.prev_state)
        
        # Calculate reward
        reward = self.state_processor.calculate_reward(car_state, self.prev_state, self.prev_action)
        self.episode_reward += reward
        
        # Add experience to memory
        self.agent.remember(prev_state_vec, self.prev_action, reward, current_state, done)
        
        # Train the agent
        if len(self.agent.replay_buffer) > self.agent.batch_size:
            self.agent.train()
        
        # Update step count
        self.step_count += 1
    
    def episode_end(self):
        """Handle end of episode"""
        self.episode_count += 1
        
        # Add episode reward to history
        self.agent.reward_history.append(self.episode_reward)
        
        # Log episode information
        print(f"Episode {self.episode_count}: Reward = {self.episode_reward:.2f}, "
              f"Steps = {self.step_count}, Epsilon = {self.agent.epsilon:.4f}")
        
        # Save model periodically
        if self.episode_count % 10 == 0:
            # Calculate average reward over last 10 episodes
            avg_reward = np.mean(self.agent.reward_history[-10:])
            self.agent.save_checkpoint(self.episode_count, avg_reward)
        
        # Reset episode counters
        self.step_count = 0
        self.episode_reward = 0.0
        self.prev_state = None
        self.prev_action = None
    
    def set_training_mode(self, mode):
        """Set whether the agent is training or evaluating"""
        self.training_mode = mode
        print(f"Training mode: {mode}")
    
    def save_results(self):
        """Save training results to file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f'results_{timestamp}.pkl')
        
        results = {
            'reward_history': self.agent.reward_history,
            'loss_history': self.agent.loss_history,
            'episodes': self.episode_count,
            'total_steps': self.agent.total_steps
        }
        
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {results_file}")