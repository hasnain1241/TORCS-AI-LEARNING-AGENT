import msgParser
import carState
import carControl
import numpy as np
import os
import datetime
import math
import time
import torch
from model import DDPGAgent
from replay_buffer import ReplayBuffer
import json

# Constants for the DDPG algorithm
MAX_EPISODES = 1000        # Maximum number of training episodes
MAX_STEPS = 10000          # Maximum steps per episode
BATCH_SIZE = 64            # Batch size for training
BUFFER_SIZE = 1000000      # Replay buffer size
TRAINING_START = 1000      # Start training after this many steps
SAVE_INTERVAL = 50         # Save models every N episodes
TRAIN_EVERY = 5            # Train after every N steps
EPSILON_DECAY = 0.999      # Decay rate for exploration noise (slower decay)
EPSILON_MIN = 0.1          # Minimum exploration noise (higher minimum)

# Anti-stalling constants - CRITICAL for preventing the car from stopping
MIN_ACCELERATION = 0.3     # Minimum acceleration to apply (prevents stopping)
SPEED_PENALTY_THRESHOLD = 20 # Speed below which heavy penalties apply
SPEED_REWARD_MULTIPLIER = 80 # Extremely high reward for speed (up from 50)
STUCK_SPEED_THRESHOLD = 5  # Speed below which car is considered stuck
STUCK_PENALTY = 300        # Heavy penalty for being stuck

class Driver(object):
    '''
    RL-enhanced driver for TORCS using DDPG algorithm
    '''

    def __init__(self, stage, training=True):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        self.training = training  # Whether we're in training mode
        
        # Parse arguments
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        # Initialize state variables
        self.steer_lock = 0.785398
        self.max_speed = 350
        self.prev_rpm = None
        
        # Anti-stall tracking variables
        self.stuck_counter = 0    # Count consecutive steps at low speed
        self.last_speed = 0       # Track previous speed
        
        # Setup logging and create directories
        self.setup_logging()
        
        # Initialize replay buffer and DDPG agent
        self.setup_reinforcement_learning()
        
        # Training variables
        self.epsilon = 0.4  # Exploration rate
        self.episode = 0    # Current episode
        self.step = 0       # Current step in episode
        self.total_reward = 0.0  # Total reward in current episode
        self.prev_state = None   # Previous state for calculating reward
        self.prev_action = None  # Previous action for reference
        
        # Performance metrics
        self.best_reward = -float('inf')
        self.last_rewards = []   # Store last 10 episode rewards
        
        # Load existing models if available
        if not self.training:
            self.agent.load_models()
            print("Running in inference mode with loaded models.")
    
    def setup_logging(self):
        '''Setup logging system for telemetry and training data'''
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
            
        # Create a timestamp for log files
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f'logs/telemetry_{self.timestamp}.csv'
        self.reward_log = f'logs/rewards_{self.timestamp}.csv'
        
        # Open reward log file
        reward_header = "episode,total_reward,avg_reward,max_speed,distance_raced,damage\n"
        with open(self.reward_log, 'w') as f:
            f.write(reward_header)
        
        print(f"Logging telemetry to: {self.log_file}")
        print(f"Logging rewards to: {self.reward_log}")

    def setup_reinforcement_learning(self):
        '''Initialize the reinforcement learning components'''
        # Define state and action dimensions
        # Verify the actual length of your state vector
        test_state = self.get_state_representation()
        self.state_dim = len(test_state)  # Use actual length instead of hardcoded value
        print(f"State vector has length: {self.state_dim}")
        
        # Action: steering, acceleration/brake
        self.action_dim = 2  # steering, acceleration/brake
        
        # Create DDPG agent
        self.agent = DDPGAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-3,
            gamma=0.99,
            tau=0.001,
            hidden1=400,
            hidden2=300,
            batch_size=BATCH_SIZE
        )
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size=BUFFER_SIZE)
        
        print("Reinforcement learning agent initialized.")
    
    def log_telemetry(self):
        '''Log telemetry data to CSV file'''
        if self.state.sensors is None:
            return
        
        # Basic telemetry data
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3]
        data = [
            timestamp,
            self.state.getAngle(),
            self.state.getCurLapTime(),
            self.state.getDamage(),
            self.state.getDistFromStart(),
            self.state.getDistRaced(),
            self.state.getFuel(),
            self.state.getGear(),
            self.state.getLastLapTime() if self.state.getLastLapTime() else 0,
            self.state.getRacePos(),
            self.state.getRpm(),
            self.state.getSpeedX(),
            self.state.getSpeedY(),
            self.state.getSpeedZ(),
            self.state.getTrackPos(),
            self.state.getZ()
        ]
        
        # Convert all values to strings and handle None values
        data_str = [str(item) if item is not None else "0" for item in data]
        
        # Add array data (track, opponents, wheelSpinVel)
        track_data = "|".join([str(x) for x in self.state.getTrack()]) if self.state.getTrack() else "0"
        opponents_data = "|".join([str(x) for x in self.state.getOpponents()]) if self.state.getOpponents() else "0"
        wheel_data = "|".join([str(x) for x in self.state.getWheelSpinVel()]) if self.state.getWheelSpinVel() else "0"
        
        # Combine all data
        log_data = ",".join(data_str) + "," + track_data + "," + opponents_data + "," + wheel_data
        
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(log_data + "\n")
    
    def log_reward(self):
        '''Log reward data at the end of an episode'''
        if len(self.last_rewards) > 0:
            avg_reward = sum(self.last_rewards) / len(self.last_rewards)
        else:
            avg_reward = 0
            
        # Get additional metrics
        max_speed = max([0] + [s['speedX'] for s in self.episode_states]) if hasattr(self, 'episode_states') else 0
        distance_raced = self.state.getDistRaced() if self.state.getDistRaced() is not None else 0
        damage = self.state.getDamage() if self.state.getDamage() is not None else 0
        
        log_data = f"{self.episode},{self.total_reward:.4f},{avg_reward:.4f},{max_speed:.2f},{distance_raced:.2f},{damage:.2f}\n"
        
        with open(self.reward_log, 'a') as f:
            f.write(log_data)
    
    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        # Set up rangefinder angles for optimal track sensing
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        # More precise angles in front for accurate steering
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        # Initialize episode states
        self.episode_states = []
        self.episode += 1
        self.step = 0
        self.total_reward = 0.0
        self.prev_state = None
        self.prev_action = None
        self.stuck_counter = 0
        self.last_speed = 0
        
        return self.parser.stringify({'init': self.angles})
    
    def get_state_representation(self):
        '''Convert car state to a state vector for RL agent'''
        # Basic state variables
        angle = self.state.getAngle() if self.state.getAngle() is not None else 0
        trackPos = self.state.getTrackPos() if self.state.getTrackPos() is not None else 0
        speedX = self.state.getSpeedX() if self.state.getSpeedX() is not None else 0
        speedY = self.state.getSpeedY() if self.state.getSpeedY() is not None else 0
        
        # Normalize speed values
        speedX = speedX / self.max_speed
        speedY = speedY / self.max_speed
        
        # Track sensors
        track = self.state.getTrack()
        if track is None:
            track = [100] * 19  # Default to max range
        
        # Normalize track sensors (they range from 0 to 200)
        track = [min(1.0, max(0.0, t / 200.0)) for t in track]
        
        # Opponent sensors
        opponents = self.state.getOpponents()
        if opponents is None:
            opponents = [200] * 36  # Default to max range
            
        # Normalize opponent sensors (they range from 0 to 200)
        opponents = [min(1.0, max(0.0, o / 200.0)) for o in opponents]
        
        # Combine all state variables
        state_vector = [angle, trackPos, speedX, speedY] + track + opponents
        
        return np.array(state_vector, dtype=np.float32)
    
    def compute_reward(self, state_vector, action):
        '''Compute reward based on current state and action'''
        # Extract state variables
        angle = state_vector[0]
        trackPos = state_vector[1]
        speedX = state_vector[2] * self.max_speed  # Denormalize
        speedY = state_vector[3] * self.max_speed  # Denormalize
        track_sensors = state_vector[4:23]
        
        # ANTI-STALL: EXTREME REWARD FOR FORWARD MOTION
        reward = max(speedX * SPEED_REWARD_MULTIPLIER, 0)  
        
        # ANTI-STALL: Higher survival bonus
        reward += 20
        
        # Minimal penalty for being off-center
        trackPos_penalty = -abs(trackPos) * 2
        reward += trackPos_penalty
        
        # Very minimal penalty for steering
        steering_penalty = -abs(action[0]) * 0.1
        reward += steering_penalty
        
        # Only heavy penalty for going completely off track
        if abs(trackPos) > 0.95:
            reward -= 100
        
        # Very small angle penalty
        angle_penalty = -abs(angle) * 1
        reward += angle_penalty
        
        # Almost no side speed penalty
        side_speed_penalty = -abs(speedY) * 0.1
        reward += side_speed_penalty
        
        # ANTI-STALL: Huge reward for staying on track while going fast
        if abs(trackPos) < 0.4:  # More lenient center requirement
            reward += speedX * 10.0  # 10x the speed bonus
        
        # ANTI-STALL: Force continuous movement with extreme penalty for slowness
        if speedX < SPEED_PENALTY_THRESHOLD:
            slow_penalty = (SPEED_PENALTY_THRESHOLD - speedX) * 15
            reward -= slow_penalty
            
            # Extra logging for debugging
            if self.step % 100 == 0:
                print(f"ANTI-STALL: Applied slow speed penalty of {slow_penalty:.1f} for speed {speedX:.1f}")
        
        # ANTI-STALL: Track if car is stuck (continuous low speed)
        if speedX < STUCK_SPEED_THRESHOLD:
            self.stuck_counter += 1
            if self.stuck_counter > 10:  # Stuck for 10+ steps
                reward -= STUCK_PENALTY
                if self.step % 10 == 0:
                    print(f"ANTI-STALL WARNING: Car stuck for {self.stuck_counter} steps! Applied penalty of {STUCK_PENALTY}")
        else:
            self.stuck_counter = 0  # Reset counter if moving
        
        # Detect if car is damaged
        damage = self.state.getDamage()
        if damage is not None and damage > 0:
            # Very minimal penalty for damage
            reward -= damage * 0.01  # Reduced penalty for damage
        
        # Check if current step is terminal
        done = False
        
        # Game over only if completely off track
        if abs(trackPos) > 0.98:
            done = True
            reward -= 100  # Penalty for going off track
            
        # Print reward breakdown occasionally
        if self.step % 500 == 0:
            print(f"Reward breakdown - Speed: {speedX * SPEED_REWARD_MULTIPLIER:.1f}, "
                  f"TrackPos: {trackPos_penalty:.1f}, Angle: {angle_penalty:.1f}, "
                  f"Total: {reward:.1f}")
            
        return reward, done
    
    def drive(self, msg):
        '''
        Main driving function that now uses the RL agent
        '''
        # Parse message to get car state
        self.state.setFromMsg(msg)
        
        # Log telemetry
        self.log_telemetry()
        
        # Get state representation for agent
        current_state = self.get_state_representation()
        
        # Store state for episode history
        self.episode_states.append({
            'angle': self.state.getAngle(),
            'trackPos': self.state.getTrackPos(),
            'speedX': self.state.getSpeedX(),
            'track': self.state.getTrack(),
            'opponents': self.state.getOpponents()
        })
        
        # Current speed for monitoring
        current_speed = self.state.getSpeedX() if self.state.getSpeedX() is not None else 0
        
        # ANTI-STALL: Check for stalling
        if current_speed < STUCK_SPEED_THRESHOLD and self.last_speed < STUCK_SPEED_THRESHOLD:
            print(f"ANTI-STALL: Detected slow speed {current_speed:.2f}, forcing acceleration")
        
        # Store speed for next iteration
        self.last_speed = current_speed
        
        # In training mode, decay exploration noise
        if self.training:
            self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        
        # Print speed info
        if self.step % 100 == 0:
            print(f"Current speed: {current_speed:.2f}, Track position: {self.state.getTrackPos():.2f}")
        
        # Get action from agent (with exploration noise in training)
        if self.training and np.random.random() < self.epsilon:
            # Random action for exploration with anti-stall bias
            steering = np.random.uniform(-0.5, 0.5)  # Limit random steering range
            
            # ANTI-STALL: Force high acceleration in all episodes
            acceleration = np.random.uniform(0.5, 1.0)  # High random acceleration
            brake = 0  # No braking
            
            # Convert to agent action format
            action = np.array([steering, acceleration - brake], dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)
        else:
            # Get action from agent
            add_noise = self.training and np.random.random() < self.epsilon
            action = self.agent.get_action(current_state, add_noise=add_noise)
        
        # ANTI-STALL: FORCE high acceleration and limited steering for ALL episodes
        # Reduce steering range for stability
        action[0] = action[0] * 0.5  # Limit steering to 50%
        
        # ANTI-STALL: Force minimum acceleration in ALL episodes
        action[1] = max(action[1], MIN_ACCELERATION)  # Minimum acceleration
        
        # ANTI-STALL: Extra force on acceleration if detected as stuck
        if self.stuck_counter > 5:
            action[1] = 1.0  # Full acceleration when stuck
            action[0] = 0.0  # Straighten steering to prevent zigzagging
            print(f"ANTI-STALL OVERRIDE: Forcing full acceleration to escape stuck state ({self.stuck_counter} steps)")
        
        # Print forced action
        if self.step % 100 == 0:
            print(f"Action: acceleration={action[1]:.2f}, steering={action[0]:.2f}")
        
        # Apply action to car control
        self.apply_action(action)
        
        # In training mode, compute reward and store transition
        if self.training and self.prev_state is not None:
            reward, done = self.compute_reward(current_state, action)
            
            # Scale rewards in early episodes to establish basic driving
            if self.episode < 30:
                # More weight to early rewards to establish basic driving
                reward = reward * (1 + (30 - self.episode) / 30)
            
            self.total_reward += reward
            
            # Print current reward every 100 steps
            if self.step % 100 == 0:
                print(f"Step {self.step}, Current Reward: {reward:.4f}, Speed: {current_speed:.2f}")
            
            # Store transition in replay buffer
            self.replay_buffer.add(
                self.prev_state, 
                self.prev_action, 
                reward, 
                current_state, 
                done
            )
            
            # Train agent if enough samples in buffer
            if (self.replay_buffer.size() > TRAINING_START and 
                self.step % TRAIN_EVERY == 0):
                
                batch = self.replay_buffer.sample(BATCH_SIZE)
                critic_loss, actor_loss = self.agent.update(batch)
                
                if self.step % 100 == 0:
                    print(f"Step {self.step}, Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}")
            
            # Check if episode is done
            if done:
                self.handle_episode_end()
        
        # Update previous state and action
        self.prev_state = current_state
        self.prev_action = action
        
        # Increment step counter
        self.step += 1
        
        # Apply ABS for better braking
        self.abs()
        
        # ANTI-STALL: Final safety check - override if car is not moving
        if current_speed < 1.0 and not self.training:
            print("ANTI-STALL SAFETY OVERRIDE: Car not moving, forcing full acceleration!")
            self.control.setAccel(1.0)
            self.control.setBrake(0.0)
            self.control.setSteer(0.0)
        
        return self.control.toMsg()
    
    def apply_action(self, action):
        '''Apply agent's action to car control'''
        # Extract steering and acceleration from action
        steering = action[0]
        accel_brake = action[1]
        
        # ANTI-STALL: FORCE ACCELERATION for ALL episodes
        # Force acceleration
        accel_brake = max(accel_brake, MIN_ACCELERATION)  # Minimum acceleration
        
        # Apply steering (already in [-1, 1] range)
        self.control.setSteer(steering)
        
        # Process gear
        self.gear()
        
        # Process acceleration/braking
        if accel_brake >= 0:
            # Positive value means acceleration
            self.control.setAccel(accel_brake)
            self.control.setBrake(0)
        else:
            # Negative value means braking
            # ANTI-STALL: Limited braking for ALL episodes
            self.control.setAccel(0.1)  # Maintain some acceleration
            self.control.setBrake(-accel_brake * 0.3)  # Reduced braking
    
    def handle_episode_end(self):
        '''Handle end of episode in training mode'''
        # Log episode reward
        self.last_rewards.append(self.total_reward)
        if len(self.last_rewards) > 10:
            self.last_rewards.pop(0)
        
        # Log reward data
        self.log_reward()
        
        # Check if this is the best episode
        if self.total_reward > self.best_reward:
            self.best_reward = self.total_reward
            self.agent.save_models(self.episode)
            print(f"New best reward: {self.best_reward:.2f} at episode {self.episode}")
        
        # Save model periodically
        if self.episode % SAVE_INTERVAL == 0:
            self.agent.save_models(self.episode)
        
        # Print episode summary
        avg_reward = sum(self.last_rewards) / len(self.last_rewards)
        max_speed = max([0] + [s['speedX'] for s in self.episode_states]) if hasattr(self, 'episode_states') else 0
        print(f"Episode {self.episode} - "
              f"Reward: {self.total_reward:.2f}, "
              f"Avg Reward: {avg_reward:.2f}, "
              f"Max Speed: {max_speed:.2f}, "
              f"Epsilon: {self.epsilon:.3f}")
        
        # Reset episode variables
        self.episode_states = []
        self.step = 0
        self.total_reward = 0.0
        self.prev_state = None
        self.prev_action = None
        self.stuck_counter = 0
        self.last_speed = 0
        
        # Increment episode counter
        self.episode += 1
    
    def gear(self):
        '''
        Enhanced gear management to prevent gear cycling issues
        '''
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        speed = self.state.getSpeedX()
        
        # Safety check
        if gear <= 0:
            self.control.setGear(1)
            return
        
        # ANTI-STALL: Force first gear at very low speeds
        if speed < 5:
            self.control.setGear(1)
            return
        
        # Use the gear logic with improved downshift prevention
        if self.prev_rpm is None:
            up = True
        else:
            if (self.prev_rpm - rpm) < 0:
                up = True
            else:
                up = False
        
        # Shift up at high RPM
        if up and rpm > 7000:
            gear += 1
        
        # Shift down at low RPM - BUT ONLY if we're not already in gear 1
        if not up and rpm < 3000 and gear > 1:
            gear -= 1
        
        # Double-check we're never below gear 1
        if gear <= 0:
            gear = 1
        
        self.control.setGear(gear)
        
        # Store current RPM for next iteration
        self.prev_rpm = rpm
    
    def abs(self):
        '''
        Anti-lock braking system to prevent wheel lockup
        '''
        speed = self.state.getSpeedX()
        wheelSpinVel = self.state.getWheelSpinVel()
        brake = self.control.getBrake()
        
        if speed < 3 or not wheelSpinVel or brake < 0.1:
            return  # ABS not needed
            
        # Calculate wheel slip
        wheel_slip = []
        for wheel_speed in wheelSpinVel:
            if wheel_speed > 0 and speed > 0:
                # Slip ratio calculation (simplified)
                wheel_speed_ms = wheel_speed * 0.3  # approximate wheel radius is 0.3m
                slip = (speed - wheel_speed_ms) / speed
                wheel_slip.append(slip)
            else:
                wheel_slip.append(0)
        
        # Check if any wheel is slipping too much
        max_slip = max(wheel_slip) if wheel_slip else 0
        if max_slip > 0.2:  # Over 20% slip
            # Reduce brake pressure to prevent lockup
            reduced_brake = brake * 0.7
            self.control.setBrake(reduced_brake)
    
    def onShutDown(self):
        '''Handle shutdown'''
        # Save final model if in training mode
        if self.training:
            self.agent.save_models(self.episode)
            print(f"Final model saved at episode {self.episode}")
            
        # Save configuration for reproducibility
        config = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_speed": self.max_speed,
            "training_params": {
                "max_episodes": MAX_EPISODES,
                "batch_size": BATCH_SIZE,
                "buffer_size": BUFFER_SIZE,
                "training_start": TRAINING_START,
                "epsilon_decay": EPSILON_DECAY,
                "epsilon_min": EPSILON_MIN
            },
            "timestamp": self.timestamp,
            "total_episodes": self.episode,
            "best_reward": self.best_reward
        }
        
        with open(f'models/config_{self.timestamp}.json', 'w') as f:
            json.dump(config, f, indent=4)
    
    def onRestart(self):
        '''Handle restart'''
        self.prev_rpm = None
        self.episode_states = []
        self.stuck_counter = 0
        self.last_speed = 0
        
        # If in training mode, consider this a new episode
        if self.training:
            self.handle_episode_end()