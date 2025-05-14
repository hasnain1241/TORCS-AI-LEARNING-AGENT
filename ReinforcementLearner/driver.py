import msgParser
import carState
import carControl
import logging
import os
import datetime
import math
import numpy as np
import learningAgent
import gearControl

class Driver(object):
    '''
    A learning-based driver for the TORCS simulator that learns from experience
    '''

    def __init__(self, stage):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        # Car constants
        self.steer_lock = 0.785398
        self.max_speed = 350
        self.prev_rpm = None
        
        # Load existing model if available or create new one
        model_path = self._find_latest_model()
        self.ai = learningAgent.RacingAI(load_model_path=model_path)
        
        # Set training mode based on stage
        self.ai.set_training_mode(self.stage != self.RACE)  # Train in non-race mode
        
        # Initialize gear controller
        self.gear_controller = gearControl.GearController()
        
        # Track memory - remembers recent states
        self.track_history = []
        self.max_history = 100
        
        # Performance tracking
        self.lap_start_time = None
        self.best_lap_time = float('inf')
        self.damage_at_lap_start = 0
        self.laps_completed = 0
        
        # Setup logging for telemetry
        self.setup_logging()
        
        # Episode tracking
        self.episode_step = 0
        self.episode_timeout = 10000  # Maximum steps per episode
        self.is_episode_end = False
        self.prev_distance = 0
        self.stuck_counter = 0
        self.max_stuck_steps = 100  # Maximum steps allowed to be stuck
        
        print(f"AI Driver initialized with model: {model_path or 'New model'}")
    
    def _find_latest_model(self):
        """Find the latest saved model file"""
        models_dir = 'models'
        best_model_path = os.path.join(models_dir, 'best_model.pt')
        
        # Check if best model exists
        if os.path.exists(best_model_path):
            return best_model_path
            
        # If not, check for any other models
        if os.path.exists(models_dir):
            models = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith('.pt')]
            if models:
                # Sort by modification time (newest first)
                models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                return models[0]
        
        return None  # No model found
    
    def setup_logging(self):
        '''Setup logging system for telemetry data'''
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        # Create a timestamp for the log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f'logs/telemetry_{timestamp}.csv'
        
        # Configure logging
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(message)s',
            filemode='w'
        )
        self.logger = logging.getLogger('telemetry')
        
        # Write header for CSV file
        header = "timestamp,angle,curLapTime,damage,distFromStart,distRaced,fuel,gear,lastLapTime," + \
                 "racePos,rpm,speedX,speedY,speedZ,trackPos,z," + \
                 "track_data,opponents_data,wheelSpinVel_data," + \
                 "steering,acceleration,brake,reward,episode,episode_step"
        self.logger.info(header)
        
        print(f"Telemetry logging started. Data will be saved to {log_file}")

    def log_telemetry(self, steering=0, accel=0, brake=0, reward=0):
        '''Log telemetry data to CSV file'''
        if self.state.sensors is None:
            return
        
        # Get timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3]
        
        # Basic telemetry data
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
        
        # Add array data (track, opponents, wheelSpinVel) - convert lists to comma-separated strings
        track_data = "|".join([str(x) for x in self.state.getTrack()]) if self.state.getTrack() else "0"
        opponents_data = "|".join([str(x) for x in self.state.getOpponents()]) if self.state.getOpponents() else "0"
        wheel_data = "|".join([str(x) for x in self.state.getWheelSpinVel()]) if self.state.getWheelSpinVel() else "0"
        
        # Add control and learning data
        control_data = [
            str(steering),
            str(accel),
            str(brake),
            str(reward),
            str(self.ai.episode_count),
            str(self.episode_step)
        ]
        
        # Combine all data
        log_data = ",".join(data_str) + "," + track_data + "," + opponents_data + "," + wheel_data + "," + ",".join(control_data)
        
        # Log to file
        self.logger.info(log_data)
        
        # Add to track history for trend analysis
        if len(self.track_history) >= self.max_history:
            self.track_history.pop(0)
        self.track_history.append({
            'angle': self.state.getAngle(),
            'trackPos': self.state.getTrackPos(),
            'speedX': self.state.getSpeedX(),
            'track': self.state.getTrack(),
            'opponents': self.state.getOpponents()
        })
    
    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        # Set up rangefinder angles for optimal track sensing
        # Wide-angle sensors for overall awareness
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        # More precise angles in front for accurate steering
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    def drive(self, msg):
        '''
        Main driving function called every update
        '''
        self.state.setFromMsg(msg)
        self.episode_step += 1
        
        # Check for episode end conditions
        self._check_episode_end()
        
        # If episode ended, handle it
        if self.is_episode_end:
            self._handle_episode_end()
            self.is_episode_end = False
            
        # Check for specific lap tracking
        self._track_lap_performance()
        
        # Check for recovery situations first
        if self.needs_recovery():
            return self.recovery_mode()
        
        # Get action from neural network
        steering, accel, brake = self.ai.get_action(self.state)
        
        # Apply the actions
        self.control.setSteer(steering)
        self.control.setAccel(accel)
        self.control.setBrake(brake)
        
        # Use learning-based gear selection
        self._handle_gear_selection()
        
        # Process learning if in training mode
        if self.ai.training_mode:
            # Let the agent learn from this step
            self.ai.learn(self.state, done=False)
        
        # Log telemetry
        self.log_telemetry(steering, accel, brake, self.ai.episode_reward)
        
        # Track previous distance for stuck detection
        self.prev_distance = self.state.getDistRaced() or 0
        
        return self.control.toMsg()
    
    def _check_episode_end(self):
        """Check if current episode should end"""
        # Get current state values
        distance = self.state.getDistRaced() or 0
        track_pos = self.state.getTrackPos() or 0
        speed = self.state.getSpeedX() or 0
        
        # Check for timeout
        if self.episode_step >= self.episode_timeout:
            print("Episode ended: Timeout")
            self.is_episode_end = True
            return
            
        # Check for off-track
        if abs(track_pos) > 1.0:
            print("Episode ended: Off track")
            self.is_episode_end = True
            return
            
        # Check for being stuck (no forward progress)
        progress = distance - self.prev_distance
        if abs(progress) < 0.001 and speed < 1.0:
            self.stuck_counter += 1
            if self.stuck_counter > 100:  # Stuck for 100 consecutive steps
                print("Episode ended: Car stuck")
                self.is_episode_end = True
                return
        else:
            self.stuck_counter = 0  # Reset stuck counter if making progress
            
        # Check for significant damage
        damage = self.state.getDamage() or 0
        if damage > 10000:  # High damage threshold
            print("Episode ended: Excessive damage")
            self.is_episode_end = True
            return
    
    def _handle_episode_end(self):
        """Handle the end of an episode"""
        # Mark this as a terminal state for learning
        if self.ai.training_mode:
            self.ai.learn(self.state, done=True)
            self.ai.episode_end()
        
        # Reset episode counters
        self.episode_step = 0
        self.stuck_counter = 0
        self.prev_distance = 0
        
        # Periodically save AI model
        if self.ai.episode_count % 5 == 0:
            # Save current model state
            self.ai.agent.save_model(self.ai.episode_count, self.ai.episode_reward)
    
    def _track_lap_performance(self):
        """Track lap times and performance metrics"""
        # Get current lap time
        cur_lap_time = self.state.getCurLapTime()
        last_lap_time = self.state.getLastLapTime()
        
        # Check for lap start
        if cur_lap_time is not None and cur_lap_time < 5.0 and self.lap_start_time is not None and cur_lap_time < self.lap_start_time:
            # Completed a lap
            self.laps_completed += 1
            
            # Record best lap time
            if last_lap_time is not None and last_lap_time > 0 and last_lap_time < self.best_lap_time:
                self.best_lap_time = last_lap_time
                print(f"New best lap time: {self.best_lap_time:.3f}s")
            
            # Calculate damage taken during lap
            current_damage = self.state.getDamage() or 0
            damage_in_lap = current_damage - self.damage_at_lap_start
            
            print(f"Lap {self.laps_completed} completed: Time={last_lap_time:.3f}s, Damage={damage_in_lap:.1f}")
            
            # Reset lap tracking
            self.damage_at_lap_start = current_damage
        
        # Update lap start time
        self.lap_start_time = cur_lap_time
    
    def _handle_gear_selection(self):
        """Handle gear selection using the gear controller"""
        # Get current state
        current_gear = self.state.getGear() or 1
        rpm = self.state.getRpm() or 0
        speed = self.state.getSpeedX() or 0
        
        # Calculate acceleration from speed history
        acceleration = 0
        if len(self.track_history) > 1:
            prev_speed = self.track_history[-1]['speedX'] or 0
            current_speed = speed
            acceleration = current_speed - prev_speed
        
        # Track position and steering
        track_pos = self.state.getTrackPos() or 0
        steering = self.control.getSteer() or 0
        
        # Get recommended gear from controller
        new_gear = self.gear_controller.get_gear(
            current_gear, rpm, speed, acceleration, track_pos, steering
        )
        
        # Set the gear
        self.control.setGear(new_gear)
    
    def needs_recovery(self):
        '''Determine if car needs recovery mode'''
        speed = self.state.getSpeedX()
        
        # Car is moving in reverse
        if speed < -1.0:
            print("RECOVERY: Car moving backward, activating recovery")
            return True
            
        # Car is stuck or very slow for longer period
        if abs(speed) < 2.0 and len(self.track_history) > 20:
            # Check if we've been slow for a while
            slow_count = 0
            for i in range(1, min(20, len(self.track_history))):
                if abs(self.track_history[-i]['speedX']) < 2.0:
                    slow_count += 1
            
            if slow_count > 15:  # Been slow for 15+ recent readings
                print("RECOVERY: Car appears to be stuck, activating recovery")
                return True
                
        return False
        
    def recovery_mode(self):
        '''Special mode to recover from stuck situations'''
        # Create fresh control object
        self.control = carControl.CarControl()
        
        # Force gear 1
        self.control.setGear(1)
        
        # Apply full throttle
        self.control.setAccel(1.0)
        
        # Apply steering toward track center
        track_pos = self.state.getTrackPos()
        if abs(track_pos) > 0.1:
            # Steer toward center
            recovery_steer = -0.5 * track_pos / max(0.1, abs(track_pos))
            self.control.setSteer(max(-0.5, min(0.5, recovery_steer)))
        else:
            # If already centered, use a small alternating steering to "wiggle" free
            wiggle = 0.2 * math.sin(datetime.datetime.now().timestamp() * 5)
            self.control.setSteer(wiggle)
            
        # Ensure zero brake
        self.control.setBrake(0.0)
        
        print("RECOVERY: Applied recovery controls")
        return self.control.toMsg()
    
    def onShutDown(self):
        """Handle shutdown - save model and results"""
        print("Shutting down driver...")
        
        # Save final AI model
        if self.ai.training_mode:
            final_model_path = self.ai.agent.save_model(
                self.ai.episode_count, 
                self.ai.episode_reward
            )
            print(f"Final model saved to: {final_model_path}")
            
            # Save results data
            self.ai.save_results()
            
        print(f"Performance summary:")
        print(f"Episodes completed: {self.ai.episode_count}")
        print(f"Laps completed: {self.laps_completed}")
        if self.best_lap_time < float('inf'):
            print(f"Best lap time: {self.best_lap_time:.3f}s")
    
    def onRestart(self):
        """Handle restart - reset episode counters"""
        print("Restarting driver...")
        
        # Reset counters
        self.episode_step = 0
        self.stuck_counter = 0
        self.prev_distance = 0
        self.prev_rpm = None
        self.track_history = []
        
        # End current episode
        if self.ai.training_mode:
            self.ai.episode_end()
            
        # Reset lap tracking
        self.lap_start_time = None
        self.damage_at_lap_start = 0