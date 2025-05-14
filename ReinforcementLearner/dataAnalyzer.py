import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import torch # type: ignore

class DataAnalyzer:
    """
    Utility class for analyzing TORCS racing data and visualizing model performance
    """
    
    def __init__(self, telemetry_dir='logs', models_dir='models', results_dir='results'):
        """Initialize the data analyzer"""
        self.telemetry_dir = telemetry_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # Ensure directories exist
        os.makedirs(telemetry_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Visualization output directory
        self.viz_dir = 'visualizations'
        os.makedirs(self.viz_dir, exist_ok=True)
        
        self.telemetry_data = None
        self.training_results = None
    
    def load_telemetry(self, file_path=None):
        """
        Load telemetry data from a CSV file
        
        Args:
            file_path: Path to CSV file. If None, loads the most recent file.
        """
        if file_path is None:
            # Find most recent telemetry file
            files = [os.path.join(self.telemetry_dir, f) for f in os.listdir(self.telemetry_dir)
                    if f.startswith('telemetry_') and f.endswith('.csv')]
            
            if not files:
                print("No telemetry files found.")
                return None
                
            # Sort by modification time (newest first)
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            file_path = files[0]
        
        try:
            # Load telemetry data
            print(f"Loading telemetry from: {file_path}")
            
            # Custom parsing for pipe-delimited nested data
            self.telemetry_data = pd.read_csv(file_path)
            
            # Process track, opponents, and wheelSpinVel data that use pipe delimiter
            for col in ['track_data', 'opponents_data', 'wheelSpinVel_data']:
                if col in self.telemetry_data.columns:
                    # Convert pipe-delimited strings to lists
                    self.telemetry_data[col] = self.telemetry_data[col].apply(
                        lambda x: [float(v) for v in str(x).split('|')] if isinstance(x, str) else []
                    )
            
            print(f"Loaded {len(self.telemetry_data)} telemetry records.")
            return self.telemetry_data
            
        except Exception as e:
            print(f"Error loading telemetry data: {e}")
            return None
    
    def load_training_results(self, file_path=None):
        """
        Load training results from a pickle file
        
        Args:
            file_path: Path to pickle file. If None, loads the most recent file.
        """
        if file_path is None:
            # Find most recent results file
            files = [os.path.join(self.results_dir, f) for f in os.listdir(self.results_dir)
                    if f.startswith('results_') and f.endswith('.pkl')]
            
            if not files:
                print("No results files found.")
                return None
                
            # Sort by modification time (newest first)
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            file_path = files[0]
        
        try:
            # Load results data
            print(f"Loading training results from: {file_path}")
            with open(file_path, 'rb') as f:
                self.training_results = pickle.load(f)
            
            print(f"Loaded training results for {self.training_results.get('episodes', 0)} episodes.")
            return self.training_results
            
        except Exception as e:
            print(f"Error loading training results: {e}")
            return None
    
    def visualize_training_progress(self, save_path=None):
        """
        Visualize training progress from the loaded results
        
        Args:
            save_path: Path to save the visualization. If None, auto-generates a filename.
        """
        if self.training_results is None:
            print("No training results loaded. Call load_training_results() first.")
            return
        
        # Extract data
        reward_history = self.training_results.get('reward_history', [])
        loss_history = self.training_results.get('loss_history', [])
        
        if not reward_history:
            print("No reward history found in training results.")
            return
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot rewards
        plt.subplot(2, 1, 1)
        plt.plot(reward_history, 'b-')
        plt.title('Training Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        # Calculate moving average
        window_size = min(10, len(reward_history))
        if window_size > 1:
            moving_avg = np.convolve(reward_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(reward_history)), moving_avg, 'r-', label=f'{window_size}-Episode Average')
            plt.legend()
        
        # Plot losses if available
        if loss_history:
            plt.subplot(2, 1, 2)
            
            # Extract critic and actor losses
            critic_losses = [x[0] for x in loss_history]
            actor_losses = [x[1] for x in loss_history]
            
            plt.plot(critic_losses, 'g-', label='Critic Loss')
            plt.plot(actor_losses, 'r-', label='Actor Loss')
            plt.title('Training Losses')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            # Create a filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.viz_dir, f'training_progress_{timestamp}.png')
        
        plt.savefig(save_path)
        print(f"Visualization saved to: {save_path}")
        plt.close()
    
    def visualize_lap_times(self, save_path=None):
        """
        Visualize lap times from telemetry data
        
        Args:
            save_path: Path to save the visualization. If None, auto-generates a filename.
        """
        if self.telemetry_data is None:
            print("No telemetry data loaded. Call load_telemetry() first.")
            return
        
        # Extract lap time data
        lap_times = []
        current_lap = 0
        prev_lap_time = 0
        
        for i, row in self.telemetry_data.iterrows():
            # Get current and last lap time
            cur_lap_time = row.get('curLapTime', 0)
            last_lap_time = row.get('lastLapTime', 0)
            
            # Check for lap completion
            if last_lap_time > 0 and last_lap_time != prev_lap_time:
                lap_times.append((current_lap, last_lap_time))
                current_lap += 1
                prev_lap_time = last_lap_time
        
        if not lap_times:
            print("No lap time data found in telemetry.")
            return
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Extract data for plotting
        lap_numbers, times = zip(*lap_times)
        
        # Plot lap times
        plt.bar(lap_numbers, times, color='blue', alpha=0.7)
        plt.axhline(y=min(times), color='r', linestyle='--', label=f'Best: {min(times):.2f}s')
        
        plt.title('Lap Times')
        plt.xlabel('Lap Number')
        plt.ylabel('Time (seconds)')
        plt.grid(True, axis='y')
        plt.legend()
        
        # Save figure
        if save_path is None:
            # Create a filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.viz_dir, f'lap_times_{timestamp}.png')
        
        plt.savefig(save_path)
        print(f"Visualization saved to: {save_path}")
        plt.close()
    
    def visualize_racing_line(self, lap_number=None, save_path=None):
        """
        Visualize the racing line for a specific lap
        
        Args:
            lap_number: Lap number to visualize. If None, uses the last complete lap.
            save_path: Path to save the visualization. If None, auto-generates a filename.
        """
        if self.telemetry_data is None:
            print("No telemetry data loaded. Call load_telemetry() first.")
            return
        
        # Find lap boundaries
        lap_boundaries = []
        current_lap = 0
        lap_start_idx = 0
        
        for i, row in self.telemetry_data.iterrows():
            # Get current and last lap time
            cur_lap_time = row.get('curLapTime', 0)
            
            # Check for lap start
            if i > 0 and cur_lap_time < 5.0 and self.telemetry_data.iloc[i-1].get('curLapTime', 0) > 5.0:
                # End of previous lap
                lap_boundaries.append((current_lap, lap_start_idx, i-1))
                current_lap += 1
                lap_start_idx = i
        
        # Add the last lap if it's not complete
        if lap_start_idx < len(self.telemetry_data) - 1:
            lap_boundaries.append((current_lap, lap_start_idx, len(self.telemetry_data) - 1))
        
        if not lap_boundaries:
            print("No complete laps found in telemetry.")
            return
        
        # Select lap to visualize
        if lap_number is None or lap_number >= len(lap_boundaries):
            # Use the last complete lap
            lap_number = len(lap_boundaries) - 1
        
        lap_info = lap_boundaries[lap_number]
        lap_data = self.telemetry_data.iloc[lap_info[1]:lap_info[2]+1]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot track position vs. distance
        plt.subplot(2, 1, 1)
        plt.plot(lap_data['distFromStart'], lap_data['trackPos'], 'b-')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.axhline(y=1, color='r', linestyle='-')
        plt.axhline(y=-1, color='r', linestyle='-')
        plt.title(f'Racing Line - Lap {lap_info[0]}')
        plt.xlabel('Distance (m)')
        plt.ylabel('Track Position [-1 to 1]')
        plt.grid(True)
        
        # Plot speed profile
        plt.subplot(2, 1, 2)
        plt.plot(lap_data['distFromStart'], lap_data['speedX'], 'g-')
        plt.title('Speed Profile')
        plt.xlabel('Distance (m)')
        plt.ylabel('Speed (m/s)')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            # Create a filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.viz_dir, f'racing_line_lap{lap_info[0]}_{timestamp}.png')
        
        plt.savefig(save_path)
        print(f"Visualization saved to: {save_path}")
        plt.close()
    
    def analyze_model_behavior(self, save_dir=None):
        """
        Analyze how the model behaves in different racing scenarios
        
        Args:
            save_dir: Directory to save visualizations. If None, uses the default viz directory.
        """
        if self.telemetry_data is None:
            print("No telemetry data loaded. Call load_telemetry() first.")
            return
        
        if save_dir is None:
            save_dir = self.viz_dir
        
        # Create subdirectory for this analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = os.path.join(save_dir, f'model_analysis_{timestamp}')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Analyze speed vs. track curvature
        self._analyze_speed_vs_curvature(analysis_dir)
        
        # Analyze steering behavior
        self._analyze_steering_behavior(analysis_dir)
        
        # Analyze obstacle avoidance
        self._analyze_obstacle_avoidance(analysis_dir)
        
        print(f"Model behavior analysis saved to: {analysis_dir}")
    
    def _analyze_speed_vs_curvature(self, save_dir):
        """Analyze speed vs. track curvature relationship"""
        try:
            # Calculate an approximate track curvature from the track sensors
            def estimate_curvature(track_data):
                if not track_data or len(track_data) < 19:
                    return 0
                    
                left_sensor = track_data[0]  # -90 degrees
                center_sensor = track_data[9]  # 0 degrees
                right_sensor = track_data[18]  # 90 degrees
                
                if left_sensor > 0 and right_sensor > 0 and center_sensor > 0:
                    # Simple curvature estimation
                    diff = left_sensor - right_sensor
                    sum_dist = left_sensor + right_sensor
                    if sum_dist > 0:
                        return diff / sum_dist
                return 0
            
            # Calculate curvature for each track data point
            curvatures = self.telemetry_data['track_data'].apply(estimate_curvature)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot speed vs. curvature
            plt.scatter(curvatures, self.telemetry_data['speedX'], 
                       alpha=0.5, c=self.telemetry_data['episode'], cmap='viridis')
            
            plt.title('Speed vs. Track Curvature')
            plt.xlabel('Estimated Curvature (-1 to 1)')
            plt.ylabel('Speed (m/s)')
            plt.colorbar(label='Training Episode')
            plt.grid(True)
            
            # Save figure
            save_path = os.path.join(save_dir, 'speed_vs_curvature.png')
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            print(f"Error analyzing speed vs. curvature: {e}")
    
    def _analyze_steering_behavior(self, save_dir):
        """Analyze steering behavior"""
        try:
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot steering vs. track position
            plt.subplot(2, 1, 1)
            plt.scatter(self.telemetry_data['trackPos'], self.telemetry_data['steering'], 
                       alpha=0.5, c=self.telemetry_data['episode'], cmap='viridis')
            
            plt.title('Steering vs. Track Position')
            plt.xlabel('Track Position (-1 to 1)')
            plt.ylabel('Steering (-1 to 1)')
            plt.colorbar(label='Training Episode')
            plt.grid(True)
            
            # Plot steering vs. angle
            plt.subplot(2, 1, 2)
            plt.scatter(self.telemetry_data['angle'], self.telemetry_data['steering'], 
                       alpha=0.5, c=self.telemetry_data['episode'], cmap='viridis')
            
            plt.title('Steering vs. Car Angle')
            plt.xlabel('Car Angle (rad)')
            plt.ylabel('Steering (-1 to 1)')
            plt.colorbar(label='Training Episode')
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save figure
            save_path = os.path.join(save_dir, 'steering_behavior.png')
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            print(f"Error analyzing steering behavior: {e}")
    
    def _analyze_obstacle_avoidance(self, save_dir):
        """Analyze obstacle avoidance behavior"""
        try:
            # Function to find nearest opponent
            def find_nearest_opponent(opponents_data):
                if not opponents_data:
                    return 200.0
                return min([o for o in opponents_data if o > 0], default=200.0)
            
            # Calculate nearest opponent for each data point
            nearest_opponents = self.telemetry_data['opponents_data'].apply(find_nearest_opponent)
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot nearest opponent vs. steering
            plt.subplot(2, 1, 1)
            mask = nearest_opponents < 100  # Filter to show only when opponents are reasonably close
            plt.scatter(nearest_opponents[mask], self.telemetry_data.loc[mask, 'steering'], 
                       alpha=0.5, c=self.telemetry_data.loc[mask, 'episode'], cmap='viridis')
            
            plt.title('Steering vs. Nearest Opponent Distance')
            plt.xlabel('Distance to Nearest Opponent (m)')
            plt.ylabel('Steering (-1 to 1)')
            plt.colorbar(label='Training Episode')
            plt.grid(True)
            
            # Plot nearest opponent vs. acceleration
            plt.subplot(2, 1, 2)
            plt.scatter(nearest_opponents[mask], self.telemetry_data.loc[mask, 'acceleration'], 
                       alpha=0.5, c=self.telemetry_data.loc[mask, 'episode'], cmap='viridis')
            
            plt.title('Acceleration vs. Nearest Opponent Distance')
            plt.xlabel('Distance to Nearest Opponent (m)')
            plt.ylabel('Acceleration (0 to 1)')
            plt.colorbar(label='Training Episode')
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save figure
            save_path = os.path.join(save_dir, 'obstacle_avoidance.png')
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            print(f"Error analyzing obstacle avoidance: {e}")
    
    def generate_summary_report(self, output_path=None):
        """
        Generate a comprehensive summary report of the model's performance
        
        Args:
            output_path: Path to save the report. If None, auto-generates a filename.
        """
        if self.telemetry_data is None:
            print("No telemetry data loaded. Call load_telemetry() first.")
            return
            
        if self.training_results is None:
            print("No training results loaded. Call load_training_results() first.")
            return
            
        # Create report
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.viz_dir, f'performance_report_{timestamp}.txt')
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TORCS AI RACING MODEL PERFORMANCE REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            
            # Training summary
            f.write("-" * 40 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total episodes: {self.training_results.get('episodes', 0)}\n")
            f.write(f"Total steps: {self.training_results.get('total_steps', 0)}\n")
            
            # Calculate average reward over different periods
            rewards = self.training_results.get('reward_history', [])
            if rewards:
                f.write(f"Average reward (all episodes): {np.mean(rewards):.2f}\n")
                if len(rewards) >= 10:
                    f.write(f"Average reward (last 10 episodes): {np.mean(rewards[-10:]):.2f}\n")
                if len(rewards) >= 50:
                    f.write(f"Average reward (last 50 episodes): {np.mean(rewards[-50:]):.2f}\n")
            
            # Lap time analysis
            f.write("\n" + "-" * 40 + "\n")
            f.write("LAP TIME ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            # Extract lap times
            lap_times = []
            current_lap = 0
            prev_lap_time = 0
            
            for i, row in self.telemetry_data.iterrows():
                last_lap_time = row.get('lastLapTime', 0)
                
                # Check for lap completion
                if last_lap_time > 0 and last_lap_time != prev_lap_time:
                    lap_times.append(last_lap_time)
                    prev_lap_time = last_lap_time
            
            if lap_times:
                f.write(f"Total laps completed: {len(lap_times)}\n")
                f.write(f"Best lap time: {min(lap_times):.3f}s\n")
                f.write(f"Average lap time: {np.mean(lap_times):.3f}s\n")
                f.write(f"Median lap time: {np.median(lap_times):.3f}s\n")
                f.write(f"Lap time standard deviation: {np.std(lap_times):.3f}s\n")
                f.write("\nIndividual lap times:\n")
                for i, time in enumerate(lap_times):
                    f.write(f"  Lap {i+1}: {time:.3f}s\n")
            else:
                f.write("No complete laps recorded.\n")
            
            # Damage analysis
            f.write("\n" + "-" * 40 + "\n")
            f.write("DAMAGE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            if 'damage' in self.telemetry_data.columns:
                max_damage = self.telemetry_data['damage'].max()
                avg_damage = self.telemetry_data['damage'].mean()
                f.write(f"Maximum damage: {max_damage:.1f}\n")
                f.write(f"Average damage: {avg_damage:.1f}\n")
                
                # Calculate damage per lap
                if lap_times:
                    f.write("\nDamage per lap:\n")
                    lap_boundaries = []
                    current_lap = 0
                    lap_start_idx = 0
                    
                    for i, row in self.telemetry_data.iterrows():
                        # Get current and last lap time
                        cur_lap_time = row.get('curLapTime', 0)
                        
                        # Check for lap start
                        if i > 0 and cur_lap_time < 5.0 and self.telemetry_data.iloc[i-1].get('curLapTime', 0) > 5.0:
                            # End of previous lap
                            lap_boundaries.append((current_lap, lap_start_idx, i-1))
                            current_lap += 1
                            lap_start_idx = i
                    
                    # Add the last lap if it's not complete
                    if lap_start_idx < len(self.telemetry_data) - 1:
                        lap_boundaries.append((current_lap, lap_start_idx, len(self.telemetry_data) - 1))
                    
                    for lap, start_idx, end_idx in lap_boundaries:
                        lap_data = self.telemetry_data.iloc[start_idx:end_idx+1]
                        if len(lap_data) > 0:
                            start_damage = lap_data['damage'].iloc[0]
                            end_damage = lap_data['damage'].iloc[-1]
                            damage_in_lap = end_damage - start_damage
                            f.write(f"  Lap {lap+1}: {damage_in_lap:.1f}\n")
            else:
                f.write("No damage data available.\n")
            
            # Speed analysis
            f.write("\n" + "-" * 40 + "\n")
            f.write("SPEED ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            if 'speedX' in self.telemetry_data.columns:
                max_speed = self.telemetry_data['speedX'].max()
                avg_speed = self.telemetry_data['speedX'].mean()
                median_speed = self.telemetry_data['speedX'].median()
                f.write(f"Maximum speed: {max_speed:.1f} m/s ({max_speed*3.6:.1f} km/h)\n")
                f.write(f"Average speed: {avg_speed:.1f} m/s ({avg_speed*3.6:.1f} km/h)\n")
                f.write(f"Median speed: {median_speed:.1f} m/s ({median_speed*3.6:.1f} km/h)\n")
                
                # Speed percentiles
                percentiles = [10, 25, 50, 75, 90]
                f.write("\nSpeed percentiles:\n")
                for p in percentiles:
                    speed_p = np.percentile(self.telemetry_data['speedX'], p)
                    f.write(f"  {p}th percentile: {speed_p:.1f} m/s ({speed_p*3.6:.1f} km/h)\n")
            else:
                f.write("No speed data available.\n")
            
            # Conclusion
            f.write("\n" + "-" * 40 + "\n")
            f.write("CONCLUSION\n")
            f.write("-" * 40 + "\n")
            
            # Calculate improvement metrics
            if rewards:
                first_10_avg = np.mean(rewards[:10]) if len(rewards) >= 10 else np.mean(rewards)
                last_10_avg = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
                improvement = ((last_10_avg - first_10_avg) / abs(first_10_avg)) * 100 if first_10_avg != 0 else 0
                
                f.write(f"Model reward improvement: {improvement:.1f}%\n")
            
            if lap_times and len(lap_times) >= 2:
                first_lap = lap_times[0]
                last_lap = lap_times[-1]
                best_lap = min(lap_times)
                lap_improvement = ((first_lap - last_lap) / first_lap) * 100
                
                f.write(f"Lap time improvement (first to last): {lap_improvement:.1f}%\n")
                f.write(f"Lap time improvement (first to best): {((first_lap - best_lap) / first_lap) * 100:.1f}%\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"Performance report saved to: {output_path}")
        return output_path


def main():
    """Main function to run data analysis"""
    analyzer = DataAnalyzer()
    
    # Load telemetry and training results
    analyzer.load_telemetry()
    analyzer.load_training_results()
    
    # Generate visualizations
    analyzer.visualize_training_progress()
    analyzer.visualize_lap_times()
    analyzer.visualize_racing_line()
    analyzer.analyze_model_behavior()
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()