#!/usr/bin/env python

"""
Training script for TORCS DDPG agent
This script provides a more convenient way to train the agent
"""

import os
import sys
import time
import argparse
import subprocess
import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

def print_banner():
    print("\n" + "="*80)
    print("  TORCS Racing AI - DDPG Training Script  ".center(80, "="))
    print("="*80 + "\n")

def find_latest_model():
    '''Find the latest model file and return its episode number'''
    model_files = glob.glob('models/actor_episode_*.pt')
    if not model_files:
        print("No model files found!")
        return None
    
    # Extract episode numbers from filenames
    episode_numbers = [int(f.split('_')[-1].split('.')[0]) for f in model_files]
    if not episode_numbers:
        return None
    
    # Return the highest episode number
    return max(episode_numbers)

def list_available_models():
    '''List all available model checkpoints'''
    model_files = glob.glob('models/actor_episode_*.pt')
    if not model_files:
        print("No model files found!")
        return []
    
    # Extract episode numbers from filenames
    episode_numbers = [int(f.split('_')[-1].split('.')[0]) for f in model_files]
    episode_numbers.sort()
    
    return episode_numbers

def plot_rewards(rewards_file, output_file=None):
    """Plot rewards from training"""
    
    try:
        data = pd.read_csv(rewards_file)
        
        plt.figure(figsize=(12, 8))
        
        # Plot total reward
        plt.subplot(2, 2, 1)
        plt.plot(data['episode'], data['total_reward'], label='Total Reward')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Rewards')
        plt.grid(True)
        
        # Plot average reward (moving average)
        plt.subplot(2, 2, 2)
        window = min(10, len(data))
        data['avg_reward_ma'] = data['avg_reward'].rolling(window=window, min_periods=1).mean()
        plt.plot(data['episode'], data['avg_reward_ma'], label=f'Moving Avg ({window} episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Avg Reward')
        plt.title(f'Average Reward (Moving Avg {window} episodes)')
        plt.grid(True)
        
        # Plot max speed
        plt.subplot(2, 2, 3)
        plt.plot(data['episode'], data['max_speed'], label='Max Speed')
        plt.xlabel('Episode')
        plt.ylabel('Speed (m/s)')
        plt.title('Maximum Speed per Episode')
        plt.grid(True)
        
        # Plot damage
        plt.subplot(2, 2, 4)
        plt.plot(data['episode'], data['damage'], label='Damage')
        plt.xlabel('Episode')
        plt.ylabel('Damage')
        plt.title('Damage per Episode')
        plt.grid(True)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            print(f"Plot saved to {output_file}")
        else:
            plt.show()
            
    except Exception as e:
        print(f"Error plotting rewards: {e}")

def main():
    parser = argparse.ArgumentParser(description='Train DDPG agent for TORCS')
    
    parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                        help='Host IP address (default: localhost)')
    parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                        help='Host port number (default: 3001)')
    parser.add_argument('--episodes', action='store', type=int, dest='episodes', default=500,
                        help='Number of episodes to train (default: 500)')
    parser.add_argument('--track', action='store', dest='track', default=None,
                        help='Track to use for training')
    parser.add_argument('--visualize', action='store_true', dest='visualize', default=False,
                        help='Show visualization during training (slower but helpful for debugging)')
    parser.add_argument('--continue', action='store_true', dest='continue_training', default=False,
                        help='Continue training from the latest saved model')
    parser.add_argument('--from-model', type=int, dest='from_model', default=None,
                        help='Continue training from a specific model episode number')
    parser.add_argument('--plot', action='store_true', dest='plot', default=False,
                        help='Plot rewards after training')
    parser.add_argument('--list-models', action='store_true', dest='list_models', default=False,
                        help='List all available trained models')
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list_models:
        print("Available model checkpoints:")
        models = list_available_models()
        if models:
            for model in models:
                print(f"  Episode {model}")
        else:
            print("  No models found")
        return
    
    print_banner()
    print("Starting TORCS training with the following settings:")
    print(f"Host: {args.host_ip}:{args.host_port}")
    print(f"Episodes: {args.episodes}")
    print(f"Track: {args.track if args.track else 'Default'}")
    print(f"Visualization: {'Enabled' if args.visualize else 'Disabled'}")
    
    # Determine which model to continue from
    continue_from_model = None
    if args.from_model is not None:
        continue_from_model = args.from_model
        print(f"Continuing from model episode: {continue_from_model}")
    elif args.continue_training:
        continue_from_model = find_latest_model()
        if continue_from_model:
            print(f"Continuing from latest model (episode {continue_from_model})")
        else:
            print("No models found to continue from. Starting fresh training.")
    else:
        print("Starting fresh training")
    
    print('-' * 80)
    
    # Create directories if needed
    for directory in ['logs', 'models', 'plots']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Run TORCS first
    torcs_cmd = ['torcs', '-nofuel', '-nodamage', '-nolaptime']
    
    if not args.visualize:
        torcs_cmd.append('-vision')  # Less visualization for faster training
    
    torcs_proc = None
    
    try:
        # Start TORCS
        print("Skipping TORCS launch on Windows. Please start TORCS manually if needed.")
        client_cmd = [
            'python', 'pyclient.py',
            '--host', args.host_ip,
            '--port', str(args.host_port),
            '--maxEpisodes', str(args.episodes),
            '--training'
        ]
        
        if args.track:
            client_cmd.extend(['--track', args.track])
            
        # Add continue flag if requested
        if continue_from_model is not None:
            client_cmd.extend(['--load', str(continue_from_model)])
        
        print("Starting training client...")
        print(f"Command: {' '.join(client_cmd)}")
        
        client_proc = subprocess.Popen(client_cmd)
        
        # Wait for training to complete
        client_proc.wait()
        
        print("Training completed!")
        
        # Plot rewards if requested
        if args.plot:
            print("Generating reward plots...")
            
            # Find the most recent rewards file
            rewards_files = [f for f in os.listdir('logs') if f.startswith('rewards_')]
            if rewards_files:
                latest_rewards_file = max(rewards_files, key=lambda x: os.path.getmtime(os.path.join('logs', x)))
                rewards_path = os.path.join('logs', latest_rewards_file)
                plot_path = os.path.join('plots', f"training_plot_{time.strftime('%Y%m%d_%H%M%S')}.png")
                plot_rewards(rewards_path, plot_path)
            else:
                print("No rewards files found!")
                
    except KeyboardInterrupt:
        print("Training interrupted by user!")
    finally:
        # Clean up processes
        # No TORCS process to shut down on Windows
        print("Training script finished.")

if __name__ == "__main__":
    main()