#!/usr/bin/env python

"""
Evaluation script for TORCS DDPG agent (Windows-safe version)
"""

import os
import sys
import time
import argparse
import subprocess

def print_banner():
    print("\n" + "="*80)
    print("  TORCS Racing AI - Model Evaluation Script  ".center(80, "="))
    print("="*80 + "\n")

def find_best_model():
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print("No models directory found!")
        return None
    model_files = [f for f in os.listdir(models_dir) if f.startswith('actor_episode_') and f.endswith('.pt')]
    if not model_files:
        print("No trained models found!")
        return None
    episode_numbers = [int(f.replace('actor_episode_', '').replace('.pt', '')) for f in model_files]
    return max(episode_numbers)

def main():
    parser = argparse.ArgumentParser(description='Evaluate DDPG agent for TORCS')
    parser.add_argument('--host', dest='host_ip', default='localhost')
    parser.add_argument('--port', type=int, dest='host_port', default=3001)
    parser.add_argument('--episodes', type=int, dest='episodes', default=5)
    parser.add_argument('--track', dest='track', default=None)
    parser.add_argument('--model', type=int, dest='model_episode', default=None)
    args = parser.parse_args()

    if args.model_episode is None:
        best_model = find_best_model()
        if best_model is None:
            print("Error: No trained models found and no model episode specified!")
            return
        args.model_episode = best_model

    print_banner()
    print(f"Host: {args.host_ip}:{args.host_port}")
    print(f"Episodes: {args.episodes}")
    print(f"Track: {args.track if args.track else 'Default'}")
    print(f"Model episode: {args.model_episode}")
    print('-' * 80)

    for directory in ['logs', 'models']:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # ❗ SKIPPING TORCS LAUNCH (Windows-safe)
    print("Skipping TORCS launch on Windows. Please start TORCS manually.")

    try:
        # Start evaluation
        client_cmd = [
            'python', 'pyclient.py',
            '--host', args.host_ip,
            '--port', str(args.host_port),
            '--maxEpisodes', str(args.episodes),
            '--load', str(args.model_episode)
        ]
        if args.track:
            client_cmd.extend(['--track', args.track])

        print("Starting evaluation client...")
        print(f"Command: {' '.join(client_cmd)}")

        client_proc = subprocess.Popen(client_cmd)
        client_proc.wait()
        print("Evaluation completed!")

    except KeyboardInterrupt:
        print("Evaluation interrupted by user!")
    finally:
        print("Evaluation script finished.")

if __name__ == "__main__":
    main()
