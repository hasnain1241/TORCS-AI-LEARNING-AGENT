import os
import sys
import argparse
import time
import subprocess
import signal
import datetime

def print_banner():
    print("\n" + "="*80)
    print("  TORCS AI Racing - Learning Agent Launcher  ".center(80, "="))
    print("="*80 + "\n")

def setup_directories():
    for directory in ['logs', 'models', 'results', 'visualizations']:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def find_latest_model():
    models_dir = 'models'
    best_model_path = os.path.join(models_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        return best_model_path
    if os.path.exists(models_dir):
        models = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith('.pt')]
        if models:
            models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return models[0]
    return None

def run_training(args):
    print("\nStarting training session...")
    print(f"Mode: {'Training' if args.train else 'Evaluation'}")

    # Match the correct long argument names from Launcher1.py
    cmd = [
        sys.executable, 'pyclient.py',
        '--host', args.host,
        '--port', str(args.port),
        '--id', args.id,
        '--episodes', str(args.max_episodes),
        '--track', args.track or "",
        '--stage', str(args.stage)
    ]

    if args.debug:
        cmd.append('--debug')

    # Timeout is not supported in Launcher1.py unless you add it there
    # So we skip it here or print a warning
    if args.timeout:
        print(f"(Note: Timeout value {args.timeout}s is not used in client directly.)")

    print(f"Command: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(cmd)
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\nInterrupted by user. Shutting down client...")
            process.send_signal(signal.CTRL_BREAK_EVENT if os.name == 'nt' else signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Client did not exit gracefully. Terminating...")
                process.terminate()
        return process.returncode
    except Exception as e:
        print(f"Error running client: {e}")
        return 1

def analyze_results(args):
    print("\nAnalyzing training results...")
    import dataAnalyzer
    analyzer = dataAnalyzer.DataAnalyzer()
    analyzer.load_telemetry()
    analyzer.load_training_results()
    analyzer.visualize_training_progress()
    analyzer.visualize_lap_times()
    analyzer.visualize_racing_line()
    analyzer.analyze_model_behavior()
    report_path = analyzer.generate_summary_report()
    print(f"\nAnalysis complete. Report saved to: {report_path}")
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i < 15 or "CONCLUSION" in line or i >= len(lines) - 10:
                    print(line.rstrip())
    return 0

def main():
    parser = argparse.ArgumentParser(description='TORCS AI Racing - Learning Agent Launcher')
    parser.add_argument('--host', default='localhost', help='TORCS server hostname')
    parser.add_argument('--port', type=int, default=3001, help='TORCS server port')
    parser.add_argument('--id', default='SCR', help='Bot ID')
    parser.add_argument('--track', default=None, help='Track name')
    parser.add_argument('--max-episodes', type=int, default=100, help='Maximum number of episodes')
    parser.add_argument('--stage', type=int, default=0, choices=[0, 1, 2, 3],
                        help='Race stage: 0=Warm-Up, 1=Qualifying, 2=Race, 3=Unknown')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--timeout', type=float, default=1.0, help='(Unused) Socket timeout in seconds')
    parser.add_argument('--train', action='store_true', help='Enable training mode')
    parser.add_argument('--analyze', action='store_true', help='Analyze results after completion')
    args = parser.parse_args()

    print_banner()
    setup_directories()

    latest_model = find_latest_model()
    if latest_model:
        print(f"Found existing model: {latest_model}")
    else:
        print("No existing model found. Will create a new one.")

    print("\nConfiguration:")
    print(f"  Server: {args.host}:{args.port}")
    print(f"  Bot ID: {args.id}")
    print(f"  Track: {args.track or 'Default'}")
    print(f"  Maximum episodes: {args.max_episodes}")
    print(f"  Stage: {['Warm-Up', 'Qualifying', 'Race', 'Unknown'][args.stage]}")
    print(f"  Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    print(f"  Training mode: {'Enabled' if args.train else 'Disabled (Evaluation only)'}")
    print(f"  Socket timeout: {args.timeout}s (not used)")

    start_time = time.time()
    result = run_training(args)
    duration = time.time() - start_time

    print(f"\nSession finished with exit code {result}")
    print(f"Duration: {datetime.timedelta(seconds=int(duration))}")

    if args.analyze:
        analyze_results(args)

    return result

if __name__ == "__main__":
    sys.exit(main())
