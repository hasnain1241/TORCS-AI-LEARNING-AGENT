import sys
import argparse
import socket
import time
import os
import datetime
import driver
import glob

def print_banner():
    '''Print a nice banner for the client'''
    print("\n" + "="*80)
    print("  TORCS Racing AI - RL-Enhanced Python Client  ".center(80, "="))
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

if __name__ == '__main__':
    # Configure the argument parser
    parser = argparse.ArgumentParser(description='Advanced Python client with reinforcement learning for TORCS.')

    parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                        help='Host IP address (default: localhost)')
    parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                        help='Host port number (default: 3001)')
    parser.add_argument('--id', action='store', dest='id', default='SCR',
                        help='Bot ID (default: SCR)')
    parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
                        help='Maximum number of learning episodes (default: 1)')
    parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                        help='Maximum number of steps (default: 0)')
    parser.add_argument('--track', action='store', dest='track', default=None,
                        help='Name of the track')
    parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                        help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')
    parser.add_argument('--debug', action='store_true', dest='debug', default=False,
                        help='Enable debug output (default: False)')
    parser.add_argument('--timeout', action='store', type=float, dest='timeout', default=1.0,
                        help='Socket timeout in seconds (default: 1.0)')
    parser.add_argument('--profile', action='store_true', dest='profile', default=False,
                        help='Enable performance profiling (default: False)')
    parser.add_argument('--training', action='store_true', dest='training', default=False,
                        help='Enable training mode (default: False)')
    parser.add_argument('--load', type=int, dest='load_episode', default=None,
                        help='Model episode to load (use -1 to load latest)')

    arguments = parser.parse_args()

    # Handle loading the latest model if -1 is specified
    if arguments.load_episode == -1:
        latest_episode = find_latest_model()
        if latest_episode:
            arguments.load_episode = latest_episode
            print(f"Found latest model: episode {latest_episode}")
        else:
            print("No model files found. Cannot load latest model.")
            arguments.load_episode = None

    # Print banner and summary
    print_banner()
    print('Connecting to server host ip:', arguments.host_ip, '@ port:', arguments.host_port)
    print('Bot ID:', arguments.id)
    print('Maximum episodes:', arguments.max_episodes)
    print('Maximum steps:', arguments.max_steps)
    print('Track:', arguments.track if arguments.track else "Default")
    print('Stage:', ["Warm-Up", "Qualifying", "Race", "Unknown"][arguments.stage])
    print('Debug:', "Enabled" if arguments.debug else "Disabled")
    print('Timeout:', arguments.timeout, "seconds")
    print('Profiling:', "Enabled" if arguments.profile else "Disabled")
    print('Training mode:', "Enabled" if arguments.training else "Disabled")
    if arguments.load_episode:
        print('Loading model from episode:', arguments.load_episode)
    print('-' * 80)

    # Set up socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except socket.error as msg:
        print('ERROR: Could not create socket -', msg)
        sys.exit(-1)

    # Set timeout
    sock.settimeout(arguments.timeout)

    # Set up performance metrics
    start_time = time.time()
    total_steps = 0
    reconnect_count = 0
    shutdown_client = False
    cur_episode = 0

    # Enable verbose debugging if requested
    verbose = arguments.debug

    # Create driver instance
    d = driver.Driver(arguments.stage, training=arguments.training)

    # Load model if specified
    if arguments.load_episode:
        success = d.agent.load_models(arguments.load_episode)
        if success:
            print(f"Successfully loaded model from episode {arguments.load_episode}")
        else:
            print(f"Failed to load model from episode {arguments.load_episode}")
            if not arguments.training:
                print("Exiting since we're in inference mode and can't load the model")
                sys.exit(1)

    # Function to handle socket reconnection
    def reconnect_to_server():
        global reconnect_count
        reconnect_count += 1
        print(f"Attempting reconnection ({reconnect_count})...")
        time.sleep(1.0)  # Wait before retry
        return False

    # Log directory setup
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"race_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Create log file
    with open(log_filename, 'w') as log_file:
        log_file.write(f"Race started at: {datetime.datetime.now()}\n")
        log_file.write(f"Track: {arguments.track if arguments.track else 'Default'}\n")
        log_file.write(f"Stage: {['Warm-Up', 'Qualifying', 'Race', 'Unknown'][arguments.stage]}\n")
        log_file.write(f"Training: {'Enabled' if arguments.training else 'Disabled'}\n")
        if arguments.load_episode:
            log_file.write(f"Loaded model from episode: {arguments.load_episode}\n")
        log_file.write("-" * 80 + "\n")

    # Main client loop
    while not shutdown_client:
        # Connection phase
        connected = False
        
        while not connected:
            try:
                print('Sending id to server:', arguments.id)
                init_string = arguments.id + d.init()
                print('Sending init string:', init_string[:50] + "..." if len(init_string) > 50 else init_string)
                
                sock.sendto(init_string.encode(), (arguments.host_ip, arguments.host_port))
                
                buf, addr = sock.recvfrom(1000)
                buf = buf.decode('utf-8')
                
                if buf.find('***identified***') >= 0:
                    print('Successfully connected!')
                    connected = True
                    with open(log_filename, 'a') as log_file:
                        log_file.write(f"Connected to server at {datetime.datetime.now()}\n")
                else:
                    print("Received unexpected response, retrying...")
                    
            except socket.timeout:
                print("Connection timed out, retrying...")
            except socket.error as msg:
                print("Socket error:", msg)
                reconnect_to_server()
            except Exception as e:
                print("Unexpected error:", e)
                reconnect_to_server()
        
        # Racing phase
        current_step = 0
        step_start_time = time.time()
        fastest_lap = float('inf')
        damage_level = 0
        
        while connected:
            # Wait for server message
            buf = None
            try:
                buf, addr = sock.recvfrom(1000)
                buf = buf.decode('utf-8')
            except socket.timeout:
                print("No response from server...")
                if reconnect_to_server():
                    break
                continue
            except socket.error as msg:
                print("Socket error:", msg)
                if reconnect_to_server():
                    break
                continue
            
            # Check for server commands
            if buf is None:
                continue
                
            if verbose:
                print('Received:', buf[:50] + "..." if len(buf) > 50 else buf)
            
            if buf.find('***shutdown***') >= 0:
                d.onShutDown()
                shutdown_client = True
                print('Client shutdown requested by server')
                with open(log_filename, 'a') as log_file:
                    log_file.write(f"Server requested shutdown at {datetime.datetime.now()}\n")
                break
            
            if buf.find('***restart***') >= 0:
                d.onRestart()
                print('Client restart requested by server')
                with open(log_filename, 'a') as log_file:
                    log_file.write(f"Server requested restart at {datetime.datetime.now()}\n")
                break
            
            # Process step
            current_step += 1
            total_steps += 1
            
            # Check maximum steps
            if arguments.max_steps > 0 and current_step == arguments.max_steps:
                print(f"Reached maximum steps ({arguments.max_steps})")
                buf = '(meta 1)'  # Request pit stop
            else:
                # Get driver response
                step_start = time.time()
                
                if arguments.profile:
                    # With profiling
                    import cProfile
                    import pstats
                    profiler = cProfile.Profile()
                    profiler.enable()
                    buf = d.drive(buf)
                    profiler.disable()
                    
                    # Collect and display profile data
                    if current_step % 100 == 0:  # Only show profiling every 100 steps
                        stats = pstats.Stats(profiler).sort_stats('cumtime')
                        print("\nPerformance Profile:")
                        stats.print_stats(10)  # Show top 10 functions
                else:
                    # Normal execution
                    buf = d.drive(buf)
                
                # Measure step time
                step_time = time.time() - step_start
                
                # Track performance metrics
                if current_step % 100 == 0:
                    # Get state information
                    try:
                        # Access car state to get metrics
                        cur_damage = d.state.getDamage()
                        cur_lap_time = d.state.getCurLapTime()
                        last_lap_time = d.state.getLastLapTime()
                        race_pos = d.state.getRacePos()
                        speed = d.state.getSpeedX()
                        
                        # Update metrics
                        if cur_damage is not None:
                            damage_level = cur_damage
                        
                        # Track best lap time
                        if last_lap_time is not None and last_lap_time > 0:
                            fastest_lap = min(fastest_lap, last_lap_time)
                        
                        # Log progress
                        print(f"Step {current_step}: Speed={speed:.1f} m/s, Position={race_pos}, " + 
                            f"Damage={damage_level:.1f}, Lap time={cur_lap_time:.1f}s, " +
                            f"Training={'Yes' if arguments.training else 'No'}")
                        
                        # Write to log file
                        with open(log_filename, 'a') as log_file:
                            log_file.write(f"Step {current_step}: Speed={speed:.1f}, Pos={race_pos}, " + 
                                        f"Damage={damage_level:.1f}, Time={cur_lap_time:.1f}\n")
                            
                    except Exception as e:
                        if verbose:
                            print("Error accessing state:", e)
            
            # Safety check: if car is stopped during inference, force acceleration
            if not arguments.training and hasattr(d, 'state') and d.state.getSpeedX() is not None and d.state.getSpeedX() < 3.0:
                print("SAFETY OVERRIDE: Car seems stuck, forcing acceleration")
                d.control.setAccel(1.0)
                d.control.setBrake(0.0)
                buf = d.control.toMsg()
            
            # Send control message to server
            if buf is not None:
                if verbose:
                    print('Sending:', buf[:50] + "..." if len(buf) > 50 else buf)
                try:
                    sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
                except socket.error as msg:
                    print("Failed to send data:", msg)
                    if reconnect_to_server():
                        break
                        
        # Episode complete
        cur_episode += 1
        
        # Check if we've reached the maximum episodes
        if cur_episode >= arguments.max_episodes:
            shutdown_client = True
            print(f"Completed {cur_episode} episodes as requested")

    # Session complete
    session_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"Session complete after {session_time:.1f} seconds")
    print(f"Total steps: {total_steps}")
    print(f"Episodes: {cur_episode}")
    if fastest_lap < float('inf'):
        print(f"Fastest lap: {fastest_lap:.3f}s")
    print(f"Final damage level: {damage_level:.1f}")
    print(f"Training mode: {'Enabled' if arguments.training else 'Disabled'}")
    print("="*80)

    # Write final statistics to log
    with open(log_filename, 'a') as log_file:
        log_file.write("\n" + "="*80 + "\n")
        log_file.write(f"Session ended at: {datetime.datetime.now()}\n")
        log_file.write(f"Total time: {session_time:.1f} seconds\n")
        log_file.write(f"Total steps: {total_steps}\n")
        log_file.write(f"Episodes: {cur_episode}\n")
        if fastest_lap < float('inf'):
            log_file.write(f"Fastest lap: {fastest_lap:.3f}s\n")
        log_file.write(f"Final damage level: {damage_level:.1f}\n")
        log_file.write(f"Training mode: {'Enabled' if arguments.training else 'Disabled'}\n")
        log_file.write("="*80 + "\n")

    # Close socket
    sock.close()