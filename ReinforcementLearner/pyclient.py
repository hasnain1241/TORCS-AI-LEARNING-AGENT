#!/usr/bin/env python

import sys
import getopt
import os
import time
from ddpg import playGame

# Initialize help messages
ophelp = 'Options:\n'
ophelp += ' --host, -H <host>    TORCS server host. [localhost]\n'
ophelp += ' --port, -p <port>    TORCS port. [3001]\n'
ophelp += ' --id, -i <id>        ID for server. [SCR]\n'
ophelp += ' --steps, -m <#>      Maximum simulation steps. 1 sec ~ 50 steps. [100000]\n'
ophelp += ' --maxEpisodes, --episodes, -e <#>   Maximum learning episodes. [1]\n'
ophelp += ' --track, -t <track>  Your name for this track. Used for learning. [unknown]\n'
ophelp += ' --stage, -s <#>      0=warm up, 1=qualifying, 2=race, 3=unknown. [3]\n'
ophelp += ' --debug, -d          Output full telemetry.\n'
ophelp += ' --train              Enable training mode.\n'
ophelp += ' --timeout <#>        Socket timeout. [1.0]\n'
ophelp += ' --help, -h           Show this help.\n'
ophelp += ' --version, -v        Show current version.'
usage = 'Usage: %s [ophelp [optargs]] \n' % sys.argv[0]
usage = usage + ophelp
version = "20250511"

def main():
    # Default settings
    host = 'localhost'
    port = 3001
    sid = 'SCR'
    max_episodes = 1
    max_steps = 100000
    track = 'unknown'
    stage = 3
    debug = False
    train = True
    timeout = 1.0
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'H:p:i:m:e:t:s:dhv',
                                  ['host=', 'port=', 'id=', 'steps=',
                                   'episodes=', 'maxEpisodes=', 'track=', 'stage=',
                                   'debug', 'train', 'timeout=', 'help', 'version'])
    except getopt.error as why:
        print('getopt error: %s\n%s' % (why, usage))
        sys.exit(-1)
        
    for opt, arg in opts:
        if opt == '-h' or opt == '--help':
            print(usage)
            sys.exit(0)
        if opt == '-d' or opt == '--debug':
            debug = True
        if opt == '-H' or opt == '--host':
            host = arg
        if opt == '-i' or opt == '--id':
            sid = arg
        if opt == '-t' or opt == '--track':
            track = arg
        if opt == '-s' or opt == '--stage':
            stage = int(arg)
        if opt == '-p' or opt == '--port':
            port = int(arg)
        if opt == '-e' or opt == '--episodes' or opt == '--maxEpisodes':
            max_episodes = int(arg)
        if opt == '-m' or opt == '--steps':
            max_steps = int(arg)
        if opt == '--train':
            train = True
        if opt == '--timeout':
            timeout = float(arg)
        if opt == '-v' or opt == '--version':
            print('%s %s' % (sys.argv[0], version))
            sys.exit(0)
    
    # Print configuration
    print("\nRunning with configuration:")
    print(f"  Host: {host}:{port}")
    print(f"  Bot ID: {sid}")
    print(f"  Track: {track}")
    print(f"  Episodes: {max_episodes}")
    print(f"  Steps: {max_steps}")
    print(f"  Stage: {stage}")
    print(f"  Debug: {debug}")
    print(f"  Training: {train}")
    print(f"  Timeout: {timeout}s\n")
    
    # Run the game
    train_indicator = 1 if train else 0
    playGame(train_indicator=train_indicator)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())