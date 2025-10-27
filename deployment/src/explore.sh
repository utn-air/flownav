#!/bin/bash

# Create a new tmux session
session_name="turtlebot_$(date +%s)"
tmux new-session -d -s $session_name

tmux set-option -g mouse on

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves

# Run the navigate.py script with command line args
tmux select-pane -t 0
# Enabling Discovery Server mode creates this file 
tmux send-keys "source /etc/turtlebot4/setup.bash" Enter
# Source venv
tmux send-keys "source .venv/bin/activate" Enter
tmux send-keys "cd deployment/src/" Enter
tmux send-keys "python explore.py $@" Enter

# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 1
# Enabling Discovery Server mode creates this file 
tmux send-keys "source /etc/turtlebot4/setup.bash" Enter
# Source venv
tmux send-keys "source .venv/bin/activate" Enter
tmux send-keys "cd deployment/src/" Enter
tmux send-keys "python pd_controller.py" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
