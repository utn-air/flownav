#!/bin/bash

# Create a new tmux session
session_name="record_bag_$(date +%s)"
tmux new-session -d -s $session_name

tmux set-option -g mouse on

# Change the directory to ../topomaps/bags and run the rosbag record command in the third pane
tmux select-pane -t 0
# Enabling Discovery Server mode creates this file
tmux send-keys "source /etc/turtlebot4/setup.bash" Enter
# Source venv
tmux send-keys "source .venv/bin/activate" Enter
tmux send-keys "cd ../topomaps/bags" Enter
tmux send-keys "ros2 bag record $1 -o $2" # change topic if necessary

# Attach to the tmux session
# terminator -x tmux attach-session -t $session_name
tmux -2 attach-session -t $session_name