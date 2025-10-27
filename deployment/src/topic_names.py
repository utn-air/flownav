# topic names for ROS communication


# Robot name space to append to all topics
# Change this to match your robot's namespace
# For example, if your robot is named "turtle1", you can set it to "/turtle1"
ROBOT_NAMESPACE = "/turtle1"    

# Image observation topics
IMAGE_TOPIC = f"{ROBOT_NAMESPACE}/image_compressed"

# exploration topics
WAYPOINT_TOPIC = f"{ROBOT_NAMESPACE}/waypoint"
REACHED_GOAL_TOPIC = f"{ROBOT_NAMESPACE}/topoplan/reached_goal"
SAMPLED_ACTIONS_TOPIC = f"{ROBOT_NAMESPACE}/sampled_actions"

# move the robot
VEL_TOPIC = f"repub/cmd_vel"
