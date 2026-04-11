#!/bin/bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source install/setup.bash

echo "Starting Gazebo Simulation..."
ros2 launch two_wheeled_robot load_world_into_gazebo.launch.py &
GAZEBO_PID=$!

sleep 3

echo "Starting Model Testing (Infinite loop)..."
python3 $(ros2 pkg prefix two_wheeled_robot)/share/two_wheeled_robot/rl/test.py

echo "Wait, cleaning up Gazebo..."
kill -INT $GAZEBO_PID 2>/dev/null
sleep 2
pkill -9 -f gzserver 2>/dev/null
pkill -9 -f gzclient 2>/dev/null
kill -9 $GAZEBO_PID 2>/dev/null

echo "Cleanup complete."
