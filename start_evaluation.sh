#!/bin/bash

# A Cyclone DDS hálózati réteg engedélyezése az instabilitások és szerviz fagyások elkerülése végett
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Source the ROS 2 setup configuration
source install/setup.bash

echo "Starting Gazebo Simulation..."
# Start Gazebo in the background
ros2 launch two_wheeled_robot load_world_into_gazebo.launch.py &
GAZEBO_PID=$!

echo "Waiting for 3 seconds to let Gazebo initialize..."
sleep 3

echo "Starting Model Evaluation (Validation)..."
# Futtatjuk az evaluate.py fájlt, ami a telepített share mappában található
python3 $(ros2 pkg prefix two_wheeled_robot)/share/two_wheeled_robot/rl/evaluate.py

# When it finishes
echo "Evaluation finished. Cleaning up Gazebo..."
kill -INT $GAZEBO_PID 2>/dev/null
sleep 2
pkill -9 -f gzserver 2>/dev/null
pkill -9 -f gzclient 2>/dev/null
kill -9 $GAZEBO_PID 2>/dev/null

echo "Cleanup complete."
