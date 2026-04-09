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

echo "Starting Training Script..."
# Start the RL training
ros2 launch two_wheeled_robot launch_rl_training.launch.py

# When the training finishes (or gets interrupted), also kill Gazebo
echo "Training finished or interrupted. Cleaning up Gazebo..."
# 1. Küldünk egy leállító jelet (mint egy Ctrl+C) a launch folyamatnak
kill -INT $GAZEBO_PID 2>/dev/null

# 2. Várunk kicsit, hogy a ROS2 szépen lezárja a node-jait
sleep 2

# 3. Biztos ami biztos: garantáltan kinyírjuk a beragadt Gazebo folyamatokat (szerver és kliens)
pkill -9 -f gzserver 2>/dev/null
pkill -9 -f gzclient 2>/dev/null
kill -9 $GAZEBO_PID 2>/dev/null

echo "Cleanup complete."
