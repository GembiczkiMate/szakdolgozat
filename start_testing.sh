#!/bin/bash

# A Cyclone DDS hálózati réteg engedélyezése az instabilitások és szerviz fagyások elkerülése végett
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=1
export MALLOC_CHECK_=0
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

echo "========================================================="
echo " TESTING (VÉGTELEN FUTÁS) - VIZUÁLIS TESZT NYITOTT GAZEBÓVAL"
echo "========================================================="
echo ""
echo "Melyik modellt (jutalmazási rendszert) szeretnéd betölteni és kiértékelni?"
echo "1) Kamerás kép alapján (Vision - eredeti verzió)"
echo "2) Pálya koordinátái alapján (Coordinate - spline mérés)"
while true; do
    read -p "Választásod (1/2): " REWARD_CHOICE
    case $REWARD_CHOICE in
        1) REWARD_MODE="vision"; break;;
        2) REWARD_MODE="coordinate"; break;;
        *) echo "Kérlek, válassz érvényes opciót (1 vagy 2).";;
    esac
done

echo ""
echo "=> Kiválasztott mód: $REWARD_MODE (Meg fog nyílni a Gazebo)"

# Source the ROS 2 setup configuration
source install/setup.bash

echo "Starting Gazebo Simulation (Látható / GUI ablakban)..."
# Start Gazebo in the background WITH HEADLESS FALSE explicitly
export GAZEBO_HEADLESS="False"
ros2 launch two_wheeled_robot load_world_into_gazebo.launch.py headless:=False &
GAZEBO_PID=$!

echo "Waiting for 3 seconds to let Gazebo initialize..."
sleep 3

echo "Starting Model Testing (Infinite loop) for $REWARD_MODE..."
# Futtatjuk a test.py fájlt, átadva a kiválasztott jutalmazási módot
python3 $(ros2 pkg prefix two_wheeled_robot)/share/two_wheeled_robot/rl/test.py --reward-mode $REWARD_MODE

# When it finishes (Ctrl+C)
echo "Testing finished or interrupted. Cleaning up Gazebo..."
kill -INT $GAZEBO_PID 2>/dev/null
sleep 2
pkill -9 -f gzserver 2>/dev/null
pkill -9 -f gzclient 2>/dev/null
kill -9 $GAZEBO_PID 2>/dev/null

echo "Cleanup complete."
