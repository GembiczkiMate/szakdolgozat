#!/bin/bash

# ==============================================================================
# ROS 2 Humble & Gazebo & Python RL Dependencies Installer Script
# ==============================================================================

set -e # Kilép, ha bármilyen hiba történik

echo "=================================================="
echo " ROS 2 és függőségek telepítésének megkezdése..."
echo "=================================================="

# 1. Rendszer frissítése és alap csomagok
echo "[1/6] Rendszer frissítése és alap eszközök telepítése..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y software-properties-common curl gnupg2 lsb-release python3-pip python3-venv build-essential

# 2. ROS 2 Repo hozzáadása
echo "[2/6] ROS 2 tároló (repository) hozzáadása..."
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 3. ROS 2 Humble és Gazebo telepítése
echo "[3/6] ROS 2 Humble, Gazebo és fordító eszközök (Colcon) telepítése..."
sudo apt update
# Desktop telepítés (tartalmazza az RViz-t és az alap ROS csomagokat)
sudo apt install -y ros-humble-desktop
# Gazebo és a ROS 2 <-> Gazebo összekötő csomagok
sudo apt install -y ros-humble-gazebo-ros-pkgs ros-humble-xacro ros-humble-robot-state-publisher ros-humble-cv-bridge ros-humble-image-transport
# Colcon build tool és rosdep
sudo apt install -y python3-colcon-common-extensions python3-rosdep

# 4. Rosdep inicializálása (ha még nem volt)
echo "[4/6] Rosdep inicializálása..."
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
fi
rosdep update

# 5. Python függőségek (Reinforcement Learning)
echo "[5/6] Python ML/RL keretrendszerek telepítése (PyTorch, Stable-Baselines3, Gym)..."
# ROS 2 környezetben a pip csomagokat érdemes felhasználói (-user) szintre tenni, hogy ne akadjanak össze az apt-tal
pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Kifejezetten a numpy < 2.0.0 kötelező a ROS 2 és a Stable Baselines 3 kompatibilitási hibáinak elkerüléséhez!
pip3 install --user stable-baselines3 gymnasium opencv-python tensorboard "numpy<2.0.0" matplotlib

# 6. Környezeti változók Bashrc-be írása
echo "[6/6] .bashrc konfigurálása..."
if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
  echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
  echo "ROS 2 Humble környezet hozzáadva a ~/.bashrc fájlhoz."
fi

# Fontos: A Colcon fordítás (ament_cmake) számára be kell tölteni a ROS környezetet a fordítás ELŐTT is!
source /opt/ros/humble/setup.bash

# A Cyclone DDS hálózati javítás telepítése, ami elengedhetetlen a stabil RL tanításhoz!
echo "A Cyclone DDS csomagok telepítése..."
sudo apt install -y ros-humble-rmw-cyclonedds-cpp

# Megkeressük az aktuális ROS 2 workspace-t, ha benne állunk
WORKSPACE_ROOT=$(cd ../.. && pwd)
WORKSPACE_SETUP="$WORKSPACE_ROOT/install/setup.bash"

echo "=================================================="
echo " A csomag felépítése (colcon build) ..."
echo "=================================================="
(cd "$WORKSPACE_ROOT" && colcon build)

if [ -f "$WORKSPACE_SETUP" ]; then
    # Hozzáadjuk a bashrc-hez az aktuális felhasználónak
    if ! grep -q "source $WORKSPACE_SETUP" ~/.bashrc; then
      echo "source $WORKSPACE_SETUP" >> ~/.bashrc
      echo "Workspace környezet ($WORKSPACE_SETUP) hozzáadva a ~/.bashrc fájlhoz."
    fi
    # És rögtön aktiváljuk is erre a sessionre
    source "$WORKSPACE_SETUP"
fi

echo "=================================================="
echo " TELEPÍTÉS SIKERESEN BEFEJEZŐDÖTT!"
echo " Kérlek zárd be ezt a terminált és nyiss egy újat,"
echo " vagy futtasd a 'source ~/.bashrc' parancsot!"
echo " Ezt követően a 'colcon build' és a launch fájlok futtathatók lesznek."
echo "=================================================="
