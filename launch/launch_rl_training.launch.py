# filepath: /home/gembi/rl_ros/src/two_wheeled_robot/launch/launch_rl_training.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_two_wheeled_robot = get_package_share_directory('two_wheeled_robot')
    
    # Környezeti változóból kiolvassuk a WATCHDOG által beállított módot
    # Ha valamiért közvetlenül futtatnánk, 'vision' az alapértelmezett.
    reward_mode = os.environ.get('TRAIN_REWARD_MODE', 'vision')

    # Note: We use ExecuteProcess to run the python script directly.
    # This is often simpler for RL training than using a ComposableNode.
    training_script = ExecuteProcess(
        cmd=['python3', os.path.join(pkg_two_wheeled_robot, 'rl', 'train.py'), '--reward-mode', reward_mode],
        output='screen'
    )

    return LaunchDescription([
        
        training_script
    ])