import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import math
import random
import os

from track_config import TRAINING_TRACKS, TESTING_TRACKS, FORBIDDEN_ZONES
from vision_processor import VisionProcessor
from reward_calculator import RewardCalculator
from gazebo_entity_manager import GazeboEntityManager



class RosLineFollowEnv(gym.Env, Node):
    """Custom Gym environment for the ROS2 line-following robot."""
    metadata = {'render_modes': ['human']}

    # --- FINISH LINE CONFIGURATION ---
    # Robot spawns at (10.0, -4.2), facing -X direction (yaw = -3.14)
    # Set these coordinates based on where your track ends
    SPAWN_X = 10.0
    SPAWN_Y = -4.2
    
    FINISH_RADIUS = 0.5     # How close robot needs to be to "cross" finish line
    FINISH_REWARD = 200.0   # Bonus reward for reaching finish line

    
    
    # Number of interpolation points between each control point
    SPLINE_RESOLUTION = 20

    def __init__(self, is_testing_mode=False):
        # Initialize the Gym Environment and the ROS2 Node
        gym.Env.__init__(self)
        Node.__init__(self, 'ros_line_follow_env')
        
        self.is_testing_mode = is_testing_mode
        self.get_logger().info(f"Environment initialized in {'TESTING' if is_testing_mode else 'TRAINING'} mode.")

        # Bind external configs
        self.PREDEFINED_TRACKS = TESTING_TRACKS if self.is_testing_mode else TRAINING_TRACKS
        self.TRACK_POINTS = self.PREDEFINED_TRACKS[0]
        self.FORBIDDEN_ZONES = FORBIDDEN_ZONES
        
        # Dynamically set finish line coordinates based on the active track
        if len(self.TRACK_POINTS) > 0:
            self.FINISH_X = self.TRACK_POINTS[-1][0]
            self.FINISH_Y = self.TRACK_POINTS[-1][1]
            self.SPAWN_X = self.TRACK_POINTS[0][0]
            self.SPAWN_Y = self.TRACK_POINTS[0][1]

        # --- Action and Observation Spaces ---
        # NOTE: Keeping the original max bounds for action_space so the pre-trained model 
        # doesn't crash on load (ValueError: Action spaces do not match).
        # Actual robotic limits are scaled in the step() function.
        self.max_speed = 0.5    
        self.max_turn = 1.5     
        self.action_space = spaces.Box(low=np.array([0.0, -self.max_turn]), high=np.array([self.max_speed, self.max_turn]), dtype=np.float32)
        
        # RAW Image observation space: Full camera image (RGB)
        # Using camera resolution - no resizing, no grayscale conversion
        # Channel-first format (C, H, W) as required by CnnPolicy
        # Values in [0, 255] as expected by NatureCNN
        self.img_height = 240  # Camera height
        self.img_width = 320   # Camera width
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(3, self.img_height, self.img_width),  # Channel-first: (C, H, W)
            dtype=np.uint8
        )

        # --- Extracted Modules ---
        self.vision_processor = VisionProcessor(self.img_height, self.img_width)
        self.reward_calculator = RewardCalculator(self.max_speed, self.max_turn, self.FINISH_REWARD)

        # --- ROS2 Connections ---
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(Image, '/line_camera/image_raw', self.image_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Path to URDF file
        self.urdf_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'urdf', 'mobile_robot.urdf'
        )
        
        # Robot and model names
        self.robot_name = 'two_wheeled_robot'
        self.line_model_name = 'track_line'
        
        # Current randomization values
        self.current_ground_color = 'Custom/Padlo'  # Default Custom Padlo color
        self.current_line_width = 0.1
        self.current_camera_pitch = 0.3

        self.gazebo_manager = GazeboEntityManager(self, self.robot_name, self.line_model_name, self.urdf_path)

        self.bridge = CvBridge()
        self.latest_image = None
        self.image_received = False
        
        # --- Robot Position Tracking ---
        self.robot_x = self.SPAWN_X
        self.robot_y = self.SPAWN_Y
        self.finished = False
        self.steps_on_line = 0  # Track consecutive steps on line
        self.current_step = 0   # Current step in episode
        self.max_steps = 1500   # Maximum steps per episode (truncation)
        
        # --- Track if initial setup done ---
        self.initial_setup_done = False

        # --- Reward Weights ---
        self.stability_weight = 0.7      # Higher weight for staying on line
        self.speed_weight = 0.3          # Lower weight for speed (learn stability first)
        self.survival_bonus = 0.1        # Small bonus for each step survived
        
        # --- Smoothness Tracking ---
        self.prev_angular_speed = 0.0    # For penalizing "vibration"

        self.get_logger().info("ROS Line Follow Environment Initialized.")
        

    def image_callback(self, msg):
        self.latest_image = msg
        self.image_received = True

    def odom_callback(self, msg):
        """Track robot position from odometry."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def check_finish_line(self):
        """Check if robot has crossed the finish line."""
        distance_to_finish = math.sqrt(
            (self.robot_x - self.FINISH_X) ** 2 + 
            (self.robot_y - self.FINISH_Y) ** 2
        )
        return distance_to_finish < self.FINISH_RADIUS

    def _setup_and_reset_environment(self):
        """Set up the environment initially, and reset the robot position on subsequent calls."""
        # Randomize track
        new_track = random.choice(self.PREDEFINED_TRACKS)
        track_changed = False
        if new_track != self.TRACK_POINTS:
            track_changed = True
            
        self.TRACK_POINTS = new_track
        
        # Update finish line configuration dynamically based on track
        if len(self.TRACK_POINTS) > 0:
            self.FINISH_X = self.TRACK_POINTS[-1][0]
            self.FINISH_Y = self.TRACK_POINTS[-1][1]
            self.SPAWN_X = self.TRACK_POINTS[0][0]
            self.SPAWN_Y = self.TRACK_POINTS[0][1]

        # First time setup
        if not self.initial_setup_done:
            self.get_logger().info("Initial setup: Spawning track line and robot...")
            self.gazebo_manager.respawn_robot(self.SPAWN_X, self.SPAWN_Y, self.current_camera_pitch)
            self.gazebo_manager.respawn_line(self.TRACK_POINTS, self.SPLINE_RESOLUTION, self.current_line_width)
            self.initial_setup_done = True
            time.sleep(0.5)  # Wait for line to appear
            return
            
        # If we selected a new track, respawn the line
        if track_changed:
            self.get_logger().info(f"Selected predefined track finish line at ({self.FINISH_X}, {self.FINISH_Y})")
            self.gazebo_manager.respawn_line(self.TRACK_POINTS, self.SPLINE_RESOLUTION, self.current_line_width)
        
        # On subsequent resets, just reset the robot position to the start of the current track
        success = self.gazebo_manager.reset_robot_position(self.SPAWN_X, self.SPAWN_Y)
        if not success:
            self.gazebo_manager.respawn_robot(self.SPAWN_X, self.SPAWN_Y, self.current_camera_pitch)

    def _get_obs(self):
        if self.latest_image is None:
            # Return empty observation if no image
            empty_obs = np.zeros((3, self.img_height, self.img_width), dtype=np.uint8)
            return empty_obs, 0.0, False

        frame = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
        return self.vision_processor.process_image(frame)

    def step(self, action):
        # 1. Send action to the robot
        # Scale down actions so the robot moves slower physically while satisfying
        # the neural network's original trained action space scale.
        # Max original speed 0.5 -> scaled to 0.3 (60%)
        # Max original turn 1.5 -> scaled to 1.0 (66.6%)
        twist = Twist()
        base_linear = float(action[0])
        base_angular = float(action[1])
        
        linear_speed = base_linear * (0.3 / 0.4)
        angular_speed = base_angular * (1.0 / 1.6)
        
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed
        self.publisher_.publish(twist)

        # 2. Wait for the next observation
        self.image_received = False
        start_time = time.time()
        while not self.image_received and (time.time() - start_time) < 1.0:
            rclpy.spin_once(self, timeout_sec=0.01)

        # 3. Get observation and calculate reward
        obs, error, terminated = self._get_obs()
        
        # --- Check for finish line ---
        crossed_finish = self.check_finish_line()
        
        # Use RewardCalculator module to handle calculating core rewards and penalties
        reward = self.reward_calculator.calculate_reward(
            error, linear_speed, angular_speed, self.prev_angular_speed
        )
                  
        # Update previous steering for the next step
        self.prev_angular_speed = angular_speed

        # --- Finish line & Termination modifications ---
        term_reward, terminated = self.reward_calculator.calculate_termination_reward(
            terminated, crossed_finish, self.current_step
        )
        
        if crossed_finish and not self.finished:
            reward += term_reward
            self.finished = True
            self.get_logger().info(f"FINISH LINE CROSSED! Bonus: +{self.FINISH_REWARD}")
            
        elif terminated and not crossed_finish:
            if term_reward < 0:
                reward = term_reward
                self.get_logger().info(f"Episode ended: Robot lost visual of line at step {self.current_step}")

        # Increment step counter
        self.current_step += 1
        # Truncation due to max steps is disabled.
        # The episode will only end if the robot falls off the line or crosses the finish line.
        truncated = False
        
        info = {}

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Stop the robot
        self.publisher_.publish(Twist())
        
        # Reset finish line state
        self.finished = False
        self.robot_x = self.SPAWN_X
        self.robot_y = self.SPAWN_Y
        self.steps_on_line = 0  # Reset step counter
        self.current_step = 0   # Reset episode step counter
        self.prev_angular_speed = 0.0 # Reset smoothness tracker

        # --- Reset Robot Environment ---
        self._setup_and_reset_environment()
        
        # Wait for models to spawn and Gazebo to render the new environment
        end_wait_time = time.time() + 1.0
        while time.time() < end_wait_time:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # Wait for new image after randomization
        self.image_received = False
        start_time = time.time()
        while not self.image_received and (time.time() - start_time) < 3.0:
            rclpy.spin_once(self, timeout_sec=0.01)

        obs, _, _ = self._get_obs()
        info = {
            'ground_color': self.current_ground_color,
            'line_width': self.current_line_width,
            'camera_pitch': math.degrees(self.current_camera_pitch)
        }
        
        return obs, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.destroy_node()
        rclpy.shutdown()