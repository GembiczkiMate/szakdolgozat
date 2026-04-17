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
from gazebo_utils import TrackGenerator
from coordinate_processor import CoordinateProcessor



class RosLineFollowEnv(gym.Env, Node):
    """Custom Gym environment for the ROS2 line-following robot."""
    metadata = {'render_modes': ['human']}

    # --- FINISH LINE CONFIGURATION ---
    # Robot spawns at (10.0, -4.2), facing -X direction (yaw = -3.14)
    # Set these coordinates based on where your track ends
    SPAWN_X = 10.0
    SPAWN_Y = -4.2
    
    FINISH_RADIUS = 0.5     # How close robot needs to be to "cross" finish line
    FINISH_REWARD = 200.0   # Eredeti bonus reward for reaching finish line

    
    
    # Number of interpolation points between each control point
    SPLINE_RESOLUTION = 20

    def __init__(self, is_testing_mode=False, reward_mode='vision'):
        # Initialize the Gym Environment and the ROS2 Node
        gym.Env.__init__(self)
        Node.__init__(self, 'ros_line_follow_env')
        
        self.is_testing_mode = is_testing_mode
        self.reward_mode = reward_mode
        self.get_logger().info(f"Environment initialized in {'TESTING' if is_testing_mode else 'TRAINING'} mode. Reward system: {self.reward_mode.upper()}")

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
        self.coordinate_processor = CoordinateProcessor(max_allowed_deviation_meters=0.1) # Távolság lecsökkentve 15 cm-re (a kamera látószögének megfelelő szélesség)
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
        
        # Frissítsük a coordinate processort is, ha valaha változik a pálya vagy az elején vagyunk
        if track_changed or not hasattr(self, 'initial_setup_done') or not self.initial_setup_done:
            spline_pts = TrackGenerator._catmull_rom_spline(self.TRACK_POINTS, self.SPLINE_RESOLUTION)
            self.coordinate_processor.update_track_spline(spline_pts)
        
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
            return empty_obs, 0.0, False, 0.0

        frame = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
        img_obs, vis_error, vis_term, area = self.vision_processor.process_image(frame)
        
        # Ide jön a varázslat: ha 'coordinate' mód van, felülírjuk a kamerás büntetéseket
        # De az img_obs persze ugyanúgy megy az agynak!
        if self.reward_mode == 'coordinate':
            coord_error, coordinate_term = self.coordinate_processor.calculate_error_and_termination(self.robot_x, self.robot_y)
            # Terminálás, ha VAGY fizikailag távolodik el túlságosan a spline-tól, VAGY a kamera is elvesztette a vonalat.
            combined_term = coordinate_term or vis_term
            return img_obs, coord_error, combined_term, area
        else:
            return img_obs, vis_error, vis_term, area

    def step(self, action):
        # 1. Send action to the robot
        # Scale down actions so the robot moves slower physically while satisfying
        # the neural network's original trained action space scale.
        # Max original speed 0.5 -> scaled to 0.3 (60%)
        # Max original turn 1.5 -> scaled to 1.3 (86.6%) - Megemelve, hogy fizikailag jobban tudjon kanyarodni!
        twist = Twist()
        base_linear = float(action[0])
        base_angular = float(action[1])
        
        linear_speed = base_linear * (0.3 / 0.4)
        # Dinamikus kormányszorzó a vonal területe alapján (Kanyar-asszisztens)
        # Ha a terület "nagy" (>4000), ez egyenes. Szorzó = 1.0
        # Ha a terület lecsökken (kanyar, <2000), a szorzó megnő (akár 1.6x-osra)
        area_multiplier = 1.0
        if hasattr(self, 'last_line_area') and self.last_line_area > 0:
            if self.last_line_area < 2500:
                area_multiplier = 1.5  # Erős boost kanyarban
            elif self.last_line_area < 3500:
                area_multiplier = 1.25 # Enyhe boost
                
        # Base multiplier and dynamic area multiplier
        angular_speed = base_angular * (1.3 / 1.5) * area_multiplier
        
        # Fizikai plafon, nehogy túlpörögjön a szervó:
        angular_speed = max(min(angular_speed, 1.8), -1.8)
        
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed
        self.publisher_.publish(twist)

        # 2. Wait for the next observation
        self.image_received = False
        start_time = time.time()
        while not self.image_received and (time.time() - start_time) < 1.0:
            rclpy.spin_once(self, timeout_sec=0.01)

        # 3. Get observation and calculate reward
        obs, error, terminated, self.last_line_area = self._get_obs()
        
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
            self.episode_reward_sum += reward
            self.finished = True
            self.get_logger().info(f"FINISH LINE CROSSED! Bonus: +{self.FINISH_REWARD} | Total Episode Reward: {self.episode_reward_sum:.2f}")
            
        elif terminated and not crossed_finish:
            if term_reward < 0:
                reward = term_reward
                self.episode_reward_sum += reward
                self.get_logger().info(f"Episode ended: Robot lost visual of line at step {self.current_step} | Total Episode Reward: {self.episode_reward_sum:.2f}")
        else:
            # Csak sima lépés történt
            self.episode_reward_sum += reward

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
        self.episode_reward_sum = 0.0 # Track total reward for the episode
        self.last_line_area = 5000.0  # Kezdeti becsült terület

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

        obs, _, _, self.last_line_area = self._get_obs()
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