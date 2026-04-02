import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import time
import math
import random
import os
import re
from gazebo_msgs.srv import SpawnEntity, DeleteEntity


class RosLineFollowEnv(gym.Env, Node):
    """Custom Gym environment for the ROS2 line-following robot."""
    metadata = {'render_modes': ['human']}

    # --- FINISH LINE CONFIGURATION ---
    # Robot spawns at (10.0, -4.2), facing -X direction (yaw = -3.14)
    # Set these coordinates based on where your track ends
    SPAWN_X = 10.0
    SPAWN_Y = -4.2
    FINISH_X = 0.5          # X coordinate of finish line (adjust to your track!)
    FINISH_Y = -8.0        # Y coordinate of finish line (adjust to your track!)
    FINISH_RADIUS = 0.5     # How close robot needs to be to "cross" finish line
    FINISH_REWARD = 200.0   # Bonus reward for reaching finish line

    # --- DOMAIN RANDOMIZATION CONFIGURATION ---
    # TEMPORARILY DISABLED for debugging reset issues
    RANDOMIZE_GROUND_COLOR = False
    RANDOMIZE_LINE_WIDTH = False
    RANDOMIZE_CAMERA_PITCH = False
    RANDOMIZE_TRACK = True  # Enable randomizing between predefined manual tracks
    
    # Available Gazebo ground colors (from Gazebo's built-in materials)
    GROUND_COLORS = ['Gray', 'Black', 'White', 'Blue', 'Green']
    
    # Line width range (meters)
    LINE_WIDTH_MIN = 0.03
    LINE_WIDTH_MAX = 0.05
    
    # Camera pitch range (radians, positive = looking down)
    CAMERA_PITCH_MIN = 0.2   # Looking slightly down
    CAMERA_PITCH_MAX = 0.5   # Looking more down
    
    # Track definition - control points for Catmull-Rom spline
    # We will randomly pick one of these predefined tracks on reset
    PREDEFINED_TRACKS = [
        # Original track requested in the beginning
        [
            (10.0, -4.2),   # Start
            (3.2, -4.2),    # First turn
            (3.2, 4.0),     # Second turn
            (-3.5, 4.0),    # Third turn
            (-3.5, 0.3),    # Fourth turn
            (-5.0, 0.3),    # Fifth turn
            (-5.0, -4.2),   # Sixth turn
            (0.5, -4.2),    # Seventh turn
            (0.5, -8.0)     # Finish
        ],
        # Track 2: Center-left winding path
        [(10.0, -4.2),
         (7.0, -3.5),
         (5.5, -1.0),
         (3.0, -1.0),
         (3.0, 4.0),
         (-2.0, 4.0),
         (-2.0, -1.0),
         (-5.0, -1.0)],
        
        # Track 3: Top roundabout path
        [(10.0, -4.2),
         (8.5, -4.0),
         (9.2, -1.0),
         (9.2, 4.0),
         (9.2, 8.5),
         (5.0, 8.5),
         (1.0, 8.5),
         (-2.0, 8.5),
         (-5.0, 8.5),
         (-8.0, 8.5),
         (-8.0, 4.0),
         (-5.0, 4.0)
         ],
        
        # Track 4: Bottom zig-zag crossing
        [(10.0, -4.2),
         (6.0, -4.0),
         (3.5, -4.0),
         (2.0, -4.0),
         (1.0, -4.0),
         (-1.5, -4.0),
         (-3.0, -4.0),
         (-5.0, -4.0),
         (-7.0, -4.0),
         (-7.0, -1.0),
         (-7.0, 0.0)],
            # Track 5 (Snake through middle) - VALIDATED ALREADY
        [(10.0, -4.2),
         (8.0, -4.0),
         (8.0, 0.0),
         (3.3, 0.0),
         (3.3, 3.5),
         (-1.5, 3.5),
         (-1.5, 0.0),
         (-5.0, 0.0),
         (-5.0, -3.5),
         (-7.0, -3.5)],
    
        # Track 6 - Fix the loop out left
        [(10.0, -4.2),
         (6.0, -4.0),
         (3.0, -4.0),
         (1.0, -4.0),
         (-1.5, -4.0),
         (-5.0, -4.0), 
         (-7.5, -4.0),
         (-7.5, 0.0),
         (-3.0, 0.0),
         (-3.0, 4.0)],
    
        # Track 7 - Fix top left curve crossing
        [(10.0, -4.2),
         (8.5, -4.0),
         (9.2, -1.0),
         (9.2, 4.0),
         (9.2, 8.5),
         (2.0, 8.5),
         (2.0, 4.0),
         (-3.0, 4.0),
         (-5.0, 4.0),
         (-8.0, 4.0),
         (-8.0, 9.0),
         (-5.0, 9.0)
         ]
        ]

    
    
    # Set the initial track points
    TRACK_POINTS = PREDEFINED_TRACKS[0]
    
    # --- FORBIDDEN ZONES FOR TRACK GENERATION ---
    # Define rectangular areas where the track line is NOT allowed to be drawn.
    # For example, obstacles or walls.
    # Format: [ ((min_x, max_x), (min_y, max_y)) ]
    FORBIDDEN_ZONES = [
        # Based on your points: (10,5), (2,5), (2,10), (10,10)
        # Min X is 2.0, Max X is 10.0
        # Min Y is 5.0, Max Y is 10.0
        ((0.2, -7.3), (5.0, 7.5)),
        
        # New points based on: (10,-5), (2,-5), (2,-10), (10,-10)
        # Min X is 2.0, Max X is 10.0
        # Min Y is -10.0, Max Y is -5.0
        ((2.0, 10.0), (-10.0, -5.0)),

        ((4.5,5.4),(-1.8,-2.8)),

        ((0.8,-0.1),(-1.8,-2.8)),

        ((-3.5,-4.4),(-1.8,-2.8)),

        ((2.8,1.4),(0.8,1.7)),

        ((4.5,8.0),(0.5,3.0)),

        ((-8.6,-10.0),(10.0,-10.0)),

        ((-4.0,-10.0),(0.8,3.2)),

        ((-0.6, -10.0), (-10.0, -6.0)),

        ((-1.8, -2.5), (-5.0, -6.0)),

        ((2.5, 8.6), (5.5, 6.5)),
    ]
    
    # Number of interpolation points between each control point
    SPLINE_RESOLUTION = 20

    def __init__(self):
        # Initialize the Gym Environment and the ROS2 Node
        gym.Env.__init__(self)
        Node.__init__(self, 'ros_line_follow_env')
        
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

        # --- ROS2 Connections ---
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(Image, '/line_camera/image_raw', self.image_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # --- Domain Randomization Service Clients ---
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')
        
        # Path to URDF file
        self.urdf_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'urdf', 'mobile_robot.urdf'
        )
        
        # Robot and model names
        self.robot_name = 'two_wheeled_robot'
        self.ground_overlay_name = 'ground_overlay'
        self.line_model_name = 'track_line'
        
        # Current randomization values
        self.current_ground_color = 'Custom/Padlo'  # Default Custom Padlo color
        self.current_line_width = 0.1
        self.current_camera_pitch = 0.3

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
        self.get_logger().info(f"Finish line at ({self.FINISH_X}, {self.FINISH_Y}), radius={self.FINISH_RADIUS}m")

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

    def _randomize_environment(self):
        """Apply domain randomization by respawning models with new parameters."""
        
        # First time setup - spawn the line (required even without randomization)
        if not self.initial_setup_done:
            self.get_logger().info("Initial setup: Spawning track line...")
            self._respawn_line()
            self.initial_setup_done = True
            time.sleep(0.5)  # Wait for line to appear
        
        # Check if any randomization is enabled
        any_randomization = (self.RANDOMIZE_LINE_WIDTH or 
                             self.RANDOMIZE_CAMERA_PITCH or
                             getattr(self, 'RANDOMIZE_TRACK', False))
        
        if not any_randomization:
            # No randomization - just reset robot position without respawning
            self._reset_robot_position()
            self.get_logger().info("Reset: Robot position reset (no domain randomization)")
            return
        
        # Generate random values
        if self.RANDOMIZE_LINE_WIDTH:
            self.current_line_width = random.uniform(self.LINE_WIDTH_MIN, self.LINE_WIDTH_MAX)
        
        if self.RANDOMIZE_CAMERA_PITCH:
            self.current_camera_pitch = random.uniform(self.CAMERA_PITCH_MIN, self.CAMERA_PITCH_MAX)
        
        # Delete and respawn robot with new camera pitch
        self._respawn_robot()
        
        # Always spawn ground overlay with padlo.jpg
        self._spawn_ground_overlay()
            
        # Select a random predefined track if track randomization is enabled
        if getattr(self, 'RANDOMIZE_TRACK', False):
            self.TRACK_POINTS = random.choice(self.PREDEFINED_TRACKS)
            # Update finish criteria based on the new track
            if len(self.TRACK_POINTS) > 0:
                self.FINISH_X = self.TRACK_POINTS[-1][0]
                self.FINISH_Y = self.TRACK_POINTS[-1][1]
                self.SPAWN_X = self.TRACK_POINTS[0][0]
                self.SPAWN_Y = self.TRACK_POINTS[0][1]
            self.get_logger().info(f"Selected predefined track starting at ({self.SPAWN_X}, {self.SPAWN_Y}) and ending at ({self.FINISH_X}, {self.FINISH_Y})")
        
        # Respawn line with new width or new track
        if self.RANDOMIZE_LINE_WIDTH or getattr(self, 'RANDOMIZE_TRACK', False):
            self._respawn_line()
        
        self.get_logger().info(
            f"Domain Randomization: ground={self.current_ground_color}, "
            f"line_width={self.current_line_width:.3f}m, "
            f"camera_pitch={math.degrees(self.current_camera_pitch):.1f}°"
        )
    
    def _reset_robot_position(self):
        """Reset robot to starting position without respawning (faster)."""
        from gazebo_msgs.srv import SetEntityState
        from gazebo_msgs.msg import EntityState
        
        # Create service client if not exists
        if not hasattr(self, 'set_state_client'):
            self.set_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        
        if not self.set_state_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('set_entity_state service not available, using respawn')
            self._respawn_robot()
            return
        
        # Create request
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = self.robot_name
        req.state.pose.position.x = self.SPAWN_X
        req.state.pose.position.y = self.SPAWN_Y
        req.state.pose.position.z = 0.05
        
        # Set orientation (yaw = -3.14)
        yaw = -3.14
        req.state.pose.orientation.w = math.cos(yaw / 2)
        req.state.pose.orientation.z = math.sin(yaw / 2)
        
        # Reset velocities
        req.state.twist.linear.x = 0.0
        req.state.twist.linear.y = 0.0
        req.state.twist.linear.z = 0.0
        req.state.twist.angular.x = 0.0
        req.state.twist.angular.y = 0.0
        req.state.twist.angular.z = 0.0
        
        future = self.set_state_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        if future.result() is not None and future.result().success:
            self.get_logger().debug("Robot position reset successfully")
        else:
            self.get_logger().warn("Failed to reset robot position, using respawn")
            self._respawn_robot()

    def _delete_entity(self, name):
        """Delete an entity from Gazebo."""
        if not self.delete_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('delete_entity service not available')
            return False
        
        req = DeleteEntity.Request()
        req.name = name
        
        future = self.delete_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        
        if future.result() is not None:
            return future.result().success
        return False

    def _spawn_entity(self, name, xml, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        """Spawn an entity in Gazebo."""
        if not self.spawn_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('spawn_entity service not available')
            return False
        
        req = SpawnEntity.Request()
        req.name = name
        req.xml = xml
        req.initial_pose.position.x = float(x)
        req.initial_pose.position.y = float(y)
        req.initial_pose.position.z = float(z)
        
        # Convert euler to quaternion
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        req.initial_pose.orientation.w = cr * cp * cy + sr * sp * sy
        req.initial_pose.orientation.x = sr * cp * cy - cr * sp * sy
        req.initial_pose.orientation.y = cr * sp * cy + sr * cp * sy
        req.initial_pose.orientation.z = cr * cp * sy - sr * sp * cy
        
        future = self.spawn_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        
        if future.result() is not None:
            return future.result().success
        return False

    def _respawn_robot(self):
        """Delete and respawn the robot with modified camera pitch."""
        max_retries = 3
        
        for attempt in range(max_retries):
            # Delete existing robot
            delete_success = self._delete_entity(self.robot_name)
            time.sleep(0.5)
            
            # Read and modify URDF with new camera pitch
            try:
                with open(self.urdf_path, 'r') as f:
                    urdf_content = f.read()
                
                # Modify camera joint pitch in URDF
                # Format: <joint name="camera_joint" type="fixed">...<origin xyz="0.16 0 0.08" rpy="0 0.0 0"/></joint>
                # The joint is on a single line, we need to change the rpy value
                pattern = r'(<joint name="camera_joint"[^>]*>.*?rpy=")[^"]*(")'
                replacement = rf'\g<1>0 {self.current_camera_pitch} 0\2'
                modified_urdf = re.sub(pattern, replacement, urdf_content)
                
                # Spawn robot at start position
                success = self._spawn_entity(
                    self.robot_name,
                    modified_urdf,
                    self.SPAWN_X, self.SPAWN_Y, 0.05,
                    yaw=-3.14  # Facing -X direction
                )
                
                if success:
                    self.get_logger().info(f"Robot respawned (attempt {attempt+1}) with camera pitch: {math.degrees(self.current_camera_pitch):.1f}°")
                    return True
                else:
                    self.get_logger().warn(f"Failed to respawn robot (attempt {attempt+1}/{max_retries})")
                    time.sleep(1.0)  # Wait before retry
                    
            except Exception as e:
                self.get_logger().error(f"Error respawning robot: {e}")
                time.sleep(1.0)
        
        self.get_logger().error("Failed to respawn robot after all retries!")
        return False

    def _spawn_ground_overlay(self):
        """Spawn a colored ground overlay plane."""
        # Delete existing overlay if any
        self._delete_entity(self.ground_overlay_name)
        time.sleep(0.3)
        
        # Create SDF for colored ground overlay
        # This is a thin plane just above the ground with the selected color
        sdf = f'''<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{self.ground_overlay_name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>100 100</size>
          </plane>
        </geometry>
        <material>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/Grey</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''
        
        # Spawn slightly above ground to overlay it
        success = self._spawn_entity(
            self.ground_overlay_name,
            sdf,
            0.0, 0.0, 0.001  # 1mm above ground
        )
        
        if success:
            self.get_logger().debug(f"Ground overlay spawned with color: {self.current_ground_color}")
        else:
            self.get_logger().warn("Failed to spawn ground overlay")

    def _respawn_line(self):
        """Delete and respawn the track line with new width using Catmull-Rom spline."""
        # Delete existing line
        self._delete_entity(self.line_model_name)
        time.sleep(0.3)
        
        # Generate spline points
        spline_points = self._catmull_rom_spline(self.TRACK_POINTS, self.SPLINE_RESOLUTION)
        
        # Generate SDF for the line segments
        segments_sdf = ""
        segment_id = 0
        
        for i in range(len(spline_points) - 1):
            x1, y1 = spline_points[i]
            x2, y2 = spline_points[i + 1]
            
            # Calculate segment properties
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < 0.001:  # Skip zero-length segments
                continue
                
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            angle = math.atan2(y2 - y1, x2 - x1)
            
            segments_sdf += f'''
      <link name="segment_{segment_id}">
        <pose>{center_x} {center_y} 0.015 0 0 {angle}</pose>
        <visual name="visual">
          <geometry>
            <box>
              <size>{length + 0.005} {self.current_line_width} 0.002</size>
            </box>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>Gazebo/Red</name>
            </script>
          </material>
        </visual>
      </link>'''
            segment_id += 1
        
        # Create complete SDF model
        sdf = f'''<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{self.line_model_name}">
    <static>true</static>
    {segments_sdf}
  </model>
</sdf>'''
        
        # Spawn the line model
        success = self._spawn_entity(
            self.line_model_name,
            sdf,
            0.0, 0.0, 0.0
        )
        
        if success:
            self.get_logger().debug(f"Track line spawned with width: {self.current_line_width:.3f}m ({segment_id} segments)")
        else:
            self.get_logger().warn("Failed to spawn track line")

    def _catmull_rom_spline(self, control_points, resolution):
        """
        Generate a Catmull-Rom spline through the given control points.
        
        Args:
            control_points: List of (x, y) tuples - the points the spline passes through
            resolution: Number of interpolation points between each pair of control points
            
        Returns:
            List of (x, y) tuples representing the spline curve
        """
        if len(control_points) < 2:
            return control_points
        
        # Extend control points for end handling (duplicate first and last)
        extended_points = [control_points[0]] + list(control_points) + [control_points[-1]]
        
        spline_points = []
        
        # Iterate through each segment (P1 to P2)
        for i in range(1, len(extended_points) - 2):
            p0 = extended_points[i - 1]
            p1 = extended_points[i]
            p2 = extended_points[i + 1]
            p3 = extended_points[i + 2]
            
            # Generate points along this segment
            for t_step in range(resolution):
                t = t_step / resolution
                
                # Catmull-Rom spline formula
                # P(t) = 0.5 * [(2*P1) + (-P0 + P2)*t + (2*P0 - 5*P1 + 4*P2 - P3)*t^2 + (-P0 + 3*P1 - 3*P2 + P3)*t^3]
                t2 = t * t
                t3 = t2 * t
                
                x = 0.5 * (
                    (2 * p1[0]) +
                    (-p0[0] + p2[0]) * t +
                    (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                    (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
                )
                
                y = 0.5 * (
                    (2 * p1[1]) +
                    (-p0[1] + p2[1]) * t +
                    (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                    (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
                )
                
                spline_points.append((x, y))
        
        # Add the last control point
        spline_points.append(control_points[-1])
        
        return spline_points

    def _get_obs(self):
        if self.latest_image is None:
            # Return empty observation if no image
            empty_obs = np.zeros((3, self.img_height, self.img_width), dtype=np.uint8)
            return empty_obs, 0.0, False

        frame = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
        height, width, _ = frame.shape
        
        # --- 1. RAW IMAGE for neural network (NO PROCESSING!) ---
        # Convert BGR to RGB, then transpose to channel-first (C, H, W)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_obs = np.transpose(rgb_frame, (2, 0, 1)).astype(np.uint8)  # (H,W,C) -> (C,H,W)
        
        # --- 2. Image processing for REWARD and TERMINATION only ---
        # (The neural network doesn't see this, it's just for reward calculation)
        crop_y = int(height / 2)
        crop_img = frame[crop_y:height, 0:width]
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        
        lower_red = np.array([0, 60, 60])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            
            if area > 80:
                M = cv2.moments(c)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
            
                    # --- MODIFIED: Use pixel error for termination ---
            
                    # 1. Calculate the error in pixels from the center
                    pixel_error = width / 2 - cx
            
                    # 2. Check if it deviates more than 30% from the center in any direction
                    # (10% was too strict for initial learning, causing immediate failure before it could move)
                    max_allowed_deviation = width * 0.30
                    terminated = abs(pixel_error) > max_allowed_deviation
            
                    # 3. Calculate the normalized error for the reward function
                    normalized_error = pixel_error / (width / 2)
            
                    norm_area = min(1.0, cv2.contourArea(c) / (width * height / 2))
            
                    # Return: raw image obs, error for reward, termination flag
                    return img_obs, normalized_error, terminated
        
        # No line detected - return image obs with error=1.0 (worst case)
        return img_obs, 1.0, True

    def step(self, action):
        # 1. Send action to the robot
        # Scale down actions so the robot moves slower physically while satisfying
        # the neural network's original trained action space scale.
        # Max original speed 0.5 -> scaled to 0.3 (60%)
        # Max original turn 1.5 -> scaled to 1.0 (66.6%)
        twist = Twist()
        base_linear = float(action[0])
        base_angular = float(action[1])
        
        linear_speed = base_linear * (0.3 / 0.5)
        angular_speed = base_angular * (1.0 / 1.5)
        
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
        
        # --- NEW Reward Function ---
        # Stability factor (0 to 1, perfectly centered = 1, at edge of 10% tolerance drops to ~0.36)
        stability_factor = math.exp(-5.0 * abs(error))

        # Speed factor (0 to 1, moving at max_speed = 1)
        speed_factor = max(0.0, linear_speed) / self.max_speed

        # Core Progress Reward: The robot MUST move forward to get ANY points.
        # It gets the most points by going fast AND staying centered.
        progress_reward = speed_factor * stability_factor

        # Penalty for standing still (moves too slow)
        stationary_penalty = 0.5 if linear_speed < 0.1 else 0.0

        # --- Smoothness and Steering Penalties ---
        # Penalize large changes in steering ("vibration")
        steering_change = abs(angular_speed - self.prev_angular_speed)
        # Scaled penalty so max change gets 0.5 penalty points
        smoothness_penalty = (steering_change / (2.0 * self.max_turn)) * 0.5 
        
        # Penalize constant high steering (zig-zagging/weaving)
        steering_penalty = (abs(angular_speed) / self.max_turn) * 0.2

        # Combined weighted reward
        reward = (progress_reward - 
                  stationary_penalty - 
                  smoothness_penalty - 
                  steering_penalty) * 10.0  # Scale up for stronger signal
                  
        # Update previous steering for the next step
        self.prev_angular_speed = angular_speed

        # --- Finish line bonus ---
        if crossed_finish and not self.finished:
            reward += self.FINISH_REWARD
            self.finished = True
            self.get_logger().info(f"FINISH LINE CROSSED! Bonus: +{self.FINISH_REWARD}")
            terminated = True  # End episode on success

        # Heavy penalty for falling off the line (scaled to match new reward range)
        
        if terminated and not crossed_finish:
            reward = -100.0  # Larger penalty to discourage falling off
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

        # --- Apply Domain Randomization (respawns robot and ground overlay) ---
        self._randomize_environment()
        
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