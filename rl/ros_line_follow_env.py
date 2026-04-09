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

    # --- ENVIRONMENT CONFIGURATION ---
    # (Domain randomization has been removed by request)
    
    # Track definition - control points for Catmull-Rom spline
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
            self._respawn_robot()
            self._respawn_line()
            self.initial_setup_done = True
            time.sleep(0.5)  # Wait for line to appear
            return
            
        # If we selected a new track, respawn the line
        if track_changed:
            self.get_logger().info(f"Selected predefined track starting at ({self.SPAWN_X}, {self.SPAWN_Y})")
            self._respawn_line()
        
        # On subsequent resets, just reset the robot position to the start of the current track
        self._reset_robot_position()
    
    def _reset_robot_position(self):
        """Reset robot to starting position without respawning (faster)."""
        from gazebo_msgs.srv import SetEntityState
        from gazebo_msgs.msg import EntityState
        import subprocess
        
        if not hasattr(self, 'set_state_client'):
            self.set_state_client = self.create_client(SetEntityState, '/set_entity_state')

        req = SetEntityState.Request()
        req.state.name = self.robot_name
        req.state.pose.position.x = self.SPAWN_X
        req.state.pose.position.y = self.SPAWN_Y
        req.state.pose.position.z = 0.05
        yaw = -3.14
        req.state.pose.orientation.w = math.cos(yaw / 2)
        req.state.pose.orientation.z = math.sin(yaw / 2)

        success = False
        # 1. Próbálkozás rclpy-vel
        if self.set_state_client.wait_for_service(timeout_sec=1.0):
            try:
                future = self.set_state_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
                if future.result() is not None and future.result().success:
                    success = True
            except Exception:
                pass
        
        # 2. Ha az rclpy eldobta (timeout vagy false), jöhet a CLI fallback
        if not success:
            cmd = [
                "ros2", "service", "call", "/set_entity_state", "gazebo_msgs/srv/SetEntityState",
                f"{{state: {{name: '{self.robot_name}', pose: {{position: {{x: {self.SPAWN_X}, y: {self.SPAWN_Y}, z: 0.05}}, orientation: {{x: 0.0, y: 0.0, z: -1.0, w: 0.0}}}}}}}}"
            ]
            for _ in range(3):
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
                    if "success=True" in res.stdout or "success=True" in res.stderr or "successs=True" in res.stderr:
                        success = True
                        break
                except Exception:
                    pass
                time.sleep(0.5)

        if not success:
            self.get_logger().warn("Mindkét teleport (API és CLI) sikertelen, használjuk a teljes respawnt!")
            self._respawn_robot()

    def _delete_entity(self, name):
        """Hívja a delete_entity service-t, robosztus újrapróbálkozással és CLI fallback-kel."""
        req = DeleteEntity.Request()
        req.name = name

        max_retries = 3
        for attempt in range(max_retries):
            if not self.delete_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warning(f"DeleteEntity service nem elérhető az rclpy-ben, próbálkozás terminál (CLI) fallback-kel: {name}")
                import subprocess
                try:
                    cmd = ["ros2", "service", "call", "/delete_entity", "gazebo_msgs/srv/DeleteEntity", f"{{name: '{name}'}}"]
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
                    if "success=True" in res.stdout or "successs=True" in res.stderr:
                        return True
                except Exception as e:
                    pass
                time.sleep(1.0)
                continue
                
            try:
                future = self.delete_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
                
                if future.result() is not None and future.result().success:
                    return True
                else:
                    self.get_logger().warning(f"Nem sikerült törölni az entitást: {name} (lehet, hogy már nem is létezik).")
                    return False
            except Exception as e:
                self.get_logger().error(f"Hiba a delete_entity hívásakor: {e}")
                time.sleep(1.0)
        
        self.get_logger().error("A DeleteEntity service tartósan nem elérhető!")
        return False

    def _spawn_entity(self, name, xml, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        """Spawn an entity in Gazebo robustly."""
        req = SpawnEntity.Request()
        req.name = name
        req.xml = xml
        req.initial_pose.position.x = float(x)
        req.initial_pose.position.y = float(y)
        req.initial_pose.position.z = float(z)
        
        cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
        
        req.initial_pose.orientation.w = cr * cp * cy + sr * sp * sy
        req.initial_pose.orientation.x = sr * cp * cy - cr * sp * sy
        req.initial_pose.orientation.y = cr * sp * cy + sr * cp * sy
        req.initial_pose.orientation.z = cr * cp * sy - sr * sp * cy

        # 1. API attempt
        if self.spawn_client.wait_for_service(timeout_sec=1.0):
            try:
                future = self.spawn_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                if future.result() is not None and future.result().success:
                    return True
            except Exception:
                pass
                
        # 2. CLI Fallback (írás fájlba majd spawn via script)
        self.get_logger().warn(f"API spawn falied for {name}, trying CLI tool...")
        import subprocess
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.xml') as temp_xml:
                temp_xml.write(xml)
                temp_xml_path = temp_xml.name
                
            cmd = [
                "ros2", "run", "gazebo_ros", "spawn_entity.py", 
                "-entity", name, "-file", temp_xml_path,
                "-x", str(x), "-y", str(y), "-z", str(z),
                "-R", str(roll), "-P", str(pitch), "-Y", str(yaw)
            ]
            for _ in range(3):
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0)
                if "Spawn status: SpawnEntity: Successfully spawned" in res.stdout or res.returncode == 0:
                    import os; os.unlink(temp_xml_path)
                    return True
                time.sleep(1.0)
                
            import os; os.unlink(temp_xml_path)
        except Exception as e:
            self.get_logger().error(f"Fallback spawn tool failed: {e}")
            
        return False

    def _respawn_robot(self):
        """Delete and respawn the robot with modified camera pitch."""
        max_retries = 3
        
        for attempt in range(max_retries):
            # Delete existing robot
            delete_success = self._delete_entity(self.robot_name)
            time.sleep(1.0)  # Növelt várakozás a stabilitás érdekében (Gazebo takarítás)
            
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

    def _respawn_line(self):
        """Delete and respawn the track line with new width using Catmull-Rom spline."""
        # Delete existing line
        self._delete_entity(self.line_model_name)
        time.sleep(1.0)  # Késleltetés a törlés után a szimulátor teljesítményéhez
        
        # DDS üzenetsor késleltetési hibájának (beragadt DELETE parancs) elkerüléséhez
        # minden újracsinálásnál sorszámozzuk a vonal nevét. Így egy későn megérkező
        # törlési parancs nem tudja a már frissen lerakott vonalat véletlenül eltüntetni!
        if not hasattr(self, 'track_version'):
            self.track_version = 0
        self.track_version += 1
        self.line_model_name = f"track_line_v{self.track_version}"
        
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
            if self.current_step < 5:
                # Grace period at the beginning of the episode to find the line
                terminated = False
            else:
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