import rclpy
import time
import math
import os
import re
import subprocess
import tempfile
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, SetEntityState

from gazebo_utils import TrackGenerator

class GazeboEntityManager:
    def __init__(self, node, robot_name='two_wheeled_robot', line_model_name='track_line', urdf_path=''):
        self.node = node
        self.robot_name = robot_name
        self.line_model_name = line_model_name
        self.urdf_path = urdf_path
        
        self.spawn_client = self.node.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.node.create_client(DeleteEntity, '/delete_entity')
        self.set_state_client = self.node.create_client(SetEntityState, '/set_entity_state')

        self.track_version = 0

    def reset_robot_position(self, spawn_x, spawn_y):
        """Reset robot to starting position without respawning (faster)."""
        req = SetEntityState.Request()
        req.state.name = self.robot_name
        req.state.pose.position.x = spawn_x
        req.state.pose.position.y = spawn_y
        req.state.pose.position.z = 0.05
        yaw = -3.14
        req.state.pose.orientation.w = math.cos(yaw / 2)
        req.state.pose.orientation.z = math.sin(yaw / 2)

        success = False
        # 1. Próbálkozás rclpy-vel
        if self.set_state_client.wait_for_service(timeout_sec=1.0):
            try:
                future = self.set_state_client.call_async(req)
                rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
                if future.result() is not None and future.result().success:
                    success = True
            except Exception:
                pass
        
        # 2. Ha az rclpy eldobta (timeout vagy false), jöhet a CLI fallback
        if not success:
            cmd = [
                "ros2", "service", "call", "/set_entity_state", "gazebo_msgs/srv/SetEntityState",
                f"{{state: {{name: '{self.robot_name}', pose: {{position: {{x: {spawn_x}, y: {spawn_y}, z: 0.05}}, orientation: {{x: 0.0, y: 0.0, z: -1.0, w: 0.0}}}}}}}}"
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
            self.node.get_logger().warn("Mindkét teleport (API és CLI) sikertelen, használjuk a teljes respawnt!")
            return False
            
        return True

    def delete_entity(self, name):
        """Hívja a delete_entity service-t, robosztus újrapróbálkozással és CLI fallback-kel."""
        req = DeleteEntity.Request()
        req.name = name

        max_retries = 3
        for attempt in range(max_retries):
            if not self.delete_client.wait_for_service(timeout_sec=2.0):
                self.node.get_logger().warning(f"DeleteEntity service nem elérhető az rclpy-ben, próbálkozás terminál (CLI) fallback-kel: {name}")
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
                rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
                
                if future.result() is not None and future.result().success:
                    return True
                else:
                    self.node.get_logger().warning(f"Nem sikerült törölni az entitást: {name} (lehet, hogy már nem is létezik).")
                    return False
            except Exception as e:
                self.node.get_logger().error(f"Hiba a delete_entity hívásakor: {e}")
                time.sleep(1.0)
        
        self.node.get_logger().error("A DeleteEntity service tartósan nem elérhető!")
        return False

    def spawn_entity(self, name, xml, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
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
                rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
                if future.result() is not None and future.result().success:
                    return True
            except Exception:
                pass
                
        # 2. CLI Fallback (írás fájlba majd spawn via script)
        self.node.get_logger().warn(f"API spawn falied for {name}, trying CLI tool...")
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
                    os.unlink(temp_xml_path)
                    return True
                time.sleep(1.0)
                
            os.unlink(temp_xml_path)
        except Exception as e:
            self.node.get_logger().error(f"Fallback spawn tool failed: {e}")
            
        return False

    def respawn_robot(self, spawn_x, spawn_y, current_camera_pitch):
        """Delete and respawn the robot with modified camera pitch."""
        max_retries = 3
        
        for attempt in range(max_retries):
            # Delete existing robot
            delete_success = self.delete_entity(self.robot_name)
            time.sleep(1.0)  # Növelt várakozás a stabilitás érdekében (Gazebo takarítás)
            
            # Read and modify URDF with new camera pitch
            try:
                with open(self.urdf_path, 'r') as f:
                    urdf_content = f.read()
                
                pattern = r'(<joint name="camera_joint"[^>]*>.*?rpy=")[^"]*(")'
                replacement = rf'\g<1>0 {current_camera_pitch} 0\2'
                modified_urdf = re.sub(pattern, replacement, urdf_content)
                
                # Spawn robot at start position
                success = self.spawn_entity(
                    self.robot_name,
                    modified_urdf,
                    spawn_x, spawn_y, 0.05,
                    yaw=-3.14  # Facing -X direction
                )
                
                if success:
                    self.node.get_logger().info(f"Robot respawned (attempt {attempt+1}) with camera pitch: {math.degrees(current_camera_pitch):.1f}°")
                    return True
                else:
                    self.node.get_logger().warn(f"Failed to respawn robot (attempt {attempt+1}/{max_retries})")
                    time.sleep(1.0)  # Wait before retry
                    
            except Exception as e:
                self.node.get_logger().error(f"Error respawning robot: {e}")
                time.sleep(1.0)
        
        self.node.get_logger().error("Failed to respawn robot after all retries!")
        
        # JELZÉS KÜLDÉSE A WATCHDOGNAK, HOGY FULL RESTART KELL!
        try:
            with open('/tmp/gazebo_fatal_error.flag', 'w') as f:
                f.write('FATAL: Respawn limit reached')
        except:
            pass
        os._exit(1) # Azonnal kinyírjuk a python scriptet hibakóddal
        
        return False

    def respawn_line(self, track_points, spline_resolution, line_width):
        # Delete existing line
        self.delete_entity(self.line_model_name)
        time.sleep(1.0)  # Késleltetés a törlés után a szimulátor teljesítményéhez
        
        self.track_version += 1
        self.line_model_name = f"track_line_v{self.track_version}"
        
        # Generate spline points
        spline_points = TrackGenerator._catmull_rom_spline(track_points, spline_resolution)
        
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
              <size>{length + 0.005} {line_width} 0.002</size>
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
        success = self.spawn_entity(
            self.line_model_name,
            sdf,
            0.0, 0.0, 0.0
        )
        
        if success:
            self.node.get_logger().debug(f"Track line spawned")
        else:
            self.node.get_logger().warn("Failed to spawn track line")
