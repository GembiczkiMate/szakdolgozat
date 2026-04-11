import math
import time
import subprocess
import tempfile
import matplotlib.pyplot as plt
import io

class TrackGenerator:
    """Helper class to abstract trajectory and model SDF generation."""
    @staticmethod
    def _catmull_rom_spline(control_points, resolution):
        """Catmull-Rom spline implementation for smoothing the track."""
        if len(control_points) < 4:
            points = [control_points[0]] + control_points + [control_points[-1]]
        else:
            points = control_points.copy()
            points.insert(0, points[0])
            points.append(points[-1])
            
        spline_points = []
        # Exclude the artificially added boundary points from loop
        for i in range(1, len(points) - 2):
            p0, p1, p2, p3 = points[i-1], points[i], points[i+1], points[i+2]
            
            for t in range(resolution):
                t_val = t / float(resolution)
                
                t2 = t_val ** 2
                t3 = t2 * t_val
                
                # Catmull-Rom equations
                x = 0.5 * ((2 * p1[0]) +
                          (-p0[0] + p2[0]) * t_val +
                          (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                          (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
                          
                y = 0.5 * ((2 * p1[1]) +
                          (-p0[1] + p2[1]) * t_val +
                          (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                          (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
                          
                spline_points.append((x, y))
                
        spline_points.append(points[-2])
        return spline_points

    @staticmethod
    def generate_track_sdf(line_model_name, track_points, resolution, line_width):
        """Generate the XML/SDF representation of the line."""
        spline_points = TrackGenerator._catmull_rom_spline(track_points, resolution)
        segments_sdf = ""
        segment_id = 0
        
        for i in range(len(spline_points) - 1):
            x1, y1 = spline_points[i]
            x2, y2 = spline_points[i + 1]
            
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < 0.001:
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
            
        sdf = f'''<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{line_model_name}">
    <static>true</static>
    {segments_sdf}
  </model>
</sdf>'''
        return sdf
