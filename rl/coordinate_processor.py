import math

class CoordinateProcessor:
    def __init__(self, max_allowed_deviation_meters=0.4):
        self.spline_points = []
        self.max_allowed_deviation = max_allowed_deviation_meters
        
    def update_track_spline(self, spline_points):
        self.spline_points = spline_points
        
    def _point_to_line_dist(self, px, py, x1, y1, x2, y2):
        px, py = float(px), float(py)
        x1, y1 = float(x1), float(y1)
        x2, y2 = float(x2), float(y2)
        
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)
            
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

    def calculate_error_and_termination(self, robot_x, robot_y):
        if not self.spline_points or len(self.spline_points) < 2:
            return 1.0, True
            
        min_dist = float('inf')
        for i in range(len(self.spline_points) - 1):
            p1 = self.spline_points[i]
            p2 = self.spline_points[i + 1]
            dist = self._point_to_line_dist(robot_x, robot_y, p1[0], p1[1], p2[0], p2[1])
            if dist < min_dist:
                min_dist = dist
                
        normalized_error = min_dist / self.max_allowed_deviation
        terminated = min_dist > self.max_allowed_deviation
        normalized_error = min(max(normalized_error, 0.0), 1.0)
        
        return normalized_error, terminated
