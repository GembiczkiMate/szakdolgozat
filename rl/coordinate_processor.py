import math

class CoordinateProcessor:
    def __init__(self, max_allowed_deviation_meters=0.4):
        """
        Kezeli a koordináta alapú hibaszámítást.
        max_allowed_deviation_meters: Milyen messze mehet fizikai méterben a vonaltól, mielőtt terminál.
        """
        self.spline_points = []
        self.max_allowed_deviation = max_allowed_deviation_meters
        
    def update_track_spline(self, spline_points):
        """
        Frissíti a jelenlegi pálya finomított (spline) pontjait, ami alapján a távolságot számoljuk.
        Ezt generáláskor hívjuk meg.
        """
        self.spline_points = spline_points
        
    def calculate_error_and_termination(self, robot_x, robot_y):
        """
        Kiszámolja a legközelebbi pályaponttól való fizikai távolságot.
        
        Returns:
            normalized_error: 0.0 (tökéletesen rajta) és 1.0 (maximális megengedett távolságon) között.
            terminated: True, ha elhagyta a pályát (messzebb van a megengedettnél).
        """
        if not self.spline_points:
            # Ha nincs pálya betöltve, azonnali terminálás (biztonsági okokból)
            return 1.0, True
            
        # Keresés a legközelebbi pontra a Catmull-Rom spline-on
        min_dist = float('inf')
        for p in self.spline_points:
            dist = math.sqrt((robot_x - p[0])**2 + (robot_y - p[1])**2)
            if dist < min_dist:
                min_dist = dist
                
        # min_dist a méterben vett távolság.
        # Normalizáljuk a max_allowed_deviation alapján, hogy kompatibilis legyen a régi reward_calculator bemenetével
        normalized_error = min_dist / self.max_allowed_deviation
        
        # Ha a távolság nagyobb, mint a megengedett, az epizód véget ér
        terminated = min_dist > self.max_allowed_deviation
        
        # Határoljuk be a normalizált hibát 0 és 1 közé bizonságképpen
        normalized_error = min(max(normalized_error, 0.0), 1.0)
        
        return normalized_error, terminated
