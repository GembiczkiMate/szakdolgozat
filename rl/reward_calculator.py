import math

class RewardCalculator:
    def __init__(self, max_speed=0.5, max_turn=1.7, finish_reward=300.0, reward_mode='vision', stationary_threshold=0.1):
        self.max_speed = max_speed
        self.max_turn = max_turn
        self.finish_reward = finish_reward
        self.reward_mode = reward_mode
        self.stationary_threshold = stationary_threshold 
        
        
    def calculate_reward(self, error, linear_speed, angular_speed, prev_angular_speed):
        """
        Calculates the immediate reward, penalties, and stability factors
        based on the robot's current speed and visually detected error.
        
        Returns:
            reward: The calculated float reward
        """
        if self.reward_mode == 'coordinate':
            return self._calculate_coordinate_reward(error, linear_speed, angular_speed, prev_angular_speed)
        else:
            return self._calculate_vision_reward(error, linear_speed, angular_speed, prev_angular_speed)

    def _calculate_coordinate_reward(self, error, linear_speed, angular_speed, prev_angular_speed):
        # Szigorúbb büntetések, melyek koordináta módhoz lettek optimalizálva (nincs zaj)
        stability_factor = math.exp(-6.5 * abs(error))
        speed_factor = max(0.0, linear_speed) / self.max_speed
        progress_reward = speed_factor * stability_factor

        survival_bonus = 0.25
        stationary_penalty = 0.5 if linear_speed < self.stationary_threshold else 0.0
        
        steering_change = abs(angular_speed - prev_angular_speed)
        smoothness_penalty = (steering_change / (2.0 * self.max_turn)) * 0.5 
        steering_penalty = (abs(angular_speed) / self.max_turn) * 0.25

        reward = (progress_reward + survival_bonus - 
                  stationary_penalty - 
                  smoothness_penalty - 
                  steering_penalty) * 5.0 
        return reward

    def _calculate_vision_reward(self, error, linear_speed, angular_speed, prev_angular_speed):
        # Megengedőbb jutalmazás a kamerás (vision) módhoz, ami zajosabb lehet
        # A -4.5 "szélesíti" az elfogadható zónát a kanyarokban a -6.5 helyett
        stability_factor = math.exp(-4.5 * abs(error))
        speed_factor = max(0.0, linear_speed) / self.max_speed
        # NAGYON erős súly a progress_reward-ra, hogy előre menjen!

        # Még erősebb súly az előrehaladásra!
        progress_reward = speed_factor * stability_factor * 4.0

        # Csökkentett túlélési bónusz, hogy ne érje meg csak "életben maradni"
        survival_bonus = 0.1

        # Kevésbé szigorú büntetés az egyhelyben állásért
        stationary_penalty = 0.25 if linear_speed < self.stationary_threshold else 0.0

        # Szigorúbb büntetés a kormányzás változásáért (0.2 a 0.1 helyett)
        steering_change = abs(angular_speed - prev_angular_speed)
        smoothness_penalty = (steering_change / (2.0 * self.max_turn)) * 0.2

        # Szigorúbb büntetés folyamatos kanyarodásért (0.08 a 0.03 helyett)
        steering_penalty = (abs(angular_speed) / self.max_turn) * 0.08

        # Sáv elhagyása elleni büntetés, ha a vonal kimegy a kényelmes középső sávból (error > 0.5)
        off_center_penalty = max(0.0, (abs(error) - 0.5)) * 1.5

        # Combined weighted reward
        reward = (progress_reward + survival_bonus - 
              stationary_penalty - 
              smoothness_penalty - 
              steering_penalty -
              off_center_penalty) * 5.0  # Scale up for stronger signal
        return reward
        
    def calculate_termination_reward(self, terminated, crossed_finish, current_step, grace_period=5):
        """
        Calculates termination modifications to the reward based on success/failure.
        
        Returns:
            bonus_or_penalty: Reward addition or subtraction
            modified_terminated: Overridden termination state
        """
        if crossed_finish:
            return self.finish_reward, True
            
        if terminated and not crossed_finish:
            if current_step < grace_period:
                # Grace period at the beginning of the episode to find the line
                return 0.0, False
            else:
                return -100.0, True  # Larger penalty to discourage falling off
                
        return 0.0, terminated
