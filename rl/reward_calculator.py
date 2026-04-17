import math

class RewardCalculator:
    def __init__(self, max_speed=0.5, max_turn=1.7, finish_reward=500.0, stationary_threshold=0.1):
        self.max_speed = max_speed
        self.max_turn = max_turn
        self.finish_reward = finish_reward
        self.stationary_threshold = stationary_threshold 
        
        
    def calculate_reward(self, error, linear_speed, angular_speed, prev_angular_speed):
        """
        Calculates the immediate reward, penalties, and stability factors
        based on the robot's current speed and visually detected error.
        
        Returns:
            reward: The calculated float reward
            smoothness_penalty: The deducted penalty for swinging/vibrating steering
        """
        # Stability factor (0 to 1) 
        # A -4.5 (a korábbi -6.5 helyett) "szélesíti" az elfogadható zónát a kanyarokban. 
        # Így egy éles kanyarban kapott ~35%-os vizuális elcsúszásra nem büntet annyira durván, nem veszi el a mozgásért kapott pontot!
        stability_factor = math.exp(-6.5 * abs(error))

        # Speed factor (0 to 1, moving at max_speed = 1)
        speed_factor = max(0.0, linear_speed) / self.max_speed

        # Core Progress Reward: The robot MUST move forward to get ANY points.
        # It gets the most points by going fast AND staying centered.
        progress_reward = speed_factor * stability_factor

        # ALAPVETŐ ÉLETBEN MARADÁSI BÓNUSZ (Survival Bonus)
        # Ez garantálja, hogy a pályán töltött minden lépés mindig fixen pozitív irányba mozdítsa a mérleget, 
        # akkor is, ha kanyarog vagy lassú!
        survival_bonus = 0.5

        # Penalty for standing still (moves too slow)
        stationary_penalty = 0.5 if linear_speed < self.stationary_threshold else 0.0
        
        # --- Smoothness and Steering Penalties ---
        # Penalize large changes in steering ("vibration")
        steering_change = abs(angular_speed - prev_angular_speed)
        # Scaled penalty so max change gets 0.5 penalty points
        smoothness_penalty = (steering_change / (2.0 * self.max_turn)) * 0.1 
        
        # Penalize constant high steering (zig-zagging/weaving)
        steering_penalty = (abs(angular_speed) / self.max_turn) * 0.03

        # Combined weighted reward
        reward = (progress_reward + survival_bonus - 
                  stationary_penalty - 
                  smoothness_penalty - 
                  steering_penalty) * 10.0  # Scale up for stronger signal
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
                return -300.0, True  # Larger penalty to discourage falling off
                
        return 0.0, terminated
