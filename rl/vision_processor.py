import numpy as np
import cv2

class VisionProcessor:
    def __init__(self, img_height=240, img_width=320):
        self.img_height = img_height
        self.img_width = img_width
        
    def process_image(self, frame):
        """
        Processes the BGR image frame from the camera to extract the observation, 
        calculate the error from the track, and check if the episode should terminate.
        
        Returns:
            img_obs: Preprocessed image for the neural network (C, H, W format, RGB)
            error: Normalized pixel error (-1.0 to 1.0)
            terminated: Boolean indicating if the robot lost the line
        """
        height, width, _ = frame.shape
        
        # Convert BGR to RGB, then transpose to channel-first (C, H, W)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_obs = np.transpose(rgb_frame, (2, 0, 1)).astype(np.uint8)
        
        # Image processing for REWARD and TERMINATION only
        # The neural network doesn't see this, it's just for reward calculation
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
            
                    # 1. Calculate the error in pixels from the center
                    pixel_error = width / 2 - cx
            
                    # 2. Check if it deviates more than 30% from the center
                    max_allowed_deviation = width * 0.30
                    terminated = abs(pixel_error) > max_allowed_deviation
            
                    # 3. Calculate the normalized error
                    normalized_error = pixel_error / (width / 2)
            
                    return img_obs, normalized_error, terminated
        
        # No line detected - return image obs with error=1.0 (worst case)
        return img_obs, 1.0, True
