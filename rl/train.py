import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from ros_line_follow_env import RosLineFollowEnv
from ament_index_python.packages import get_package_share_directory
import os
import torch

# ============================================================================
# HYPERPARAMETERS - Modify these to tune your training
# ============================================================================

# --- Training Parameters ---
TOTAL_TIMESTEPS = 5000       # Total training steps (Must be larger than n_steps to log anything!)
LOG_INTERVAL = 1              # Log every N episodes

# --- PPO Algorithm Parameters ---
PPO_HYPERPARAMS = {
    "learning_rate": 3e-4,    # Learning rate (default: 3e-4)
    "n_steps": 1024,           # Steps per update (Reduced so it logs multiple times during training)
    "batch_size": 64,         # Minibatch size (default: 64)
    "n_epochs": 10,           # Epochs per update (default: 10)
    "gamma": 0.99,            # Discount factor (default: 0.99)
    "gae_lambda": 0.95,       # GAE lambda (default: 0.95)
    "clip_range": 0.2,        # PPO clip range (default: 0.2)
    "ent_coef": 0.01,         # Entropy coefficient - exploration (default: 0.0)
    "vf_coef": 0.5,           # Value function coefficient (default: 0.5)
    "max_grad_norm": 0.5,     # Max gradient norm (default: 0.5)
}

# --- CNN Policy Architecture ---
# CnnPolicy automatically extracts features from images
# Then uses fully connected layers for policy and value heads
POLICY_KWARGS = {
    "net_arch": dict(
        pi=[256, 128],        # Policy network after CNN features
        vf=[256, 128]         # Value network after CNN features
    ),
    # Use smaller CNN for 84x84 grayscale images (faster training)
    "features_extractor_kwargs": {
        "features_dim": 256   # Output dimension of CNN feature extractor
    }
}

# ============================================================================

def main(args=None):
    rclpy.init(args=args)
    
    # --- 1. Create and check the environment ---
    env = RosLineFollowEnv()
    # It's good practice to check that your environment complies with the gym interface
    # check_env(env) # This check can be slow and sometimes has issues with ROS, use if needed

    # --- 2. Create the PPO agent ---
    # Define paths dynamically using the package share directory
    package_share_dir = get_package_share_directory('two_wheeled_robot')
    # Save slightly outside the install space so they are easily accessible 
    # Usually users run this from their workspace root
    workspace_dir = os.path.join(package_share_dir, '..', '..', '..', '..')
    # Changed to v2 to force learning from scratch with the new reward logic!
    save_dir = os.path.abspath(os.path.join(workspace_dir, "ppo_line_follower_v2"))
    
    os.makedirs(save_dir, exist_ok=True)
    
    model_save_path = os.path.join(save_dir, "model")
    tensorboard_log_dir = os.path.join(save_dir, "logs")
    
    # Check if a pre-trained model exists
    if os.path.exists(model_save_path + ".zip"):
        print("Loading existing model...")
        model = PPO.load(model_save_path, env=env, tensorboard_log=tensorboard_log_dir)
    else:
        print("Creating new CNN-based model...")
        print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        model = PPO(
            "CnnPolicy",  # Changed from MlpPolicy to CnnPolicy for image input
            env,
            verbose=1,
            tensorboard_log=tensorboard_log_dir,
            policy_kwargs=POLICY_KWARGS,
            device="auto",  # Use GPU if available
            **PPO_HYPERPARAMS
        )

    # --- 3. Train the agent ---
    try:
        # Train for a specified number of timesteps
        # reset_num_timesteps=False allows continuing paths on tensorboard
        model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=LOG_INTERVAL, reset_num_timesteps=False)
        print("Training finished. Saving model...")
        model.save(model_save_path)
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        model.save(model_save_path)
    finally:
        env.close()

if __name__ == '__main__':
    main()