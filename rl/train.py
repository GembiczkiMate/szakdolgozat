import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from ros_line_follow_env import RosLineFollowEnv
from ament_index_python.packages import get_package_share_directory
import os
import torch
import math

import argparse

# ============================================================================
# HYPERPARAMETERS - Modify these to tune your training
# ============================================================================

# --- Training Parameters ---
TOTAL_TIMESTEPS = 10000       # Jelentősen megemelve, hogy legyen ideje tanulni (10k -> 300k)
LOG_INTERVAL = 1              # Log every N episodes

# --- PPO Algorithm Parameters ---
PPO_HYPERPARAMS = {
    "learning_rate": 2.5e-4,    # Megnövelt tanulási ráta, hogy gyorsabban tanuljon az új környezetből
    "n_steps": 512,           
    "batch_size": 128,         
    "n_epochs": 10,           
    "gamma": 0.99,            
    "gae_lambda": 0.95,       
    "clip_range": 0.2,        
    "ent_coef": 0.05,        # Magasabb entrópia ezen a ponton (0.005 -> 0.05), hogy merjen kísérletezni az új jutalommal!
    "vf_coef": 0.5,           
    "max_grad_norm": 0.5,     
}

# --- CNN Policy Architecture ---
# CnnPolicy automatically extracts features from images
# Then uses fully connected layers for policy and value heads
POLICY_KWARGS = {
    "net_arch": dict(
        pi=[64, 64],        # Policy network after CNN features
        vf=[64, 64]         # Value network after CNN features
    ),
    # Use smaller CNN for 84x84 grayscale images (faster training)
    "features_extractor_kwargs": {
        "features_dim": 256   # Output dimension of CNN feature extractor
    }
}

# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train PPO on Line Follower.')
    parser.add_argument('--reward-mode', type=str, default='vision', choices=['vision', 'coordinate'],
                        help="Reward calculation mode: 'vision' (based on camera) or 'coordinate' (based on odometry distance to spline).")
    # A ROS2 miatt ki kell szűrnünk a rosnak szánt argumentumokat (mint pl. a --ros-args)
    args, unknown = parser.parse_known_args()

    rclpy.init()
    
    # Környezet példányosítása a bemeneti paraméterrel
    env = RosLineFollowEnv(is_testing_mode=False, reward_mode=args.reward_mode)
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
        # Fontos: Custom objects-ben adjuk át a megváltozott hiperparamétereket a .zip-nek!
        model = PPO.load(model_save_path, env=env, custom_objects=PPO_HYPERPARAMS)
        
        # --- ERŐLTETETT KÍSÉRLETEZÉS KIKAPCSOLVA ---
        # A korábbi "variance hack" (log_std kényszerítése) itt ki lett véve, 
        # mert ha minden újraindításnál lefut (amit a watchdog generál), a robot folyamatosan 
        # "elfelejti" a finommozgásokat, és a reward rohamosan csökkeni kezd.
        
        # Ha logolni is akarjuk a folytatást
        model.tensorboard_log = tensorboard_log_dir
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