import os

import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from ros_line_follow_env import RosLineFollowEnv
from ament_index_python.packages import get_package_share_directory
import os
import torch
import math

import argparse
from stable_baselines3.common.logger import configure

# ============================================================================
# HYPERPARAMETERS - Modify these to tune your training
# ============================================================================

# --- Training Parameters ---
TOTAL_TIMESTEPS = 20000    # Nagyon magasra állítva, hogy folyamatosan fusson és naplózzon egyetlen vonalként
LOG_INTERVAL = 1              # Log every N episodes

# --- PPO Algorithm Parameters ---
PPO_HYPERPARAMS = {
    "learning_rate": 2.5e-4,    # Megnövelt tanulási ráta, hogy gyorsabban tanuljon az új környezetből
    "n_steps": 2048,             # Nagyobb n_steps a stabilabb tanulásért (pl. 2048 -> 8192)
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

# ============================================================================

class CustomSaveCallback(BaseCallback):
    """
    Silently overwrites the same model.zip so we don't spam the directory with checkpoins,
    but we still survive crashes securely.
    """
    def __init__(self, save_path, save_freq, verbose=0):
        super(CustomSaveCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)
            if self.verbose > 0:
                print(f"Silently auto-saved model at {self.num_timesteps} steps.")
        return True

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
    
    # Dinamikusan kiválasztjuk a mappát a választott mód (vision vagy coordinate) alapján, hogy ne keveredjenek össze
    folder_name = f"ppo_line_follower_{args.reward_mode}"
    save_dir = os.path.abspath(os.path.join(workspace_dir, folder_name))
    
    os.makedirs(save_dir, exist_ok=True)
    
    model_save_path = os.path.join(save_dir, "model")
    tensorboard_log_dir = os.path.join(save_dir, "logs")
    
    # Check if a pre-trained model exists and is not an empty file
    import zipfile
    is_valid_zip = False
    if os.path.exists(model_save_path + ".zip") and os.path.getsize(model_save_path + ".zip") > 0:
        try:
            with zipfile.ZipFile(model_save_path + ".zip") as zf:
                is_valid_zip = True
        except zipfile.BadZipFile:
            print("WARNING: Existing model.zip is corrupted (e.g. interrupted save). Ignoring it and creating a new one.")
            
    if is_valid_zip:
        print("Loading existing model...")
        # Fontos: Custom objects-ben adjuk át a megváltozott hiperparamétereket a .zip-nek!
        try:
            model = PPO.load(model_save_path, env=env, custom_objects=PPO_HYPERPARAMS)
        except AssertionError as e:
            if "No data found in the saved file" in str(e):
                print("WARNING: Existing model is missing data (AssertionError). Creating a new one...")
                model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir, **PPO_HYPERPARAMS)
            else:
                raise e
        
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
    # Try to clean up logger logic to append natively
    try:
        # Train for a specified number of timesteps
        # We enforce reset_num_timesteps=False so it resumes counting from where it left off,
        # but to successfully append to Tensorboard exactly, tb_log_name must remain stable
        # AND reset_num_timesteps=False inside learn().
        auto_save_callback = CustomSaveCallback(save_path=model_save_path, save_freq=1000)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=LOG_INTERVAL, reset_num_timesteps=False, tb_log_name="PPO_unified", callback=auto_save_callback)
        print("Training finished. Saving model...")
        tmp_path = model_save_path + "_tmp"
        model.save(tmp_path)
        if os.path.exists(tmp_path + ".zip"):
            os.replace(tmp_path + ".zip", model_save_path + ".zip")
    except (KeyboardInterrupt, BaseException) as e:
        print(f"Training interrupted ({e}). Saving model...")
        tmp_path = model_save_path + "_tmp"
        model.save(tmp_path)
        if os.path.exists(tmp_path + ".zip"):
            os.replace(tmp_path + ".zip", model_save_path + ".zip")
    finally:
        env.close()

if __name__ == '__main__':
    main()