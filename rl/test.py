import rclpy
import os
import argparse
from stable_baselines3 import PPO
from ros_line_follow_env import RosLineFollowEnv
from ament_index_python.packages import get_package_share_directory

def main():
    parser = argparse.ArgumentParser(description='Test PPO on Line Follower.')
    parser.add_argument('--reward-mode', type=str, default='vision', choices=['vision', 'coordinate'], help="Model type to test.")
    args, unknown = parser.parse_known_args()
    
    rclpy.init()
    
    print("==================================================")
    print(" VÉGTELEN TESZTELÉS (INFERENCE / DEPLOYMENT) INDÍTÁSA")
    print("==================================================")
    print("Ebben a módban a robot végtelen ideig próbálkozik a befejezett pályákon.")
    print("Nyomj CTRL+C -t a leállításhoz!")
    print("==================================================\n")

    # Create the environment in testing mode
    env = RosLineFollowEnv(is_testing_mode=True, reward_mode=args.reward_mode)
    
    package_share_dir = get_package_share_directory('two_wheeled_robot')
    workspace_dir = os.path.join(package_share_dir, '..', '..', '..', '..')
    folder_name = f"ppo_line_follower_{args.reward_mode}"
    save_dir = os.path.abspath(os.path.join(workspace_dir, folder_name))
    model_path = os.path.join(save_dir, "model.zip")
    
    if not os.path.exists(model_path):
        print(f"Hiba: Nincs kész modell. Kélek előbb taníts be egyet!")
        return
        
    print("Modell betöltése...")
    model = PPO.load(model_path, env=env)
    
    try:
        obs, info = env.reset()
        while True:
            # Csak kigenerálja a mozdulatot az eddigi TUDÁSA (nem próbálgathat új dolgokat -> deterministic=True)
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Ha felborul vagy beér, új pályát kérünk, hadd tekerjen a végtelenségig
            if terminated or truncated:
                print(" -> Epizód vége! Újraindítás...")
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\n\nKilépés a felhasználó kérésére...")
    finally:
        env.close()

if __name__ == '__main__':
    main()
