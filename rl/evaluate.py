import os

import rclpy
import os
import numpy as np
import argparse
from stable_baselines3 import PPO
from ros_line_follow_env import RosLineFollowEnv
from ament_index_python.packages import get_package_share_directory

def main():
    parser = argparse.ArgumentParser(description='Evaluate PPO on Line Follower.')
    parser.add_argument('--reward-mode', type=str, default='vision', choices=['vision', 'coordinate'], help="Model type to evaluate.")
    args, unknown = parser.parse_known_args()
    
    rclpy.init()
    
    print("==================================================")
    print(" VALIDÁCIÓ (KIPRÓBÁLÁS STATISZTIKÁKKAL) INDÍTÁSA")
    print("==================================================")

    # Létrehozzuk a környezetet a Validációhoz (ugyanaz a Gazebo pálya)
    # Itt is_testing_mode=False azért kell, hogy azokon a pályákon értékelje ki magát,
    # amiken a betanítás is zajlott. A modell itt már Hálózatot nem frissít, csak prediktál!
    env = RosLineFollowEnv(is_testing_mode=False, reward_mode=args.reward_mode)
    env.sequential_mode = True # Sorban menjen végig
    env.current_track_index = 0 # Kezdje a legelsővel
    env.sequential_mode = True # Sorban menjen végig
    env.current_track_index = 0 # Kezdje a legelsővel
    
    # Megkeressük az elmentett modellfájlt (model.zip) a mapparendszerben
    package_share_dir = get_package_share_directory('two_wheeled_robot')
    workspace_dir = '/media/gembi/4a3cde59-7329-4c35-983a-99689c6819c0/rl_ros/src/two_wheeled_robot'
    folder_name = f"ppo_line_follower_{args.reward_mode}"
    save_dir = os.path.abspath(os.path.join(workspace_dir, folder_name))
    model_path = os.path.join(save_dir, "model.zip")
    
    if not os.path.exists(model_path):
        print(f"Hiba: Nem találtam betanított modellt itt: {model_path}")
        print("Előbb indítsd el a tanítást és várd meg a modell első mentését!")
        return
        
    print(f"Betöltés: {model_path}...")
    # Betöltjük a mesterséges intelligenciát a korábban tanult "súlyokkal" (tudással)
    model = PPO.load(model_path, env=env)
    
    NUM_EPISODES = len(env.PREDEFINED_TRACKS)  # Pontosan egyszer menjen végig minden pályán!
    total_rewards = []
    success_count = 0
    
    print(f"\nA modell validálása {NUM_EPISODES} véletlenszerű epizódon (pályán) keresztül...")
    
    # 10 epizódon keresztül lefuttatja
    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        
        while not done:
            # DETERMINISTIC=TRUE -> Ez a legfontosabb! Itt már NEM tanul, NEM próbálkozik véletlenszerű dolgokkal, csak "robot módjára" a legtöbb pontot akarja kapni a tudása alapján.
            action, _states = model.predict(obs, deterministic=True)
            
            # A mozgást belerakja a Gazebóba
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            done = terminated or truncated
            
            # Megnézzük a végén leesett-e, vagy sikeresen megkapta a célvonal bónuszát
            if done and reward >= env.FINISH_REWARD - 50: # Hozzávetőleges check, ha a büntetés ellenére még így is magas a jutalma a cél miatt
                success_count += 1
                
        total_rewards.append(episode_reward)
        print(f"Epizód {episode + 1}/{NUM_EPISODES} - Lépések száma: {steps} - Késői pont (Jutalom): {episode_reward:.2f}")
        
    # Eredmények kiszámolása (statisztika)
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    success_rate = (success_count / NUM_EPISODES) * 100
    
    print("\n" + "="*40)
    print(" VALIDÁCIÓ EREDMÉNYEI (EVALUATION)")
    print("="*40)
    print(f"Átlagos Jutalom:  {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Sikerességi ráta: {success_rate:.1f}% (10-ből {success_count}x célba ért be!)")
    print("="*40)
    
    env.close()

if __name__ == '__main__':
    main()
