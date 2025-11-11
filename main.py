from environment.wmn_env import WMNEnvironment
from marl_agents.self_maddpg import SelfCoordinatedMADDPG
import matplotlib.pyplot as plt
import numpy as np

# --- Hyperparameters ---
NUM_NODES = 5
MAX_EPISODES = 500
MAX_STEPS_PER_EPISODE = 50
BATCH_SIZE = 64
TRAINING_START_STEP = 500

def smooth(y, box_pts=5):
    """Simple moving average for smoothing curves."""
    if len(y) < box_pts:
        return y
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode='same')

def main():
    # 1. Initialize Environment and Agent
    env = WMNEnvironment(num_nodes=NUM_NODES)
    agent = SelfCoordinatedMADDPG(env)
    
    global_step = 0
    all_rewards, all_delays, all_powers = [], [], []
    
    print(f"Starting Training for {MAX_EPISODES} Episodes...")

    # 2. Main Training Loop
    for episode in range(1, MAX_EPISODES + 1):
        states = env.reset()
        episode_reward = 0
        episode_delay, episode_power = 0, 0
        
        for t in range(MAX_STEPS_PER_EPISODE):
            global_step += 1

            # Agents choose actions
            actions = agent.choose_actions(states)
            next_states, reward, done, info = env.step(actions)

            # Store transition
            agent.buffer.push(states, actions, reward, next_states)

            states = next_states
            episode_reward += reward
            episode_delay = info["total_delay"]
            episode_power = info["total_power"]

            if global_step > TRAINING_START_STEP:
                agent.learn(BATCH_SIZE)

            if done:
                break

        all_rewards.append(episode_reward)
        all_delays.append(episode_delay)
        all_powers.append(episode_power)

        if episode % 10 == 0:
            avg_reward = sum(all_rewards[-10:]) / 10
            print(f"Episode: {episode:4d} | Global Step: {global_step:6d} | "
                  f"Avg Reward (10): {avg_reward:.2f} | "
                  f"Last Delay: {episode_delay:.2f} | Last Power: {episode_power:.2f}")

    print("Training Complete ✅")

    # --- After training: plotting section ---
    episodes = np.arange(1, len(all_rewards) + 1)
    rewards_smooth = smooth(all_rewards, 5)
    delays_smooth = smooth(all_delays, 5)
    powers_smooth = smooth(all_powers, 5)

    # --- Plot 1: Reward ---
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, rewards_smooth, label="Reward (smoothed)")
    plt.scatter(episodes, all_rewards, s=10, alpha=0.4, color='gray')
    plt.title("Training Convergence: Average Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("reward_curve.png", dpi=300, bbox_inches='tight')

    # --- Plot 2: Power ---
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, powers_smooth, color="green", label="Total Power (smoothed)")
    plt.scatter(episodes, all_powers, s=10, alpha=0.4, color="gray")
    plt.title("Average Total Power per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Power")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("power_curve.png", dpi=300, bbox_inches='tight')

    # --- Plot 3: Delay ---
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, delays_smooth, color="red", label="Delay (smoothed)")
    plt.scatter(episodes, all_delays, s=10, alpha=0.4, color="gray")
    plt.title("Average Delay per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Delay")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("delay_curve.png", dpi=300, bbox_inches='tight')

    # --- Optional combined summary plot ---
    plt.figure(figsize=(9, 6))
    plt.plot(episodes, rewards_smooth, label="Reward")
    plt.plot(episodes, powers_smooth, label="Power")
    plt.plot(episodes, delays_smooth, label="Delay")
    plt.title("Training Summary: Reward vs Power vs Delay")
    plt.xlabel("Episode")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("summary_curve.png", dpi=300, bbox_inches='tight')

    plt.show()
    print("Plots saved: reward_curve.png, power_curve.png, delay_curve.png, summary_curve.png")

if __name__ == '__main__':
    main()
