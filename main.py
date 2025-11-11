from environment.wmn_env import WMNEnvironment
from marl_agents.self_maddpg import SelfCoordinatedMADDPG
import matplotlib.pyplot as plt
import numpy as np
import torch
import csv

# --- Hyperparameters ---
NUM_NODES = 5
MAX_EPISODES = 100
MAX_STEPS_PER_EPISODE = 50
BATCH_SIZE = 64
TRAINING_START_STEP = 500
EVAL_EPISODES = 20
EVAL_STEPS = 50

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

    print("Training Complete!!!")

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

    # =======================
    #  EVALUATION PHASE
    # =======================
    print("\nStarting Deterministic Evaluation Run...")

    # Set all actors to eval mode
    for actor in agent.actors:
        actor.eval()

    eval_results = []
    per_agent_logs = {i: {"powers": [], "delays": [], "actions": []} for i in range(NUM_NODES)}

    for ep in range(1, EVAL_EPISODES + 1):
        states = env.reset()
        episode_reward, episode_delay, episode_power = 0.0, 0.0, 0.0

        for t in range(EVAL_STEPS):
            # deterministic (argmax) actions
            states_tensor = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
            coord_signals = agent._get_coordination_signals(states_tensor)
            actions = []

            for i in range(agent.N):
                local_state = states_tensor[:, i, :]
                coord_signal = coord_signals[:, i, :]
                probs = agent.actors[i](local_state, coord_signal)
                action = torch.argmax(probs, dim=1).item()
                actions.append(action)
                per_agent_logs[i]["actions"].append(action)

            next_states, reward, done, info = env.step(actions)
            episode_reward += reward
            episode_delay = info["total_delay"]
            episode_power = info["total_power"]

            for i in range(agent.N):
                per_agent_logs[i]["powers"].append(float(next_states[i][0]))
                per_agent_logs[i]["delays"].append(float(next_states[i][1]))

            states = next_states
            if done:
                break

        eval_results.append({
            "episode": ep,
            "total_reward": episode_reward,
            "total_power": episode_power,
            "total_delay": episode_delay
        })
        print(f"[Eval] Episode: {ep:3d} | Reward: {episode_reward:.2f} | Power: {episode_power:.2f} | Delay: {episode_delay:.2f}")

    # Save episode-level evaluation logs
    with open("evaluation_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "TotalReward", "TotalPower", "TotalDelay"])
        for res in eval_results:
            writer.writerow([res["episode"], res["total_reward"], res["total_power"], res["total_delay"]])

    # Per-agent summaries
    print("\nPer-Agent Evaluation Summary:")
    for i in range(NUM_NODES):
        avg_power = np.mean(per_agent_logs[i]["powers"])
        avg_delay = np.mean(per_agent_logs[i]["delays"])
        action_counts = np.bincount(per_agent_logs[i]["actions"], minlength=3)
        print(f"  Agent {i+1}: AvgPower={avg_power:.2f}, AvgDelay={avg_delay:.2f}, ActionDist={action_counts}")

    print("\n Evaluation complete. Results saved to 'evaluation_log.csv'.")

if __name__ == '__main__':
    main()
