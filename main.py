from environment.wmn_env import WMNEnvironment
from marl_agents.self_maddpg import SelfCoordinatedMADDPG

# --- Hyperparameters ---
NUM_NODES = 5             # N (Number of Agents/Nodes)
MAX_EPISODES = 500
MAX_STEPS_PER_EPISODE = 50 # T
BATCH_SIZE = 64
TRAINING_START_STEP = 500  # Start learning after collecting some initial experience

def main():
    # 1. Initialize Environment and Agent
    env = WMNEnvironment(num_nodes=NUM_NODES)
    agent = SelfCoordinatedMADDPG(env)
    
    global_step = 0
    all_rewards = []
    
    print(f"Starting Training for {MAX_EPISODES} Episodes...")

    # 2. Main Training Loop (Algorithm 1, Step 2)
    for episode in range(1, MAX_EPISODES + 1):
        # Reset environment (Step 3)
        states = env.reset() 
        episode_reward = 0
        
        # Inner loop for steps (Step 4)
        for t in range(MAX_STEPS_PER_EPISODE):
            global_step += 1
            
            # Agents choose actions (Decentralized Execution)
            actions = agent.choose_actions(states)
            
            # Execute actions in the environment
            next_states, reward, done, info = env.step(actions)
            
            # Store transition in replay buffer (Step 11)
            agent.buffer.push(states, actions, reward, next_states)
            
            states = next_states
            episode_reward += reward
            
            # Check if it's time to start training
            if global_step > TRAINING_START_STEP:
                # Sample batch and perform update (Steps 14-22)
                agent.learn(BATCH_SIZE)
            
            if done:
                break
        
        all_rewards.append(episode_reward)

        # 3. Reporting and Logging
        if episode % 10 == 0:
            avg_reward = sum(all_rewards[-10:]) / 10
            print(f"Episode: {episode:4d} | Global Step: {global_step:6d} | Avg Reward (10): {avg_reward:.2f} | Last Delay: {info['total_delay']:.2f} | Last Power: {info['total_power']:.2f}")

    print("Training Complete.")
    
if __name__ == '__main__':
    main()