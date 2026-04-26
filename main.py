import csv
from matplotlib import pyplot as plt
from DDPG_agent import DDPG
from buffer import ReplayBuffer
from ouNoise import OUNoise
from smart_temp_env import SmartTemperatureEnv




EPISODES = 300
BUFFER_SIZE = 100000
MAX_STEPS = 200

# =====================
# Training Loop
# =====================
def main():
    env = SmartTemperatureEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPG(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    rewards_history = []

    # noise
    noise = OUNoise(action_dim)

    for episode in range(EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        noise.reset()

        for step in range(MAX_STEPS):
            action = agent.select_action(state)
            noise_scale = max(0.1, 1 - episode / EPISODES)
            action = (action + noise_scale * noise.sample()).clip(
                env.action_space.low, env.action_space.high
            )

            next_state, reward, done, truncated, _ = env.step(action)
            done_bool = float(done or truncated)

            replay_buffer.push(state, action, reward, next_state, done_bool)

            state = next_state
            episode_reward += reward

            agent.train(replay_buffer)

            if done or truncated:
                break

        rewards_history.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")

    env.close()

    # =====================
    # Save to CSV
    # =====================
    with open("rewards.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(rewards_history):
            writer.writerow([i + 1, r])

    print("Saved rewards to rewards.csv")

    # =====================
    # Plot rewards
    # =====================
    fig, ax = plt.subplots(figsize=(11.5 / 2.54, 8.5 / 2.54))  # size in inches (converted from cm)
    ax.plot(rewards_history)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("DDPG Room Temperature Rewards")
    fig.tight_layout()
    fig.savefig("rewards.png", dpi=300, bbox_inches="tight")

    # plt.show()

if __name__ == "__main__":
    main()