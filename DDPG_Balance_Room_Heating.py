import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
from collections import deque
import csv
import matplotlib.pyplot as plt
from smart_temp_env import SmartTemperatureEnv


# =====================
# Hyperparameters
# =====================
ENV_NAME = "Pendulum-v1"
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
BUFFER_SIZE = 100000
BATCH_SIZE = 64
EPISODES = 300
MAX_STEPS = 200
NOISE_STD = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Replay Buffer
# =====================
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# =====================
# Actor Network
# =====================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

# =====================
# Critic Network
# =====================
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

# =====================
# DDPG Agent
# =====================
class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer):
        if len(replay_buffer) < BATCH_SIZE:
            return

        state, action, reward, next_state, done = replay_buffer.sample(BATCH_SIZE)

        state = torch.FloatTensor(state).to(DEVICE)
        action = torch.FloatTensor(action).to(DEVICE)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(DEVICE)
        next_state = torch.FloatTensor(next_state).to(DEVICE)
        done = torch.FloatTensor(done).unsqueeze(1).to(DEVICE)

        # Target Q
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * GAMMA * target_Q

        # Critic loss
        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)



class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


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