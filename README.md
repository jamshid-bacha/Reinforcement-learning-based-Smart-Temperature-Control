# 🧠 DDPG for Smart Temperature Control

This project implements a **Deep Deterministic Policy Gradient (DDPG)** agent to solve a custom reinforcement learning environment: **Smart Temperature Control**.

The agent learns how to maintain a comfortable indoor temperature while minimizing energy usage. This is a **continuous control problem** inspired by real-world HVAC (heating and cooling) systems.

---

# 🌡️ Environment: SmartTemperatureEnv

## 🎯 Objective

The goal of the agent is:

> Maintain the room temperature close to a target (22°C) while using minimal heating/cooling energy.

This creates a trade-off between:
- Comfort (temperature accuracy)
- Efficiency (energy usage)

---

# 📥 State Space

The agent observes:
[current_temperature, outside_temperature]


### 🧠 Why this state design?

- `current_temperature` → tells the agent the current condition  
- `outside_temperature` → tells the agent how the system will naturally evolve  

This allows the agent to:
- React to the current temperature  
- Anticipate future changes caused by the environment  

Without outside temperature, the agent would behave **reactively only**, not proactively.

This follows the **Markov property**:
> The current state contains enough information to predict the future.

---

# 🎮 Action Space
action ∈ [-1, 1]


- `+1` → maximum heating  
- `0` → no action  
- `-1` → maximum cooling  

### 🧠 Why continuous actions?

Real systems are not just ON/OFF:
- Heating and cooling can be adjusted smoothly  

DDPG is designed specifically for:
> Continuous action spaces

---

# ⚙️ Environment Dynamics

At each timestep:
current_temp += (action * heating_power) + (outside_temp - current_temp) * heat_transfer


### 🧠 Interpretation

Two forces influence the temperature:

1. **Agent control (heating/cooling)**
   - Directly controlled by the action  
   - Stronger action → stronger effect  

2. **Environment (outside temperature)**
   - Temperature naturally moves toward outside conditions  
   - Simulates heat loss or gain  

---

# 🏆 Reward Function
reward = -|current_temp - target_temp| - 0.1 * (action^2)


## 🧠 Why this reward design?

The reward combines two objectives:

### 1. Temperature accuracy
-|current_temp - target_temp|

- Penalizes deviation from 22°C  
- Encourages comfort  

### 2. Energy efficiency
-0.1 * (action^2)
- Penalizes large actions  
- Encourages smooth and efficient control  

---

## ⚖️ Trade-off

The agent learns:

> Keep the temperature stable while minimizing energy usage

---

## 🎯 What the agent learns

A well-trained agent will:
- Heat when temperature is too low  
- Cool when temperature is too high  
- Use minimal energy near the target  
- Anticipate outside temperature effects  

---

# ⏹️ Episode Design

- Fixed length (e.g., 200 steps)
- No early termination

### 🧠 Why?

This ensures:
- Stable training
- Learning long-term control behavior instead of short-term fixes

---

# 🤖 Algorithm: DDPG

## 📌 What is DDPG?

DDPG (Deep Deterministic Policy Gradient) is a reinforcement learning algorithm for **continuous action spaces**.

---

## 🔑 Key Components

- **Actor Network** → maps state → action  
- **Critic Network** → evaluates action quality (Q-value)  
- **Replay Buffer** → stores past experiences  
- **Target Networks** → stabilize training  

---

## ⚙️ How it works

1. Actor proposes an action  
2. Critic evaluates the action  
3. Actor updates to improve performance  
4. Experiences are reused via replay buffer  

---

## 🔊 Exploration

The agent explores using noise:
- Gaussian noise (simple)
- Ornstein–Uhlenbeck noise (smooth, better for control tasks)

---

# 🚀 How to Run

## 1. Install dependencies
pip install torch gym numpy matplotlib

---

## 2. Run training
python DDPG_Balance_Room_Heating.py


---

# 📊 Outputs

- `rewards.csv` → Episode rewards  
- Training plot → Reward vs Episode  

---

# 📁 Project Structure
├── DDPG_Balance_Room_Heating.py # Training script

├── smart_temp_env.py # Custom environment

├── rewards.csv # Training results

└── README.md



---

# 💡 Real-World Applications

This project demonstrates concepts used in:

- 🏠 Smart home temperature control  
- ⚡ Energy optimization systems  
- 🚗 Autonomous control systems  
- 🏭 Industrial process control  

---

# 🔧 Possible Improvements

- Use **TD3** (more stable than DDPG)  
- Normalize state inputs  
- Add realistic physics (thermal inertia, delays)  
- Add dynamic outside temperature  
- Log metrics using TensorBoard  

---

# 🧭 Key Takeaways

- Reinforcement learning can solve real-world control problems  
- Environment design is as important as the algorithm  
- Reward shaping strongly influences agent behavior  
- Continuous control requires specialized algorithms like DDPG  

---

# Reward and Result

<img width="800" alt="rewards" src="https://github.com/user-attachments/assets/59ffd787-daea-4fb8-b006-6d23a07c12af" />
# Reward and Result

<img src="https://github.com/user-attachments/assets/59ffd787-daea-4fb8-b006-6d23a07c12af" alt="Reward Result" width="500">
