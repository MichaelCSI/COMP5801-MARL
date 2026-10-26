import numpy as np
import torch

from citylearn.citylearn import CityLearnEnv

# Configuration


DATASET = "citylearn_challenge_2020_climate_zone_1"
EPISODE_STEPS = 24


# 1. Check PyTorch / GPU



print("1. PYTORCH / GPU CHECK")


print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU: CPU only")

# 2. Create decentralized CityLearn environment



print("2. CITYLEARN ENVIRONMENT CHECK")


env = CityLearnEnv(
    DATASET,
    central_agent=False,
    episode_time_steps=EPISODE_STEPS
)

num_agents = len(env.action_space)

print("Dataset:", DATASET)
print("Number of agents:", num_agents)

assert num_agents > 1, "The environment is not multi-agent."

print("Decentralized multi-agent environment: OK")

# 3. Inspect observations and action spaces



print("3. AGENT OBSERVATION / ACTION CHECK")


obs, info = env.reset()

for i in range(num_agents):

    observation_size = len(obs[i])
    action_space = env.action_space[i]

    print(
        f"Agent {i}: "
        f"observation={observation_size}, "
        f"action_shape={action_space.shape}, "
        f"action_low={action_space.low}, "
        f"action_high={action_space.high}"
    )

# 4. Run decentralized joint actions



print("4. MULTI-AGENT INTERACTION CHECK")


total_rewards = np.zeros(num_agents)

for timestep in range(EPISODE_STEPS):

    # Each agent independently selects an action
    actions = [
        env.action_space[i].sample()
        for i in range(num_agents)
    ]

    # Joint action is submitted to CityLearn
    next_obs, rewards, terminated, truncated, info = env.step(actions)

    # Store cumulative reward for every building
    total_rewards += np.asarray(rewards, dtype=np.float32)

    print(
        f"Step {timestep + 1:02d}: "
        f"rewards={[round(float(r), 2) for r in rewards]}"
    )

    if terminated or truncated:
        print("Episode ended early.")
        break

    obs = next_obs


# 5. Results


print("5. RESULTS")


for i, reward in enumerate(total_rewards):
    print(f"Agent {i} cumulative reward: {reward:.3f}")

print("\nEnvironment interaction: PASSED")
print("Multi-agent decentralized control: PASSED")

env.close()


print("CITYLEARN MARL ENVIRONMENT TEST PASSED")
