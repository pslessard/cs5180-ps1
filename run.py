from env import ApartmentEnv
from policies import (OptimalPolicy, RandomPolicy, ThresholdPolicy)

import numpy as np
import pandas as pd

NUM_EPISODES = 10000

def run_episode(env, policy) -> int | np.int64:
    obs, _ = env.reset()

    final_reward = 0
    terminated = False

    while not terminated:
        action = policy.act(obs)

        obs, reward, terminated, _, _ = env.step(action)

        # reward will only ever be nonzero in a terminal state, so we can just overwrite
        final_reward = reward
    
    return final_reward

def evaluate_policy(env, policy):
    rewards = []

    for _ in range(NUM_EPISODES):
        reward = run_episode(env, policy)
        rewards.append(reward)
    
    return pd.Series(rewards, dtype=np.int64)

policies = {
    "Random": {"policy": RandomPolicy(4)},
    "Threshold_1": {"policy": ThresholdPolicy(1)},
    "Threshold_2": {"policy": ThresholdPolicy(2)},
    "Threshold_3": {"policy": ThresholdPolicy(3)},
    "Threshold_4": {"policy": ThresholdPolicy(4)},
    "Optimal": { "policy": OptimalPolicy()}
}

env = ApartmentEnv(4, 4)

for policy in policies.values():
    rewards = evaluate_policy(env, policy["policy"])

    policy["mean"] = rewards.mean()
    policy["std_error"] = rewards.sem()
    policy["full_rejection_ratio"] = rewards[rewards == 0].size / NUM_EPISODES

env.close()

df = pd.DataFrame(policies)
df = df.drop("policy")
print(df)