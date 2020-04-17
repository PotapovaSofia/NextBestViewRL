import numpy as np
import itertools
import trimesh
import math
import k3d
from time import sleep
from tqdm import tqdm
import torch
from rl.dqn import *

USE_CUDA = torch.cuda.is_available()


def compute_metrics(env, agent_func, iter_cnt=10, max_iter=30):
    rewards, final_rewards, novp = [], [], []
    for _ in range(iter_cnt):
        state, action = env.reset()
        episode_reward = 0.0
        for t in range(max_iter):
            action = agent_func(state)
            state, reward, done, info = env.step(action)
            # print("REWARD: ", reward)
            env.render(action, state)
            episode_reward += reward

            if done:
                break

        final_reward = 0.0
        # final_reward = env.final_reward()
        # episode_reward += 1.0 / final_reward
        rewards.append(episode_reward)
        final_rewards.append(final_reward)
        novp.append(t + 1)
    return np.mean(rewards), np.mean(final_rewards), np.mean(novp)

from rl.environment import *

def create_env(model_path=None):
    env = Environment(illustrate=False, models_path="./data/10abc/",
                      model_path=model_path, number_of_view_points=100, image_size=512)
    env = CombiningObservationsWrapper(env)
    env = StepPenaltyRewardWrapper(env)
    env = DepthMapWrapper(env, memory_size=10)
    return env


# env = create_env()

# model = torch.load("./models/10abc-100vp34500.pt")
# if USE_CUDA:
#     model = model.cuda()

# agent_func = lambda s : model.act(s, epsilon=0.0)


models_path = "./data/1kabc/val/"
result = {}

for model_path in tqdm(sorted(os.listdir(models_path))):
    agent_func = lambda s : env.action_space.sample()
    env = create_env(os.path.join(models_path, model_path))
    result[model_path] = compute_metrics(env, agent_func, max_iter=30)
    print(model_path, result[model_path])

a, b, c = [], [], []
for name, res in result.items():
    a.append(res[0])
    b.append(res[1])
    c.append(res[2])

print("MEAN RESULT")
print(np.mean(a), np.mean(b), np.mean(c))
