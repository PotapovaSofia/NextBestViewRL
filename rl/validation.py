import os
import numpy as np
from rl.environment import *
from tqdm import tqdm

def compute_metrics(env, agent, iter_cnt=10, max_iter=30):
    rewards, final_rewards, novp = [], [], []
    for _ in range(iter_cnt):
        state, action, mask = env.reset()
        episode_reward = 0.0
        for t in range(max_iter):
            action =  agent.act(state, mask)
            state, reward, done, info, mask = env.step(action)
            episode_reward += reward

            if done:
                break
        
        rewards.append(episode_reward)
        # final_reward = env.final_reward()
        final_reward = 0
        final_rewards.append(final_reward)
        novp.append(t + 1)
    return np.mean(rewards), np.mean(final_rewards), np.mean(novp)


def create_env(model_path=None):
    env = Environment(illustrate=False, 
                      model_path=model_path,
                      image_size=128,
                      number_of_view_points=100)
    
    # env = CombiningObservationsWrapper(env)
    # env = StepPenaltyRewardWrapper(env, weight=1.0)
    # env = DepthMapWrapper(env)

    env = MeshReconstructionWrapper(env, reconstruction_depth=7)
    env = VoxelGridWrapper(env)
    env = CombiningObservationsWrapper(env)
    env = VoxelWrapper(env)
    env = StepPenaltyRewardWrapper(env)
    env = FrameStackWrapper(env, num_stack=4, lz4_compress=False)
    env = ActionMaskWrapper(env)
    return env


def validate(agent, models_path="./data/1kabc/simple/val/",
             max_iter=50, iter_cnt=1):
    metrics = []
    for model_path in tqdm(sorted(os.listdir(models_path))):
        env = create_env(os.path.join(models_path, model_path))
        result = compute_metrics(env, agent, max_iter=max_iter,
                                 iter_cnt=iter_cnt)
        metrics.append(result)
    return np.mean(metrics, axis=0)
