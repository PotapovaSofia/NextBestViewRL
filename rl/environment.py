import numpy as np
import os
import random
import gym
import k3d
from gym import spaces
from pypoisson import poisson_reconstruction

from geometry.model import Model, combine_observations, get_mesh
from geometry.voxel_grid import VoxelGrid
from geometry.utils.visualisation import illustrate_points, illustrate_mesh, illustrate_voxels


class EnvError(Exception):
    pass


class Environment(gym.Env):
    def __init__(self,
                 models_path=None,
                 model_path=None,
                 number_of_view_points=100,
                 similarity_threshold=0.95,
                 image_size=512,
                 illustrate=False):
        super().__init__()

        self.model_path = model_path
        self.models_path = models_path
        self.number_of_view_points = number_of_view_points
        self.image_size = image_size
        self.illustrate = illustrate

        self.action_space = spaces.Discrete(number_of_view_points)
        self.observation_space = spaces.Dict({
            # 'points':  spaces.Box(-np.inf, np.inf, (MAX_POINS_CNT, 3), dtype=np.float32),
            # 'normals': spaces.Box(-np.inf, np.inf, (MAX_POINS_CNT, 3), dtype=np.float32),
            # 'vertex_indexes': spaces.Box(-np.inf, np.inf, (MAX_POINS_CNT, 3), dtype=np.int32),
            # 'face_indexes': spaces.Box(-np.inf, np.inf, (MAX_POINS_CNT, 3), dtype=np.int32),
            'depth_map': spaces.Box(-np.inf, np.inf, (image_size, image_size), dtype=np.float32),
            # 'normals_image': spaces.Box(-np.inf, np.inf, (VIEW_POINTS_CNT, image_size, image_size), dtype=np.float32)
        })

        self._similarity_threshold = similarity_threshold
        self._reconstruction_depth = 10

        self.model = None
        self.plot = None

    def reset(self):
        """
        Reset the environment for new episode.
        Randomly (or not) generate CAD model for this episode.
        """
        if self.model_path is not None:
            model_path = self.model_path
        elif self.models_path is not None:
            model_path = os.path.join(self.models_path,
                                      random.sample(os.listdir(self.models_path), 1)[0])
        self.model = Model(model_path, resolution_image=self.image_size)
        self.model.generate_view_points(self.number_of_view_points)
        
        if self.illustrate:
            self.model.illustrate().display()
        
        init_action = self.action_space.sample()
        observation = self.model.get_observation(init_action)
        return observation, init_action

    def step(self, action):
        """
        Get new observation from current position (action), count step reward, decide whether to stop.
        Args:
            action: int
        return: 
            next_state: List[List[List[int, int, int]]]
            reward: float
            done: bool
            info: Tuple
        """
        assert self.action_space.contains(action)
        observation = self.model.get_observation(action)

        reward = self.step_reward(observation)
        done = reward >= self._similarity_threshold

        return observation, reward, done, {}
    
    def render(self, action, observation, plot=None):
        if plot is None:
            plot = self.plot
            
        plot = illustrate_points(
           [self.model.get_point(action)], size=0.5, plot=plot)
        
        plot = observation.illustrate(plot, size=0.03)
        return plot
    
    def step_reward(self, observation):
        # THINK ABOUT yet another reward
        return self.model.observation_similarity(observation)
    
    def final_reward(self, observation):
        vertices, faces = self._get_mesh(observation)
        reward = self.model.surface_similarity(vertices, faces)
        
        if self.illustrate:
            illustrate_mesh(vertices, faces).display()
        return reward
        
    def _get_mesh(self, observation):
        faces, vertices = poisson_reconstruction(observation.points,
                                                 observation.normals,
                                                 depth=self._reconstruction_depth)
        return vertices, faces

    
class StepPenaltyRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, weight=1.0):
        super().__init__(env)
        
        self._similarity_reward_weight = weight
            
    def reward(self, reward):
        # THINK ABOUT
        reward = -1 + self._similarity_reward_weight * reward
        return reward
    
    def render(self, action, observation):
        self.env.render(action, observation)
        
    def final_reward(self):
        return self.env.final_reward()

    
class DepthMapWrapper(gym.ObservationWrapper):
    def __init__(self, env, memory_size=10):
        super().__init__(env)
        
        self.image_size = self.observation_space['depth_map'].shape[0]
        self.memory_size = memory_size
        
        self.observation_space = spaces.Box(-np.inf, np.inf,
                (memory_size, self.image_size, self.image_size), dtype=np.float32)
        
        self.last_observation = None

    def reset(self):
        observation, action = self.env.reset()
        return self.observation(observation), action

    def observation(self, observation):
        self.last_observation = observation

        depth_map = observation.depth_map
        assert depth_map.ndim == 3

        new_size = self.memory_size - depth_map.shape[0]
        if new_size >= 0:
            depth_map = np.pad(depth_map,
                               ((0,new_size),(0,0), (0,0)),
                               mode='constant')
        else:
            depth_map = depth_map[-new_size:]
            
        return depth_map
    
    def render(self, action, observation):
        if self.last_observation is not None:
            self.env.render(action, self.last_observation)
    
    def final_reward(self):
        return self.env.final_reward()


class VoxelGridWrapper(gym.ObservationWrapper):
    def __init__(self, env, grid_shape=(64, 64, 64)):
        super().__init__(env)

        self.grid_shape = grid_shape
        self.observation_space = spaces.Box(0, 1, grid_shape, dtype=bool)

        self.mesh_grid = None
        self.gt_size = None
        self.bounds = None
        self.plot = k3d.plot(name='wrapper')

    def reset(self):
        observation, action = self.env.reset()
        self.bounds = self.env.model.mesh.bounds
        
        self.mesh_grid = VoxelGrid()
        self.mesh_grid.build(self.env.model.mesh.vertices, self.bounds)
        self.gt_size = np.count_nonzero(self.mesh_grid.surface_grid)

        return self.observation(observation), action

    def observation(self, observation):
        grid = VoxelGrid()
        grid.build(observation.points, self.bounds, observation.occluded_points)
        return grid

    def render(self, action, observation):
        self.plot.close()
        # self.plot = observation.illustrate()
        self.plot = illustrate_voxels(observation)
        self.plot.display()

    def final_reward(self):
        return self.env.final_reward()

    def step_reward(self, observation):
        intersection = np.count_nonzero(np.logical_and(self.mesh_grid.surface_grid,
                                                       observation.surface_grid))
        return intersection / self.gt_size 


class CombiningObservationsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self._similarity_threshold = 0.95

        self.combined_observation = None

    def reset(self):
        observation, action = self.env.reset()
        self.combined_observation = observation

        return self.combined_observation, action

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        self._combine_observations(observation)

        combined_reward = self.env.step_reward(self.combined_observation)
        done = done or combined_reward >= self._similarity_threshold

        new_reward = combined_reward - reward
        print(reward, new_reward, combined_reward)
        return self.combined_observation, new_reward, done, info

    def render(self, action, observation):
        self.env.render(action, observation)

    def final_reward(self):
        return self.env.final_reward(self.combined_observation)

    def _combine_observations(self, observation):
        if self.combined_observation is None:
            raise EnvError("Environment wasn't reset")
        self.combined_observation += observation


class VoxelWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.grid_shape = self.env.grid_shape
        self.observation_space = spaces.Box(0, 2, self.grid_shape, dtype=np.uint32)

    def reset(self):
        observation, action = self.env.reset()
        return self.observation(observation), action

    def observation(self, observation):
        return observation.grid()

    def render(self, action, observation):
        self.env.render(action, observation)

    def final_reward(self):
        return self.env.final_reward()

