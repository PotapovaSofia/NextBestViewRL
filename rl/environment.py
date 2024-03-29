import numpy as np
import os
import random
import gym
import k3d
from gym import spaces
from gym.wrappers import FrameStack, LazyFrames
from pypoisson import poisson_reconstruction

from geometry.model import Model, combine_observations, get_mesh
from geometry.voxel_grid import VoxelGrid, VoxelGridBuilder
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

    def reset(self, init_action=None):
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

            self.plot = k3d.plot()
            self.plot.display()
        
        if init_action is None:
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
    
        if self.illustrate:
            illustrate_mesh(vertices, faces).display()
        return reward
        
    def _get_mesh(self, observation):
        faces, vertices = poisson_reconstruction(observation.points,
                                                 observation.normals,
                                                 depth=self._reconstruction_depth)
        return vertices, faces

    def final_reward(self, observation):
        vertices, faces = self._get_mesh(observation)
        reward = self.model.surface_similarity(vertices, faces)
        return reward


class MeshReconstructionWrapper(gym.Wrapper):
    def __init__(self, env, reconstruction_depth=8, final_depth=10,
                 scale_factor=1, final_scale_factor=1, done_thresh=0.1,
                 do_step_reconstruction=False, illustrate=False):
        super().__init__(env)

        self.points = []
        self.normals = []

        self._depth = reconstruction_depth
        self._final_depth = final_depth
        self._scale_factor = scale_factor
        self._final_scale_factor = final_scale_factor
        self._done_thresh = done_thresh

        self._do_step_reconstruction = do_step_reconstruction
        self._illustrate = illustrate

    def reset(self, init_action=None):
        observation, action = self.env.reset(init_action)
        self.points = [observation.points]
        self.normals = [observation.normals]

        return observation, action

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        self.points.append(observation.points[::self._final_scale_factor])
        self.normals.append(observation.normals[::self._final_scale_factor])

        if self._do_step_reconstruction:
            done = done or self.done()

        return observation, reward, done, info

    def render(self, action, observation):
        self.env.render(action, observation)

    def final_reward(self, observation):
        points, normals = self.get_combined_points()
        reward = self.compute_reward(points, normals, self._final_depth)
        return reward

    def done(self):
        points, normals = self.get_combined_points()
        step_reward = self.compute_reward(points[::self._scale_factor],
                                          normals[::self._scale_factor],
                                          depth=self._depth)
        return step_reward < self._done_thresh

    def compute_reward(self, points, normals, depth):
        faces, vertices = poisson_reconstruction(points, normals, depth=depth)
        reward = self.env.model.surface_similarity(vertices, faces)

        if self._illustrate:
            illustrate_mesh(vertices, faces).display()
        return reward

    def get_combined_points(self):
        points = np.concatenate(self.points)
        normals = np.concatenate(self.normals)
        return points, normals


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

    def reset(self, init_action=None):
        observation, action = self.env.reset(init_action)
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
    def __init__(self, env, grid_size=64, occlusion_reward=False):
        super().__init__(env)

        self.grid_shape = ((grid_size, grid_size, grid_size))
        self.observation_space = spaces.Box(0, 1, self.grid_shape, dtype=bool)

        self.builder = VoxelGridBuilder(grid_size)

        self.mesh_grid = None
        self.gt_size = None
        self.bounds = None
        self.plot = k3d.plot(name='wrapper')

    def reset(self, init_action=None):
        observation, action = self.env.reset(init_action)
        self.bounds = self.env.model.bounds
        
        self.mesh_grid = self.builder.build(self.env.model.mesh.vertices, self.bounds)
        self.gt_size = np.count_nonzero(self.mesh_grid.surface_grid)

        return self.observation(observation), action

    def observation(self, observation):
        return self.builder.build(observation.points, self.bounds, observation.direction)

    def render(self, action, observation):
        self.plot.close()
        self.plot = illustrate_voxels(observation)
        self.plot.display()

    def final_reward(self, observation):
        return self.env.final_reward(observation)

    def step_reward(self, observation):
        intersection = np.count_nonzero(np.logical_and(self.mesh_grid.surface_grid,
                                                       observation.surface_grid))
        return intersection / self.gt_size


class CombiningObservationsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self._similarity_threshold = 0.95

        self.combined_observation = None

    def reset(self, init_action=None):
        observation, action = self.env.reset(init_action)
        self.combined_observation = observation

        return self.combined_observation, action

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        self._combine_observations(observation)

        combined_reward = self.env.step_reward(self.combined_observation)
        done = done or combined_reward >= self._similarity_threshold

        new_reward = combined_reward - reward
        # print(reward, new_reward, combined_reward)
        return self.combined_observation, new_reward, done, info

    def render(self, action, observation):
        self.env.render(action, observation)

    def final_reward(self):
        return self.env.final_reward(self.combined_observation)

    def _combine_observations(self, observation):
        if self.combined_observation is None:
            raise EnvError("Environment wasn't reset")
        self.combined_observation += observation


class VoxelWrapper(gym.Wrapper):
    def __init__(self, env, occlusion_reward=False, weight=10.):
        super().__init__(env)

        self.grid_shape = self.env.grid_shape
        self.observation_space = spaces.Box(0, 2, self.grid_shape, dtype=np.uint8)

        self._occlusion_reward = occlusion_reward
        self._weight = weight
        self._grid_size = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]

    def reset(self, init_action=None):
        observation, action = self.env.reset(init_action)
        return self.observation(observation), action

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        reward -= self.step_reward(observation)
        # print(reward)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        return observation.grid()

    def render(self, action, observation):
        self.env.render(action, observation)

    def final_reward(self):
        return self.env.final_reward()

    def step_reward(self, observation):
        occlusion_reward = 0
        if self._occlusion_reward:
            occlusion_reward = np.count_nonzero(
                observation.grid()==observation._occlusion_id)
            occlusion_reward *= self._weight
            occlusion_reward /= self._grid_size
        return occlusion_reward


class FrameStackWrapper(FrameStack):
    def __init__(self, env, num_stack, lz4_compress=False):
        super().__init__(env, num_stack, lz4_compress)

        low = np.concatenate(self.observation_space.low)
        high = np.concatenate(self.observation_space.high)
        self.observation_space = spaces.Box(low=low, high=high,
                                            dtype=self.observation_space.dtype)

    def _get_observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        observation = LazyFrames(list(self.frames), self.lz4_compress)
        return np.concatenate(observation)

    def reset(self, **kwargs):
        observation, action = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return self._get_observation(), action

    def render(self, action, observation):
        self.env.render(action, observation)

    def final_reward(self):
        return self.env.final_reward()


class ActionMaskWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.mask = None
        
    def reset(self, **kwargs):
        observation, action = self.env.reset(**kwargs)
        self.mask = np.ones(self.number_of_view_points).astype(bool)
        self.mask[action] = False
        return observation, action, self.mask
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        self.mask[action] = False
        done = done or self.done()
        return observation, reward, done, info, self.mask

    def done(self):
        return np.count_nonzero(self.mask) == 0

    def render(self, action, observation):
        self.env.render(action, observation)

    def final_reward(self):
        return self.env.final_reward()
