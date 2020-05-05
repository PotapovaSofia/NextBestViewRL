import numpy as np
import k3d
from geometry.utils.visualisation import illustrate_voxels


class VoxelGrid:
    def __init__(self, surface_grid=None, occlusion_grid=None,
                 free_grid=None, size=64):

        if surface_grid is None:
            self.surface_grid = np.zeros((size, size, size))
        else:
            self.surface_grid = surface_grid
        if occlusion_grid is None:
            self.occlusion_grid = np.zeros((size, size, size))
        else:
            self.occlusion_grid = occlusion_grid

        if free_grid is None:
            self.free_grid = np.zeros((size, size, size))
        else:
            self.free_grid = free_grid

        self._size = size
        self._surface_id = 2
        self._occlusion_id = 1
        self._free_id = -1

    def __add__(self, other):
        occluded_voxels = np.logical_and(self.occlusion_grid, other.occlusion_grid)
        occluded_voxels = occluded_voxels.astype(int)
        occluded_voxels *= self._occlusion_id

        surface_voxels = np.logical_or(self.surface_grid, other.surface_grid)
        surface_voxels = surface_voxels.astype(int)
        surface_voxels *= self._surface_id

        free_voxels = np.logical_or(self.free_grid, other.free_grid)
        free_voxels = free_voxels.astype(int)
        free_voxels *= self._free_id

        return VoxelGrid(surface_voxels, occluded_voxels, free_voxels)

    def grid(self):
        grid = self.surface_grid + self.occlusion_grid
        grid = np.clip(grid, 0, max(self._occlusion_id, self._surface_id))
        free_mask = np.logical_and(self.free_grid == self._free_id,
                              grid != self._surface_id)
        grid[free_mask] = 0
        return grid

    def illustrate(self, plot=None):
        if plot is None:
            plot = k3d.plot()

        return illustrate_voxels(self.grid(), plot)


class VoxelGridBuilder:
    def __init__(self, size=64):
        self._size = size
        self._shape = (size, size, size)
        self._surface_id = 2
        self._occlusion_id = 1
        self._free_indices = -1
        self._bounds_eps = 0.1

    def build(self, points, bounds, direction=None):

        surface_indices = self._get_surface_indices(points, bounds)
        surface_grid = self._get_grid_from_indices(surface_indices,
                                                   id=self._surface_id)

        occluded_grid = None
        free_grid = None
        if direction is not None:
            occluded_indices = self._get_space_indices(surface_indices, direction, n=1)
            occluded_grid = self._get_grid_from_indices(occluded_indices,
                                                        id=self._occlusion_id)
            free_indices = self._get_space_indices(surface_indices, -direction, n=3)
            free_grid = self._get_grid_from_indices(free_indices, id=self._free_indices)
        return VoxelGrid(surface_grid, occluded_grid, free_grid)

    def _get_surface_indices(self, points, bounds):
        indices = (((points - bounds[0]) * self._shape) /
                   (bounds[1] - bounds[0])).astype(np.int32)
        indices = np.clip(indices, 0, self._size - 1)
        # mask = np.all(np.logical_and(indices >= 0, indices < self._size), axis=1)
        # indices = indices[mask]
        if len(indices) > 0:
            indices = np.unique(indices, axis=0)
        return indices

    def _get_space_indices(self, surface_indices, direction, n=1):
        if len(surface_indices) == 0:
            return []
        cum_sums = np.tile(np.arange(self._size * n),
                           (len(surface_indices), 3, 1)) \
                          .transpose(0, 2, 1) \
                          * direction / n
        cum_sums = np.floor(cum_sums).astype(int)

        occluded_indices = np.tile(surface_indices, (self._size * n, 1, 1)) \
                           .transpose(1, 0, 2)
        occluded_indices += cum_sums
        occluded_indices = np.concatenate(occluded_indices, axis=0)

        mask = np.all(np.logical_and(occluded_indices >= 0,
                                     occluded_indices < self._size), axis=1)
        occluded_indices = occluded_indices[mask]

        return occluded_indices

    def _get_grid_from_indices(self, indices, id=1):
        grid = np.zeros(self._shape)
        
        if len(indices) > 0:
            flat_index_array = np.ravel_multi_index(
                indices.transpose(),
                grid.shape)
            np.ravel(grid)[flat_index_array] = id
        return grid
