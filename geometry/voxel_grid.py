import numpy as np
import k3d
from geometry.utils.visualisation import illustrate_voxels

class VoxelGrid:
    def __init__(self, points=None, bounds=None, direction=None,
                 surface_grid=None, occlusion_grid=None, size=64):
        
        self.surface_grid = surface_grid
        self.occlusion_grid = occlusion_grid

        self._size = size
        self._surface_id = 2
        self._occlusion_id = 1

        if surface_grid is None and points is not None:
            self.surface_grid, self.occlusion_grid = self._build(points, bounds, direction)

        
    def __add__(self, other):
        occluded_voxels = np.logical_and(self.occlusion_grid, other.occlusion_grid)
        occluded_voxels = occluded_voxels.astype(int)
        occluded_voxels *= self._occlusion_id

        surface_voxels = np.logical_or(self.surface_grid, other.surface_grid)
        surface_voxels = surface_voxels.astype(int)
        surface_voxels *= self._surface_id
        
        return VoxelGrid(surface_grid=surface_voxels, occlusion_grid=occluded_voxels)

    def _build(self, points, bounds, direction=None):
        surface_grid = np.zeros((self._size, self._size, self._size))
        occlusion_grid = np.zeros((self._size, self._size, self._size))

        indices = (((points - bounds[0]) * surface_grid.shape) /
                   (bounds[1] - bounds[0])).astype(np.int32)
        mask = np.all(np.logical_and(indices >= 0, indices < 64), axis=1)
        indices = indices[mask]
        # indices = np.unique(indices, axis=0)

        for ind in indices:
            grid[ind[0], ind[1], ind[2]] = self._surface_id

        if direction is not None:
            occluded_indices = self._get_occluded_grid_indices(indices, direction)
            for ind in occluded_indices:
                if occlusion_grid[ind[0], ind[1], ind[2]] != self._surface_id:
                    occlusion_grid[ind[0], ind[1], ind[2]] = self._occlusion_id
        return surface_grid, occlusion_grid

    def illustrate(self, plot=None):
        if plot is None:
            plot = k3d.plot()

        return illustrate_voxels(self.grid(), plot)

    def grid(self):
        grid = self.surface_grid + self.occlusion_grid
        grid = np.clip(grid, 0, max(self._occlusion_id, self._surface_id))
        return grid

    def _get_occluded_grid_indices(self, surface_indices, direction, n=2):
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

    def _fill_holes(self, arr):
        first, last = None, None
        for i in range(len(arr)):
            if arr[i] == 1:
                first = i
                break
        for i in range(len(arr) - 1, 0, -1):
            if arr[i] == 1:
                last = i
                break
        if first is not None and last is not None:
            arr[first:last + 1] = 1
        return arr

