import numpy as np
import k3d
from geometry.utils.visualisation import illustrate_voxels

class VoxelGrid:
    def __init__(self, surface_grid=None, occlusion_grid=None,
                 size=(64, 64, 64)):
        self.size = size
        
        self.surface_id = 2
        self.occlusion_id = 1
        
        self.surface_grid = surface_grid
        self.occlusion_grid = occlusion_grid
        
    def __add__(self, other):
        occluded_voxels = np.logical_and(self.occlusion_grid, other.occlusion_grid)
        occluded_voxels = occluded_voxels.astype(int)
        occluded_voxels *= self.occlusion_id

        surface_voxels = np.logical_or(self.surface_grid, other.surface_grid)
        surface_voxels = surface_voxels.astype(int)
        surface_voxels *= self.surface_id
        
        return VoxelGrid(surface_voxels, occluded_voxels)

    def build(self, surface_points, bounds, occluded_points=None):
        self.surface_grid = self._build_grid(surface_points,
                                             bounds,
                                             self.surface_id)
        if occluded_points is not None:
            self.occlusion_grid = self._build_grid(occluded_points,
                                                   bounds,
                                                   self.occlusion_id)
        else:
            self.occlusion_grid = np.zeros(self.size)

    def illustrate(self, plot=None):
        if plot is None:
            plot = k3d.plot()

        return illustrate_voxels(self.grid(), plot)

    def grid(self):
        grid = self.surface_grid + self.occlusion_grid
        grid = np.clip(grid, 0, max(self.occlusion_id, self.surface_id))
        return grid

    def _build_grid(self, points, bounds, id=1):
        grid = np.zeros(self.size)

        indices = (((points - bounds[0]) * self.size) /
                   (bounds[1] - bounds[0])).astype(np.int32)
        mask = np.all(np.logical_and(indices >= 0, indices < 64), axis=1)
        indices = indices[mask]
        # indices = np.unique(indices, axis=0)

        for ind in indices:
            grid[ind[0], ind[1], ind[2]] = id
        return grid

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

