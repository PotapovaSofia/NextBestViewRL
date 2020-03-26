import numpy as np
from trimesh.ray.ray_pyembree import RayMeshIntersector

class RaycastingImaging:
    def __init__(self, resolution_image=512, resolution_3d=0.02, projection="ortho"):
        self.resolution_image = resolution_image
        self.resolution_3d = resolution_3d
        self.projection = projection
        # self.fov = [115, 85, 80]

        self.rays_screen_coords, self.rays_origins, self.rays_directions = None, None, None

        
    def prepare(self, scanning_radius):
        # scanning radius is determined from the mesh extent
        self.rays_screen_coords, self.rays_origins, self.rays_directions = generate_rays(
            self.resolution_image, self.resolution_3d, radius=scanning_radius)

    def get_image(self, mesh): #, features):
        if any(value is None for value in [self.rays_screen_coords, self.rays_origins, self.rays_directions]):
            raise DataGenerationException('Raycasting was not prepared')

        # get a point cloud with corresponding indexes
        mesh_face_indexes, ray_indexes, points = ray_cast_mesh(
            mesh, self.rays_origins, self.rays_directions)

        # extract normals
        normals = mesh.face_normals[mesh_face_indexes]

        # compute indexes of faces and vertices in the original mesh
        # hit_surfaces_face_indexes = []
        # for idx, surface in enumerate(features['surfaces']):
        #     surface_face_indexes = np.array(surface['face_indices'])
        #     if np.any(np.isin(surface_face_indexes, mesh_face_indexes, assume_unique=True)):
        #         hit_surfaces_face_indexes.extend(surface_face_indexes)
        # mesh_face_indexes = np.unique(hit_surfaces_face_indexes)
        mesh_face_indexes = np.unique(mesh_face_indexes)
        mesh_vertex_indexes = np.unique(mesh.faces[mesh_face_indexes])
        return ray_indexes, points, normals, mesh_vertex_indexes, mesh_face_indexes

        # assemble mesh fragment into a submesh
        # nbhood = reindex_zerobased(mesh, mesh_vertex_indexes, mesh_face_indexes)
        # return ray_indexes, points, normals, nbhood, mesh_vertex_indexes, mesh_face_indexes

    def points_to_image(self, points, ray_indexes, assign_channels=None):
        xy_to_ij = self.rays_screen_coords[ray_indexes]
        # note that `points` can have many data dimensions
        if None is assign_channels:
            assign_channels = [2]
        data_channels = len(assign_channels)
        image = np.zeros((self.resolution_image, self.resolution_image, data_channels))
        image[xy_to_ij[:, 0], xy_to_ij[:, 1]] = points[:, assign_channels]
        return image.squeeze()




def generate_rays(image_resolution, resolution_3d, radius=1.0):
    """Creates an array of rays and ray directions used for mesh raycasting.
    :param image_resolution: image resolution in pixels
    :type image_resolution: int or tuple of ints
    :param resolution_3d: pixel 3d resolution in mm/pixel
    :type resolution_3d: float
    :param radius: Z coordinate for the rays origins
    :type resolution_3d: float
    :return:
        rays_screen_coords:     (W * H, 2): screen coordinates for the rays
        rays_origins:           (W * H, 3): world coordinates for rays origins
        ray_directions:         (W * H, 3): unit norm vectors pointing in directions of the rays
    """
    if isinstance(image_resolution, tuple):
        assert len(image_resolution) == 2
    else:
        image_resolution = (image_resolution, image_resolution)
    image_width, image_height = image_resolution

    # generate an array of screen coordinates for the rays
    # (rays are placed at locations [i, j] in the image)
    rays_screen_coords = np.mgrid[0:image_height, 0:image_width].reshape(
        2, image_height * image_width).T    # [h, w, 2]

    # place rays physically in locations determined by screen coordinates,
    # then shift so that camera origin is in the midpoint in the image,
    # and linearly stretch so each pixel takes exactly resolution_3d space
    screen_aspect_ratio = image_width / image_height
    rays_origins = (rays_screen_coords / np.array([[image_height, image_width]]))   # [h, w, 2], in [0, 1]
    factor = image_height / 2 * resolution_3d
    rays_origins[:, 0] = (-2 * rays_origins[:, 0] + 1) * factor  # to [-1, 1] + aspect transform
    rays_origins[:, 1] = (-2 * rays_origins[:, 1] + 1) * factor * screen_aspect_ratio
    rays_origins = np.concatenate([
        rays_origins,
        radius + np.zeros_like(rays_origins[:, [0]])
    ], axis=1)  # [h, w, 3]

    # ray directions are always facing towards Z axis
    ray_directions = np.tile(np.array([0, 0, -1]), (rays_origins.shape[0], 1))

    return rays_screen_coords, rays_origins, ray_directions


def ray_cast_mesh(mesh, rays_origins, ray_directions):
    intersector = RayMeshIntersector(mesh)
    index_triangles, index_ray, point_cloud = intersector.intersects_id(
        ray_origins=rays_origins, ray_directions=ray_directions,
        multiple_hits=False, return_locations=True)
    return index_triangles, index_ray, point_cloud


def make_noise(points, normals, scale=0.0, z_direction=None, **kwargs):
    assert z_direction is not None
    noise = np.random.normal(size=(len(points), 1), scale=scale) * z_direction
    noisy_points = points + noise * z_direction
    return noisy_points
