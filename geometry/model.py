import numpy as np
import trimesh

from pypoisson import poisson_reconstruction
from igl import hausdorff
# from pytorch3d.loss import chamfer_distance

from geometry.utils.raycasting import RaycastingImaging, make_noise
from geometry.utils.sampling import generate_sunflower_sphere_points
from geometry.utils.transform import from_euler, transform_mesh, transform_points
from geometry.utils.visualisation import illustrate_points, illustrate_mesh



class ViewPoint:
    def __init__(self, point, phi, theta):
        self.point = point
        self.phi = phi
        self.theta = theta
        
    def get_transform_matrix(self):
        return from_euler(-self.phi, -self.theta)
        

class Observation:
    def __init__(self, points, occluded_points, normals,
                 vertex_indexes, face_indexes, depth_map,
                 normals_image):
        self.points = points
        self.occluded_points = occluded_points
        self.normals = normals
        self.vertex_indexes = vertex_indexes
        self.face_indexes = face_indexes
        
        if len(depth_map.shape) == 2:
            self.depth_map = np.expand_dims(depth_map, 0)
            self.normals_image = np.expand_dims(normals_image, 0)
        else:
            self.depth_map = depth_map
            self.normals_image = normals_image

    def __del__(self):
        del self.points
        del self.normals
        del self.vertex_indexes
        del self.face_indexes
        del self.depth_map
        del self.normals_image

    def transform(self, transform):
        self.points = transform_points(self.points, transform)
        self.occluded_points = transform_points(self.occluded_points,
                                                transform)
        self.normals = transform_points(self.normals, transform,
                                        translation=None)
        
    def __add__(self, other):
        points = np.concatenate([self.points, other.points])
        occluded_points = np.concatenate([self.occluded_points,
                                          other.occluded_points])
        normals = np.concatenate([self.normals, other.normals])
        
        vertex_indexes = np.unique(np.concatenate([self.vertex_indexes,
                                                   other.vertex_indexes]))
        face_indexes = np.unique(np.concatenate([self.face_indexes,
                                                 other.face_indexes]))
        
        depth_map = np.concatenate([self.depth_map, other.depth_map])
        normals_image = np.concatenate([self.normals_image, other.normals_image])
        
        return Observation(points, occluded_points, normals,
                           vertex_indexes, face_indexes,
                           depth_map, normals_image)
        
    def illustrate(self, plot=None, size=0.05):
        return illustrate_points(self.points, plot=plot, size=size)
    
    
class Model:
    def __init__(self, model_path, resolution_image=512):
        self.mesh = self.load_mesh(model_path)
        self.bounds = self.mesh.bounds
        self.transform = np.eye(4)
        self.resolution_image = resolution_image
 
        self.raycaster = self.prepare_raycaster()
        
        self.view_points = []

    def __del__(self):
        del self.mesh
        del self.raycaster
    
    def load_mesh(self, mesh_path, shape_fabrication_extent=10.0):
        mesh = trimesh.load_mesh(mesh_path)
        mesh_extent = np.max(mesh.bounding_box.extents)
        mesh = mesh.apply_scale(shape_fabrication_extent / mesh_extent)
        # TODO compute lengths of curves + quantiles
        mesh = mesh.apply_translation(-mesh.vertices.mean(axis=0))
        return mesh
    
    def prepare_raycaster(self):
        raycaster = RaycastingImaging(resolution_image=self.resolution_image)
        raycaster.prepare(scanning_radius=np.max(self.mesh.bounding_box.extents) + 1.0)
        return raycaster
    
    def generate_view_points(self, num_points=100):
        sphere_points, phis, thetas = generate_sunflower_sphere_points(num_points)

        dists = self.mesh.vertices - self.mesh.center_mass
        radius = np.abs(dists).max()
        radius *= 1.20

        sphere_points *= radius
        sphere_points += self.mesh.center_mass

        for point, phi, theta in zip(sphere_points, phis, thetas):
            self.view_points.append(ViewPoint(point, phi, theta))
            
            
    def get_point(self, view_point_idx):
        return self.view_points[view_point_idx].point
            

    def illustrate(self):
        plot = illustrate_mesh(self.mesh.vertices, self.mesh.faces)
        plot = illustrate_points([vp.point for vp in self.view_points],
                                 plot, size=0.1)
        return plot


    def rotate_to_view_point(self, view_point):
        self.transform = view_point.get_transform_matrix()
        self.mesh = transform_mesh(self.mesh, self.transform, reverse=True)
        
        
    def rotate_to_origin(self):
        self.mesh = transform_mesh(self.mesh, self.transform, reverse=False)
        self.transform = np.eye(4)

        
    def raycast(self, visibility_eps = 1e-6):
        
        (ray_indexes,
         points,
         normals,
         mesh_vertex_indexes,
         mesh_face_indexes) = self.raycaster.get_image(self.mesh)
        
        noisy_points = make_noise(points, normals, z_direction=np.array([0., 0., -1.]))
        depth_map = self.raycaster.points_to_image(noisy_points, ray_indexes)
        normals_image = self.raycaster.points_to_image(normals, ray_indexes,
                                                       assign_channels=[0, 1, 2])

        occluded_points = self.get_occluded_points(points[::4], size=64)

        observation = Observation(noisy_points,
                                  occluded_points,
                                  normals,
                                  mesh_vertex_indexes,
                                  mesh_face_indexes,
                                  depth_map,
                                  normals_image)
        return observation

    def get_observation(self, view_point_idx):
        view_point = self.view_points[view_point_idx]
        self.rotate_to_view_point(view_point)
        
        observation = self.raycast()
        observation.transform(self.transform)
        
        self.rotate_to_origin()
        
        return observation
    
    def get_occluded_points(self, surface_points, size=64):
        step = (self.mesh.bounds[1] - self.mesh.bounds[0]).max() / size
        # size //= 2
        cum_sums = np.tile(np.arange(1, size + 1), (len(surface_points), 1)) * step
        occluded_points = np.tile(surface_points, (size, 1, 1)).transpose(1, 0, 2)
        occluded_points[:,:,2] -= cum_sums
        occluded_points = np.concatenate(occluded_points, axis=0)
        return np.asarray(occluded_points)

    def surface_similarity(self, reconstructed_vertices, reconstructed_faces):
        return hausdorff(self.mesh.vertices,
                         self.mesh.faces,
                         reconstructed_vertices,
                         reconstructed_faces.astype(np.int64))
        

    def observation_similarity(self, observation):
        # TODO: We don't want to deal with GPU tensors right now, so simplify to this
        # chamfer
        # gt = FloatTensor(np.expand_dims(self.mesh.vertices, axis=0))
        # pred = FloatTensor(np.expand_dims(observation.vertices, axis=0))
        # return chamfer_distance(gt, pred)

        # area ratio
        return observation.face_indexes.shape[0] * 1.0 / self.mesh.faces.shape[0]

    
def get_mesh(observation):
    faces, vertices = poisson_reconstruction(
        observation.points, observation.normals, depth=10)
    return vertices, faces


def combine_observations(observations):
    points = np.concatenate([observation.points for observation in observations])
    occluded_points = np.concatenate([observation.occluded_points for observation in observations])
    normals = np.concatenate([observation.normals for observation in observations])

    vertex_indexes = np.unique(np.concatenate([observation.vertex_indexes
                                               for observation in observations]))
    face_indexes = np.unique(np.concatenate([observation.face_indexes
                                             for observation in observations]))

    depth_map = np.concatenate([observation.depth_map for observation in observations])
    normals_image = np.concatenate([observation.normals_image for observation in observations])

    return Observation(points, occluded_points, normals,
                       vertex_indexes, face_indexes,
                       depth_map, normals_image)


