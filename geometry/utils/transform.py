import numpy as np
import math  
from .matrix_torch import (create_rotation_matrix_x, create_rotation_matrix_y, create_rotation_matrix_z, create_translation_matrix)
import torch

def from_extrinsic(extrinsic, fix_axes=False):
    if fix_axes:
        extrinsic[:, 1:3] *= -1
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]
    view_matrix = np.identity(4)
    view_matrix[:3,:3] = R.T
    view_matrix[:3, 3] = -R.T@T
    return view_matrix


def from_euler(phi, theta):
    rotation = torch.mm(create_rotation_matrix_y(phi),
                        create_rotation_matrix_z(theta))

    translation = create_translation_matrix(0, 0, 0)
    transform = torch.mm(rotation, translation)
    return transform.t()


def from_pose(local_rotation, local_position):
    rotation_x = create_rotation_matrix_x(-local_rotation[0])
    rotation_y = create_rotation_matrix_y(-local_rotation[1])
    rotation_z = create_rotation_matrix_z(-local_rotation[2])
    
    translation = create_translation_matrix(-local_position[0], -local_position[1], -local_position[2])
    rotation = torch.mm(rotation_x, rotation_y)
    rotation = torch.mm(rotation, rotation_z)
    transform = torch.mm(rotation, translation)
    
    return transform.t()


def to_pose(view_matrix):
    R = view_matrix[:3,:3]
    T = view_matrix[3, :3]
    T = -np.linalg.inv(R.T)@T
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    
    return np.array([x, y, z]), T


def rotate_around(transform, rotate_around, angle):
    
    ra_translation = create_translation_matrix(rotate_around[0], \
                                             rotate_around[1], \
                                             rotate_around[2])
    ra_translation_inverse = create_translation_matrix(-rotate_around[0], \
                                             -rotate_around[1], \
                                             -rotate_around[2])
    ra_rotation = create_rotation_matrix_y(-angle[1])
    ra_rotation = torch.mm(ra_rotation, create_rotation_matrix_z(-angle[2]))
    
    ra_transform = torch.mm(transform, ra_translation)
    ra_transform = torch.mm(ra_transform, ra_rotation)
    ra_transform = torch.mm(ra_transform, ra_translation_inverse)
    
    return ra_transform.t()


def transform_mesh(mesh, rotation, translation=None, reverse=False):
    # rotation = from_pose([x, y, z], [0, 0, 0]) 
    mesh_ = mesh.copy()
    if reverse:
        if translation is not None:
            mesh_.apply_translation(-translation)
        mesh_.apply_transform(rotation.T)
    else:
        mesh_.apply_transform(rotation)
        if translation is not None:
            mesh_.apply_translation(translation)

    return mesh_


def transform_points(points, rotation, translation=None, reverse=False):
    if translation is None:
        translation = np.zeros((3))
    rotation = rotation[:3, :3]
    
    if reverse:
        return (points - translation).dot(rotation)
    return points.dot(rotation.T) + translation

