import os
import numpy as np
from tqdm import tqdm

from geometry.model import Model, combine_observations, get_mesh
from geometry.utils.visualisation import illustrate_points, illustrate_mesh


def generate_observations(model):
    observations = []
    for view_point_idx in range(len(model.view_points)):
        observation = model.get_observation(view_point_idx)
        observations.append(set(observation.face_indexes))
    return observations

def find_greedy_optimal(model, area_threshold=0.95):
    observations = generate_observations(model)
    
    uncovered = set(np.arange(0, model.mesh.faces.shape[0]))
    covered = set()

    uncovered_area_threshold = len(uncovered) * (1 - area_threshold)

    view_points = []
    while len(uncovered) > uncovered_area_threshold and \
            len(view_points) < len(observations):
        view_point_index = np.argmax([len(covered.union(ob)) for ob in observations])
        new = observations[view_point_index]
        covered = covered.union(new)
        uncovered -= new
        view_points.append(view_point_index)
        
    return view_points

def main():
    for num_points in  [100]:
        models_path = "./data/10abc"
        optimal_numbers = []
        # for model_name in tqdm(sorted(os.listdir(models_path))):
        # model_path = os.path.join(models_path, model_name)
        model_path = "./data/Pyramid.obj"
        model = Model(model_path)
        model.generate_view_points(num_points)
        optimal = find_greedy_optimal(model)
        optimal_numbers.append(len(optimal))

        print("Model: ", model_path, "Number of view_points: ", num_points, "Optimal number: ",  np.mean(optimal_numbers))


if __name__ == "__main__":
    main()
