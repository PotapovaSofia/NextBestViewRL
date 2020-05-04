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

def find_greedy_optimal(model, area_threshold=0.95, do_rec=False):
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

    if not do_rec:
        return view_points, 0.0

    combined_observation = None
    for vp in view_points:
        observation = model.get_observation(vp)
        if combined_observation is None:
            combined_observation = observation
        else:
            combined_observation += observation

    reconstructed_vertices, reconstructed_faces = get_mesh(combined_observation)
    loss = model.surface_similarity(reconstructed_vertices, reconstructed_faces)

    return view_points, loss

def main():
    for num_points in  [100]:
        models_path = "./data/1kabc/simple/val"
        optimal_numbers, losses = [], []
        for model_name in tqdm(sorted(os.listdir(models_path))):
            model_path = os.path.join(models_path, model_name)
        # model_path = "./data/1kabc/simple/train/00070090_73b2f35a88394199b6fd1ab8_003.obj"
            model = Model(model_path)
            model.generate_view_points(num_points)
            optimal, loss = find_greedy_optimal(model, do_rec=True)
            optimal_numbers.append(len(optimal))
            losses.append(loss)

        print("Model: ", model_path, "Optimal number: ",  np.mean(optimal_numbers), "Loss: ", np.mean(losses))


if __name__ == "__main__":
    main()
