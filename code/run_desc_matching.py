import os

import numpy as np
import scipy.optimize
from utils import load_landmark_file, load_mesh_info, visualize_mat
from hungarian_impl import *
from scipy.spatial.distance import cdist


current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(current_dir)

mesh_dir = '.\data_p2\meshes'
mesh_dir = os.path.join(current_dir, mesh_dir)
# print(mesh_dir)

output_dir = '.\outputs_p2\desc_matching'
output_dir = os.path.join(current_dir, output_dir)
# print(output_dir)

landmark_filepath = '.\data_p2\landmark_vids.txt'
landmark_filepath = os.path.join(current_dir, landmark_filepath)
# print(landmark_filepath)

os.makedirs(output_dir, exist_ok=True)

# # Load landmark points
landmark_vids, landmark_names = load_landmark_file(landmark_filepath)
num_landmarks = len(landmark_vids)

# # Load all meshes
template_mesh_filepath = os.path.join(mesh_dir, 'template', 'mesh000.obj')
template_mesh_info = load_mesh_info(template_mesh_filepath)

test_mesh_filepaths = [os.path.join(mesh_dir, 'test', fname) for fname in os.listdir(os.path.join(mesh_dir, 'test')) if fname.endswith('.obj')]
num_test_meshes = len(test_mesh_filepaths)
test_mesh_infos = [load_mesh_info(test_mesh_filepath) for test_mesh_filepath in test_mesh_filepaths]



C_acc = np.zeros((num_landmarks, num_landmarks))

num_features = 200
templateLD = np.zeros((num_landmarks, num_features))

for i in range(num_landmarks):
    landmark_idx = landmark_vids[i]
    templateLD[i,:] = template_mesh_info['feature'][landmark_idx, :]


for k in range(num_test_meshes):
    test_mesh_info = test_mesh_infos[k]
    test_name = test_mesh_info['name']
    print(test_name)

    # # Find point correspondences

    # ---- To Do ---- #
    # Compute binary correspondence matrix C by comparing point
    # descriptors.
    #
    # Use descriptor vectors stored in mesh_info['feature'].
    # Each row is a feature vector for each vertex
    # The vertex IDs of landmark points are stored in landmark_vids
    #
    # For finding best correspondences, we recommend using 'Hungarian algorithm'
    # (https: // en.wikipedia.org / wiki / Hungarian_algorithm).
    # But any reasonable heuristic algorithm is also accepted.

    # ---- FILL HERE - --- $

    testLD = np.zeros((num_landmarks, num_features))
    for i in range(num_landmarks):
        landmark_idx = landmark_vids[i]
        testLD[i, :] = test_mesh_info['feature'][landmark_idx, :]

    distanceMatrix = cdist(templateLD, testLD, 'euclidean') 

    C = hungarian_algorithm(distanceMatrix)


    # -------- $

    C_acc += C

    visualize_mat(C, test_name, os.path.join(output_dir, f'{test_name}.png'))


# Compute overall accuracy.
accuracy = np.sum(np.diagonal(C_acc)) / np.sum(C_acc)
print(f'Overall accuracy = {accuracy:.4f}')

# Save overall frequency
visualize_mat(C_acc, 'All frequency', os.path.join(output_dir, 'all_frequency.png'))
