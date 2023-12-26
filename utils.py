import os
import cv2 
import numpy as np

def get_all_paths(folder_path):
    # Initialize an empty list to store paths
    all_paths = []
    # Iterate through all elements in the folder
    for root, _, files in os.walk(folder_path):
        # Add paths of files
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.split('.')[-1]=='png':
                all_paths.append(file_path)

    return all_paths

def edge_keeping_index(original_image, filtered_image):
    # Compute Laplacian of the original and filtered images
    laplacian_original = (cv2.Laplacian((original_image*255).astype(np.uint8), cv2.CV_64F)[:, :, :1]).reshape(original_image.shape[0:2])
    laplacian_filtered = (cv2.Laplacian((filtered_image*255).astype(np.uint8), cv2.CV_64F)[:, :, :1]).reshape(filtered_image.shape[0:2])


    # Compute the edge-keeping index
    num = np.sum(np.dot((laplacian_original - np.mean(laplacian_original)).T, (laplacian_filtered - np.mean(laplacian_filtered))))
    denom = np.sum(np.dot((laplacian_original - np.mean(laplacian_original)).T, (laplacian_original - np.mean(laplacian_original))))*np.sum(np.dot((laplacian_filtered - np.mean(laplacian_filtered)).T, (laplacian_filtered - np.mean(laplacian_filtered))))
    eki = num /np.sqrt(denom)

    return eki