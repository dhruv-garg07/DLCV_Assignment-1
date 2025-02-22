import os
import cv2
import numpy as np

# Custom Local Binary Pattern function
def lbp(image):
    height, width = image.shape
    lbp_image = np.zeros((height - 2, width - 2), dtype=np.uint8)  # Output LBP matrix

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            center = image[i, j]  # Central pixel value
            binary_string = ""

            # 8 Neighbor Pixels (Clockwise: Top-left to Left)
            neighbors = [
                image[i-1, j-1], image[i-1, j], image[i-1, j+1],  # Top 3 pixels
                image[i, j+1],                                      # Right
                image[i+1, j+1], image[i+1, j], image[i+1, j-1],  # Bottom 3 pixels
                image[i, j-1]                                      # Left
            ]

            # Convert neighborhood values to binary pattern
            for neighbor in neighbors:
                binary_string += '1' if neighbor >= center else '0'

            # Convert binary to decimal
            lbp_value = int(binary_string, 2)
            print(lbp_value)
            lbp_image[i-1, j-1] = lbp_value  # Store LBP value (ignoring border pixels)

    return lbp_image

# Paths
dataset_path = "ChessDataset"
output_base = "FeatureDatabase"

# Iterate through dataset and process images
for subdir, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(subdir, file)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Compute LBP using custom function
            lbp_matrix = lbp(img)

            # Compute histogram (1×256 feature vector)
            hist, _ = np.histogram(lbp_matrix.ravel(), bins=256, range=(0, 256), density=True)

            # Save the histogram in FeatureDatabase_Custom while maintaining folder structure
            relative_path = os.path.relpath(subdir, dataset_path)
            output_folder = os.path.join(output_base, relative_path)
            os.makedirs(output_folder, exist_ok=True)

            save_filename = os.path.splitext(file)[0] + ".npy"
            save_path = os.path.join(output_folder, save_filename)
            np.save(save_path, hist)

            print(f"Processed: {image_path} → Saved: {save_path}")
