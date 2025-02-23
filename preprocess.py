import os
import cv2
import numpy as np

# Optimized Local Binary Pattern (LBP) function
import numpy as np
import cv2

def lbp_rotation_invariant(image, radius=1, neighbors=8):
    h, w = image.shape
    lbp_image = np.zeros((h, w), dtype=np.uint8)

    # Offsets for circular neighborhood
    angles = np.linspace(0, 2 * np.pi, neighbors, endpoint=False)
    dx = np.round(radius * np.cos(angles)).astype(int)
    dy = np.round(radius * np.sin(angles)).astype(int)

    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            center_pixel = image[y, x]
            binary_pattern = []

            for i in range(neighbors):
                neighbor_x = x + dx[i]
                neighbor_y = y + dy[i]

                if 0 <= neighbor_x < w and 0 <= neighbor_y < h:
                    binary_pattern.append(1 if image[neighbor_y, neighbor_x] >= center_pixel else 0)
                else:
                    binary_pattern.append(0)

            # Convert list to binary string and find the minimum rotation
            binary_string = ''.join(map(str, binary_pattern))
            min_rotation = min(int(binary_string[i:] + binary_string[:i], 2) for i in range(len(binary_string)))

            lbp_image[y, x] = min_rotation

    hist = cv2.calcHist([lbp_image.astype(np.float32)], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # Normalize and flatten
    return hist



 


if __name__ == "__main__":

    # Paths
    dataset_path = "Dataset"
    output_base = "FeatureDatabase"

    # Iterate through dataset and process images
    for subdir, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg", ".pgm")):  # Added .pgm support
                image_path = os.path.join(subdir, file)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print("Failed to read image: {image_path}")
                    continue

                # Compute LBP
                hist = lbp_rotation_invariant(img)

                # Save feature vector (LBP histogram) while maintaining folder structure
                relative_path = os.path.relpath(subdir, dataset_path)
                output_folder = os.path.join(output_base, relative_path)
                os.makedirs(output_folder, exist_ok=True)

                save_filename = os.path.splitext(file)[0] + ".npy"
                save_path = os.path.join(output_folder, save_filename)
                np.save(save_path, hist)

                print(f"Processed: {image_path} → Saved: {save_path}")
