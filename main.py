import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from preprocess import lbp_rotation_invariant 

# Paths
FEATURE_DATABASE_PATH = "FeatureDatabase"
IMAGE_DATABASE_PATH = "Dataset"
IMAGE_EXTENSIONS = [".jpg", ".png", ".JPG", ".jpeg", ".pgm"]

# Load feature database
def load_feature_database():
    feature_vectors, labels, image_paths = [], [], []

    for class_name in os.listdir(FEATURE_DATABASE_PATH):
        class_path = os.path.join(FEATURE_DATABASE_PATH, class_name)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                if file_name.endswith(".npy"):
                    feature_path = os.path.join(class_path, file_name)
                    feature_vector = np.load(feature_path)

                    base_name = os.path.splitext(file_name)[0]  
                    image_path = find_image_path(IMAGE_DATABASE_PATH, class_name, base_name)

                    if image_path:
                        feature_vectors.append(feature_vector)
                        labels.append(class_name)
                        image_paths.append(image_path)

    return np.array(feature_vectors), np.array(labels), image_paths

def find_image_path(root_path, class_name, base_name):
    class_path = os.path.join(root_path, class_name)
    if not os.path.exists(class_path):
        return None
    
    for ext in IMAGE_EXTENSIONS:
        potential_path = os.path.join(class_path, base_name + ext)
        if os.path.exists(potential_path):
            return potential_path
    
    return None

# Load dataset
features, labels, image_paths = load_feature_database()
# features = features.reshape(features.shape[0], -1)


# Training KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(features, labels)

def find_and_display_k_nearest_images(query_image_path, max_k=5):
    """ Finding and display the K most similar images using KNN """
    if not os.path.exists(query_image_path):
        print(f"Error: Image path '{query_image_path}' does not exist.")
        return

    try:
        image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
        query_feature = lbp_rotation_invariant(image).reshape(1, -1)
    except ValueError as e:
        print(e)
        return

    # Find K nearest neighbors
    max_k = min(max_k, len(image_paths))
    distances, indices = knn.kneighbors(query_feature, n_neighbors=max_k)

    nearest_images = [(image_paths[i], distances[0][j]) for j, i in enumerate(indices[0])]

    # Display results
    fig, axes = plt.subplots(1, len(nearest_images) + 1, figsize=(15, 5))

    # Show query image
    query_img = cv2.imread(query_image_path)
    if query_img is not None:
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(query_img)
        axes[0].set_title("Query Image")
        axes[0].axis("off")
    else:
        print(f"Error loading query image: {query_image_path}")
        return

    # Show nearest images
    for i, (img_path, dist) in enumerate(nearest_images, start=1):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
            axes[i].set_title(f"Dist: {dist:.4f}")
            axes[i].axis("off")
        else:
            print(f"Error loading image: {img_path}")

    plt.show()

# User Input for Image Path and Neighbors
query_image = input("Enter the path of the query image: ").strip()
while not os.path.exists(query_image):
    print("Invalid path. Please enter a valid image path.")
    query_image = input("Enter the path of the query image: ").strip()

while True:
    try:
        max_k = int(input("Enter the number of nearest neighbors to find: ").strip())
        if max_k > 0:
            break
        else:
            print("Please enter a positive integer.")
    except ValueError:
        print("Invalid input. Enter a number.")

# Perform KNN search and display results
find_and_display_k_nearest_images(query_image, max_k)
