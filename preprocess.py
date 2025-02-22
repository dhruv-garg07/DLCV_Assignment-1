import os
import cv2

# Define paths
input_folder = "ChessDataset"  # Change this to your dataset path
output_folder = "preprocessed_dataset"  # Folder to save processed images

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define fixed image size
IMAGE_SIZE = (128, 128)

# Process images
for class_name in os.listdir(input_folder):  
    class_path = os.path.join(input_folder, class_name)
    output_class_path = os.path.join(output_folder, class_name)

    # Create class folder in output
    if not os.path.exists(output_class_path):
        os.makedirs(output_class_path)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        
        # Read and process image
        img = cv2.imread(img_path)  
        if img is None:
            print(f"Skipping {img_name}, unable to read.")
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img_resized = cv2.resize(img_gray, IMAGE_SIZE)  # Resize

        # Save preprocessed image
        output_img_path = os.path.join(output_class_path, img_name)
        cv2.imwrite(output_img_path, img_resized)

print("✅ Dataset preprocessing completed. Check the 'preprocessed_dataset' folder.")
