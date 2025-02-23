Image Retrieval using Rotation-Invariant LBP and KNNThis project implements an image retrieval system using Rotation-Invariant Local Binary Pattern (LBP) for feature extraction and K-Nearest Neighbors (KNN) for similarity search.

📂 Project StructureDLCV_Assignment-1/
│── Dataset/                   # Contains the images for feature extraction
│── FeatureDatabase/           # Stores extracted LBP feature vectors as .npy files
│── TestImages/                # Contains query images for retrieval
│── main.py                    # Main script to run the image retrieval
│── preprocess.py              # Feature extraction using LBP
│── requirements.txt           # Dependencies for the project
│── venv/                      # Virtual environment (optional) 
Dependencies: Make sure you have the required libraries installed before running the project.

The key dependencies are:
    OpenCV (cv2)
    NumPy
    Matplotlib
    scikit-learn

🔄 Preprocessing (Feature Extraction)Before retrieving images, extract LBP features from the dataset:
    python preprocess.pyThis will create .npy feature files inside FeatureDatabase/.
🔍 Running Image RetrievalThe retrieval system will compare images in TestImages/ against the dataset:
        python main.py
The script will:
Load precomputed LBP feature vectors from FeatureDatabase

Train a KNN classifier

Process all images inside TestImages/

Retrieve and display the top-K most similar images

Troubleshooting:
Ensure that Dataset/ and TestImages/ contain valid images.
If FeatureDatabase/ is empty, rerun preprocess.py to generate features.
If an error occurs related to feature dimensions, check lbp_rotation_invariant() implementation.