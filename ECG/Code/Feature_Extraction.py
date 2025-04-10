import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib  # For saving k-means model

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Path to your dataset
covid_path = "ECG/COVID"
non_covid_path = "ECG/NORMAL"

# File paths for saved features
bovw_features_path = "ECG/bovw_features.npy"
labels_path = "ECG/labels.npy"
kmeans_model_path = "ECG/kmeans_model.pkl"
scaler_path = "ECG/scaler.pkl"

def extract_sift_features(image_path):
    """Extracts SIFT keypoints and descriptors from an image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    _, descriptors = sift.detectAndCompute(image, None)
    return descriptors

# Check if saved features exist
if os.path.exists(bovw_features_path) and os.path.exists(labels_path) and os.path.exists(kmeans_model_path):
    print("Loading precomputed features and K-Means model...")
    
    X = np.load(bovw_features_path)
    y = np.load(labels_path)
    kmeans = joblib.load(kmeans_model_path)
    scaler = joblib.load(scaler_path)

else:
    print("Extracting SIFT features from dataset...")

    # Extract features from dataset
    covid_features = [extract_sift_features(os.path.join(covid_path, f)) for f in os.listdir(covid_path)]
    non_covid_features = [extract_sift_features(os.path.join(non_covid_path, f)) for f in os.listdir(non_covid_path)]

    # Remove None values (if any images failed to load)
    covid_features = [desc for desc in covid_features if desc is not None]
    non_covid_features = [desc for desc in non_covid_features if desc is not None]

    # Stack all descriptors together for clustering
    all_descriptors = np.vstack(covid_features + non_covid_features)

    # Define number of clusters (visual words)
    K = 50  # You can tune this parameter

    # Apply K-Means clustering
    print("Running K-Means clustering...")
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(all_descriptors)

    # Function to create BoVW histogram for each image
    def create_bovw_histogram(descriptors, kmeans):
        if descriptors is None:
            return np.zeros(K)
        labels = kmeans.predict(descriptors)
        histogram, _ = np.histogram(labels, bins=np.arange(K + 1))
        return histogram

    # Convert images to BoVW histograms
    print("Creating BoVW histograms...")
    covid_histograms = [create_bovw_histogram(desc, kmeans) for desc in covid_features]
    non_covid_histograms = [create_bovw_histogram(desc, kmeans) for desc in non_covid_features]

    # Prepare dataset
    X = np.vstack((covid_histograms, non_covid_histograms))
    y = np.hstack((np.ones(len(covid_histograms)), np.zeros(len(non_covid_histograms))))  # 1 for COVID, 0 for Non-COVID

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Save processed features
    print("Saving features and model...")
    np.save(bovw_features_path, X)
    np.save(labels_path, y)
    joblib.dump(kmeans, kmeans_model_path)
    joblib.dump(scaler, scaler_path)

print(f"Feature shape: {X.shape}, Labels shape: {y.shape}")
