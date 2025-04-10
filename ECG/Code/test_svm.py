import cv2
import os
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Paths to saved models
kmeans_model_path = "ECG/kmeans_model.pkl"
scaler_path = "ECG/scaler.pkl"
svm_model_path = "ECG/svm_model.pkl"

# Paths to new test dataset
new_covid_path = "ECG/NEW_COVID"   # Change to your test dataset folder
new_non_covid_path = "ECG/NEW_NORMAL"

# Load trained models
print("Loading trained models...")
kmeans = joblib.load(kmeans_model_path)
scaler = joblib.load(scaler_path)
svm = joblib.load(svm_model_path)

# Initialize SIFT detector
sift = cv2.SIFT_create()

def extract_sift_features(image_path):
    """Extracts SIFT keypoints and descriptors from an image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    _, descriptors = sift.detectAndCompute(image, None)
    return descriptors

def create_bovw_histogram(descriptors, kmeans):
    """Creates a BoVW histogram using the trained K-Means model."""
    K = kmeans.n_clusters
    if descriptors is None:
        return np.zeros(K)
    labels = kmeans.predict(descriptors)
    histogram, _ = np.histogram(labels, bins=np.arange(K + 1))
    return histogram

# Extract features from the new dataset
print("Extracting features from new dataset...")
# Only process up to 100 images from each category
covid_files = os.listdir(new_covid_path)[:100]
non_covid_files = os.listdir(new_non_covid_path)[:100]

covid_features = [extract_sift_features(os.path.join(new_covid_path, f)) for f in covid_files]
non_covid_features = [extract_sift_features(os.path.join(new_non_covid_path, f)) for f in non_covid_files]


# Remove None values (if any images failed to load)
covid_features = [desc for desc in covid_features if desc is not None]
non_covid_features = [desc for desc in non_covid_features if desc is not None]

# Convert images to BoVW histograms
covid_histograms = [create_bovw_histogram(desc, kmeans) for desc in covid_features]
non_covid_histograms = [create_bovw_histogram(desc, kmeans) for desc in non_covid_features]

# Prepare dataset
X_test = np.vstack((covid_histograms, non_covid_histograms))
y_test = np.hstack((np.ones(len(covid_histograms)), np.zeros(len(non_covid_histograms))))  # 1 for COVID, 0 for Non-COVID

# Normalize features
X_test = scaler.transform(X_test)

# Predict with the trained SVM
print("Making predictions...")
y_pred = svm.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy on New Dataset: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-COVID", "COVID"]))
