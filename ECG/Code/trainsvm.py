import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# File paths for saved features
bovw_features_path = "ECG/bovw_features.npy"
labels_path = "ECG/labels.npy"
svm_model_path = "ECG/svm_model.pkl"

# Load features and labels
print("Loading features and labels...")
X = np.load(bovw_features_path)
y = np.load(labels_path)

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM classifier
print("Training SVM classifier...")
svm = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
svm.fit(X_train, y_train)

# Evaluate the model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-COVID", "COVID"]))

# Save trained model
print("Saving trained SVM model...")
joblib.dump(svm, svm_model_path)

print("SVM training complete! Model saved as 'svm_model.pkl'.")

