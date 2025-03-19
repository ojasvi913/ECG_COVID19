# **ECG Classification using SIFT and SVM**

This project classifies ECG images into two categories: **COVID** and **Non-COVID** using the **Bag of Visual Words (BoVW) model** with **SIFT features** and an **SVM classifier**.

## **Project Overview**
The process follows these steps:
1. **Extract SIFT features** from ECG images.
2. **Create a Bag of Visual Words (BoVW)** representation using K-Means clustering.
3. **Train an SVM classifier** on the BoVW features.
4. **Test the trained SVM** on a new dataset.

## **Scripts and Their Functions**
### 1️⃣ **Feature Extraction (`extract_sift_features.py`)**
- Extracts **SIFT features** from all ECG images.
- Saves the extracted features for later use.

#### **How to Run**
```bash
python extract_sift_features.py
```

#### **Output Files**
- `sift_features.npy` → Contains extracted SIFT descriptors.

---

### 2️⃣ **Training the SVM (`train_svm.py`)**
- Uses **K-Means clustering** to form visual words (BoVW model).
- Normalizes the features and **trains an SVM model**.
- Saves the trained models for later classification.

#### **How to Run**
```bash
python train_svm.py
```

#### **Output Files**
- `kmeans_model.pkl` → Trained K-Means model for BoVW.
- `scaler.pkl` → Normalization scaler for feature standardization.
- `svm_model.pkl` → Trained SVM classifier.

---

### 3️⃣ **Testing the SVM (`test_svm.py`)**
- Loads the trained **K-Means, Scaler, and SVM models**.
- Extracts **SIFT features from a new test dataset**.
- Converts test images into **BoVW histograms**.
- Classifies test images using the **pretrained SVM**.
- Prints **accuracy and classification report**.

#### **How to Run**
```bash
python test_svm.py
```

#### **Expected Output**
- Classification accuracy.
- Precision, recall, and F1-score for each class.

---

## **Dataset Structure**
Ensure your dataset is structured as follows:
```
📂 Project Folder
 ├── 📂 COVID          # Training images of COVID-positive ECGs
 ├── 📂 NORMAL         # Training images of normal ECGs
 ├── 📂 NEW_COVID      # New test dataset (COVID)
 ├── 📂 NEW_NORMAL     # New test dataset (Normal)
```

---

## **Next Steps**
✅ Test the model on a different dataset to check generalization.  
✅ If accuracy is low, consider retraining on a larger dataset.  
✅ Experiment with different values of `K` in K-Means clustering.  

Feel free to improve the model or integrate deep learning for better accuracy! 🚀

