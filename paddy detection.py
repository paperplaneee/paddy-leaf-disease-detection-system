import os
import cv2
import numpy as np
import joblib
import json
import pickle
import shutil
import albumentations as A
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_PATH = "C:/dataset/aftertrain"

TRAIN_PATH = os.path.join(BASE_PATH, "train")
VAL_PATH = os.path.join(BASE_PATH, "val")
TEST_PATH = os.path.join(BASE_PATH, "test")
TEST_RAW_PATH = os.path.join(BASE_PATH, "test_raw")

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))
    return cv2.bitwise_and(image, image, mask=mask)


def segment_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 24) & (hsv[:, :, 1] >= 32) & (hsv[:, :, 1] <= 255)
    segmented = image.copy()
    segmented[~mask] = 0
    return segmented


def extract_features(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    lbp = local_binary_pattern(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)
    gabor_res = gabor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), frequency=0.6)
    gabor_features = np.array([gabor_res[0].mean(), gabor_res[0].std()])
    return np.concatenate((hist, lbp_hist, gabor_features))

def process_dataset(folder):
    features, labels = [], []
    for label in os.listdir(folder):
        class_path = os.path.join(folder, label)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            preprocessed = preprocess_image(img)
            segmented = segment_image(preprocessed)
            features.append(extract_features(segmented))
            labels.append(label)
    return np.array(features), np.array(labels)

def train_model():
    print("🔄 Loading datasets...")
    train_features, train_labels = process_dataset(TRAIN_PATH)
    val_features, val_labels = process_dataset(VAL_PATH)
    test_features, test_labels = process_dataset(TEST_PATH)
    print(f"Feature vector shape: {train_features.shape}")
    scaler = StandardScaler()
    train_features, val_features, test_features = scaler.fit_transform(train_features), scaler.transform(val_features), scaler.transform(test_features)
    pca = PCA(n_components=50)
    #train_features, val_features, test_features = pca.fit_transform(train_features), pca.transform(val_features), pca.transform(test_features)
    
    clf = VotingClassifier([
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)),
        ('svc', SVC(kernel='rbf', C=200, gamma=0.001, probability=True, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ], voting='soft')
    
    print("🔄 Training model...")
    clf.fit(train_features, train_labels)
    
    print("✅ Training complete! Saving model...")
    pickle.dump(clf, open('model.pkl', 'wb'))
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(pca, 'pca.pkl')
    
    return clf, val_features, val_labels, test_features, test_labels

def evaluate_model(clf, val_features, val_labels, test_features, test_labels):
    val_preds = clf.predict(val_features)
    val_report = classification_report(val_labels, val_preds)
    test_preds = clf.predict(test_features)
    print("Validation Accuracy:", accuracy_score(val_labels, val_preds))
    print("Test Accuracy:", accuracy_score(test_labels, test_preds))
    print(f"Validation Classification Report:\n{val_report}")

if __name__ == "__main__":
    clf, val_features, val_labels, test_features, test_labels = train_model()
    evaluate_model(clf, val_features, val_labels, test_features, test_labels)