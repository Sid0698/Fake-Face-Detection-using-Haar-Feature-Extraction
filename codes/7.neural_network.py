import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPClassifier


def extract_features(image_path):
    # Read image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Apply Haar feature extraction
    haar = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    faces = haar.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    features = []
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, (50, 50))
        features.append(face.flatten())
    return features


def load_data(directory):
    # Load real and fake images from directory
    real_images = []
    fake_images = []
    for filename in os.listdir(directory):
        if filename.startswith("real"):
            real_images.extend(extract_features(os.path.join(directory, filename)))
        elif filename.startswith("fake"):
            fake_images.extend(extract_features(os.path.join(directory, filename)))
    # Create labels for real and fake images
    real_labels = [0] * len(real_images)
    fake_labels = [1] * len(fake_images)
    # Merge data and labels
    features = np.concatenate([real_images, fake_images])
    labels = np.concatenate([real_labels, fake_labels])
    return features, labels


def main():
    # Load data
    features, labels = load_data("data/testing")
    # Apply variance thresholding
    sel = VarianceThreshold(threshold=100)
    filtered_features = sel.fit_transform(features)
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(filtered_features, labels, test_size=0.2, random_state=42)
    # Create and train neural network
    clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", max_iter=100)
    clf.fit(X_train, y_train)
    # Evaluate model
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print("Training set score: {:.2f}".format(train_score))
    print("Test set score: {:.2f}".format(test_score))


if __name__ == "__main__":
    main()
