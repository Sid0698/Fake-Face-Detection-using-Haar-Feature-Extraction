import cv2
import numpy as np
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# specify the path to the directory containing the images
path_to_images = "data/testing"

# create a list of all image filenames in the directory
filenames = [os.path.join(path_to_images, f) for f in os.listdir(path_to_images) if os.path.isfile(os.path.join(path_to_images, f))]

# create lists to store the preprocessed images and their Haar features
images = []
features = []
labels = []

# create a Haar cascade classifier for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

# loop over each image and extract Haar features from detected faces
for filename in filenames:
    # load the image and convert it to grayscale
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # extract Haar features from each face and store them in the 'features' list
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        feature = cv2.resize(face, (100, 100)).flatten()
        features.append(feature)
        # set the label to 0 for real images and 1 for fake images
        label = 0 if "real" in filename else 1
        labels.append(label)

    # store the original image with the detected faces in the 'images' list
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    images.append(image)

# convert the 'features' and 'labels' lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# compute the variance of each feature dimension
variances = np.var(features, axis=0)

# set a threshold for the minimum variance to retain
variance_threshold = 1000

# filter out the features with variance below the threshold
selected_features = features[:, variances >= variance_threshold]

# print the number of retained features
print("Number of retained features:", selected_features.shape[1])

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(selected_features, labels, test_size=0.2, random_state=42)

# create an AdaBoost classifier and fit it to the training data
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)

# make predictions on the test data
y_pred = ada.predict(X_test)

# compute the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
