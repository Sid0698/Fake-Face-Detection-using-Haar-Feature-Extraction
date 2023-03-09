import cv2
import numpy as np
import os
# from google.colab.patches import cv2_imshow

# specify the path to the directory containing the images
path_to_images = "data/haar"

# create a list of all image filenames in the directory
filenames = [os.path.join(path_to_images, f) for f in os.listdir(path_to_images) if os.path.isfile(os.path.join(path_to_images, f))]

# create lists to store the preprocessed images and their Haar features
images = []
features = []

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

    # store the original image with the detected faces in the 'images' list
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    images.append(image)

# convert the 'features' list to a numpy array
features = np.array(features)

# compute the variance of each feature dimension
variances = np.var(features, axis=0)

# set a threshold for the minimum variance to retain
variance_threshold = 1000

# filter out the features with variance below the threshold
selected_features = features[:, variances >= variance_threshold]

# print the number of retained features
print("Number of retained features:", selected_features.shape[1])

# create a new directory for storing the images with retained faces
output_directory = "data/variance"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# loop over the images and save the ones with retained faces
for i, image in enumerate(images):
    # detect faces in the grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # create a copy of the image and draw rectangles around the retained faces
    output_image = image.copy()
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        feature = cv2.resize(face, (100, 100)).flatten()
        if np.var(feature) >= variance_threshold:
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # get the original filename of the image and create a path for the output image
    filename = os.path.basename(filenames[i])
    output_path = os.path.join(output_directory, filename)

    # write the output image to disk
    cv2.imwrite(output_path, output_image)
