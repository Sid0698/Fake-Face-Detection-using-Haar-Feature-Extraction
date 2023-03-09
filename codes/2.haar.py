import numpy as np
import cv2
import os
# from google.colab.patches import cv2_imshow

def normalize_and_scale(image):
    # Convert the image to floating point format
    image = np.float32(image)

    # Normalize the image
    image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)

    # Scale the image
    image *= 255

    # Convert the image back to 8-bit format
    image = np.uint8(image)

    return image

def haar_feature_extraction(image):
    # Load the Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return image

# Directory containing the images
image_dir = "data/normalize/"

# List of filenames in the image directory
filenames = os.listdir(image_dir)

for filename in filenames:
    # Load the normalized and scaled image
    image = cv2.imread(os.path.join(image_dir, filename))

    # Apply Haar feature extraction on the image
    image = haar_feature_extraction(image)

    # Save the image with Haar features to disk
    cv2.imwrite(os.path.join("data/haar", filename), image)

    # Display the first image with Haar features
    cv2.imshow(image)
    cv2.waitKey(0)
cv2.destroyAllWindows()
