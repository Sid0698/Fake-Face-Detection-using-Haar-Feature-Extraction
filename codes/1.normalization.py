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

# Directory containing the images
image_dir = "data/training_real/"

# List of filenames in the image directory
filenames = os.listdir(image_dir)

output_directory = "data/normalize"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
# List to store the normalized and scaled images
images = []

for filename in filenames:
    # Load the image
    image = cv2.imread(os.path.join(image_dir, filename))

    # Normalize and scale the image
    image = normalize_and_scale(image)

    # Add the image to the list of images
    images.append(image)

    # Save the normalized image to disk
    cv2.imwrite(os.path.join(output_directory, filename), image)

# Display the first normalized image
    cv2.imshow(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
