import os
import cv2
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def haar_feature_extraction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

def apply_variance_threshold(features, threshold=0.95):
    sel = VarianceThreshold(threshold=threshold)
    filtered_features = sel.fit_transform(features)
    return filtered_features

def get_image_features(img_path):
    img = cv2.imread(img_path)
    faces = haar_feature_extraction(img)
    features = []
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(face_gray, (50, 50))
        features.extend(resized_face.flatten())
    return features

def train_bayesian_network(images_dir):
    labels = []
    features = []
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(images_dir, filename)
            label = filename.split("_")[0] # assuming the label is before the underscore in the filename
            labels.append(label)
            img_features = get_image_features(img_path)
            features.append(img_features)

    filtered_features = apply_variance_threshold(features)
    model = BayesianModel([('f1', 'label')])
    n_features = filtered_features.shape[1]
    cpd_f1 = TabularCPD(variable='f1', variable_card=2, values=[[1/n_features, 1/n_features]])
    cpd_label = TabularCPD(variable='label', variable_card=2, values=[[0.5, 0.5], [0.5, 0.5]], evidence=['f1'], evidence_card=[2])
    model.add_cpds(cpd_f1, cpd_label)
    model.check_model()
    infer = VariableElimination(model)

    return model, infer, filtered_features, labels

def predict_bayesian_network(image_path, model, infer, filtered_features):
    img_features = get_image_features(image_path)
    if len(img_features) == 0:
        return None
    img_features = np.array(img_features).reshape(1, -1)
    img_features = apply_variance_threshold(img_features, threshold=0.95)
    evidence = {'f1': img_features.tolist()[0]}
    query = infer.query(['label'], evidence=evidence)
    return query['label'].values[1]

images_dir = "data/variance"
model, infer, filtered_features, labels = train_bayesian_network(images_dir)
test_image_path = "testing/*.jpg"
prediction = predict_bayesian_network(test_image_path, model, infer, filtered_features)
print(prediction)
