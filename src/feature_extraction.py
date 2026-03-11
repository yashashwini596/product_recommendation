import cv2
import numpy as np

def extract_features(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print("Error: Could not read image:", image_path)
        return None

    img = cv2.resize(img, (224,224))

    features = img.flatten()

    return features