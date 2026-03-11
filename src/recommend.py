import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from feature_extraction import extract_features
from tkinter import Tk
from tkinter.filedialog import askopenfilename

dataset_path = "dataset/images"

images = os.listdir(dataset_path)

features = []

# Extract features from dataset
for img in images:

    path = os.path.join(dataset_path, img)

    feat = extract_features(path)

    features.append(feat)

features = np.array(features)

# Open file chooser
Tk().withdraw()

print("Select an input image")

query_path = askopenfilename()

# Extract query features
query_features = extract_features(query_path)
if query_features is None:
    print("Invalid image selected. Please try again.")
    exit()

# Calculate similarity
similarity = cosine_similarity([query_features], features)

similar_images = np.argsort(similarity[0])[::-1]

print("Showing Recommended Products")

plt.figure(figsize=(10,6))

count = 0

for idx in similar_images:

    img_path = os.path.join(dataset_path, images[idx])

    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(1,5,count+1)

    plt.imshow(img)

    plt.title(images[idx])

    plt.axis("off")

    count += 1

    if count == 5:
        break

plt.show()