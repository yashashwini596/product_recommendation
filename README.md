Product recommendation using image processing

## Project Overview

This project recommends visually similar products using image similarity techniques.
The system extracts features from images and compares them with a dataset to suggest similar items.

## Technologies Used

* Python
* OpenCV
* NumPy
* Scikit-learn

## How It Works

1. The user selects an input product image.
2. The system extracts visual features from the image.
3. These features are compared with images in the dataset.
4. Cosine similarity is used to find the most similar products.
5. The top matching products are displayed.

## Project Structure

```
product-recommendation-system
│
├── dataset/
│   └── images/
│
├── src/
│   ├── feature_extraction.py
│   └── recommend.py
│
├── requirements.txt
└── README.md
```

## Installation

Install required libraries:

pip install -r requirements.txt

## Run the Project

python src/recommend.py

## Future Improvements

* Use deep learning models (ResNet/VGG) for better feature extraction
* Build a web interface using Streamlit
* Use a larger dataset for better recommendations
