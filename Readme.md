# Image Classification using Unsupervised Learning
This project focuses on classifying images of famous personalities including Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli using unsupervised learning techniques. We utilize Support Vector Machines (SVM) and PyWavelet to extract features and perform classifications.

## Overview
The classification process involves training an SVM model on extracted features from the images. We employ PyWavelet to extract relevant features from the images, which are then fed into the SVM for classification.

## Frontend UI
The user interface is developed using Streamlit, a Python library for building interactive web applications. Users can upload an image, and the system will classify it into one of the predefined classes.

## Requirements
To run the application locally, make sure you have the following dependencies installed:

1. Python 3.x
2. Streamlit
3. PyWavelet
4. OpenCV
5. SVM (from scikit-learn or any other library)
6. Haar Cascade Face and Eye XML files


You can install the Python dependencies using pip:

```
pip install -r requirements.txt
```

## Usage
Clone this repository to your local machine.
Navigate to the project directory.

Ensure you have the Haar Cascade XML files for face and eye detection in the appropriate directory.

Run the Streamlit app using the following command:

```
streamlit run app.py
```

Once the app is running, you can upload an image and click on the "Process" button to classify it.

## Dataset
The dataset consists of images of Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli. Ensure that your dataset is appropriately labeled and organized for training.
