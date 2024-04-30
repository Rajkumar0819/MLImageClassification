import streamlit as st 
import numpy as np 
import joblib
import pywt
import cv2
from PIL import Image

face_cascade = cv2.CascadeClassifier("opencv/haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("opencv/haarcascades/haarcascade_eye.xml")

def get_cropped_images():
    img = cv2.imread("img.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces =  face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >=2:
            return roi_color

def w2d(img, mode='haar', level=1):
    imArray = img
    imArray = cv2.cvtColor( imArray, cv2.COLOR_RGB2GRAY)
    #convert to float
    imArray = np.float32(imArray)
    imArray /= 255

    coeffs=pywt.wavedec2 (imArray, mode, level=level)
    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0
    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H

def final_image_output():
    cropped_image = get_cropped_images()
    if cropped_image is None:
        st.error("No faces detected in the uploaded image.")
        return None
    img_har = w2d(cropped_image, 'db1',5)
    scalled_raw_img = cv2.resize(cropped_image, (32, 32))
    scalled_img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1), scalled_img_har.reshape(32*32,1)))
    reshaped_features = combined_img.reshape(1, -1)

    # Ensure the reshaped features have the same number of features as the training data
    if reshaped_features.shape[1] != 4096:
        reshaped_features = np.pad(reshaped_features, ((0, 0), (0, 4096 - reshaped_features.shape[1])), mode='constant')
    
    return reshaped_features

model_input = None
predictions = 0 
pipeline_model = joblib.load('pipeline_model.pkl')

if __name__ == "__main__":
    
    st.sidebar.header("Image classification between Lionel Messi, Lionel Messi, Roger Federer, Serena Williams, Virat Kohli")

    st.sidebar.write("#### Make sure the face and both eyes are visible")

    img_file_buffer = st.sidebar.file_uploader('Upload a image ', type=["jpg", "png"])

    if img_file_buffer is not None:
        img = Image.open(img_file_buffer)
        img = img.save("img.jpg")

        button = st.sidebar.button("Process")
        if button == True:
            model_input = final_image_output()
            predictions = pipeline_model.predict(model_input)

            if predictions == 0:
                st.write("The person is Lionel Messi")
                st.image("UI/messi.jpg")
            elif predictions == 1:
                st.write("The person is MariaSharapova")
                st.image("UI/MariaSharapova.jpg")
            elif predictions == 2:
                st.write("The person is Roger Federer")
                st.image("UI/Roger.jpg")
            elif predictions == 3:
                st.write("The person is Serena Williams")
                st.image("UI/serena.jpg")
            elif predictions == 4:
                st.write("The person is Virat Kohli")
                st.image("UI/virat.jpg")
