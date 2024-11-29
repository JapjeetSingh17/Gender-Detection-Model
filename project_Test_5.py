# import streamlit as st
# import cv2
# import boto3
# import time
# import uuid
# import numpy as np
# import os
# import requests
# from contextlib import closing
# import subprocess
# import sys

# # Install opencv-python if not already installed
# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install("opencv-python")


# # Check if model files exist, and download only if they are missing
# def download_file_if_not_exists(url, destination):
#     if not os.path.exists(destination):
#         with closing(requests.get(url, stream=True)) as response:
#             if response.status_code == 200:
#                 with open(destination, 'wb') as file:
#                     for chunk in response.iter_content(chunk_size=1024):
#                         if chunk:
#                             file.write(chunk)
#             else:
#                 st.error(f"Error downloading file from {url}: Status code {response.status_code}")
#                 return False
#         st.write(f"{destination} downloaded successfully.")
#     return True

# # Model URLs and destination filenames
# model_files = {
#     "opencv_face_detector.pbtxt": "https://raw.githubusercontent.com/spmallick/opencv-dnn-face-detection/master/opencv_face_detector.pbtxt",
#     "opencv_face_detector_uint8.pb": "https://raw.githubusercontent.com/spmallick/opencv-dnn-face-detection/master/opencv_face_detector_uint8.pb",
#     "gender_deploy.prototxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
#     "gender_net.caffemodel": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_net.caffemodel"
# }

# # Download model files if not present
# for filename, url in model_files.items():
#     if not download_file_if_not_exists(url, filename):
#         st.error(f"Error: Unable to download {filename}.")
#         exit(1)

# # Load models
# faceProto = "opencv_face_detector.pbtxt"
# faceModel = "opencv_face_detector_uint8.pb"
# faceNet = cv2.dnn.readNet(faceModel, faceProto)

# genderProto = "gender_deploy.prototxt"
# genderModel = "gender_net.caffemodel"
# genderNet = cv2.dnn.readNet(genderModel, genderProto)

# genderList = ['Male', 'Female']

# # Initialize DynamoDB client
# os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
# dynamodb = boto3.resource('dynamodb')
# table = dynamodb.Table('HotspotZone')

# def store_metadata_in_dynamodb(face_id, image_id, gender, confidence):
#     try:
#         response = table.put_item(
#             Item={
#                 'Face_id': face_id,
#                 'ImageID': image_id,
#                 'Gender': gender,
#                 'Confidence': str(confidence),
#                 'Timestamp': str(time.time())
#             }
#         )
#         return response
#     except Exception as e:
#         st.error(f"Error storing item: {e}")

# def getFace(faceDetectionModel, inputImage, conf_threshold=0.7):
#     cpy_input_image = inputImage.copy()
#     frameWidth = cpy_input_image.shape[1]
#     frameHeight = cpy_input_image.shape[0]
#     blob = cv2.dnn.blobFromImage(cpy_input_image, scalefactor=1.0, size=(227, 227), mean=(104, 117, 123), crop=False)
#     faceDetectionModel.setInput(blob)
#     detections = faceDetectionModel.forward()

#     bounding_boxes = []
#     for i in range(detections.shape[2]):
#         confidence_score = detections[0, 0, i, 2]
#         if confidence_score > conf_threshold:
#             x1 = int(detections[0, 0, i, 3] * frameWidth)
#             y1 = int(detections[0, 0, i, 4] * frameHeight)
#             x2 = int(detections[0, 0, i, 5] * frameWidth)
#             y2 = int(detections[0, 0, i, 6] * frameHeight)
#             x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frameWidth, x2), min(frameHeight, y2)
#             if x2 > x1 and y2 > y1:
#                 bounding_boxes.append([x1, y1, x2, y2])
#                 cv2.rectangle(cpy_input_image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

#     return cpy_input_image, bounding_boxes

# # Streamlit UI
# st.title("Live Face Detection and Gender Prediction")

# ip_url = st.text_input("Enter IP Webcam URL", "http://10.12.60.193:8080/video")

# if ip_url:
#     cap = cv2.VideoCapture("http://10.12.60.193:8080/video")

#     # Check if the video stream is opened successfully
#     if not cap.isOpened():
#         st.error("Error: Could not open video stream.")
    
#     # Streamlit live video display loop
#     frame_placeholder = st.empty()

#     while cap.isOpened():
#         ret, inputImage = cap.read()
#         if not ret:
#             st.error("Failed to capture frame from IP webcam.")
#             break

#         detected_image, bounding_boxes = getFace(faceNet, inputImage)

#         if bounding_boxes:
#             for bounding_box in bounding_boxes:
#                 x1, y1, x2, y2 = bounding_box
#                 detected_face_box = inputImage[y1:y2, x1:x2]

#                 if detected_face_box.size == 0:
#                     continue

#                 detected_face_blob = cv2.dnn.blobFromImage(
#                     detected_face_box,
#                     scalefactor=1,
#                     size=(227, 227),
#                     mean=([78.4263377603, 87.7689143744, 114.895847746]),
#                     crop=False
#                 )

#                 genderNet.setInput(detected_face_blob)
#                 genderPrediction = genderNet.forward()
#                 gender = genderList[genderPrediction[0].argmax()]
#                 confidence = genderPrediction[0].max()

#                 # Generate unique IDs
#                 face_id = str(uuid.uuid4())  
#                 image_id = f"frame-{int(time.time())}"

#                 # Store metadata in DynamoDB
#                 store_metadata_in_dynamodb(face_id=image_id,
#                                            image_id=image_id,
#                                            gender=gender,
#                                            confidence=confidence)

#                 # Display result on the image
#                 cv2.putText(img=detected_image,
#                             text=f"{gender} {confidence:.2f}",
#                             org=(x1,y1 - 10),
#                             fontFace=cv2.FONT_HERSHEY_COMPLEX,
#                             fontScale=1.1,
#                             color=(0 ,255 ,0),
#                             thickness=2)

#         # Convert BGR to RGB for Streamlit display and show the image in real-time
#         frame_placeholder.image(cv2.cvtColor(detected_image ,cv2.COLOR_BGR2RGB))

#         # Break loop if user presses 'Stop'
#         if st.button("Stop"):
#             break

#     cap.release()

import streamlit as st
import cv2
import boto3
import time
import uuid
import numpy as np
import os
import requests
from contextlib import closing

# Check if model files exist, and download only if they are missing
def download_file_if_not_exists(url, destination):
    if not os.path.exists(destination):
        with closing(requests.get(url, stream=True)) as response:
            if response.status_code == 200:
                with open(destination, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            file.write(chunk)
            else:
                st.error(f"Error downloading file from {url}: Status code {response.status_code}")
                return False
        st.write(f"{destination} downloaded successfully.")
    return True

# Model URLs and destination filenames
model_files = {
    "opencv_face_detector.pbtxt": "https://raw.githubusercontent.com/spmallick/opencv-dnn-face-detection/master/opencv_face_detector.pbtxt",
    "opencv_face_detector_uint8.pb": "https://raw.githubusercontent.com/spmallick/opencv-dnn-face-detection/master/opencv_face_detector_uint8.pb",
    "gender_deploy.prototxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
    "gender_net.caffemodel": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_net.caffemodel"
}

# Download model files if not present
for filename, url in model_files.items():
    if not download_file_if_not_exists(url, filename):
        st.error(f"Error: Unable to download {filename}.")
        exit(1)

# Load models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProto)

genderList = ['Male', 'Female']

# Initialize DynamoDB client
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('HotspotZone')

def store_metadata_in_dynamodb(face_id, image_id, gender, confidence):
    try:
        response = table.put_item(
            Item={
                'Face_id': face_id,
                'ImageID': image_id,
                'Gender': gender,
                'Confidence': str(confidence),
                'Timestamp': str(time.time())
            }
        )
        return response
    except Exception as e:
        st.error(f"Error storing item: {e}")

def getFace(faceDetectionModel, inputImage, conf_threshold=0.7):
    cpy_input_image = inputImage.copy()
    frameWidth = cpy_input_image.shape[1]
    frameHeight = cpy_input_image.shape[0]
    blob = cv2.dnn.blobFromImage(cpy_input_image, scalefactor=1.0, size=(227, 227), mean=(104, 117, 123), crop=False)
    faceDetectionModel.setInput(blob)
    detections = faceDetectionModel.forward()

    bounding_boxes = []
    for i in range(detections.shape[2]):
        confidence_score = detections[0, 0, i, 2]
        if confidence_score > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frameWidth, x2), min(frameHeight, y2)
            if x2 > x1 and y2 > y1:
                bounding_boxes.append([x1, y1, x2, y2])
                cv2.rectangle(cpy_input_image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

    return cpy_input_image, bounding_boxes

# Streamlit UI
st.title("Live Face Detection and Gender Prediction")

ip_url = st.text_input("Enter IP Webcam URL", "http://10.12.34.84:8080/video")

if ip_url:
    cap = cv2.VideoCapture(ip_url,)

    # Check if the video stream is opened successfullyx
    if not cap.isOpened():
        st.error("Error: Could not open video stream.")
    
    # Streamlit live video display loop
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, inputImage = cap.read()
        if not ret:
            st.error("Failed to capture frame from IP webcam.")
            break

        detected_image, bounding_boxes = getFace(faceNet, inputImage)

        if bounding_boxes:
            for bounding_box in bounding_boxes:
                x1, y1, x2, y2 = bounding_box
                detected_face_box = inputImage[y1:y2, x1:x2]

                if detected_face_box.size == 0:
                    continue

                detected_face_blob = cv2.dnn.blobFromImage(
                    detected_face_box,
                    scalefactor=1,
                    size=(227, 227),
                    mean=([78.4263377603, 87.7689143744, 114.895847746]),
                    crop=False
                )

                genderNet.setInput(detected_face_blob)
                genderPrediction = genderNet.forward()
                gender = genderList[genderPrediction[0].argmax()]
                confidence = genderPrediction[0].max()

                # Generate unique IDs
                face_id = str(uuid.uuid4())  
                image_id = f"frame-{int(time.time())}"

                # Store metadata in DynamoDB
                store_metadata_in_dynamodb(face_id=image_id,
                                           image_id=image_id,
                                           gender=gender,
                                           confidence=confidence)

                # Display result on the image
                cv2.putText(img=detected_image,
                            text=f"{gender} {confidence:.2f}",
                            org=(x1,y1 - 10),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=1.1,
                            color=(0 ,255 ,0),
                            thickness=2)

        # Convert BGR to RGB for Streamlit display and show the image in real-time
        frame_placeholder.image(cv2.cvtColor(detected_image ,cv2.COLOR_BGR2RGB))

        # Break loop if user presses 'Stop'
        # Add a unique key to avoid duplicate element ID error.
        # stop_key_unique_value = f"stop_button_{time.time()}"
        
        # if st.button("Stop", key=stop_key_unique_value):
        #     break

    cap.release()