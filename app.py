import streamlit as st
import cv2
import boto3
import time
import uuid  # Import uuid for generating unique Face_id
import numpy as np

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('HotspotZone')

def store_metadata_in_dynamodb(face_id, image_id, gender, confidence):
    try:
        response = table.put_item(
            Item={
                'Face_id': face_id,  # Include Face_id as a key
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
    blob = cv2.dnn.blobFromImage(cpy_input_image, scalefactor=1, size=(227, 227), mean=(104, 117, 123), crop=False)
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

# Load models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProto)

genderList = ['Male', 'Female']

# Streamlit UI
st.title("Face Detection and Gender Prediction")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    inputImage = cv2.imdecode(file_bytes, 1)

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

        st.image(cv2.cvtColor(detected_image ,cv2.COLOR_BGR2RGB))

    else:
        st.write("No faces detected.")
        