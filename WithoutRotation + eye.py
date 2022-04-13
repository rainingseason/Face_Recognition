import cv2
import os
import numpy as np


training_data_folder_path = '/Users/charlesbeh/Desktop/School/IS/Database'

#Face Detection
haarcascade_frontalface = 'Face_Recognition/haarcascade_frontalface_alt2.xml'
def detect_face(input_img):
    image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('Face_Recognition/haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return -1, -1
    (x, y, w, h) = faces[0]
    return image[y:y+w, x:x+h], faces[0]


#Preparing training dataset
def prepare_training_data(training_data_folder_path):
    detected_faces = []
    face_labels = []
    traning_image_dirs = os.listdir(training_data_folder_path)
    for dir_name in traning_image_dirs:
        label = int(dir_name)
        training_image_path = training_data_folder_path + "/" + dir_name
        training_images_names = os.listdir(training_image_path)

        for image_name in training_images_names:
            image_path = training_image_path + "/" + image_name
            image = cv2.imread(image_path)
            face, rect = detect_face(image)
            if face is not -1:
                resized_face = cv2.resize(face, (250, 300), interpolation=cv2.INTER_AREA)
                detected_faces.append(resized_face)
                face_labels.append(label)
                print(image_path)

    return detected_faces, face_labels

detected_faces, face_labels = prepare_training_data('Database')
print("Total faces: ", len(detected_faces))

#Init a Face recognition
eigenfaces_recognizer = cv2.face.EigenFaceRecognizer_create()

#Train the face recognizer model
eigenfaces_recognizer.train(detected_faces, np.array(face_labels))


# video_capture.set(3, 1920)  # width
# video_capture.set(4, 1080)  # height

width = 250
height = 300
dim = (width, height)

namelist = ['Truong', 
            'Charles', 
            'User1', 
            'User2', 
            'User3', 
            'User4',
            'User5', 
            'User6', 
            'User7', 
            'User8',
            'User9', 
            'User10', 
            'User11', 
            'User12',
            'User13', 
            'User14', 
            'User15', 
            'User16',]

color = {
    0: (238, 238, 0),  
    1: (255, 62, 191), 
    2: (255, 0, 0),  
    3: (255, 0, 255),
    4: (0, 255, 0),
    5: (0, 0, 255),
    6: (255, 255, 255),
    7: (255, 255, 255),
    8: (255, 255, 255),
    9: (255, 255, 255),
    10: (255, 255, 255),
    11: (255, 255, 255),
    12: (255, 255, 255),
    13: (255, 255, 255),
    14: (255, 255, 255),
    15: (255, 255, 255),
    16: (255, 255, 255),
    17: (255, 255, 255),
    18: (255, 255, 255),
    19: (255, 255, 255),}

video_capture = cv2.VideoCapture(0)
# 0 means webcam
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
face_cascade = cv2.CascadeClassifier('Face_Recognition/haarcascade_frontalface_alt2.xml')
while True:
    ret, frames = video_capture.read()
    # converted = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    eyes = eye_cascade.detectMultiScale(gray, 1.2, 5)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(frames,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        for (x, y, w, h) in faces:
            
            # resize image and convert to grayscale
            resized = cv2.resize(gray[y:y+h, x:x+w], (250, 300), interpolation=cv2.INTER_AREA)

            
            prediction = eigenfaces_recognizer.predict(resized)
            confidence_score = 100 - int(prediction[1]/200) # confidence score 0-20000; 0 means perfectly match;
            name = namelist[prediction[0]]  # retrieve name from the list based on the prediction index
            final_label = name + ' ' + str(confidence_score) + '%'
            thecolor = color[prediction[0]] if color[prediction[0]] else (255, 255, 255)
            if(confidence_score < 65):
                final_label = 'Unknown'
                thecolor = (255, 255, 255)

            # Draw a rectangle around the faces
            rec = cv2.rectangle(frames, (x, y), (x + w, y + h), thecolor, 2)
            cv2.putText(rec, final_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, thecolor, 2)

    # Display the resulting frame
    cv2.imshow('Video', frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
