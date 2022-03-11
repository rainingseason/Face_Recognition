import cv2
import numpy
import os
import time
from mtcnn import MTCNN

# fn_haar = 'haarcascade_frontalface_default.xml' # higher recall
fn_haar = 'haarcascade_frontalface_alt2.xml'  # higher precise
fn_dir = 'database'
counter = 0


print('Training...')
# Create a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(fn_dir):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(fn_dir, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
            counter += 1
        id += 1
# (im_width, im_height) = (100, 100)

# Create a Numpy array from the two lists above
(images, lables) = [numpy.array(lis) for lis in [images, lables]]

# OpenCV trains a model from the images
model = cv2.face.EigenFaceRecognizer_create()
# model = cv2.face.FisherFaceRecognizer_create() # not available now
# model = cv2.face.LBPHFaceRecognizer_create() # percentage need to change


model.train(images, lables)

print(str(counter) + ' images for training done!')
#################### Training done !!!! #####################

#################### predefined dictionary !!!! #####################
namelist = ['alyna', 'charles', 'Kailun', 'Truong']  # index 0 1 2...
color = {
    0: (238, 238, 0),  # alyna cyan
    1: (255, 62, 191),  # charles purple
    2: (255, 255, 255),  # kailun white
    3: (255, 0, 0),  # truong blue
    4: (0, 255, 0),  # unused green
    5: (0, 0, 255),  # unused red
}

# notes: (B,G,R)

#################### prediction !!!! #####################
# Choose either 1
# faceCascade = cv2.CascadeClassifier(fn_haar)
detector = MTCNN()

video_capture = cv2.VideoCapture(0)  # 0 means webcam
# video_capture.set(3, 1920)  # width
# video_capture.set(4, 1080)  # height

width = 100
height = 100
dim = (width, height)

# For CV2 classifier
'''
while True:
    # time.sleep(0.05)
    ret, frames = video_capture.read()

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        try:
            # resize image and convert to grayscale
            resized = cv2.resize(gray[x:x + w, y:y + h], dim)
            # converted = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            prediction = model.predict(resized)

            name = namelist[prediction[0]]  # retrieve name from the list based on the prediction index
            confidence_score = str(
                round((100 - (prediction[1] / 200)), 1)) + '%'  # confidence score 0-20000; 0 means perfectly match;
            final_lable = name + ' ' + confidence_score
            thecolor = color[prediction[0]] if color[prediction[0]] else (255, 255, 255)

            rec = cv2.rectangle(frames, (x, y), (x + w, y + h), thecolor, 2)
            cv2.putText(rec, final_lable, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, thecolor, 2)
        except:
            pass
    # Display the resulting frame
    cv2.imshow('Video', frames)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
'''

# For MTCNN classifier
while (True):
    # time.sleep(0.5)
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.resize(frame, (600, 400))
    faces = detector.detect_faces(frame)
    for face in faces:
        box = face['box']
        conf = face['confidence']
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 1)
        if conf > 0:
            try:
                gray_resized_face = cv2.resize(gray[x:x + w, y:y + h], dim)
                prediction = model.predict(gray_resized_face)

                name = namelist[prediction[0]]  # retrieve name from the list based on the prediction index
                confidence_score = str(
                    round((100 - (prediction[1] / 200)), 1)) + '%'  # confidence score 0-20000; 0 means perfectly match;
                final_lable = name + ' ' + confidence_score
                thecolor = color[prediction[0]] if color[prediction[0]] else (255, 255, 255)

                rec = cv2.rectangle(frame, (x, y), (x + w, y + h), thecolor, 2)
                cv2.putText(rec, final_lable, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, thecolor, 2)
            except:
                pass
    cv2.imshow('Video', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break