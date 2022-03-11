import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

training_data_folder_path = 'C:/Users/phuct/PycharmProjects/Project/dataset/training-data'
test_data_folder_path = 'C:/Users/phuct/PycharmProjects/Project/dataset/test-data'

random_image = cv2.imread('C:/Users/phuct/PycharmProjects/Project/dataset/training-data/7/Truong_1.jpeg')
fig = plt.figure()
ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))
ax1.set_title('Image from category 7')# change category name accordingly
plt.imshow(cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB))
plt.show()

#Face Detection
haarcascade_frontalface = 'C:/Users/phuct/PycharmProjects/Project/opencv_xml_files/haarcascade_frontalface.xml'
def detect_face(input_img):
    image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('C:/Users/phuct/PycharmProjects/Project/opencv_xml_files/haarcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5);
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
                resized_face = cv2.resize(face, (121, 121), interpolation=cv2.INTER_AREA)
                detected_faces.append(resized_face)
                face_labels.append(label)

    return detected_faces, face_labels

detected_faces, face_labels = prepare_training_data('C:/Users/phuct/PycharmProjects/Project/dataset/training-data')
print("Total faces: ", len(detected_faces))
print("Total labels: ", len(face_labels))

#Init a Face recognition
eigenfaces_recognizer = cv2.face.EigenFaceRecognizer_create()

#Train the face recognizer model
eigenfaces_recognizer.train(detected_faces, np.array(face_labels))

def draw_rectangle(test_image, rect):
    (x, y, w, h) = rect
    cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(test_image, label_text, x, y):
    cv2.putText(test_image, label_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

#Predict Output on test data
def predict(test_image):
    detected_face, rect = detect_face(test_image)
    resized_test_image = cv2.resize(detected_face, (121,121), interpolation = cv2.INTER_AREA)
    label= eigenfaces_recognizer.predict(resized_test_image)
    label_text = tags[label[0]]
    draw_rectangle(test_image, rect)
    draw_text(test_image, label_text, rect[0], rect[1]-5)
    return test_image, label_text

tags = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

#Change the path of test image
test_image = cv2.imread("C:/Users/phuct/PycharmProjects/Project/dataset/test-data/9/Charles_6.jpeg")

predicted_image, label = predict(test_image)

fig = plt.figure()
ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))
ax1.set_title('actual class: ' + tags[9]+ ' | ' + 'predicted class: ' + label)
plt.axis("off")
plt.imshow(cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB))
plt.show()

