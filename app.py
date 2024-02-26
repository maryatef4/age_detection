# import the necessary packages
import numpy as np
import argparse
from imutils import face_utils
import imutils
import cv2
import dlib
from tracker import Tracker
from ultralytics import YOLO
import pandas as pd

#Loading count model
count_model = YOLO('yolov8s.pt')
# Read Coco.txt to get class names objects for that we can detect it using yolo model
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="deploy.prototxt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="res10_300x300_ssd_iter_140000.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
# init age parameters 
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
# load our serialized model from disk
print("[INFO] loading models...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

age_net = cv2.dnn.readNetFromCaffe(
                        "age_gender_models/deploy_age.prototxt", 
                        "age_gender_models/age_net.caffemodel")

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")


ct = Tracker()
cap =cv2.VideoCapture("C:\\Users\\MISHO TECHNOLOGY\\Downloads\\vid5.mp4")
while True:
    success, frame = cap.read()
    if not success:
        break
    else:
        frame = cv2.resize(frame, (700, 500))
        results = count_model.predict(frame) 
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        list = []
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
        
            c = class_list[d]
            if 'person' in c:
                list.append([x1, y1, x2, y2]) 
                 # Check if faces are detected before processing
                if list:
                    bbox_id = ct.update(list)
                    for bbox in bbox_id:
                        #box axis and id
                        x3, y3, x4, y4, id = bbox
                        cx = (x3 + x4) // 2
                        cy = y3
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # frame = imutils.resize(frame, width=400)

                        detector = dlib.get_frontal_face_detector()
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        rects = detector(gray, 0)

                        objects_rect = []

                        for rect in rects:
                            (x, y, w, h) = face_utils.rect_to_bb(rect)
                            if y < 0:
                                print("a")
                                continue
                            objects_rect.append((x, y, x + w, y + h))

                            face_img = frame[y:y+h, x:x+w].copy()

                            # if face_img is not None and not face_img.size == (0, 0):
                            if face_img is not None and not face_img.size == (0, 0) and face_img.shape[0] > 0 and face_img.shape[1] > 0:

                                blob2 = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                                age_net.setInput(blob2)
                                age_preds = age_net.forward()
                                predicted_age_index = age_preds[0].argmax()
                                predicted_age_range = age_list[predicted_age_index]
                                lower_bound_age = int(''.join(filter(str.isdigit, predicted_age_range.split('-')[0])))

                                if lower_bound_age >= 0 and lower_bound_age < 18:
                                    age_category = "child"
                                    print('child')
                                else:
                                    age_category = "adult"
                                    print('adult')

                                cv2.putText(frame, age_category, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                                # cv2.rectangle(face_img, (x1, y1), (x2, y2), (255, 255, 0), 1)
                            else:
                                print("Invalid image or image size is (0, 0)")


                        # Update the tracker with the detected faces
                        tracked_objects = ct.update(objects_rect)
                else:
                    tracked_objects = []  # No faces detected, reset tracked_objects

              
    cv2.imshow("frame",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break
   

cap.release()
cv2.destroyAllWindows()