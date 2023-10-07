# from turtle import color
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 648)
cap.set(4, 480)

classNames = []
classFile = 'coco.names'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(328, 328)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()

    classIds,confs,bbox = net.detect(img, confThreshold=0.6)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color = (0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0, 255, 0),2)
            cv2.putText(img, str(confidence),(box[0]+100, box[1]+20),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)



    cv2.imshow('output', img)
    cv2.waitKey(1)