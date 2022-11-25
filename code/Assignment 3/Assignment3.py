import cv2
import numpy as np
import time

cap = cv2.VideoCapture(2)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

classesFile = 'coco.names'
classNames = []

with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# modelConfiguration = 'yolov3-320.cfg'
# modelWeights = 'yolov3.weights'
modelConfiguration = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    
    for i in indices:
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x-50,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

sum_time = 0
sec_time = time.time_ns()
text=""
while True:
    start_time = time.time_ns()
    success, img = cap.read()
    img = cv2.flip(img,1)
    # img = cv2.imread("stop.jpg")
    # img = cv2.imread("stopsmall.jpg")
    # img = cv2.imread("abb.jpg")
    # img = cv2.imread("dogcatsmall.jpg")
    # print(img.shape)
    # img = img [0:int(img.shape[0]*0.3),0:int(img.shape[1]*0.3)]
    # print(img.shape)

    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)

    findObjects(outputs,img)

    end_time = time.time_ns()
    
    tot_time_per_frame = end_time - start_time

    # print(tot_time_per_frame)
    sum_time += 1

    if time.time_ns() - sec_time > 1e9:
        text = str(sum_time)
        sum_time = 0
        sec_time = time.time_ns()

    cv2.putText(img,text,(17,37),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(img,text,(15,35),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),2,cv2.LINE_AA)

    # img = cv2.resize(img, (640,640))
    cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
