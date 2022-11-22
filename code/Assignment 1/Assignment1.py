import cv2
import time
import numpy as np


cap = cv2.VideoCapture(1)
sum_time = 0
sec_time = time.time_ns()
text = ""

while(True):
    start_time = time.time_ns()
    ret, frame = cap.read()

    ## brightest point - CV
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (1, 1), 0)
    # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    # cv2.circle(frame, maxLoc, 1, (255, 0, 0), 2)

    ## reddest point - CV
    # image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower = np.array([155,25,0])
    # upper = np.array([179,255,255]) 
    # mask = cv2.inRange(image, lower, upper)
    # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image[:,:,1], mask=mask)
    # cv2.circle(frame, maxLoc, 1, (255, 0, 0), 2)

    ## brightest point - double for
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # max_val = 0
    # for i in range(len(gray)):
    #     for j in range(len(gray[1])):
    #         if ((gray[i][j]) > max_val):
    #             max_val = (gray[i][j])
    #             loc = (j,i)
    # cv2.circle(frame, loc, 1, (255, 0, 0), 2)

    ## reddest red - double for
    # image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # max_val = [0,100,100]
    # for i in range(len(image)):
    #     for j in range(len(image[1])):
    #         # print(temp_val[0],temp_val[1],temp_val[2])
    #         if (image[i][j][1] >= max_val[1] & image[i][j][2] >= max_val[2] & (image[i][j][0] <= 10)):
    #             max_val = image[i][j]
    #             loc = (j,i)
    # print(max_val,loc)
    # cv2.circle(frame, loc, 1, (255, 0, 0), 2)
    
    
    # loc = (0, 0)
    # max_val = [0,0,100]
    # (x,y,z) = (image.shape)
    # for i in range(x):
    #     for j in range(y):
    #         if (image[i, j][2] >= max_val[2] and image[i,j][1] >= max_val[1] and image[i,j][0] == 0):
    #             max_val = image[i, j]
    #             loc = (j, i)
    # cv2.circle(frame, loc, 1, (255, 0, 0), 2)


    end_time = time.time_ns()
    
    tot_time_per_frame = end_time - start_time

    print(tot_time_per_frame)
    sum_time += 1

    if time.time_ns() - sec_time > 1e9:
        text = str(sum_time)
        sum_time = 0
        sec_time = time.time_ns()

    text = str(int(1e9/tot_time_per_frame))
    cv2.putText(frame,text,(17,37),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame,text,(15,35),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),2,cv2.LINE_AA)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()