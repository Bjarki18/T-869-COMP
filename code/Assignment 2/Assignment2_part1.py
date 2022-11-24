import cv2
import numpy as np

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = frame[100:480, 0:640]
    # frame = cv2.imread("arg.png",1)
    
    ###### CODE TO ROTATE IMAGE 45 DEG FOR TESTING PURPOSES
    # (h, w) = frame.shape[:2]
    # (cX, cY) = (w // 2, h // 2)
    # M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
    # frame = cv2.warpAffine(frame, M, (w, h))

    edges = cv2.Canny(frame,100,200)
    xy_vals = np.argwhere(edges != 0)
    best_inside = 0
    best_p1 = (0,0)
    best_p2 = (0,0)
    best_inliers = []
    for i in range(50):
        inliers = []
        if len(xy_vals) == 0:
            break
        p1 = xy_vals[np.random.randint(0,len(xy_vals))]
        p2 = xy_vals[np.random.randint(0,len(xy_vals))]
        inliers.append(p1)
        inliers.append(p2)
        inside = 0
        random_xy_vals = xy_vals[np.random.randint(len(xy_vals), size=int(0.02*len(xy_vals)))]
        for p in random_xy_vals:
            # p = xy_vals[j]
            d = np.linalg.norm(np.cross(p2-p1,p1-p))/np.linalg.norm(p2-p1)
            # print(d)
            if (d < 2):
                inside += 1
                inliers.append(p)
        if inside > best_inside:
            best_inside = inside
            best_p1 = p1
            best_p2 = p2
            best_inliers = inliers
    
    # fit = np.polynomial.Polynomial(np.polynomial.Polynomial.fit(inliers[:,1],inliers[:,0],1),t)
    # polytrend = np.poly1d(np.polyfit(inliers[:,1],inliers[:,0],1))
    # print("polytrend: ", polytrend, "\n")
    best_inliers = np.array(best_inliers)
    draw_x = np.arange(0, edges.shape[1], 1)

    ## CHECK IF LINE IS VERTICAL, IF SO THEN COMPUTE POLYFIT AS HORIZONTAL LINE THEN FLIP X Y AXIS TO SHOW VERTICAL
    if (max(best_inliers[:,1]) - min(best_inliers[:,1]) < 50):
        draw_y = np.polyval(np.polyfit(best_inliers[:,0],best_inliers[:,1],1),draw_x)
        draw_points = (np.asarray([draw_y, draw_x]).T).astype(np.int32) 
        cv2.polylines(frame, [draw_points], False, (255,0,0),2) 
    else:
        draw_y = np.polyval(np.polyfit(best_inliers[:,1],best_inliers[:,0],1),draw_x)
        draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32) 
        cv2.polylines(frame, [draw_points], False, (255,0,0),2) 
        
    # print(tuple(reversed(best_p1)), best_p1)
    # frame[100,100] = [255,0,0]
    # frame[draw_points[:,0],draw_points[:,1]] = [0,255,0]
    # frame[best_inliers[:,0],best_inliers[:,1]] = [0,255,0]
    # frame[best_inliers[:,0]+1,best_inliers[:,1]+1] = [0,255,0]
    # frame[best_inliers[:,0]+1,best_inliers[:,1]-1] = [0,255,0]
    # frame[best_inliers[:,0]-1,best_inliers[:,1]+1] = [0,255,0]
    # frame[best_inliers[:,0]-1,best_inliers[:,1]-1] = [0,255,0]
    for x in best_inliers:
        cv2.circle(frame, tuple(reversed(x)), 1, (0, 255, 0), 2)
    # cv2.circle(frame, inliers[10:20], 1, (0, 255, 0), 2)

    # print("\n inliers: ", inliers.sort())
    # print("draw points : ", draw_points)
    
    cv2.line(frame,tuple(reversed(best_p1)),tuple(reversed(best_p2)),(0,0,255),2)
    
    # cv2.line(frame,draw_points[0],draw_points[-1],(255,125,125),2)
    
    # if best_inside > super_best_inside:
    # super_best_points = (tuple(reversed(best_p1)), tuple(reversed(best_p2)))
    # super_best_inside = best_inside
    # print(super_best_inside, super_best_points)
    # print(best_p1, best_p2)
    # print(np.cross(best_p1,best_p2))
    # print(best_inside, best_p1, best_p2)
    # line = np.cross(p1,p2)


    # cv2.putText(frame,text,(17,37),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),2,cv2.LINE_AA)
    # cv2.putText(frame,text,(15,35),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),2,cv2.LINE_AA)

    cv2.imshow('frame',frame)
    # cv2.imshow('edges',edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# cap.release()
cv2.destroyAllWindows()