import cv2
import keyboard

#not good for detecting crowded places!
#really good for detecting fast moving objects 

personColor = (0,0,200)
areaColor = (0,200,200)
startDL = (0,995)#(40,1100)#DL stands for detection line, a.k.a the place where people will be detected
endDL = (1198,5)#(100,1200)
presize = 250 # defines the quaility of the detection

#initialize camera
cap = cv2.VideoCapture("in.avi")# change to any video

#object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history = 800,varThreshold=127)

while(cap.isOpened()):
    #capture corruption
    ret, frame = cap.read()
    if not ret:
        print("Corrupted Frame!")
        continue
   
    width  = cap.get(3)
    height = cap.get(4)
    print(height, width)
    
    #area of interest
    roi = frame[40 :950,150 :1100]
    ##roi = cv2.rectangle(frame, startDL, endDL, areaColor, 2) # allows detection only in the shopping line
    
   
    #object detection
    mask = object_detector.apply(roi)
    _,mask = cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    contures, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contures:
        #calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > presize:
            #cv2.drawContours(roi, [cnt],-1,(255,0,0),2)
            x,y,w,h = cv2.boundingRect(cnt) 
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,200),3)
   
    cv2.imshow("frame",frame)
    cv2.imshow("mask",mask)
    key = cv2.waitKey(1)
    if key == 27:    #quit the program if esc is pressed <--------------------------------------------------------
        cap.release()
        cv2.destroyAllWindows()
        break