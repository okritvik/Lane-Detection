import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('whiteline.mp4')

mask = np.zeros((540,960),dtype=np.uint8) #masking the top half of the frame
points = np.array([(0,345),(960,345),(960,540),(0,540)])
cv2.fillPoly(mask, pts = [points], color=(255,255,255))

# cv2.imshow("Mask",mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

prev_line_points = None #To store the previous better detected points
prev_line_slope = None #To store the previous better detected corresponding slope
out1  = cv2.VideoWriter('Hough_Lines_Q2.avi',cv2.VideoWriter_fourcc(*'XVID'),20,(960,540)) #Video writer for detection using Hough Lines
out2  = cv2.VideoWriter('Contours_Q2.avi',cv2.VideoWriter_fourcc(*'XVID'),20,(960,540)) #Video writer for detection using contours

while cap.isOpened():
    ret, frame = cap.read()
    # print(frame.shape)
    
    if ret:
        frame_copy = frame.copy()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converting to gray image
        gray_frame = cv2.bitwise_and(gray_frame, mask) #Masking the top part of the image
        ok, thresholded_frame = cv2.threshold(gray_frame, 230, 255, cv2.THRESH_BINARY) #Thresholding the gray image
        cv2.imshow("THRESH Frame",thresholded_frame)
        # cv2.waitKey(100)
        
        edge =  cv2.Canny(thresholded_frame, 200, 255) #Canny Edge Detection of the thresholded frame
        edge = cv2.GaussianBlur(edge, (5,5), 3) #Blurring the detected edges
        cv2.imshow("Canny",edge)
        # cv2.waitKey(100)
        
        #Detecting the lines using Hough Lines
        lines = cv2.HoughLinesP(
            edge, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=75, # Min number of votes for valid line
            minLineLength=5, # Min allowed length of line
            maxLineGap=5 # Max allowed gap between line for joining them
            )
        # print(lines)
        solid_points = [] #List to store the solid line points detected using Hough Lines
        solid_dist = [] #Corresponding Distance of the line
        dash_points = [] #List to store the dashed line points detected using Hough Lines
        dash_slope= []#Corresponding Slope
        
        #Extracting each points from the detected lines
        for points in lines:
            # Extracted points nested in the list
            x1,y1,x2,y2=points[0]
            #Calculating the eucledian distance
            dist  = np.sqrt(np.power(x1-x2,2) + np.power(y1-y2,2))
            #Initial Seggragation using the distance (Maximum Distance is for solid line)
            if(dist>200):
                solid_points.append([x1,y1,x2,y2])
                solid_dist.append(dist)
                # cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),5)
            if(dist<100):
                dash_points.append([x1,y1,x2,y2])
        
        #Extracting the points with maximum eucledian distance.
        x1,y1,x2,y2 = solid_points[solid_dist.index(max(solid_dist))]
        solid_slope = (y2-y1)/(x2-x1) #Calculating the corresponding slope
        
        # print("Solid Line Slope: ", solid_slope)
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),5)    
        
        #Note that the slope of the other line (dashed) is of opposite sign to the solid line
        #That means, slope1*slope2 is always negative.
        #Seggragating the computed dashed lines from the above using the slopes.
        if len(dash_points)!=0:
            for i in range(0,len(dash_points)):
                x3,y3,x4,y4 = dash_points[i]
                d_slope = (y4-y3)/(x4-x3)
                if (d_slope*solid_slope)<0:
                    dash_slope.append(d_slope)
                    prev_line_points = (x3,y3,x4,y4)
            if(len(dash_slope)!=0):
                prev_line_slope = np.mean(dash_slope)
        
        #Computing the minimum and maximum pixel coordinates from the equation of the line
        if prev_line_slope is not None:
            # print(prev_line_slope)
            # print(prev_line_points)
            # z = np.polyfit([prev_line_points[0],prev_line_points[2]],[prev_line_points[1],prev_line_points[3]],1)
            # print("Z = ",z)
            y_points = []
            x_points = []
            for i in range(0,400):
                if(prev_line_slope*i + prev_line_points[1]-(prev_line_points[0]*prev_line_slope))<frame.shape[0]:
                    y_points.append(prev_line_slope*i + prev_line_points[1]-(prev_line_points[0]*prev_line_slope))
                    x_points.append(i)
            min_index = y_points.index(min(y_points))
            max_index = y_points.index(max(y_points))
            cv2.line(frame,(int(x_points[min_index]),int(y_points[min_index])),(int(x_points[max_index]),int(y_points[max_index])),(0,0,255),5)
        
            road_points = np.array([(x1,y1),(x2,y2),(int(x_points[max_index]),int(y_points[max_index])),(int(x_points[min_index]),int(y_points[min_index]))])
            cv2.fillPoly(frame, pts = [road_points], color=(135,239,225))
        
       
        #Detection using Contours
        
        #Find contours of the thresholded image
        contours, hierarchy = cv2.findContours(image=thresholded_frame, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        #Find the areas
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas) #Finding index of the maximum area                              
    	# draw contours on the original image
        cv2.drawContours(image=frame_copy, contours=contours, contourIdx=max_index, color=(0, 255, 0), thickness=5, lineType=cv2.LINE_AA)                
        # print("Contour",contours[max_index])
        areas.sort()
        second_max_index = len(areas)-2 #Finding the second maximum and third maximum areas (Dashed lines)
        third_max_index = len(areas)-3
        
        #Drawing the contours
        for c in contours:
            if cv2.contourArea(c) == areas[second_max_index]:
                cv2.drawContours(image=frame_copy,contours=c,contourIdx = -1,color=(0,0,255),thickness=5,lineType = cv2.LINE_AA)
                # print("C",c)
                break
        for c in contours:
            if cv2.contourArea(c) == areas[third_max_index]:
                cv2.drawContours(image=frame_copy,contours=c,contourIdx = -1,color=(0,0,255),thickness=5,lineType = cv2.LINE_AA)
                # print("C",c)
                break
        
        # see the results
        cv2.imshow('Using Contours', frame_copy)
        cv2.imshow("Using Hough Lines",frame)
        out1.write(frame)
        out2.write(frame_copy)
        cv2.waitKey(10)
    
    else:
        break
    
cap.release()
out1.release()
out2.release()
cv2.destroyAllWindows()