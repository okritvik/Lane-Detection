import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('challenge.mp4')

mask = np.zeros((720,1280),dtype=np.uint8) #Masking the upper part of the image
points = np.array([(0,425),(1280,425),(1280,720),(0,720)])
cv2.fillPoly(mask, pts = [points], color=(255,255,255))

# cv2.imshow("Mask",mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# src_points = np.array([[575,480],[326,660],[1062,660],[762,480]])
# dst_points = np.array([[326,480],[326,660],[1062,660],[1062,480]])

src_points = np.array([[410,585],[290,673],[1080,673],[930,585]]) #Source Points to find the homography
dst_points = np.array([[290,585],[290,673],[1080,673],[1080,585]]) #Destination points to find the homography

# src_points = np.array([[575,480],[326,660],[1062,660],[762,480]]) #Source Points to find the homography
# dst_points = np.array([[320,707], [320,10], [970,10], [970,707]]) #Destination points to find the homography

# src_points = np.array([[265,650],[597,448],[753,448],[1135,650]]) #Source Points to find the homography
# dst_points = np.array([[320,707], [320,10], [970,10], [970,707]]) #Destination points to find the homography


H = cv2.findHomography(src_points, dst_points)
H = H[0] #Homography Matrix
print(H)

#16th frame stored to tune the HSV parameters for solid yellow line detection
for i in range(0,16):
    ret,frame = cap.read()
    if ret:
        if(i==15):        
            cv2.imwrite("Q3_Frame.png",frame)

cap.release()

out  = cv2.VideoWriter('Curve_Detection.avi',cv2.VideoWriter_fourcc(*'XVID'),20,(1280,720))
cap = cv2.VideoCapture('challenge.mp4')

#Tuned HSV Parameter values
#(hMin = 19 , sMin = 72, vMin = 195), (hMax = 24 , sMax = 255, vMax = 255)

#Lists to store the previous best values
prev_solid_points = [] 
prev_dashed_points =  []
prev_solid_z = [] #Curve coefficients

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # frame = cv2.rotate
        # print(frame.shape)
        frame_copy = frame.copy()
        cv2.imshow("Original Frame",frame)
        
        #HSV Conversion
        hsv_image = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2HSV)
        lower_range = np.array([19,72,195])
        upper_range = np.array([24,255,255])
        
        hsv_mask = cv2.inRange(hsv_image,lower_range,upper_range) #Creating mask
        # cv2.imshow("HSV Mask",hsv_mask)
        # cv2.imshow("Frame",frame_copy)
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converting to gray Frame
        gray_frame = cv2.GaussianBlur(gray_frame, (3,3), 2) #Applying gaussian blur
        gray_frame = cv2.bitwise_and(gray_frame, mask) #Extracting the required region
        ok, thresholded_frame = cv2.threshold(gray_frame, 225, 255, cv2.THRESH_BINARY) #Thresholding gray frame
        
        morphology_frame = cv2.morphologyEx(thresholded_frame, cv2.MORPH_CLOSE, (7,7)) #Applying the morphology close operator
        # cv2.findHomography(srcPoints, dstPoints)
        # cv2.imshow("Gray Frame",morphology_frame)
        # cv2.waitKey(10)
        projected_solid = cv2.warpPerspective(hsv_mask, H, (1280,720)) #Warping the image acquired from HSV
        projected_dashed = cv2.warpPerspective(morphology_frame, H, (1280,720)) #Warping the image acquired from thresholding
        
        # abc = cv2.warpPerspective(frame_copy, H, (1280,720))
        
        projected = cv2.bitwise_or(projected_solid, projected_dashed) #Combining both for better visual purposes
        
        solid_indices = np.where(projected_solid == 255) #Extracting all the points where there is white in the HSV warped image
        # print(solid_indices)
        
        dashed_indices = np.where(projected_dashed == 255) #Extracting all the points where there is white in the warped image from thresholding
        # print(max(dashed_indices[0])-min(dashed_indices[0]))
        z_solid = np.polyfit(solid_indices[1],solid_indices[0],2) #Curve fitting for solid using polyfit
        # print(z_solid)
        z_dashed = np.polyfit(dashed_indices[1],dashed_indices[0],2) #Curve fitting for dashed using polyfit
        # print(z_dashed)
        
        #Lists to store the coordinates
        solid_x = []
        solid_y = []
        dashed_x = []
        dashed_y = []
        
        #Finding the coordinates from the acquired curve fitting parameters
        for i in range(np.min(solid_indices[1]),np.max(solid_indices[1])):
            y = int(z_solid[0]*i*i + z_solid[1]*i + z_solid[2])
            if(y<projected_solid.shape[0] and y>=0):
                solid_x.append(i)
                solid_y.append(y)
        
        #Finding the coordinates from the acquired curve fitting parameters
        for i in range(np.min(dashed_indices[1]),np.max(dashed_indices[1])):
            y = int(z_dashed[0]*i*i + z_dashed[1]*i + z_dashed[2])
            if(y<projected_dashed.shape[0] and y>=0):
                dashed_x.append(i)
                dashed_y.append(y)
        
        #Creating a blank canvas
        canvas = np.zeros((720,1280,3),dtype=np.uint8)
        solid_draw_points = []
        dashed_draw_points = []
        
        #Appending the originating bottom pixel
        max_y = max(solid_y)
        corr_x = solid_x[solid_y.index(max(solid_y))]
        solid_draw_points.append((corr_x,720))
        
        #Appending the points to a list for polyline drawing
        for i in range(0,len(solid_x)):
            solid_draw_points.append((solid_x[i],solid_y[i]))
            # canvas[solid_y[i]][solid_x[i]] = 255
        
        #Storing the best result incase the pipeline doesn't detect the lines in some frames
        if(len(solid_draw_points)>90):
            prev_solid_points = solid_draw_points.copy()
            prev_solid_z = z_solid.copy()
        
        #Draw the curve
        if(len(prev_solid_points)>0):
            cv2.polylines(canvas, [np.array(prev_solid_points)], False, (255,0,0),20)  # args: image, points, closed, color  
        
        #Appending the originating bottom pixel
        max_y = max(dashed_y)
        corr_x = dashed_x[dashed_y.index(max(dashed_y))]
        dashed_draw_points.append((corr_x,720))
        
        #Appending the points to a list for polyline drawing
        for i in range(0,len(dashed_x)):
            dashed_draw_points.append((dashed_x[i],dashed_y[i]))
        
        max_y = max(dashed_y)
        min_y = min(dashed_y)
        # print(max_y - min_y)
        #Storing the best result incase the pipeline doesn't detect the lines in some frames
        if((len(dashed_indices[0])>1600) and (max(dashed_indices[0])-min(dashed_indices[0]))>400):
            prev_dashed_points = dashed_draw_points.copy()
        
        #Draw the curve
        if(len(prev_dashed_points)>0):
            cv2.polylines(canvas, [np.array(prev_dashed_points)], False, (255,0,0),20)  # args: image, points, closed, color  
        # cv2.polylines(canvas, [np.array(dashed_draw_points)], False, (255,0,0),5)  # args: image, points, closed, color  
        temp = cv2.addWeighted(cv2.cvtColor(projected, cv2.COLOR_GRAY2BGR),1,canvas,0.7,0)
        cv2.imshow("Projected",temp)
        #Filling the space between the two curves
        poly_fill_points = []
        poly_x = []
        poly_y = []
        for i in range(0,len(prev_solid_points)):
            x,y = prev_solid_points[i]
            poly_x.append(x)
            poly_y.append(y)
        #maximum and minimum points of the solid line curve
        poly_fill_points.append([poly_x[poly_y.index(min(poly_y))],min(poly_y)])
        poly_fill_points.append([poly_x[poly_y.index(max(poly_y))],max(poly_y)])
        
        poly_x = []
        poly_y = []
        for i in range(0,len(prev_dashed_points)):
            x,y = prev_dashed_points[i]
            poly_x.append(x)
            poly_y.append(y)
        #Maximum and minimum points of the dashed line curve    
        poly_fill_points.append([poly_x[poly_y.index(max(poly_y))],max(poly_y)])
        poly_fill_points.append([poly_x[poly_y.index(min(poly_y))],min(poly_y)])
        
        #Filling the polygon
        cv2.fillPoly(canvas, [np.array(poly_fill_points)], (255,0,0))
        #inverse warping the canvas
        lanes = cv2.warpPerspective(canvas, np.linalg.inv(H), (1280,720))
        #Combining both the original frame and the inverse warped image
        dst = cv2.addWeighted(frame_copy,1,lanes,0.7,0)
        
        
        ######### Finding Radius of Curvature at the mid point of the curve ########
        
        r_point = prev_solid_points[int(len(prev_solid_points)/2)]
        
        R = (np.sqrt(np.power(( 1 + np.power(2*prev_solid_z[0]*r_point[0],2)),2)))/(2*prev_solid_z[0])
        R = round(R,2)
        # print(R)
         
        cv2.putText(dst,"Radius-"+str(R),(50,50),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2,(0,0,255),2,cv2.LINE_AA)        
        
        cv2.imshow("Canvas",dst)
        # temp = cv2.addWeighted(cv2.cvtColor(projected, cv2.COLOR_GRAY2BGR),1,canvas,0.7,0)
        # cv2.imshow("Projected",temp)
        cv2.waitKey(10)
        out.write(dst)
        
        # print(len(solid_draw_points))
        # print(len(dashed_draw_points))
        # print()
        # print(solid_x)
        # print(solid_y)
        # break
    else:
        break
        
        
cap.release()
out.release()
cv2.destroyAllWindows()
