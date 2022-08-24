# Lane Detection
Part of the Second Project - ENPM 673 - Perception for Autonomous Robots

## Required Libraries:  
* OpenCV : Version 4.1
* NumPy
* Matplotlib

## Pipeline for lane detection using contours: 
<p align="center">
  <img src="https://user-images.githubusercontent.com/40200916/186532304-51264980-49cf-4285-99ee-6cc6577e1945.png" width="50%">
</p>

## Pipeline for lane detection using hough-lines: 
<p align="center">
  <img src="https://user-images.githubusercontent.com/40200916/186532386-e0dfb8b9-46f7-4ed1-8ab4-841a739a60a1.png" width="50%">
</p>

## Pipeline for curved lane detection:
<p align="center">
  <img src="https://user-images.githubusercontent.com/40200916/186532476-0efd47b3-0e9c-4421-a7ba-4e238b80d6d9.png" width="50%">
</p>

## Given Dataset:
* whiteline.mp4 file is required for the lane detection using contours and hough-lines.
* challenge.mp4 file is required for the curved lane detection.

## Running the Code:

**Format/Syntax:**  
* ```python3 lane_detection.py```
* ```python3 curve_detection.py```

## Result:
### Lane Detection using Contours
<p align="center">
  <img src="https://user-images.githubusercontent.com/40200916/186533464-cc0ced39-2d11-4cd6-b636-b801991f94c9.gif" width="50%">
</p>

### Lane Detection using Hough Lines
<p align="center">
  <img src="https://user-images.githubusercontent.com/40200916/186533835-6eb40843-329d-4eea-9ebc-bde969bb04e9.gif" width="50%">
</p>

### Curved Lane Detection with Solid Yellow Line
<p align="center">
  <img src="https://user-images.githubusercontent.com/40200916/186534204-600c01c6-a5a8-4d79-ae9b-79b0330e9cd8.gif" width="50%">
</p>

## Exiting From the Code:

1. Press any key after the visualization is done. Make sure that the image window is selected while giving any key as an input.

2. You can always use ctrl+c to exit from the program.

### Note:
* Use HSV.py script with Q3_Frame.png to find HSV values for the yellow solid line masking. The files are available in the repository. Otherwise, choose any random frame from the challenge.mp4 and save it as an image to run HSV on top of it for further fine tuning.
* Uncomment some of the lines in the code to vizualize intermediate results.
