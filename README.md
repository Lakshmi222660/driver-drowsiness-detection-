# Driver Drowsiness Detection System

Driver Drowsiness Detection is a real-time computer vision project developed using Python, OpenCV, and Dlib. The system continuously monitors the driver’s eye movements using facial landmarks and provides visual and audio alerts when drowsiness or sleep is detected.

## Project Logic
This project uses Dlib’s 68 facial landmark detector along with OpenCV for real-time video processing. Using eye landmarks, the Eye Aspect Ratio (EAR) is calculated to determine the driver’s alertness level.

## Eye Aspect Ratio (EAR)
EAR is calculated using the ratio of vertical eye distances to horizontal eye distance. A lower EAR indicates closed eyes.

## Working of the Project
- Webcam captures live video
- Face and facial landmarks are detected
- EAR is computed in real time
- Time-based logic detects sleeping and drowsiness
- Audio alerts are triggered on state change
- Driver state is logged with timestamps
- Dashboard UI displays status and metrics

## Technologies Used
Python, OpenCV, Dlib, NumPy, Imutils, Playsound

## How to Run
1. Install dependencies  
   `pip install opencv-python dlib imutils numpy playsound`
2. Run  
   `python main.py`
3. Press Q or ESC to exit

## Conclusion
This system helps prevent accidents by detecting driver fatigue in real time using computer vision.