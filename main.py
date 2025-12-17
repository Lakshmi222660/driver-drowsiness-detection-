import cv2
import numpy as np
import dlib
from imutils import face_utils
from playsound import playsound
import threading
import os

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SLEEP_SOUND = os.path.join(BASE_DIR, "sleep.wav")
DROWSY_SOUND = os.path.join(BASE_DIR, "drowsy.wav")
ACTIVE_SOUND = os.path.join(BASE_DIR, "active.wav")  # optional sound
PREDICTOR = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")

# Check if files exist
for file_path in [SLEEP_SOUND, DROWSY_SOUND, PREDICTOR]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

# ---------------- SOUND FUNCTION ----------------
def play_sound(path):
    if os.path.exists(path):
        threading.Thread(target=playsound, args=(path,), daemon=True).start()

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR)

# ---------------- THRESHOLDS ----------------
EAR_ACTIVE = 0.28
EAR_DROWSY = 0.22
MAR_YAWN = 0.70

SLEEP_FRAMES = 35
DROWSY_FRAMES = 20
YAWN_FRAMES = 15

sleep_cnt = drowsy_eye_cnt = yawn_cnt = 0
status = "Active :)"
prev_status = ""

# ---------------- FUNCTIONS ----------------
def dist(a, b):
    return np.linalg.norm(a - b)

def eye_aspect_ratio(eye):
    A = dist(eye[1], eye[5])
    B = dist(eye[2], eye[4])
    C = dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist(mouth[2], mouth[6])
    B = dist(mouth[3], mouth[5])
    C = dist(mouth[0], mouth[4])
    return (A + B) / (2.0 * C)

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # draw landmarks (optional, like first screenshot)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        leftEye = shape[36:42]
        rightEye = shape[42:48]
        mouth = shape[60:68]

        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2
        mar = mouth_aspect_ratio(mouth)

        # ---------------- LOGIC ----------------
        if ear < EAR_DROWSY:
            sleep_cnt += 1
            drowsy_eye_cnt = yawn_cnt = 0
            if sleep_cnt >= SLEEP_FRAMES:
                status = "SLEEPING !!!"
        elif EAR_DROWSY <= ear < EAR_ACTIVE:
            drowsy_eye_cnt += 1
            sleep_cnt = 0
            if drowsy_eye_cnt >= DROWSY_FRAMES:
                status = "Drowsy !"
        elif mar > MAR_YAWN:
            yawn_cnt += 1
            sleep_cnt = drowsy_eye_cnt = 0
            if yawn_cnt >= YAWN_FRAMES:
                status = "Drowsy !"
        else:
            sleep_cnt = drowsy_eye_cnt = yawn_cnt = 0
            status = "Active :)"

        # ---------------- ALERT ----------------
        if status != prev_status:
            if status == "SLEEPING !!!":
                play_sound(SLEEP_SOUND)
            elif status == "Drowsy !":
                play_sound(DROWSY_SOUND)
            elif status == "Active :)":
                # optional: uncomment if you want sound for active state
                # play_sound(ACTIVE_SOUND)
                pass
            prev_status = status

        # ---------------- DISPLAY ----------------
        color = (0, 255, 0)  # Green for active
        if status == "Drowsy !":
            color = (0, 0, 255)  # Red for drowsy
        if status == "SLEEPING !!!":
            color = (255, 0, 0)  # Blue for sleeping

        cv2.putText(frame, status, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()