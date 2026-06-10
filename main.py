import cv2
import mediapipe as mp
import os
from datetime import datetime, timedelta
import time

# MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

# Attendance file
csv_file = "attendance.csv"

if not os.path.exists(csv_file):
    with open(csv_file, "w") as f:
        f.write("Status,Time\n")

print("CSV Location:", os.path.abspath(csv_file))

# Webcam
cap = cv2.VideoCapture(0)

last_mark_time = None  # Track the last time attendance was marked
min_interval = 30      # Minimum seconds between marks

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb)

    if results.detections:
        print("Face Detected")

        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

        cv2.putText(
            frame,
            "FACE DETECTED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        current_time = datetime.now()
        
        # Check if enough time has passed since the last mark
        if last_mark_time is None or (current_time - last_mark_time).total_seconds() >= min_interval:
            current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
            with open(csv_file, "a") as f:
                f.write(f"Present,{current_time_str}\n")

            print(f"Attendance Marked: {current_time_str}")
            last_mark_time = current_time

    cv2.imshow("Attendance System", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()