import cv2
import numpy as np
from scipy.spatial import distance as dist

# Constants
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES_THRESHOLD = 20

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to calculate EAR
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # Calculate the EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize variables
frame_counter = 0
alarm_on = False

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract region of interest (ROI) for the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Apply eye detection within the face ROI
        eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml').detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # Extract region of interest (ROI) for the eye
            eye_roi_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_roi_color = roi_color[ey:ey+eh, ex:ex+ew]

            # Resize the eye ROI for better EAR calculation
            eye_roi_gray = cv2.resize(eye_roi_gray, (64, 32))
            eye_roi_gray = cv2.GaussianBlur(eye_roi_gray, (5, 5), 0)

            # Calculate the EAR for the eye
            eye_ear = eye_aspect_ratio(eye_roi_gray)

            # Draw the eyes on the frame
            eye_x = x + ex
            eye_y = y + ey
            cv2.rectangle(frame, (eye_x, eye_y), (eye_x + ew, eye_y + eh), (0, 255, 0), 2)

            # Check if EAR falls below the threshold
            if eye_ear < EAR_THRESHOLD:
                frame_counter += 1

                # If the eyes are closed for consecutive frames, display a red light
                if frame_counter >= CONSECUTIVE_FRAMES_THRESHOLD and not alarm_on:
                    # Add a red light on the top of the frame
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
                    alarm_on = True

                    # Add any additional alert mechanisms here (e.g., visual alert, haptic feedback)

            else:
                frame_counter = 0
                alarm_on = False

            # Display the EAR on the frame
            cv2.putText(frame, f"EAR: {eye_ear:.2f}", (eye_x, eye_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Drowsy Driver Detector', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
