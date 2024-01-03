import cv2
import numpy as np
import time

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture object with the camera (0 is usually the built-in camera)
cap = cv2.VideoCapture(0)

# Initialize variables for optical flow
old_frame = None
old_gray = None
p0 = None
last_update_time = time.time()
face_center_history = []
history_length = 10  # Adjust this value for the desired length of the history

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection and optical flow
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if old_frame is not None:
        # Optical flow
        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None)

            # Ensure that st is not None and is reshaped to a 1D array if needed
            if st is not None and len(st.shape) == 2 and st.shape[1] == 1:
                st = st.reshape(-1)

            # Filter out valid points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                frame = cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 0, 255), 2)

            # Update face center history
            if len(faces) > 0:
                x, y, w, h = faces[0]
                x_center = x + w // 2
                y_center = y + h // 2
                face_center_history.append((x_center, y_center))

                # Keep only the last N positions in the history
                if len(face_center_history) > history_length:
                    face_center_history = face_center_history[-history_length:]

            # Draw a rectangle around the detected face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Draw a continuous line based on the smoothed face movement history
            for i in range(1, len(face_center_history)):
                cv2.line(frame, face_center_history[i - 1], face_center_history[i], (0, 255, 0), 2)

    # Update the old frame and old_gray
    old_frame = frame.copy()
    old_gray = gray.copy()

    # Update the feature points for optical flow
    if len(faces) > 0:
        # Use the center of the first detected face for tracking
        x, y, w, h = faces[0]
        x_center = x + w // 2
        y_center = y + h // 2
        p0 = np.array([[x_center, y_center]], dtype=np.float32)

    # Check if 5 seconds have elapsed, reset the history
    current_time = time.time()
    if current_time - last_update_time >= 5:
        last_update_time = current_time
        face_center_history = []

    # Display the result
    cv2.imshow('Face Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()