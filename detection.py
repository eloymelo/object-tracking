import cv2

# Load the face model from our local file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open MacBook camera (index 1)
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert to grayscale (detection works on single-channel)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces — returns list of (x, y, w, h) tuples
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

    # Draw a box around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'Face', (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO

# Load a pre-trained YOLO model
model = YOLO('yolov8n.pt')  # 'n' = nano (fastest, good for learning)

# Detect
results = model(frame)  # frame = BGR image from OpenCV