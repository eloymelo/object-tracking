import cv2

# Try camera indices 0-4
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w, _ = frame.shape
            print(f"Camera {i}: ✅ WORKS ({w}x{h})")
        else:
            print(f"Camera {i}: ❌ No image")
        cap.release()
    else:
        print(f"Camera {i}: ❌ Not found")