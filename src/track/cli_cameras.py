import cv2


def main(max_index: int = 5) -> None:
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w, _ = frame.shape
                print(f"Camera {i}: works ({w}x{h})")
            else:
                print(f"Camera {i}: no image")
            cap.release()
        else:
            print(f"Camera {i}: not found")


if __name__ == "__main__":
    main()