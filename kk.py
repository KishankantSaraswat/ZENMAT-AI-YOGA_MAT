import cv2

def test_camera():
    for i in range(5):  # Check multiple indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera is available at index {i}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                cv2.imshow('Camera Test', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
            return
        else:
            print(f"No camera at index {i}")
        cap.release()

    print("No working camera found")

test_camera()
