import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("videos/test_video2.mov")

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, verbose=False)
    annotated = results[0].plot()
    cv2.imshow("Tennis Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()