import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
BALL_CLASS_ID = 32  # The class ID for tennis balls in the YOLO model
ball_y_positions = []
cap = cv2.VideoCapture("videos/test_video5.mov")

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, verbose=False)

    for box in results[0].boxes:
        if int(box.cls[0]) == BALL_CLASS_ID:
            cx, cy, w, h = box.xywh[0]
            ball_y_positions.append(float(cy))  
            print("ball y:", round(float(cy), 1))

    annotated = results[0].plot()
    cv2.imshow("Tennis Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



bounces = 0
direction = None

for i in range(1, len(ball_y_positions)):
    diff = ball_y_positions[i] - ball_y_positions[i - 1]

    if diff < 0:
        new_direction = "falling"
    elif diff > 0:
        new_direction = "rising"
    else:
        continue

    if direction == "falling" and new_direction == "rising":
        bounces += 1
    
    direction = new_direction


print("Frames where the ball was detected:", len(ball_y_positions))
print("Bounces counted:", bounces)