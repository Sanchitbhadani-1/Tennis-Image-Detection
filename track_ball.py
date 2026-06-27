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


THRESHOLD = 8 #tuneable threshold for bounce detection
bounces = 0
last_extreme = ball_y_positions[0]
looking_for = "low"

for y in ball_y_positions[1:]:
    if looking_for == "low":
        
        if y > last_extreme:
            last_extreme = y                    
        elif last_extreme - y >= THRESHOLD:    
            bounces += 1                       
            looking_for = "high"
            last_extreme = y
    else:  
        
        if y < last_extreme:
            last_extreme = y                    
        elif y - last_extreme >= THRESHOLD:     
            looking_for = "low"               
            last_extreme = y

print("Bounces counted:", bounces)



print("Frames where the ball was detected:", len(ball_y_positions))
print("Bounces counted:", bounces)