import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")        

BALL_CLASS_ID = 32
THRESHOLD = 10


bounces = 0
last_extreme = None
looking_for = "low"

cap = cv2.VideoCapture(0)          

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, verbose=False)

    for box in results[0].boxes:
        if int(box.cls[0]) == BALL_CLASS_ID:
            cx, cy, w, h = box.xywh[0]
            cy = float(cy)

            if last_extreme is None:
                last_extreme = cy                  
            elif looking_for == "low":
                if cy > last_extreme:
                    last_extreme = cy             
                elif last_extreme - cy >= THRESHOLD:
                    bounces += 1                  
                    looking_for = "high"
                    last_extreme = cy
            else:  
                if cy < last_extreme:
                    last_extreme = cy              
                elif cy - last_extreme >= THRESHOLD:
                    looking_for = "low"            
                    last_extreme = cy

   
    annotated = results[0].plot()
    cv2.putText(annotated, f"Bounces: {bounces}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Live Bounce Counter", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

