import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

BALL_CLASS_ID = 32
THRESHOLD = 10


ACCENT = (79, 255, 223)   
PANEL = (28, 28, 28)      
WHITE = (240, 240, 240)


def draw_counter(img, bounces):
   
    h, w = img.shape[:2]
    
    s = w / 1280.0

    pad = int(24 * s)
    panel_w = int(300 * s)
    panel_h = int(130 * s)
    x0, y0 = pad, pad
    x1, y1 = x0 + panel_w, y0 + panel_h

    
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), PANEL, -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
   
    cv2.rectangle(img, (x0, y0), (x0 + int(8 * s), y1), ACCENT, -1)

    
    cv2.putText(img, "BOUNCES", (x0 + int(28 * s), y0 + int(38 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, WHITE, max(1, int(2 * s)),
                cv2.LINE_AA)

    
    cv2.putText(img, str(bounces), (x0 + int(26 * s), y1 - int(22 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 2.6 * s, ACCENT, max(2, int(6 * s)),
                cv2.LINE_AA)


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
    draw_counter(annotated, bounces)

    cv2.imshow("Live Bounce Counter", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
