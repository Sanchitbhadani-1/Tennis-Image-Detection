from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("images/test_image.jpg")
result = results[0]

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    name = model.names[class_id]
    print(name, round(confidence, 2))

result.save(filename="output.jpg")