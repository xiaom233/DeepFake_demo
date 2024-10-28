import cv2
from ultralytics import YOLO
import math

model = YOLO('yolov8n-face.pt')
print(model.names)
webcamera = cv2.VideoCapture(0)
# webcamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# webcamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    success, img = webcamera.read()
    results = model(img, conf=0.5, stream=True)
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.rectangle(img, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)

            # Draw a label with a name below the face


            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            # print("Confidence --->", confidence)

            # class name
            cls = int(box.cls[0])

            # object details
            org = [x1+6, y2-10]
            fontScale = 0.5
            color = (255, 255, 255)
            thickness = 1

            # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, f"Face:{confidence}", org, font, fontScale, color, thickness)

            # cv2.putText(img, f"Face:{confidence}", org, font, fontScale, color, thickness)
    cv2.namedWindow("Meeting", cv2.WINDOW_FREERATIO)
    # cv2.namedWindow("Webcam", 0)
    cv2.imshow('Meeting', img)
    if cv2.waitKey(1) == ord('q'):
        break

webcamera.release()
cv2.destroyAllWindows()