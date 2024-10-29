import cv2
from ultralytics import YOLO
import math
import os
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from network import MFF_MoE
import numpy as np
from modules.processors.frame.face_swapper import process_image_numpy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
local_weight_dir = r""


class NetInference():
    def __init__(self):
        self.net = MFF_MoE(pretrained=False)
        self.net.load(path=local_weight_dir)
        self.net = nn.DataParallel(self.net).cuda()
        self.net.eval()
        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((512, 512), antialias=True),
        ])

    def infer(self, x):
        # x = cv2.imread(input_path)[..., ::-1]
        x = Image.fromarray(np.uint8(x))
        x = self.transform_val(x).unsqueeze(0).cuda()
        pred = self.net(x)
        pred = pred.detach().cpu().numpy()
        return pred

deepfake_detector = NetInference()

detect_model = YOLO('yolov8n-face.pt').cuda()
source_path = r"imgs/musk.jpg"
print(detect_model.names)
webcamera = cv2.VideoCapture(0)
# webcamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# webcamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    success, img = webcamera.read()
    fake = process_image_numpy(source_path, img)
    deep_fake_result = deepfake_detector.infer(fake)
    results = detect_model(img, conf=0.5, stream=True)
    fake_results = detect_model(fake, conf=0.5, stream=True)

    rect_img = img
    rect_fake = fake
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(rect_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.rectangle(rect_img, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)

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
            cv2.putText(rect_img, f"Face:{confidence}", org, font, fontScale, color, thickness)

    for r in fake_results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(rect_fake, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.rectangle(rect_fake, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)

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
            cv2.putText(rect_fake, f"Fake face:{deep_fake_result}", org, font, fontScale, color, thickness)

    cv2.namedWindow("input", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("fake", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("input_detect", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("deepfake_detect", cv2.WINDOW_FREERATIO)

    # cv2.namedWindow("Webcam", 0)
    cv2.imshow('input', img)
    cv2.imshow('fake', fake)
    cv2.imshow('input_detect', rect_img)
    cv2.imshow('deepfake_detect', rect_fake)
    if cv2.waitKey(1) == ord('q'):
        break

webcamera.release()
cv2.destroyAllWindows()