import sys
import cv2
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QMessageBox
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
from ultralytics import YOLO
import onnxruntime
from modules.processors.frame.face_swapper import process_image_numpy
from time import time
import math
from torchvision import transforms
from PIL import Image
from network import MFF_MoE
import numpy as np
import torch
from copy import deepcopy


local_weight_dir = r""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NetInference():
    def __init__(self):
        self.net = MFF_MoE(pretrained=False)
        self.net.load(path=local_weight_dir)
        # self.net = nn.DataParallel(self.net).cuda()
        self.net = self.net.to(device)
        self.net.eval()
        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((512, 512), antialias=True),
        ])

    def infer(self, x):
        # x = cv2.imread(input_path)[..., ::-1]
        x = Image.fromarray(np.uint8(x))
        x = self.transform_val(x).unsqueeze(0).to(device)
        pred = self.net(x)
        pred = pred.detach().cpu().numpy()
        return pred


class CameraApp(QWidget):
    def __init__(self):
        super().__init__()

        # Layouts
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()
        self.main_layout = QHBoxLayout()

        # Labels for images
        self.image_label1 = QLabel()
        self.image_label2 = QLabel()
        self.image_label3 = QLabel()

        # Button for deepfake function
        self.deepfake_button = QPushButton("Deepfake")
        self.deepfake_button.setFixedSize(960, 60)
        self.deepfake_button.clicked.connect(self.deepfake)

        # Add labels and button to left layout
        self.left_layout.addWidget(self.image_label1)
        self.left_layout.addWidget(self.image_label2)
        self.left_layout.addWidget(self.deepfake_button)

        # Add label to right layout
        self.right_layout.addWidget(self.image_label3)

        # Add left and right layouts to main layout
        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.right_layout)

        # Set main layout
        self.setLayout(self.main_layout)

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # model setup
        self.is_faker = False
        self.detect_model = YOLO('yolov8n-face.pt')
        self.faker = process_image_numpy
        self.deepfake_detector = NetInference()

        # source
        source_path = r"imgs/musk.jpg"
        self.source_arr = cv2.imread(source_path)
        # faker model initiate
        ret, camera_frame = self.cap.read()
        fake = process_image_numpy(self.source_arr, camera_frame)

    def update_frame(self):
        ret, camera_frame = self.cap.read()
        if ret:
            # Convert frame to QImage
            camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            if self.is_faker:
                fake = process_image_numpy(self.source_arr, camera_frame)
                results = self.detect_model(fake, conf=0.5, stream=False)
                rect_fake = self.rec_img(fake, results)
            else:
                fake = camera_frame
                rect_fake = camera_frame
            qt_camera_frame = self.img2qt_image(camera_frame)
            qt_fake = self.img2qt_image(fake)
            qt_rect_fake = self.img2qt_image(rect_fake)

            # Update labels
            left_image_size = (960, 720)
            right_image_size = (1440, 1080)
            self.image_label1.setPixmap(QPixmap.fromImage(qt_camera_frame).scaled(*left_image_size))
            self.image_label2.setPixmap(QPixmap.fromImage(qt_fake).scaled(*left_image_size))
            self.image_label3.setPixmap(QPixmap.fromImage(qt_rect_fake).scaled(*right_image_size))

    def deepfake(self):
        print("Deepfake function called!")
        self.is_faker = not self.is_faker
        if self.is_faker:
            warning = QMessageBox()
            warning.setIcon(QMessageBox.Icon.Warning)
            warning.setText("⚠⚠⚠ Warning: Deepfake face is detected !")
            warning.setWindowTitle("Warning")
            warning.setStandardButtons(QMessageBox.StandardButton.Ok)
            warning.exec()

    def closeEvent(self, event):
        self.cap.release()

    def img2qt_image(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return qt_image

    def rec_img(self, input_img, results):
        rect_img = deepcopy(input_img)
        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                box_start_time = time()
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                # put box in cam
                cv2.rectangle(rect_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.rectangle(rect_img, (x1, y2 - 25), (x2, y2), (255, 0, 0), cv2.FILLED)

                # Draw a label with a name below the face

                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                # print("Confidence --->", confidence)

                # class name
                cls = int(box.cls[0])

                # object details
                org = [x1 + 6, y2 - 10]
                fontScale = 0.5
                color = (255, 255, 255)
                thickness = 1

                # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(rect_img, f"Fake:{confidence}", org, font, fontScale, color, thickness)

                box_end_time = time()
        return rect_img


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())
