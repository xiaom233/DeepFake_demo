import sys
import cv2
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QMessageBox
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QSizePolicy
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
        self.main_layout = QVBoxLayout()
        self.top_layout = QVBoxLayout()
        self.bottom_layout = QHBoxLayout()

        # Title and description
        self.title = QLabel("面向深度伪造检测的线上会议实景展示DEMO")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet("font-size: 32px; font-weight: bold;")
        self.description = QLabel(
            "这是一个面向深度伪造检测的实景demo, 画面中展示了一个线上会议的场景，当会议室2中成员尝试使用deepfake技术进行造假时，我们的系统会弹出相关的警告，提示会议室1中的参会者谨慎辨别，以防上当受骗。")
        self.description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description.setStyleSheet("font-size: 18px;")
        self.description.setWordWrap(True)  # 启用自动换行

        # Labels for images
        self.image_label1 = QLabel()
        self.image_label2 = QLabel()
        self.image_label3 = QLabel()

        self.caption1 = QLabel("会议室1实时画面")
        self.caption1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption1.setStyleSheet("font-size: 18px;")

        self.caption2 = QLabel("会议室2实时画面")
        self.caption2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption2.setStyleSheet("font-size: 18px;")

        self.caption3 = QLabel("会议室1传回画面")
        self.caption3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption3.setStyleSheet("font-size: 18px;")

        # Button for deepfake function
        self.deepfake_button = QPushButton("Deepfake")
        self.deepfake_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.deepfake_button.setFixedHeight(60)
        self.deepfake_button.clicked.connect(self.deepfake)

        # Add title and description to top layout
        self.top_layout.addWidget(self.title)
        self.top_layout.addWidget(self.description)

        # Add labels and captions to left layout
        left_image_layout1 = QVBoxLayout()
        left_image_layout1.addWidget(self.image_label1)
        left_image_layout1.addWidget(self.caption1)

        left_image_layout2 = QVBoxLayout()
        left_image_layout2.addWidget(self.image_label2)
        left_image_layout2.addWidget(self.caption2)

        left_images_layout = QVBoxLayout()
        left_images_layout.addLayout(left_image_layout1)
        left_images_layout.addLayout(left_image_layout2)

        # Add button to left layout
        left_images_layout.addWidget(self.deepfake_button)

        # Add label and caption to right layout
        right_image_layout = QVBoxLayout()
        right_image_layout.addWidget(self.image_label3)
        right_image_layout.addWidget(self.caption3)

        # Add left and right layouts to bottom layout
        self.bottom_layout.addLayout(left_images_layout)
        self.bottom_layout.addLayout(right_image_layout)

        # Add top and bottom layouts to main layout
        self.main_layout.addLayout(self.top_layout)
        self.main_layout.addLayout(self.bottom_layout)

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

        # Timer for deepfake detection
        self.infer_timer = QTimer()
        self.infer_timer.timeout.connect(self.run_infer)
        self.infer_interval = 3000  # 3 seconds
        self.last_infer_time = 0

    def update_frame(self):  # Need to be modified if we have two cameras.
        ret, camera_frame = self.cap.read()
        if ret:
            # Convert frame to QImage
            camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            if self.is_faker:
                self.fake = process_image_numpy(self.source_arr, camera_frame)
                self.results = self.detect_model(self.fake, conf=0.5, stream=False)
                self.rect_fake = self.rec_img(self.fake, self.results)
            else:
                self.fake = camera_frame
                self.rect_fake = camera_frame
            qt_camera_frame = self.img2qt_image(camera_frame)
            qt_fake = self.img2qt_image(self.fake)
            qt_rect_fake = self.img2qt_image(self.rect_fake)

            # Calculate image sizes
            window_width = self.width()
            left_image_width = window_width // 3
            left_image_height = left_image_width * 9 // 16
            right_image_width = window_width * 9 // 16
            right_image_height = right_image_width * 3 // 4

            # Update labels
            self.image_label1.setPixmap(QPixmap.fromImage(qt_camera_frame).scaled(left_image_width, left_image_height,
                                                                                  Qt.AspectRatioMode.KeepAspectRatio))
            self.image_label1.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_label2.setPixmap(QPixmap.fromImage(qt_fake).scaled(left_image_width, left_image_height,
                                                                          Qt.AspectRatioMode.KeepAspectRatio))
            self.image_label2.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_label3.setPixmap(QPixmap.fromImage(qt_rect_fake).scaled(right_image_width, right_image_height,
                                                                               Qt.AspectRatioMode.KeepAspectRatio))
            self.image_label3.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Update button size
            button_width = window_width // 4
            self.deepfake_button.setFixedWidth(button_width)

    def img2qt_image(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return qt_image

    def closeEvent(self, event):
        self.cap.release()

    def deepfake(self):
        print("Deepfake function called!")
        self.is_faker = not self.is_faker

        if self.is_faker:
            # Start the timer for deepfake detection
            self.infer_timer.start(self.infer_interval)
        else:
            # Stop the timer if deepfake detection is turned off
            self.infer_timer.stop()

    def run_infer(self):
        # Use the 'fake' frame from the update_frame method
        fake = self.fake

        # Call the infer method from NetInference class
        pred = self.deepfake_detector.infer(fake)

        # Print or process the prediction result
        print("Prediction result:", pred)

        # Check if pred value exceeds the threshold
        if pred > 0.1:
            warning = QMessageBox()
            warning.setIcon(QMessageBox.Icon.Warning)
            warning.setText("⚠⚠⚠ Warning: Deepfake face is detected !")
            warning.setWindowTitle("Warning")
            warning.setStandardButtons(QMessageBox.StandardButton.Ok)
            warning.exec()


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
