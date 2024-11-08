import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk, ImageEnhance

class FancyCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fancy Camera App")
        self.root.geometry("1000x700")

        # 使用grid布局管理器
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # 创建左侧的两个摄像头显示窗口
        self.left_frame = ttk.Frame(self.root)
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        self.left_frame.grid_rowconfigure(0, weight=1)
        self.left_frame.grid_rowconfigure(1, weight=1)

        self.top_left_label = ttk.Label(self.left_frame)
        self.top_left_label.grid(row=0, column=0, sticky="nsew")

        self.bottom_left_label = ttk.Label(self.left_frame)
        self.bottom_left_label.grid(row=1, column=0, sticky="nsew")

        # 创建右侧的截图显示窗口
        self.right_frame = ttk.Frame(self.root)
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        self.screenshot_label = ttk.Label(self.right_frame)
        self.screenshot_label.pack(fill=tk.BOTH, expand=True)

        # 打开DroidCam虚拟摄像头
        self.cap = cv2.VideoCapture(1)  # 假设DroidCam是设备索引1

        # 开始更新画面
        self.update_frames()

    def update_frames(self):
        ret, frame = self.cap.read()
        if ret:
            # 将图像转换为tkinter兼容的格式
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            # 增强图像效果
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)  # 增加对比度

            imgtk = ImageTk.PhotoImage(image=img)

            # 更新左侧的两个窗口
            self.top_left_label.imgtk = imgtk
            self.top_left_label.configure(image=imgtk)

            self.bottom_left_label.imgtk = imgtk
            self.bottom_left_label.configure(image=imgtk)

            # 更新右侧的截图窗口
            self.screenshot_label.imgtk = imgtk
            self.screenshot_label.configure(image=imgtk)

        # 每10毫秒更新一次画面
        self.root.after(10, self.update_frames)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FancyCameraApp(root)
    root.mainloop()
