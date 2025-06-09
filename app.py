import sys
import json
import cv2
import os
import tempfile
import threading
from datetime import datetime
from gtts import gTTS
import pygame
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QFrame
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO

pygame.mixer.init()


def load_label_map(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class TrafficSignApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(" Nhận diện biển báo giao thông - YOLOv8 ")
        self.resize(900, 700)
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QWidget {
                background-color: #F5F7FA;
                color: #2C3E50;
            }
            QPushButton {
                background-color: #3498DB;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 16px;
                color: white;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QLabel#titleLabel {
                font-size: 24px;
                font-weight: bold;
                color: #E67E22;
            }
            QLabel#statusLabel {
                font-size: 16px;
                color: #34495E;
            }
            QFrame#imageFrame {
                border: 3px solid #3498DB;
                border-radius: 10px;
                background-color: #ECF0F1;
            }
        """)

        self.model = YOLO("models/best.pt")
        self.last_spoken = set()

        self.title_label = QLabel("Nhận diện biển báo giao thông")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel("Chọn ảnh hoặc video để bắt đầu nhận diện.")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.count_label = QLabel("Số lượng biển báo: 0")
        self.count_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.count_label.setStyleSheet("font-size: 16px; color: #3399FF;")

        self.detail_label = QLabel("")
        self.detail_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet("font-size: 14px; color: #2C3E50;")

        self.image_frame = QFrame()
        self.image_frame.setObjectName("imageFrame")
        self.image_frame.setFixedSize(720, 540)
        self.image_label = QLabel("Chưa có dữ liệu")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("font-size: 18px;")
        self.image_label.setFixedSize(710, 530)

        frame_layout = QVBoxLayout()
        frame_layout.setContentsMargins(5, 5, 5, 5)
        frame_layout.addWidget(self.image_label)
        self.image_frame.setLayout(frame_layout)

        self.img_btn = QPushButton(" Chọn ảnh")
        self.img_btn.clicked.connect(self.load_image)

        self.vid_btn = QPushButton(" Chọn video")
        self.vid_btn.clicked.connect(self.load_video)

        self.stop_btn = QPushButton(" Dừng video")
        self.stop_btn.clicked.connect(self.stop_video)

        self.resume_btn = QPushButton(" Tiếp tục video")
        self.resume_btn.clicked.connect(self.resume_video)

        self.save_btn = QPushButton(" Lưu kết quả")
        self.save_btn.clicked.connect(self.save_result)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.img_btn)
        btn_layout.addSpacing(20)
        btn_layout.addWidget(self.vid_btn)
        btn_layout.addSpacing(20)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addSpacing(20)
        btn_layout.addWidget(self.resume_btn)
        btn_layout.addSpacing(20)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch()

        # Khung hiển thị thông tin bên phải
        info_layout = QVBoxLayout()
        info_layout.addWidget(self.count_label)
        info_layout.addWidget(self.detail_label)
        info_layout.addStretch()

        right_frame = QFrame()
        right_frame.setLayout(info_layout)
        right_frame.setStyleSheet("border: 2px solid gray; padding: 10px;")

        # Bố cục ngang: trái là ảnh, phải là khung thông tin
        content_layout = QHBoxLayout()
        content_layout.addWidget(self.image_frame)
        content_layout.addWidget(right_frame)

        # Layout chính
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title_label)
        main_layout.addSpacing(10)
        main_layout.addWidget(self.status_label)
        main_layout.addSpacing(10)
        main_layout.addLayout(content_layout)
        main_layout.addSpacing(20)
        main_layout.addLayout(btn_layout)
        main_layout.addSpacing(30)

        self.setLayout(main_layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.label_map_vi = load_label_map("data/label_name.json")
        self.last_frame = None

    def load_image(self):
        self.timer.stop()
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.status_label.setText(f"Đang xử lý ảnh: {os.path.basename(file_path)}")
            result = self.model(file_path)[0]
            img = result.plot()

            names = [result.names[int(cls)] for cls in result.boxes.cls] if result.boxes else []
            self.count_label.setText(f"Số lượng biển báo: {len(names)}")
            self.update_detail_label(names)
            self.speak_labels(names)

            self.display_image(img)
            self.last_frame = img
            self.status_label.setText(f"Hoàn thành nhận diện ảnh: {os.path.basename(file_path)}")

    def load_video(self):
        self.timer.stop()
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn video", "", "Videos (*.mp4 *.avi *.mov)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.last_spoken.clear()
            self.status_label.setText(f"Phát video: {os.path.basename(file_path)}")
            self.timer.start(30)

    def stop_video(self):
        if self.cap:
            self.timer.stop()
            self.status_label.setText("Video đã được tạm dừng.")

    def resume_video(self):
        if self.cap and not self.timer.isActive():
            self.status_label.setText("Tiếp tục phát video.")
            self.timer.start(30)

    def save_result(self):
        if self.last_frame is not None:
            save_path, _ = QFileDialog.getSaveFileName(self, "Lưu kết quả",
                                                       f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                                                       "Images (*.jpg *.png *.bmp)")
            if save_path:
                cv2.imwrite(save_path, self.last_frame)
                self.status_label.setText(f"Đã lưu kết quả tại: {save_path}")
        else:
            self.status_label.setText("Không có ảnh/video nào để lưu.")

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                result = self.model(frame)[0]
                frame = result.plot()

                names = [result.names[int(cls)] for cls in result.boxes.cls] if result.boxes else []
                self.count_label.setText(f"Số lượng biển báo: {len(names)}")
                self.update_detail_label(names)
                self.speak_labels(names)

                self.display_image(frame)
                self.last_frame = frame
            else:
                self.cap.release()
                self.cap = None
                self.timer.stop()
                self.status_label.setText("Kết thúc video.")

    def display_image(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    def update_detail_label(self, names):
        if not names:
            self.detail_label.setText("Không phát hiện biển báo nào.")
            return
        vi_labels = [self.label_map_vi.get(label, label) for label in names]
        detail_text = "\n".join(f"- {label}" for label in vi_labels)
        self.detail_label.setText("Chi tiết:\n" + detail_text)

    def speak_labels(self, names):
        new_labels = [name for name in names if name not in self.last_spoken]
        if new_labels:
            vi_labels = [self.label_map_vi.get(label, label) for label in new_labels]
            text = ', '.join(vi_labels)
            print("Speak:", text)

            def run_tts(text, labels_to_update):
                try:
                    tts = gTTS(text=text, lang='vi')
                    temp_path = os.path.join(tempfile.gettempdir(), "temp_audio.mp3")
                    tts.save(temp_path)
                    pygame.mixer.music.load(temp_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    os.remove(temp_path)
                    self.last_spoken.update(labels_to_update)
                except Exception as e:
                    print("Lỗi phát âm:", e)

            threading.Thread(target=run_tts, args=(text, new_labels)).start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrafficSignApp()
    window.show()
    sys.exit(app.exec_())
