
import os
from utils.detect import load_model_and_labels, detect_on_image, detect_on_video

INPUT_PATH = 'data/input/videos/video1.mp4'
OUTPUT_IMAGE = 'data/output/result.jpg'

ext = os.path.splitext(INPUT_PATH)[1].lower()
model, label_map = load_model_and_labels()

if ext in ['.jpg', '.jpeg', '.png']:
    path = detect_on_image(model, label_map, INPUT_PATH, OUTPUT_IMAGE)
    os.system(f'start {path}')
elif ext in ['.mp4', '.avi', '.mov']:
    detect_on_video(model, label_map, INPUT_PATH)
    os.system("start output.avi")
else:
    print("❌ Định dạng không hỗ trợ. Vui lòng dùng ảnh hoặc video.")


