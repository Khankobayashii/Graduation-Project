
import cv2
import json
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO
from utils.config import *
from utils.tts import speak_label


def load_model_and_labels():
    model = YOLO(MODEL_PATH)
    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    return model, label_map


def draw_text_with_pil(image, x, y, text):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    draw.text((x, y), text, font=font, fill=TEXT_COLOR)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


def detect_on_image(model, label_map, input_path, output_path):
    results = model(input_path)[0]
    image = cv2.imread(input_path)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label_code = model.names[cls_id]
        label_name = label_map.get(label_code, label_code)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        image = draw_text_with_pil(image, x1, y1 - 25, label_name)
        print(f"Phát hiện: {label_name}")
        speak_label(label_name)

    cv2.imwrite(output_path, image)
    return output_path


def detect_on_video(model, label_map, input_path, output_path="output.avi"):
    cap = cv2.VideoCapture(input_path)
    out = None
    speak_counts = {}
    FRAME_SKIP = 2
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        results = model(frame)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label_code = model.names[cls_id]
            label_name = label_map.get(label_code, label_code)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
            frame = draw_text_with_pil(frame, x1, y1 - 25, label_name)
            print(f"Phát hiện: {label_name}")

            # Kiểm tra số lần đã phát giọng nói
            if speak_counts.get(label_name, 0) < 3:
                speak_label(label_name)
                speak_counts[label_name] = speak_counts.get(label_name, 0) + 1

        # Hiển thị khung hình
        cv2.imshow("YOLOv8 Video Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 10.0, (w, h))

        out.write(frame)

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
