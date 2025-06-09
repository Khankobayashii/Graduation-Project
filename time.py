# import time
# import os
# from ultralytics import YOLO
# from PIL import Image
#
# # Load model YOLO11s
# model = YOLO('models/best.pt')
#
# # Đường dẫn tới thư mục ảnh test
# image_dir = "Traffic-Sign-Dataset-2/valid/images"
#
# # Lấy danh sách tất cả ảnh trong thư mục
# image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
#
# # Chỉ test ví dụ với 50 ảnh đầu
# image_paths = image_paths[:50]
#
# start = time.time()
#
# for img_path in image_paths:
#     # Bạn có thể truyền path trực tiếp hoặc mở ảnh bằng PIL
#     # results = model.predict(img_path)  # Cách đơn giản nhất
#     results = model.predict(img_path, verbose=False)
#
# end = time.time()
#
# total_time = end - start
# fps = len(image_paths) / total_time
#
# print(f"Processed {len(image_paths)} images in {total_time:.2f} seconds")
# print(f"FPS: {fps:.2f}")

from ultralytics import YOLO

model = YOLO("models/best.pt")
print(f"Total parameters: {sum(p.numel() for p in model.model.parameters()):,}")
