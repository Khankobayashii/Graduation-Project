from collections import Counter
import os
import matplotlib.pyplot as plt
import json

label_dirs = [
    "Traffic-Sign-Dataset-2/test/labels",
    "Traffic-Sign-Dataset-2/train/labels",
    "Traffic-Sign-Dataset-2/valid/labels"
]

# Load file ánh xạ JSON (giả sử dạng: "0": "Biển cấm quay đầu")
with open("data/label_name.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

all_labels = []

for label_dir in label_dirs:
    if not os.path.exists(label_dir):
        print(f"⚠️ Không tìm thấy thư mục: {label_dir}")
        continue

    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            with open(os.path.join(label_dir, file), "r") as f:
                for line in f:
                    class_id = line.strip().split()[0]  # ví dụ: "16"
                    label_vi = label_map.get(class_id, f"Không rõ (class {class_id})")
                    all_labels.append(label_vi)

# Đếm số lượng ảnh theo tên tiếng Việt
counts = Counter(all_labels)

# Sắp xếp theo số lượng giảm dần
sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

# Vẽ biểu đồ cột nằm ngang
plt.figure(figsize=(12, max(6, len(sorted_counts) * 0.5)))  # Tự động tăng chiều cao theo số label
plt.barh(list(sorted_counts.keys()), list(sorted_counts.values()), color='skyblue')
plt.xlabel("Số lượng ảnh")
plt.ylabel("Loại biển báo")
plt.title("Thống kê số lượng ảnh theo từng loại biển báo")
plt.gca().invert_yaxis()  # Label có số lượng lớn hiển thị ở trên cùng
plt.tight_layout()
plt.show()
