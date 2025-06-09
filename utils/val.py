from ultralytics import YOLO
import pandas as pd


model = YOLO("../models/best.pt")


metrics = model.val(data="Traffic-Sign-Dataset-2/data.yaml")


per_class_data = {
    "Class": [model.names[i] for i in range(len(model.names))],
    "Precision": metrics.box.p.tolist(),
    "Recall": metrics.box.r.tolist(),
    "mAP@0.5": metrics.box.map50.tolist(),
    "mAP@0.5:0.95": metrics.box.map.tolist()
}

df = pd.DataFrame(per_class_data)
df.to_csv("per_class_metrics.csv", index=False)
print(df)
