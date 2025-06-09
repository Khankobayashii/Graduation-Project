
from roboflow import Roboflow
rf = Roboflow(api_key="FhYL2VBYofP96EAP86e5")
project = rf.workspace("khangnp63cnttntueduvn").project("traffic-sign-dataset-he5op")
version = project.version(2)
dataset = version.download("yolov8")
