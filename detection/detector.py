from ultralytics import YOLO

class Detector:
    def __init__(self, conf=0.25):
        self.model = YOLO("yolov8n.pt")  # lightweight
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        return results[0].plot()  # returns annotated frame