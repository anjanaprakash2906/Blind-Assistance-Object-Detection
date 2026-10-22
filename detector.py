"""
Thin wrapper around Ultralytics YOLOv8 so the main loop doesn't need to know
about model internals, and so target-class filtering lives in one place.
"""

from typing import List, Optional
from ultralytics import YOLO


class Detection:
    __slots__ = ("label", "confidence", "x1", "y1", "x2", "y2")

    def __init__(self, label: str, confidence: float, x1: float, y1: float, x2: float, y2: float):
        self.label = label
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class ObjectDetector:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[str]] = None,
    ):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = set(target_classes) if target_classes else None

        # Map class-id -> class-name once, from the model's own metadata.
        self.class_names = self.model.names

    def detect(self, frame) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls[0])
            label = self.class_names.get(class_id, str(class_id))

            if self.target_classes is not None and label not in self.target_classes:
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

            detections.append(Detection(label, confidence, x1, y1, x2, y2))

        return detections
