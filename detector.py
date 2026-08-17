import os
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

try:
    import torch
    if torch.cuda.is_available():
        DEFAULT_DEVICE = "cuda"
    else:
        DEFAULT_DEVICE = "cpu"
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
except Exception:
    DEFAULT_DEVICE = "cpu"

import config


class YOLODetector:
    """
    Ultralytics YOLO wrapper for high-performance traffic object detection.
    Detects persons, cars, motorcycles, buses, and trucks.
    Integrates a dedicated Helmet-trained YOLO model for helmet / no-helmet detection.
    """

    def __init__(self, model_path: str = config.MODEL_PATH, conf: float = config.CONFIDENCE_THRESHOLD, device: str = DEFAULT_DEVICE):
        self.model_path = model_path
        self.conf = conf
        self.device = device
        print(f"[YOLODetector] Loading Base YOLO model from '{self.model_path}' on device '{self.device}'...")
        try:
            self.model = YOLO(self.model_path)
            self.class_names = self.model.names
            print(f"[YOLODetector] Base YOLO model loaded successfully. Classes: {len(self.class_names)}")
        except Exception as e:
            print(f"[YOLODetector] Error loading Base YOLO model: {e}")
            raise

        # Dedicated Helmet YOLO Model setup
        self.helmet_model = None
        self.helmet_model_available = False
        self.helmet_class_names = {}
        self.helmet_classes = set()
        self.no_helmet_classes = set()

        helmet_path = getattr(config, "HELMET_MODEL_PATH", None)
        if helmet_path and os.path.isfile(helmet_path):
            try:
                print(f"[YOLODetector] Loading Helmet model from '{helmet_path}'...")
                self.helmet_model = YOLO(helmet_path)
                self.helmet_class_names = getattr(self.helmet_model, "names", {})
                self.helmet_model_available = True
                print(f"[YOLODetector] ✓ Helmet model loaded successfully!")
                print(f"[YOLODetector]   Inspected Helmet Classes: {self.helmet_class_names}")

                # Analyze and categorize classes
                for cid, cname in self.helmet_class_names.items():
                    cname_clean = str(cname).lower().strip().replace("-", "_").replace(" ", "_")
                    if any(w in cname_clean for w in ["no_helmet", "without_helmet", "none_helmet", "nohelmet", "head", "bare_head"]):
                        self.no_helmet_classes.add(int(cid))
                    elif any(w in cname_clean for w in ["helmet", "with_helmet", "wearing_helmet", "helmeted"]):
                        self.helmet_classes.add(int(cid))
                    else:
                        # Default fallback
                        self.helmet_classes.add(int(cid))

                print(f"[YOLODetector]   Class Mapping -> Helmet: {list(self.helmet_classes)}, No-Helmet: {list(self.no_helmet_classes)}")
            except Exception as e:
                print(f"[YOLODetector] Failed to load Helmet model: {e}")
                self.helmet_model = None
                self.helmet_model_available = False
        else:
            print(f"[YOLODetector] Helmet model NOT found at '{helmet_path}'.")
            print(f"[YOLODetector]   To enable real helmet detection, place your trained YOLO model at: models/helmet_model.pt")
            print(f"[YOLODetector]   (Helmet detection disabled gracefully without faking)")

    def detect(self, frame: np.ndarray, conf: float = None, imgsz: int = 640) -> sv.Detections:
        """
        Runs object detection on a single frame and returns filtered sv.Detections.
        """
        if frame is None:
            return sv.Detections.empty()

        c = conf if conf is not None else self.conf
        # Run Ultralytics inference with device optimization and imgsz
        results = self.model.predict(
            frame,
            conf=c,
            imgsz=imgsz,
            device=self.device,
            verbose=False
        )[0]

        # Convert to supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        if len(detections) == 0:
            return detections

        # Filter to only relevant classes (person, car, motorcycle, bus, truck)
        filter_mask = np.isin(detections.class_id, list(config.ALL_TRACKED_CLASS_IDS))
        return detections[filter_mask]

    def get_class_name(self, class_id: int) -> str:
        """Returns the human-readable class name from base model."""
        return self.class_names.get(int(class_id), f"class_{class_id}")

    def check_rider_helmet(self, frame: np.ndarray, rider_box: np.ndarray, bike_box: np.ndarray = None) -> dict:
        """
        Runs real inference using the loaded Helmet YOLO model on the individual rider's upper body/head.
        Inspects detected classes and returns:
          {
             "available": bool,
             "status": "HELMET" | "NO HELMET" | "UNKNOWN",
             "confidence": float,
             "class_name": str,
             "message": str
          }
        """
        if not self.helmet_model_available or self.helmet_model is None or frame is None:
            return {
                "available": False,
                "status": "UNKNOWN",
                "confidence": 0.0,
                "class_name": None,
                "message": "Helmet model not configured"
            }

        rx1, ry1, rx2, ry2 = map(int, rider_box)
        h_frame, w_frame = frame.shape[:2]

        # Crop rider head region: top 45% of rider bounding box + small margin
        rw = rx2 - rx1
        rh = ry2 - ry1

        hx1 = max(0, int(rx1 - rw * 0.15))
        hy1 = max(0, int(ry1 - rh * 0.20))
        hx2 = min(w_frame, int(rx2 + rw * 0.15))
        hy2 = min(h_frame, int(ry1 + rh * 0.50))

        head_crop = frame[hy1:hy2, hx1:hx2]
        if head_crop.size == 0 or head_crop.shape[0] < 5 or head_crop.shape[1] < 5:
            head_crop = frame[max(0, ry1):min(h_frame, ry2), max(0, rx1):min(w_frame, rx2)]

        if head_crop.size == 0:
            return {
                "available": True,
                "status": "UNKNOWN",
                "confidence": 0.0,
                "class_name": None,
                "message": "Invalid rider crop area"
            }

        try:
            results = self.helmet_model(head_crop, conf=config.HELMET_CONFIDENCE_THRESHOLD, verbose=False)[0]
            boxes = results.boxes

            if len(boxes) == 0:
                # If single-class helmet model: no helmet detected in rider head region -> NO HELMET
                if len(self.no_helmet_classes) == 0 and len(self.helmet_classes) > 0:
                    return {
                        "available": True,
                        "status": "NO HELMET",
                        "confidence": 0.75,
                        "class_name": "no_helmet",
                        "message": "No helmet detected on rider"
                    }
                else:
                    return {
                        "available": True,
                        "status": "NO HELMET",
                        "confidence": 0.70,
                        "class_name": "no_helmet",
                        "message": "No helmet detected on rider"
                    }

            # Choose the highest confidence detection in rider head region
            best_conf = -1.0
            best_cls = None
            for box in boxes:
                c = float(box.conf[0])
                cls_id = int(box.cls[0])
                if c > best_conf:
                    best_conf = c
                    best_cls = cls_id

            cname = self.helmet_class_names.get(best_cls, str(best_cls))
            if best_cls in self.no_helmet_classes:
                return {
                    "available": True,
                    "status": "NO HELMET",
                    "confidence": round(best_conf, 2),
                    "class_name": cname,
                    "message": f"Detected {cname}"
                }
            elif best_cls in self.helmet_classes:
                return {
                    "available": True,
                    "status": "HELMET",
                    "confidence": round(best_conf, 2),
                    "class_name": cname,
                    "message": f"Detected {cname}"
                }
            else:
                if "helmet" in str(cname).lower() and "no" not in str(cname).lower():
                    return {
                        "available": True,
                        "status": "HELMET",
                        "confidence": round(best_conf, 2),
                        "class_name": cname,
                        "message": f"Detected {cname}"
                    }
                else:
                    return {
                        "available": True,
                        "status": "NO HELMET",
                        "confidence": round(best_conf, 2),
                        "class_name": cname,
                        "message": f"Detected {cname}"
                    }

        except Exception as e:
            return {
                "available": True,
                "status": "UNKNOWN",
                "confidence": 0.0,
                "class_name": None,
                "message": f"Helmet inference error: {e}"
            }

    def check_helmet(self, rider_crop: np.ndarray) -> dict:
        """Backwards compatibility hook for raw image crops."""
        if not self.helmet_model_available or self.helmet_model is None:
            return {"available": False, "has_helmet": None, "message": "Helmet model not configured"}
        try:
            results = self.helmet_model(rider_crop, conf=config.HELMET_CONFIDENCE_THRESHOLD, verbose=False)[0]
            has_helmet = len(results.boxes) > 0
            return {
                "available": True,
                "has_helmet": has_helmet,
                "message": "Helmet detected" if has_helmet else "No helmet detected"
            }
        except Exception as e:
            return {"available": False, "has_helmet": None, "message": str(e)}
