"""
3_accident_stopped_vehicle.py
================================
EVENT: Accident / Stopped Vehicle Detection
Flags a vehicle that was moving and then stays still for too long
(a common proxy for accidents, breakdowns, or stalled vehicles on a road).

Install:
    pip install ultralytics supervision opencv-python numpy

SETUP REQUIRED:
    - STOP_TIME_SEC: how long a vehicle must stay still before flagging (default 5s)
    - STOP_SPEED_PX_PER_FRAME: movement below this = "not moving" (tune to your
      video resolution / camera distance -- a vehicle far from camera moves fewer
      pixels per frame than one close up)
    - (Optional) EXCLUDE_ZONES: polygons where stopping is normal (e.g. traffic
      signal, toll booth) so those don't get falsely flagged.

Run:
    python 3_accident_stopped_vehicle.py --source video.mp4 --output accident_out.mp4
"""

import argparse
import json
import time
from collections import deque, defaultdict

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# ---------------- CONFIG (edit these) ----------------
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.35

COCO_CAR, COCO_MOTORCYCLE, COCO_BUS, COCO_TRUCK = 2, 3, 5, 7
VEHICLE_CLASSES = {COCO_CAR, COCO_MOTORCYCLE, COCO_BUS, COCO_TRUCK}

STOP_SPEED_PX_PER_FRAME = 2.0   # <-- TUNE THIS to your video
STOP_TIME_SEC = 5.0             # <-- TUNE THIS: how long "stopped" must persist to flag
TRACK_HISTORY_LEN = 30

# Optional: zones where stopping is expected/normal (e.g. red light, toll booth).
# Vehicles stopped inside these zones will NOT be flagged. Leave empty list if not needed.
EXCLUDE_ZONES = [
    # np.array([[100, 400], [400, 400], [400, 600], [100, 600]]),  # e.g. signal-stop area
]
# -------------------------------------------------------


def point_in_any_zone(point, zones):
    for zone in zones:
        if cv2.pointPolygonTest(zone.astype(np.int32), point, False) >= 0:
            return True
    return False


def bbox_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def log_event(logfile, track_id, frame_idx, stopped_for_sec):
    record = {"event": "accident_or_stopped_vehicle", "track_id": int(track_id),
              "frame": int(frame_idx), "stopped_for_sec": round(stopped_for_sec, 1),
              "time": time.time()}
    logfile.write(json.dumps(record) + "\n")
    logfile.flush()
    print(f"[ALERT] stopped_vehicle | track_id={track_id} | frame={frame_idx} | "
          f"stopped_for={stopped_for_sec:.1f}s")


def run(source, output_path, show=False):
    model = YOLO(MODEL_PATH)
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    logfile = open("accident_events.jsonl", "a")

    positions = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_LEN))
    stopped_since = {}     # track_id -> timestamp (sec) when it first became "stopped"
    flagged = set()        # track_ids already logged (avoid duplicate logs)
    was_moving = defaultdict(bool)  # track_id -> has this vehicle ever moved meaningfully

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = frame_idx / fps

            results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[np.isin(detections.class_id, list(VEHICLE_CLASSES))]
            detections = tracker.update_with_detections(detections)

            labels = []
            for xyxy, class_id, tracker_id in zip(detections.xyxy, detections.class_id, detections.tracker_id):
                name = model.names.get(int(class_id), str(class_id))
                if tracker_id is None:
                    labels.append(name)
                    continue

                center = bbox_center(xyxy)
                positions[tracker_id].append(center)

                tag = ""
                if len(positions[tracker_id]) >= 2:
                    speed = np.linalg.norm(positions[tracker_id][-1] - positions[tracker_id][-2])

                    if speed >= STOP_SPEED_PX_PER_FRAME:
                        was_moving[tracker_id] = True
                        stopped_since.pop(tracker_id, None)
                        flagged.discard(tracker_id)
                    else:
                        current_pos = tuple(positions[tracker_id][-1])
                        in_excluded_zone = point_in_any_zone(current_pos, EXCLUDE_ZONES)

                        # only flag vehicles that were previously moving (skip parked cars
                        # that were never moving in view) and are outside excluded zones
                        if was_moving[tracker_id] and not in_excluded_zone:
                            if tracker_id not in stopped_since:
                                stopped_since[tracker_id] = now
                            elapsed = now - stopped_since[tracker_id]
                            if elapsed >= STOP_TIME_SEC:
                                tag = " [STOPPED/ACCIDENT?]"
                                if tracker_id not in flagged:
                                    flagged.add(tracker_id)
                                    log_event(logfile, tracker_id, frame_idx, elapsed)

                labels.append(f"#{tracker_id} {name}{tag}")

            annotated = box_annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)
            for zone in EXCLUDE_ZONES:
                cv2.polylines(annotated, [zone.astype(np.int32)], True, (255, 255, 0), 2)

            writer.write(annotated)
            if show:
                cv2.imshow("Accident / Stopped Vehicle Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        logfile.close()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import os
    default_src = "videos/accident.mp4" if os.path.exists("videos/accident.mp4") else "videos/input.mp4"
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=default_src, help=f"Video source path (default: {default_src})")
    parser.add_argument("--output", default="output/accident_out.mp4")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    src = int(args.source) if str(args.source).isdigit() else args.source
    run(src, args.output, show=args.show)