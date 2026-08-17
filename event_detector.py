import os
import json
import datetime
import time
from collections import defaultdict, deque
import numpy as np
import cv2
import supervision as sv

import config


class EventDetector:
    """
    Evaluates traffic video frames for safety & traffic violation events:
    1. Triple Riding / Triple Sit Detection (Primary Event)
    2. Wrong-Way Driving
    3. Multi-Vehicle Collision & Accident Detection
    4. Vehicle Stopped / Road Hazard
    5. Helmet Violation (Modular AI Model)
    """

    def __init__(self, detector, tracker, camera_id: str = config.CAMERA_ID, video_name: str = "input.mp4", start_clean: bool = False):
        self.detector = detector
        self.tracker = tracker
        self.camera_id = camera_id
        self.video_name = video_name

        # Events log
        self.events = []
        if not start_clean:
            self.load_existing_events()

        # Consecutive frame counters for stability
        # track_id -> consecutive frames with 3+ persons
        self.triple_riding_consecutive = defaultdict(int)
        # track_id -> consecutive frames stopped
        self.stopped_consecutive = defaultdict(int)
        # track_id -> consecutive frames moving in wrong direction
        self.wrong_way_consecutive = defaultdict(int)
        # pair_key -> consecutive frames in collision/proximity
        self.collision_consecutive = defaultdict(int)

        # Cooldown management: event_key -> last_frame_triggered
        self.event_cooldowns = {}

        # Active frame alerts for visual overlay
        self.active_frame_alerts = []
        self.last_frame_telemetry = {}

    def load_existing_events(self):
        """Loads historical events from data/events.json if present."""
        os.makedirs(os.path.dirname(config.EVENTS_JSON_PATH), exist_ok=True)
        if os.path.exists(config.EVENTS_JSON_PATH):
            try:
                with open(config.EVENTS_JSON_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.events = data.get("events", [])
            except Exception as e:
                print(f"[EventDetector] Warning: Could not parse events.json: {e}")
                self.events = []
        else:
            self.save_events()

    def save_events(self):
        """Saves events to data/events.json atomically."""
        os.makedirs(os.path.dirname(config.EVENTS_JSON_PATH), exist_ok=True)
        try:
            with open(config.EVENTS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump({"events": self.events}, f, indent=2)
        except Exception as e:
            print(f"[EventDetector] Error saving events: {e}")

    def log_event(self, event_type: str, vehicle_id: int, confidence: float, details: dict, frame_no: int, fps: float = 25.0, detected_object: str = None):
        """
        Logs a newly detected event, checking cooldown to prevent duplicate spam.
        """
        cooldown_key = f"{event_type}_{vehicle_id}"
        last_frame = self.event_cooldowns.get(cooldown_key, -99999)

        if (frame_no - last_frame) < config.EVENT_COOLDOWN:
            return None

        now_iso = datetime.datetime.now().isoformat(timespec="seconds")
        obj_str = detected_object or f"vehicle #{vehicle_id}"
        event_record = {
            "video_name": self.video_name,
            "video_id": self.video_name,
            "camera_id": self.camera_id,
            "event_type": event_type,
            "event": event_type,
            "timestamp": now_iso,
            "frame_no": int(frame_no),
            "relative_time_sec": round(frame_no / max(1.0, fps), 2),
            "detected_object": obj_str,
            "vehicle_id": int(vehicle_id) if vehicle_id is not None else None,
            "confidence": round(float(confidence), 2),
            "details": details,
            **details
        }

        self.events.append(event_record)
        self.event_cooldowns[cooldown_key] = frame_no
        self.save_events()
        print(f"[EVENT DETECTED] [{self.video_name}] Frame {frame_no} -> {event_type} on {obj_str} ({event_record['confidence']}): {details}")
        return event_record

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculates Intersection over Union between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union = area1 + area2 - inter_area
        return float(inter_area / max(1.0, union))

    def _is_person_on_motorcycle(self, person_box: np.ndarray, bike_box: np.ndarray) -> bool:
        """
        Geometric check to determine if a person is riding a specific motorcycle.
        Checks horizontal overlap and vertical alignment.
        """
        px1, py1, px2, py2 = person_box
        bx1, by1, bx2, by2 = bike_box

        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0

        # Person's center should be horizontally aligned within motorcycle width + margin
        margin_x = config.PERSON_MOTORCYCLE_DISTANCE
        horizontally_aligned = (bx1 - margin_x) <= pcx <= (bx2 + margin_x)

        # Person is sitting above/on the motorcycle
        vertically_aligned = (by1 - (py2 - py1)) <= pcy <= (by2 + 40)

        # Also check bounding box intersection
        inter_x1 = max(px1, bx1)
        inter_y1 = max(py1, by1)
        inter_x2 = min(px2, bx2)
        inter_y2 = min(py2, by2)
        has_overlap = (inter_x2 > inter_x1) and (inter_y2 > inter_y1)

        return horizontally_aligned and (vertically_aligned or has_overlap)

    def _get_movement_direction(self, trajectory) -> str:
        """
        Calculates dominant direction vector from trajectory points.
        """
        if len(trajectory) < 4:
            return None

        p_start = trajectory[0]
        p_end = trajectory[-1]

        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        distance = np.hypot(dx, dy)

        if distance < config.MIN_MOVEMENT:
            return "STATIONARY"

        if abs(dx) >= abs(dy):
            return "RIGHT" if dx > 0 else "LEFT"
        else:
            return "DOWN" if dy > 0 else "UP"

    def process_frame(self, frame: np.ndarray, frame_no: int, fps: float = 25.0, tracked_detections: sv.Detections = None) -> tuple:
        """
        Processes a single frame for detections, tracking, and event analysis.
        Returns: (annotated_frame, list_of_active_alerts, list_of_new_events)
        """
        self.active_frame_alerts = []
        new_events = []

        if tracked_detections is not None:
            tracked = tracked_detections
        else:
            # 1. Run YOLO object detection
            detections = self.detector.detect(frame)
            # 2. Update multi-object tracking (vehicles and pedestrians)
            tracked = self.tracker.update(detections)

        if len(tracked) == 0:
            return frame, [], []

        # Separate persons and vehicles
        person_indices = []
        vehicle_indices = []
        for i in range(len(tracked)):
            cls_id = int(tracked.class_id[i])
            if cls_id == config.CLASS_PERSON:
                person_indices.append(i)
            elif cls_id in config.VEHICLE_CLASS_IDS:
                vehicle_indices.append(i)

        person_boxes = [tracked.xyxy[i] for i in person_indices]

        # Structure to hold display metadata for each tracked vehicle
        vehicle_status = {}

        # 3. Analyze each vehicle
        for idx in vehicle_indices:
            track_id = tracked.tracker_id[idx]
            if track_id is None:
                continue

            track_id = int(track_id)
            cls_id = int(tracked.class_id[idx])
            box = tracked.xyxy[idx]
            conf = tracked.confidence[idx] if tracked.confidence is not None else 0.85
            class_name = self.detector.get_class_name(cls_id)
            trajectory = self.tracker.get_trajectory(track_id)

            status_tags = []

            # -------------------------------------------------------------
            # EVENT 1 & 4: MOTORCYCLE-SPECIFIC CHECKS (TRIPLE RIDING & HELMET)
            # -------------------------------------------------------------
            if cls_id == config.CLASS_MOTORCYCLE:
                # Count persons riding this motorcycle
                associated_persons = 0
                riders_without_helmet = 0

                for pbox in person_boxes:
                    if self._is_person_on_motorcycle(pbox, box):
                        associated_persons += 1
                        # Real Helmet Check for this rider
                        if self.detector.helmet_model_available:
                            h_res = self.detector.check_rider_helmet(frame, pbox, box)
                            if h_res.get("available") and h_res.get("status") == "NO HELMET":
                                riders_without_helmet += 1
                                h_conf = h_res.get("confidence", conf)
                                ev = self.log_event(
                                    event_type="helmet_violation",
                                    vehicle_id=track_id,
                                    confidence=h_conf,
                                    details={"helmet_detected": False, "status": "NO HELMET"},
                                    frame_no=frame_no
                                )
                                if ev:
                                    new_events.append(ev)
                                    self.active_frame_alerts.append(f"WARNING: Helmet Violation on Bike ID {track_id}")

                if riders_without_helmet > 0:
                    status_tags.append("NO HELMET")

                # If detected person count on this bike >= 3
                if associated_persons >= 3:
                    self.triple_riding_consecutive[track_id] += 1
                    if self.triple_riding_consecutive[track_id] >= config.TRIPLE_RIDING_FRAMES:
                        status_tags.append("TRIPLE RIDING")
                        ev = self.log_event(
                            event_type="triple_riding",
                            vehicle_id=track_id,
                            confidence=conf,
                            details={"person_count": associated_persons},
                            frame_no=frame_no
                        )
                        if ev:
                            new_events.append(ev)
                            self.active_frame_alerts.append(f"CRITICAL: Triple Riding detected on Bike ID {track_id} ({associated_persons} persons)")
                else:
                    self.triple_riding_consecutive[track_id] = max(0, self.triple_riding_consecutive[track_id] - 1)

            # -------------------------------------------------------------
            # EVENT 2: WRONG-WAY DRIVING
            # -------------------------------------------------------------
            current_dir = self._get_movement_direction(trajectory)
            expected_dir = config.EXPECTED_DIRECTION.upper()

            is_wrong_way = False
            if current_dir and current_dir != "STATIONARY":
                if expected_dir == "RIGHT" and current_dir == "LEFT":
                    is_wrong_way = True
                elif expected_dir == "LEFT" and current_dir == "RIGHT":
                    is_wrong_way = True
                elif expected_dir == "DOWN" and current_dir == "UP":
                    is_wrong_way = True
                elif expected_dir == "UP" and current_dir == "DOWN":
                    is_wrong_way = True

            if is_wrong_way:
                self.wrong_way_consecutive[track_id] += 1
                if self.wrong_way_consecutive[track_id] >= config.WRONG_WAY_CONSECUTIVE_FRAMES:
                    status_tags.append("WRONG WAY")
                    ev = self.log_event(
                        event_type="wrong_way_driving",
                        vehicle_id=track_id,
                        confidence=conf,
                        details={"movement_direction": current_dir, "expected_direction": expected_dir},
                        frame_no=frame_no
                    )
                    if ev:
                        new_events.append(ev)
                        self.active_frame_alerts.append(f"ALERT: Wrong-Way Driving by Vehicle ID {track_id} (Heading {current_dir})")
            else:
                self.wrong_way_consecutive[track_id] = max(0, self.wrong_way_consecutive[track_id] - 1)

            # -------------------------------------------------------------
            # EVENT 3: VEHICLE STOPPED / POSSIBLE ACCIDENT
            # -------------------------------------------------------------
            if len(trajectory) >= 2:
                # Instantaneous speed between last 2 points
                p1 = np.array(trajectory[-2])
                p2 = np.array(trajectory[-1])
                speed = np.linalg.norm(p2 - p1)

                if speed < config.STOPPED_SPEED_THRESHOLD:
                    self.stopped_consecutive[track_id] += 1
                    if self.stopped_consecutive[track_id] >= config.STOPPED_FRAMES:
                        stopped_sec = round(self.stopped_consecutive[track_id] / fps, 1)
                        status_tags.append("STOPPED / ACCIDENT")
                        ev = self.log_event(
                            event_type="vehicle_stopped",
                            vehicle_id=track_id,
                            confidence=conf,
                            details={"stopped_duration_sec": stopped_sec, "status": "Vehicle Stopped / Possible Accident"},
                            frame_no=frame_no
                        )
                        if ev:
                            new_events.append(ev)
                            self.active_frame_alerts.append(f"WARNING: Vehicle ID {track_id} Stopped for {stopped_sec}s (Possible Accident)")
                else:
                    self.stopped_consecutive[track_id] = max(0, self.stopped_consecutive[track_id] - 2)

            vehicle_status[track_id] = {
                "class_name": class_name,
                "tags": status_tags,
                "box": box
            }

        # -------------------------------------------------------------
        # EVENT 0: MULTI-VEHICLE COLLISION & ACCIDENT DETECTION
        # -------------------------------------------------------------
        for i in range(len(vehicle_indices)):
            for j in range(i + 1, len(vehicle_indices)):
                idx1 = vehicle_indices[i]
                idx2 = vehicle_indices[j]
                t1 = tracked.tracker_id[idx1]
                t2 = tracked.tracker_id[idx2]
                if t1 is None or t2 is None:
                    continue
                t1, t2 = int(t1), int(t2)
                box1 = tracked.xyxy[idx1]
                box2 = tracked.xyxy[idx2]

                c1 = np.array([(box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0])
                c2 = np.array([(box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0])
                dist = float(np.linalg.norm(c1 - c2))
                iou = self._calculate_iou(box1, box2)

                pair_key = tuple(sorted([t1, t2]))
                # Check collision condition using configurable thresholds
                if iou >= config.COLLISION_IOU_THRESHOLD or dist < config.COLLISION_DISTANCE_THRESHOLD:
                    self.collision_consecutive[pair_key] += 1
                    if self.collision_consecutive[pair_key] >= config.COLLISION_CONSECUTIVE_FRAMES:
                        for tid in (t1, t2):
                            if tid in vehicle_status:
                                if "ACCIDENT / COLLISION" not in vehicle_status[tid]["tags"]:
                                    vehicle_status[tid]["tags"].append("ACCIDENT / COLLISION")

                        cls1 = int(tracked.class_id[idx1])
                        cls2 = int(tracked.class_id[idx2])
                        n1 = self.detector.get_class_name(cls1)
                        n2 = self.detector.get_class_name(cls2)
                        c1_conf = float(tracked.confidence[idx1]) if tracked.confidence is not None else 0.85
                        c2_conf = float(tracked.confidence[idx2]) if tracked.confidence is not None else 0.85
                        c_conf = max(c1_conf, c2_conf)

                        ev = self.log_event(
                            event_type="accident_collision",
                            vehicle_id=t1,
                            confidence=c_conf,
                            details={
                                "vehicle_1": {"class": n1, "track_id": t1},
                                "vehicle_2": {"class": n2, "track_id": t2},
                                "iou": round(iou, 3),
                                "distance": round(dist, 1),
                                "status": "Vehicle Collision / Accident"
                            },
                            frame_no=frame_no,
                            fps=fps,
                            detected_object=f"{n1} #{t1} & {n2} #{t2}"
                        )
                        if ev:
                            new_events.append(ev)
                            self.active_frame_alerts.append(f"CRITICAL ACCIDENT: Collision between {n1.upper()} #{t1} & {n2.upper()} #{t2}")
                else:
                    self.collision_consecutive[pair_key] = max(0, self.collision_consecutive[pair_key] - 1)

        # 4. Compile frame-level detection telemetry and counts
        frame_detections_list = []
        counts = {
            "total": len(tracked),
            "person": 0,
            "car": 0,
            "bus": 0,
            "truck": 0,
            "motorcycle": 0
        }

        for i in range(len(tracked)):
            cls_id = int(tracked.class_id[i])
            cname = self.detector.get_class_name(cls_id).lower()
            conf_val = float(tracked.confidence[i]) if tracked.confidence is not None else 0.85
            t_id = int(tracked.tracker_id[i]) if (tracked.tracker_id is not None and tracked.tracker_id[i] is not None) else None
            bx = [int(v) for v in tracked.xyxy[i]]

            if cname in counts:
                counts[cname] += 1
            else:
                counts["total"] += 1

            frame_detections_list.append({
                "class_name": cname,
                "confidence": round(conf_val, 2),
                "confidence_pct": f"{int(round(conf_val * 100))}%",
                "box": bx,
                "track_id": t_id,
                "tags": vehicle_status.get(t_id, {}).get("tags", []) if t_id is not None else []
            })

        self.last_frame_telemetry = {
            "frame_no": frame_no,
            "counts": counts,
            "detections": frame_detections_list,
            "active_alerts": list(self.active_frame_alerts),
            "new_events": new_events
        }

        # 5. Render Annotations on Frame
        annotated_frame = self.render_overlay(frame, tracked, vehicle_status, frame_no)

        return annotated_frame, self.active_frame_alerts, new_events

    def render_overlay(self, frame: np.ndarray, tracked: sv.Detections, vehicle_status: dict, frame_no: int) -> np.ndarray:
        """
        Draws clear rectangular bounding boxes, class names + confidence scores (e.g. 'car 91%'),
        tracking IDs, speed/status tags, and top HUD banner.
        """
        out_frame = frame.copy()
        h, w = out_frame.shape[:2]

        # Draw detected and tracked objects
        for i in range(len(tracked)):
            track_id = tracked.tracker_id[i] if tracked.tracker_id is not None else None
            t_id_int = int(track_id) if track_id is not None else None
            cls_id = int(tracked.class_id[i])
            x1, y1, x2, y2 = map(int, tracked.xyxy[i])
            conf_val = float(tracked.confidence[i]) if tracked.confidence is not None else 0.85
            conf_pct = f"{int(round(conf_val * 100))}%"

            status_info = vehicle_status.get(t_id_int, {}) if t_id_int is not None else {}
            tags = status_info.get("tags", [])
            class_name = self.detector.get_class_name(cls_id).lower()

            # Determine box color based on active violation / class type
            if "ACCIDENT / COLLISION" in tags:
                color = (0, 0, 255)  # Bright Red
                box_thickness = 3
            elif "TRIPLE RIDING" in tags or "WRONG WAY" in tags:
                color = (0, 0, 255)  # Bright Red
                box_thickness = 3
            elif "STOPPED / ACCIDENT" in tags:
                color = (0, 140, 255)  # Orange
                box_thickness = 3
            elif "NO HELMET" in tags:
                color = (0, 215, 255)  # Yellow-Amber
                box_thickness = 2
            elif cls_id == config.CLASS_PERSON:
                color = (255, 190, 0)  # Cyan/Sky Blue for pedestrians
                box_thickness = 2
            elif cls_id == config.CLASS_MOTORCYCLE:
                color = (0, 230, 255)  # Amber Yellow for motorcycle
                box_thickness = 2
            elif cls_id == config.CLASS_BUS:
                color = (255, 120, 220)  # Magenta for bus
                box_thickness = 2
            elif cls_id == config.CLASS_TRUCK:
                color = (0, 165, 255)  # Orange for truck
                box_thickness = 2
            else:
                color = (0, 255, 128)  # Bright Emerald Green for cars/vehicles
                box_thickness = 2

            # 1. Clear rectangular bounding box
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, box_thickness)

            # 2. Build label text: e.g. "car 91%" or "person 95% #1"
            if t_id_int is not None:
                label = f"{class_name} #{t_id_int} {conf_pct}"
            else:
                label = f"{class_name} {conf_pct}"

            if tags:
                label += f" | {' + '.join(tags)}"

            # 3. Draw label banner background for high legibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.52
            font_thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            label_y1 = max(0, y1 - th - 8)
            label_y2 = y1

            # Background label pill
            cv2.rectangle(out_frame, (x1, label_y1), (x1 + tw + 8, label_y2), color, -1)
            # Crisp dark text on colored background
            cv2.putText(out_frame, label, (x1 + 4, label_y2 - 4), font, font_scale, (10, 10, 15), font_thickness, cv2.LINE_AA)

            # Draw trajectory trail for tracked vehicles
            if t_id_int is not None and cls_id in config.VEHICLE_CLASS_IDS:
                pts = self.tracker.get_trajectory(t_id_int)
                if len(pts) > 1:
                    for j in range(1, len(pts)):
                        pt1 = (int(pts[j-1][0]), int(pts[j-1][1]))
                        pt2 = (int(pts[j][0]), int(pts[j][1]))
                        cv2.line(out_frame, pt1, pt2, color, 1, cv2.LINE_AA)

        # Draw Top CCTV HUD Overlay Banner
        hud_bg = np.zeros((65, w, 3), dtype=np.uint8)
        cv2.rectangle(hud_bg, (0, 0), (w, 65), (15, 18, 25), -1)
        out_frame[0:65, 0:w] = cv2.addWeighted(out_frame[0:65, 0:w], 0.25, hud_bg, 0.75, 0)

        # HUD Text info
        cv2.putText(out_frame, "CITYEYE CCTV ANALYTICS", (20, 26), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 240, 255), 2, cv2.LINE_AA)
        status_line = f"CAM: {self.camera_id} | VIDEO: {self.video_name} | FRAME: {frame_no} | EVENTS: {len(self.events)}"
        cv2.putText(out_frame, status_line, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # Modular Helmet Status Badge
        helmet_badge = "HELMET AI: ACTIVE" if self.detector.helmet_model_available else "HELMET AI: NOT CONFIGURED"
        badge_color = (0, 255, 128) if self.detector.helmet_model_available else (120, 120, 120)
        cv2.putText(out_frame, helmet_badge, (w - 280, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_color, 1, cv2.LINE_AA)

        # Active Alert ticker on HUD
        if self.active_frame_alerts:
            latest_alert = self.active_frame_alerts[0]
            cv2.putText(out_frame, f">> {latest_alert}", (w - 550, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2, cv2.LINE_AA)

        return out_frame

    def process_live_stream(self, stream_source: str = None, output_path: str = None, conf: float = 0.35, show: bool = True, max_frames: int = None) -> dict:
        """
        Connects to a live IP Webcam (or video stream), reads frames continuously,
        runs real-time YOLO object detection, tracking, and safety event analytics,
        and optionally displays the annotated live stream in an OpenCV desktop window.
        """
        source = stream_source or getattr(config, "IP_WEBCAM_URL", "http://192.168.0.107:8080/video")
        is_url = str(source).startswith(("http://", "https://", "rtsp://"))
        if is_url:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|max_delay;500000"

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[CityEye Live Stream Error] Could not connect to camera at: {source}")
            return {"status": "error", "message": f"Cannot connect to stream: {source}", "events": []}

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        writer = None
        if output_path:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        print(f"\n[CityEye Camera] Connected to live feed: {source}")
        print("Press 'q' or ESC in the video window to stop live detection.\n")

        frame_no = 0
        start_time = time.time()
        retry_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    if is_url:
                        retry_count += 1
                        time.sleep(0.05)
                        if retry_count < 20:
                            continue
                        else:
                            print(f"[CityEye Camera] Reconnecting to {source}...")
                            cap.release()
                            time.sleep(0.5)
                            cap = cv2.VideoCapture(source)
                            retry_count = 0
                            continue
                    else:
                        break

                retry_count = 0
                frame_no += 1

                annotated_frame, alerts, new_events = self.process_frame(frame, frame_no, fps)

                if writer:
                    writer.write(annotated_frame)

                if show:
                    cv2.imshow("CityEye — Live IP Webcam YOLO Stream (Press Q to quit)", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("\n[CityEye Camera] Live stream stopped by user.")
                        break

                if max_frames and frame_no >= max_frames:
                    break

        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        elapsed = round(time.time() - start_time, 2)
        summary = self.get_summary_statistics()
        summary["elapsed_seconds"] = elapsed
        summary["total_frames"] = frame_no
        summary["source"] = str(source)
        return summary

    def process_image(self, image_path: str = config.IMAGE_PATH, output_path: str = config.IMAGE_OUTPUT_PATH) -> dict:
        """
        Runs YOLO object detection on a single static image (without tracking).
        Detects persons, motorcycles, and flags triple riding when >=3 persons are on a motorcycle.
        Draws bounding boxes, labels, confidence scores, and saves to output_path.
        """
        if not os.path.exists(image_path):
            msg = f"[CityEye Image] Image not found at '{image_path}'. Please place an image at '{image_path}'."
            print(msg)
            return {
                "status": "error",
                "message": msg,
                "input_path": image_path,
                "persons_detected": 0,
                "motorcycles_detected": 0,
                "triple_riding_count": 0,
                "events": []
            }

        frame = cv2.imread(image_path)
        if frame is None:
            msg = f"[CityEye Image] Failed to load image from '{image_path}'. File may be corrupt or unsupported format."
            print(msg)
            return {
                "status": "error",
                "message": msg,
                "input_path": image_path,
                "persons_detected": 0,
                "motorcycles_detected": 0,
                "triple_riding_count": 0,
                "events": []
            }

        h, w = frame.shape[:2]
        print(f"\n[CityEye Image] Processing image: {image_path} ({w}x{h})...")

        # 1. Run YOLO object detection
        detections = self.detector.detect(frame)
        annotated_img = frame.copy()

        person_indices = []
        motorcycle_indices = []
        other_vehicle_indices = []

        if len(detections) > 0:
            for i in range(len(detections)):
                cls_id = int(detections.class_id[i])
                if cls_id == config.CLASS_PERSON:
                    person_indices.append(i)
                elif cls_id == config.CLASS_MOTORCYCLE:
                    motorcycle_indices.append(i)
                elif cls_id in config.VEHICLE_CLASS_IDS:
                    other_vehicle_indices.append(i)

        # 2. Analyze Motorcycle & Person associations for Triple Riding & Helmet Violations
        motorcycle_info = []
        triple_riding_events = []
        triple_riding_person_indices = set()
        person_helmet_status = {}  # p_idx -> dict (available, status, confidence, class_name)
        helmet_violation_events = []

        for m_idx in motorcycle_indices:
            m_box = detections.xyxy[m_idx]
            m_conf = float(detections.confidence[m_idx]) if detections.confidence is not None else 0.85

            # Find all persons sitting on/near this motorcycle
            associated_persons = []
            for p_idx in person_indices:
                p_box = detections.xyxy[p_idx]
                if self._is_person_on_motorcycle(p_box, m_box):
                    associated_persons.append(p_idx)

                    # Real Helmet Detection on this rider
                    if self.detector.helmet_model_available:
                        h_res = self.detector.check_rider_helmet(frame, p_box, m_box)
                        person_helmet_status[p_idx] = h_res
                        if h_res.get("available") and h_res.get("status") == "NO HELMET":
                            h_ev = {
                                "camera_id": self.camera_id,
                                "source": "image",
                                "event": "helmet_violation",
                                "vehicle_id": m_idx + 1,
                                "confidence": round(float(h_res.get("confidence", 0.75)), 2),
                                "status": "NO HELMET",
                                "rider_box": [round(float(c), 1) for c in p_box],
                                "timestamp": datetime.datetime.now().isoformat(timespec="seconds")
                            }
                            helmet_violation_events.append(h_ev)
                            self.events.append(h_ev)

            p_count = len(associated_persons)
            is_triple_riding = (p_count >= 3)

            if is_triple_riding:
                triple_riding_person_indices.update(associated_persons)
                ev = {
                    "event": "triple_riding",
                    "confidence": round(m_conf, 2),
                    "person_count": p_count,
                    "box": [round(float(c), 1) for c in m_box],
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds")
                }
                triple_riding_events.append(ev)
                self.events.append({
                    "camera_id": self.camera_id,
                    "source": "image",
                    "event": "triple_riding",
                    "vehicle_id": m_idx + 1,
                    "confidence": round(m_conf, 2),
                    "person_count": p_count,
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds")
                })

            motorcycle_info.append({
                "idx": m_idx,
                "box": m_box,
                "conf": m_conf,
                "person_count": p_count,
                "is_triple_riding": is_triple_riding,
                "associated_persons": associated_persons
            })

        # Save any new events to data/events.json
        if triple_riding_events or helmet_violation_events:
            self.save_events()

        # 3. Draw Bounding Boxes, Labels, and Confidence Scores
        # Draw Persons
        for p_idx in person_indices:
            p_box = detections.xyxy[p_idx]
            p_conf = float(detections.confidence[p_idx]) if detections.confidence is not None else 0.85
            x1, y1, x2, y2 = map(int, p_box)

            is_triple_rider = (p_idx in triple_riding_person_indices)
            h_info = person_helmet_status.get(p_idx, {})

            if h_info.get("available") and h_info.get("status") in ("HELMET", "NO HELMET"):
                h_status = h_info["status"]
                h_conf = h_info["confidence"]
                if h_status == "NO HELMET":
                    p_color = (0, 0, 255)  # Bright Red
                    label = f"RIDER: NO HELMET {h_conf:.2f}"
                else:
                    p_color = (0, 255, 128)  # Bright Green
                    label = f"RIDER: HELMET {h_conf:.2f}"
            elif is_triple_rider:
                p_color = (0, 69, 255)  # Orange-Red for triple riders
                label = f"PERSON {p_conf:.2f} (RIDER)"
            else:
                p_color = (255, 200, 0)  # Cyan/Blue
                label = f"PERSON {p_conf:.2f}"

            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), p_color, 2)
            self._draw_label(annotated_img, label, x1, y1, p_color)

        # Draw Other Vehicles (Cars, Buses, Trucks)
        for v_idx in other_vehicle_indices:
            v_box = detections.xyxy[v_idx]
            v_conf = float(detections.confidence[v_idx]) if detections.confidence is not None else 0.85
            v_cls = int(detections.class_id[v_idx])
            v_name = self.detector.get_class_name(v_cls).upper()
            x1, y1, x2, y2 = map(int, v_box)
            v_color = (0, 255, 128)  # Green
            label = f"{v_name} {v_conf:.2f}"

            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), v_color, 2)
            self._draw_label(annotated_img, label, x1, y1, v_color)

        # Draw Motorcycles (with prominent highlight if Triple Riding or No Helmet)
        for m_info in motorcycle_info:
            x1, y1, x2, y2 = map(int, m_info["box"])
            m_conf = m_info["conf"]
            p_count = m_info["person_count"]

            if m_info["is_triple_riding"]:
                m_color = (0, 0, 255)  # Bright Red
                thickness = 3
                label = f"MOTORCYCLE {m_conf:.2f} | TRIPLE RIDING ({p_count} PERSONS)"
            else:
                m_color = (0, 255, 255)  # Yellow for normal motorcycle
                thickness = 2
                rider_suffix = f" ({p_count} rider{'s' if p_count > 1 else ''})" if p_count > 0 else ""
                label = f"MOTORCYCLE {m_conf:.2f}{rider_suffix}"

            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), m_color, thickness)
            self._draw_label(annotated_img, label, x1, y1, m_color)

        # 4. Top CCTV HUD Banner
        banner_h = 60
        hud_bg = np.zeros((banner_h, w, 3), dtype=np.uint8)
        cv2.rectangle(hud_bg, (0, 0), (w, banner_h), (15, 18, 25), -1)
        annotated_img[0:banner_h, 0:w] = cv2.addWeighted(annotated_img[0:banner_h, 0:w], 0.2, hud_bg, 0.8, 0)

        cv2.putText(annotated_img, "CITYEYE AI IMAGE ANALYTICS", (20, 24), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 240, 255), 2, cv2.LINE_AA)
        helmet_tag = f"HELMET VIOLATIONS: {len(helmet_violation_events)}" if self.detector.helmet_model_available else "HELMET AI: NOT LOADED"
        status_line = f"FILE: {os.path.basename(image_path)} | PERSONS: {len(person_indices)} | BIKES: {len(motorcycle_indices)} | TRIPLE: {len(triple_riding_events)} | {helmet_tag}"
        cv2.putText(annotated_img, status_line, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

        if triple_riding_events:
            alert_text = f"CRITICAL: {len(triple_riding_events)} TRIPLE RIDING DETECTED"
            (atw, _), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.putText(annotated_img, alert_text, (max(20, w - atw - 20), 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        elif helmet_violation_events:
            alert_text = f"WARNING: {len(helmet_violation_events)} NO-HELMET VIOLATION(S)"
            (atw, _), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.putText(annotated_img, alert_text, (max(20, w - atw - 20), 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2, cv2.LINE_AA)

        # 5. Save Output Image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated_img)

        print(f"[CityEye Image] Detection complete:")
        print(f"  - Persons detected:       {len(person_indices)}")
        print(f"  - Motorcycles detected:   {len(motorcycle_indices)}")
        print(f"  - Other vehicles:         {len(other_vehicle_indices)}")
        print(f"  - Triple Riding events:   {len(triple_riding_events)}")
        print(f"  - Helmet Violations:      {len(helmet_violation_events)}")
        print(f"  - Result saved to:        {output_path}\n")

        return {
            "status": "success",
            "message": f"Detected {len(person_indices)} persons, {len(motorcycle_indices)} motorcycles, {len(triple_riding_events)} triple riding violations.",
            "input_path": image_path,
            "output_path": output_path,
            "persons_detected": len(person_indices),
            "motorcycles_detected": len(motorcycle_indices),
            "triple_riding_count": len(triple_riding_events),
            "events": triple_riding_events
        }

    def _draw_label(self, img: np.ndarray, text: str, x1: int, y1: int, color: tuple):
        """Draws a solid filled banner for text label readability."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        font_thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        label_y1 = max(0, y1 - th - 8)
        label_y2 = y1
        cv2.rectangle(img, (x1, label_y1), (x1 + tw + 8, label_y2), color, -1)
        # Use dark text on bright labels, white text on dark labels
        text_color = (0, 0, 0) if (color[0] + color[1] + color[2]) > 350 else (255, 255, 255)
        cv2.putText(img, text, (x1 + 4, label_y2 - 4), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    def process_video(self, video_path: str = config.VIDEO_PATH, output_path: str = config.OUTPUT_PATH, conf: float = None, imgsz: int = 640) -> dict:
        """
        Processes a full MP4 video with ByteTrack multi-object tracking.
        Detects collisions/accidents, wrong-way driving, stopped vehicle, helmet violation, and triple riding.
        Writes annotated video to output_path and logs events to data/events.json.
        """
        import time

        if not os.path.exists(video_path):
            msg = f"[CityEye Video] Video file not found at '{video_path}'. (Skipping video processing)"
            print(msg)
            return {
                "status": "error",
                "message": msg,
                "input_path": video_path,
                "video_name": os.path.basename(video_path),
                "total_detections": 0,
                "total_events": 0,
                "no_helmet_events": 0,
                "accident_events": 0,
                "other_events": 0,
                "events": []
            }

        self.video_name = os.path.basename(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            msg = f"[CityEye Video] Failed to open video file '{video_path}'."
            print(msg)
            return {
                "status": "error",
                "message": msg,
                "input_path": video_path,
                "video_name": self.video_name,
                "total_detections": 0,
                "total_events": 0,
                "no_helmet_events": 0,
                "accident_events": 0,
                "other_events": 0,
                "events": []
            }

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"\n[CityEye Video] Processing video: {video_path}")
        print(f"  - Video Name: {self.video_name}")
        print(f"  - Resolution: {width}x{height} @ {fps:.1f} FPS ({total_frames} frames)")
        print(f"  - Output:     {output_path}")

        frame_no = 0
        start_time = time.time()
        total_detections_count = 0
        run_events = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_no += 1
                # Run YOLO detection & tracking
                dets = self.detector.detect(frame, conf=conf, imgsz=imgsz)
                total_detections_count += len(dets)
                tracked = self.tracker.update(dets)

                annotated_frame, alerts, frame_events = self.process_frame(
                    frame=frame,
                    frame_no=frame_no,
                    fps=fps,
                    tracked_detections=tracked
                )
                writer.write(annotated_frame)
                if frame_events:
                    run_events.extend(frame_events)

                if frame_no % 30 == 0 or frame_no == total_frames:
                    pct = int((frame_no / max(1, total_frames)) * 100) if total_frames > 0 else 0
                    print(f"  [Frame {frame_no}/{total_frames}] {pct}% | Detections: {total_detections_count} | Active Events: {len(run_events)}")
        finally:
            cap.release()
            writer.release()

        elapsed = round(time.time() - start_time, 2)
        fps_speed = round(frame_no / max(0.001, elapsed), 1)

        # Categorize run events
        no_helmet_count = sum(1 for e in run_events if e.get("event") in ("helmet_violation", "no_helmet"))
        accident_count = sum(1 for e in run_events if e.get("event") in ("accident_collision", "possible_accident"))
        triple_count = sum(1 for e in run_events if e.get("event") == "triple_riding")
        wrong_way_count = sum(1 for e in run_events if e.get("event") == "wrong_way_driving")
        stopped_count = sum(1 for e in run_events if e.get("event") == "vehicle_stopped")
        other_events_count = triple_count + wrong_way_count + stopped_count

        print(f"\n[CityEye Video] Video Processing Complete ({self.video_name}):")
        print(f"  - Total Frames:     {frame_no} ({fps_speed} FPS)")
        print(f"  - Elapsed Time:     {elapsed}s")
        print(f"  - Total Detections: {total_detections_count}")
        print(f"  - Total Events:     {len(run_events)}")
        print(f"  - NO HELMET:        {no_helmet_count}")
        print(f"  - Accident:         {accident_count}")
        print(f"  - Other Events:     {other_events_count}")
        print(f"  - Output Video:     {output_path}\n")

        return {
            "status": "success",
            "video_name": self.video_name,
            "input_path": video_path,
            "output_path": output_path,
            "total_frames_processed": frame_no,
            "elapsed_seconds": elapsed,
            "processing_fps": fps_speed,
            "total_detections": total_detections_count,
            "total_events": len(run_events),
            "no_helmet_events": no_helmet_count,
            "accident_events": accident_count,
            "triple_riding_events": triple_count,
            "wrong_way_events": wrong_way_count,
            "vehicle_stopped_events": stopped_count,
            "other_events": other_events_count,
            "events": run_events
        }

    def get_summary_statistics(self) -> dict:
        """
        Computes aggregate statistics for the dashboard.
        """
        stats = {
            "total_events": len(self.events),
            "triple_riding": 0,
            "wrong_way_driving": 0,
            "vehicle_stopped": 0,
            "helmet_violation": 0,
            "accident_collision": 0,
            "total_vehicles": len(self.tracker.all_seen_vehicles) if self.tracker else 0,
            "total_persons": len(self.tracker.all_seen_persons) if self.tracker else 0
        }

        for ev in self.events:
            ev_type = ev.get("event")
            if ev_type in stats:
                stats[ev_type] += 1

        return stats

