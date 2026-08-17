import os
import time
import json
import datetime
import threading
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import config
from detector import YOLODetector
from tracker import MultiObjectTracker
from event_detector import EventDetector

# Ensure required directories exist
for directory in [config.VIDEOS_DIR, config.OUTPUT_DIR, config.DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

app = FastAPI(
    title="CityEye AI Video Analytics",
    description="Real-time CCTV & Traffic AI Event Detection System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static and output directories
STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/output", StaticFiles(directory=str(config.OUTPUT_DIR)), name="output")

# Processing state tracking
processing_lock = threading.Lock()
pipeline_state = {
    "is_processing": False,
    "progress_pct": 0,
    "current_frame": 0,
    "total_frames": 0,
    "latest_message": "Idle",
    "last_run_summary": None
}

# Live Streaming & Telemetry State
live_stream_lock = threading.Lock()
active_stream_session_id = 0
live_telemetry_state = {
    "is_streaming": False,
    "video_file": None,
    "frame_no": 0,
    "total_frames": 0,
    "fps": 25.0,
    "counts": {
        "total": 0,
        "person": 0,
        "car": 0,
        "bus": 0,
        "truck": 0,
        "motorcycle": 0,
        "cumulative_vehicles": 0,
        "cumulative_persons": 0,
        "cumulative_events": 0
    },
    "current_detections": [],
    "active_alerts": [],
    "recent_events": []
}

# Global instances (lazy-loaded or initialized)
detector_instance = None
tracker_instance = None
event_detector_instance = None


def get_detector():
    global detector_instance
    if detector_instance is None:
        detector_instance = YOLODetector()
    return detector_instance


def process_video_pipeline(video_path: str, output_path: str) -> dict:
    """
    Core video processing pipeline: reads MP4, detects objects with YOLO,
    tracks with ByteTrack, detects 4 traffic events, and writes output video.
    """
    global pipeline_state, tracker_instance, event_detector_instance

    if not os.path.exists(video_path):
        err_msg = "traffic.mp4 not found. Put an MP4 traffic video inside videos/traffic.mp4"
        print(f"[CityEye Error] {err_msg}")
        return {
            "status": "error",
            "message": err_msg,
            "events_count": 0
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        err_msg = f"Failed to open video file: {video_path}"
        print(f"[CityEye Error] {err_msg}")
        return {"status": "error", "message": err_msg, "events_count": 0}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer (using mp4v / avc1 compatible codec)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    det = get_detector()
    tracker_instance = MultiObjectTracker()
    event_detector_instance = EventDetector(det, tracker_instance, config.CAMERA_ID)

    pipeline_state["is_processing"] = True
    pipeline_state["total_frames"] = total_frames
    pipeline_state["current_frame"] = 0
    pipeline_state["latest_message"] = "Processing video frames..."

    frame_no = 0
    start_time = time.time()
    events_logged = 0

    print(f"\n[CityEye Pipeline] Started processing: {video_path} ({total_frames} frames, {width}x{height} @ {fps:.1f} FPS)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_no += 1
            annotated_frame, alerts, new_events = event_detector_instance.process_frame(frame, frame_no, fps)
            writer.write(annotated_frame)

            events_logged += len(new_events)

            if total_frames > 0:
                pct = int((frame_no / total_frames) * 100)
                pipeline_state["progress_pct"] = min(100, pct)
            pipeline_state["current_frame"] = frame_no

            if frame_no % 30 == 0:
                print(f"[CityEye] Frame {frame_no}/{total_frames} ({pipeline_state['progress_pct']}%) - Events so far: {len(event_detector_instance.events)}")

    finally:
        cap.release()
        writer.release()

    elapsed = round(time.time() - start_time, 2)
    stats = event_detector_instance.get_summary_statistics()

def generate_live_video_stream(video_filename: str, conf_threshold: float = config.CONFIDENCE_THRESHOLD, loop: bool = True):
    """
    Frame-by-frame generator for live MJPEG video streaming with real-time YOLO object detection.
    Supports both live IP Webcam camera streams (http://192.168.0.107:8080/video) and offline video files.
    Draws rectangular bounding boxes with class name + confidence % (e.g. 'car 91%', 'person 95%'),
    updates live detection telemetry, and yields multipart JPEG frames.

    Performance optimizations for live feeds:
    - Frame skipping: runs YOLO only every Nth frame (config.LIVE_DETECTION_INTERVAL)
    - Smaller inference size: uses config.LIVE_IMGSZ (416) for faster detection
    - Reduced display resolution: 640px width for faster JPEG encoding
    """
    global live_telemetry_state, active_stream_session_id

    # Register current streaming session
    session_id = time.time()
    active_stream_session_id = session_id

    # Check if video_filename is an IP stream URL or keyword
    is_url_stream = str(video_filename).startswith(("http://", "https://", "rtsp://")) or video_filename == "ip_webcam"
    
    if is_url_stream:
        video_path = getattr(config, "IP_WEBCAM_URL", "http://192.168.0.107:8080/video") if video_filename == "ip_webcam" else video_filename
        camera_id = "cam_ip_webcam"
        stream_label = "IP Webcam (Phone Camera)"
    else:
        # Resolve local video path
        video_path = os.path.join(config.VIDEOS_DIR, video_filename)
        if not os.path.exists(video_path):
            video_path = config.VIDEO_PATH
        camera_id = f"cam_{Path(video_filename).stem}"
        stream_label = video_filename

    det = get_detector()
    trk = MultiObjectTracker()
    # Start with a clean event list — no stale events from previous runs
    ev_det = EventDetector(det, trk, camera_id=camera_id, video_name=stream_label, start_clean=True)

    # Open video capture with network optimizations if URL
    if is_url_stream:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|max_delay;500000"
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[CityEye Stream Error] Cannot connect to source: {video_path}")
        # Yield single placeholder error frame
        err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(err_frame, "CANNOT CONNECT TO CAMERA", (60, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(err_frame, f"URL: {video_path}", (60, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(err_frame, "Please ensure phone and PC are on the same Wi-Fi.", (60, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1)
        _, buf = cv2.imencode('.jpg', err_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        return

    total_frames = 0 if is_url_stream else (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    frame_delay = 0.001 if is_url_stream else (1.0 / max(1.0, min(fps, 30.0)))

    # Live-mode detection interval: only run YOLO every Nth frame
    detection_interval = config.LIVE_DETECTION_INTERVAL if is_url_stream else 1
    live_imgsz = config.LIVE_IMGSZ if is_url_stream else 640

    frame_no = 0
    # Clear stale telemetry from previous sessions
    with live_stream_lock:
        live_telemetry_state["is_streaming"] = True
        live_telemetry_state["video_file"] = stream_label
        live_telemetry_state["total_frames"] = total_frames
        live_telemetry_state["fps"] = round(fps, 1)
        live_telemetry_state["recent_events"] = []

    print(f"[CityEye Live Stream] Started stream for '{stream_label}' (Source: {video_path}, Conf: {conf_threshold}, DetInterval: {detection_interval}, ImgSz: {live_imgsz})")

    retry_count = 0
    last_tracked = None  # Cache last tracked detections for frame-skipping
    try:
        while active_stream_session_id == session_id:
            loop_start = time.time()
            ret, frame = cap.read()

            if not ret or frame is None:
                if is_url_stream:
                    retry_count += 1
                    time.sleep(0.05)
                    if retry_count < 15:
                        continue
                    else:
                        print(f"[CityEye Live Stream] Reconnecting to IP camera at {video_path}...")
                        cap.release()
                        time.sleep(0.3)
                        cap = cv2.VideoCapture(video_path)
                        retry_count = 0
                        continue
                elif loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_no = 0
                    continue
                else:
                    break

            retry_count = 0
            frame_no += 1

            # Resize for display efficiency: 640px width for live camera, keeps aspect ratio
            if is_url_stream and frame.shape[1] > 720:
                scale = 640.0 / frame.shape[1]
                frame = cv2.resize(frame, (640, int(frame.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)

            # Frame skipping: only run full YOLO + tracking on detection frames
            run_detection = (frame_no % detection_interval == 0) or (last_tracked is None)

            if run_detection:
                # Full pipeline: YOLO detect -> ByteTrack -> Event analysis
                detections = det.detect(frame, conf=conf_threshold, imgsz=live_imgsz)
                tracked = trk.update(detections)
                last_tracked = tracked
                annotated_frame, alerts, new_events = ev_det.process_frame(frame, frame_no, fps, tracked_detections=tracked)
            else:
                # Reuse last tracked detections — just redraw overlay without re-running YOLO
                annotated_frame, alerts, new_events = ev_det.process_frame(frame, frame_no, fps, tracked_detections=last_tracked)

            # Retrieve telemetry data compiled in process_frame
            telemetry = getattr(ev_det, "last_frame_telemetry", {})
            trk_stats = trk.get_stats()
            frame_counts = telemetry.get("counts", {
                "total": 0, "person": 0, "car": 0, "bus": 0, "truck": 0, "motorcycle": 0
            })

            # Update live telemetry state for real-time frontend polling
            with live_stream_lock:
                live_telemetry_state["is_streaming"] = True
                live_telemetry_state["frame_no"] = frame_no
                live_telemetry_state["counts"] = {
                    **frame_counts,
                    "cumulative_vehicles": trk_stats.get("total_vehicles", 0),
                    "cumulative_persons": trk_stats.get("total_persons", 0),
                    "cumulative_events": len(ev_det.events)
                }
                live_telemetry_state["current_detections"] = telemetry.get("detections", [])
                live_telemetry_state["active_alerts"] = alerts
                if new_events:
                    live_telemetry_state["recent_events"] = (new_events + live_telemetry_state["recent_events"])[:50]

            # Encode annotated frame as JPEG (quality 85 for clear bounding boxes)
            success, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )

            # Delay regulation
            elapsed = time.time() - loop_start
            if not is_url_stream:
                sleep_duration = max(0.005, frame_delay - elapsed)
                time.sleep(sleep_duration)
            else:
                time.sleep(0.005)

    except GeneratorExit:
        pass
    except Exception as e:
        print(f"[CityEye Live Stream] Exception in streaming loop: {e}")
    finally:
        cap.release()
        if active_stream_session_id == session_id:
            with live_stream_lock:
                live_telemetry_state["is_streaming"] = False
        print(f"[CityEye Live Stream] Stopped stream for '{stream_label}' at frame {frame_no}")


@app.get("/videos")
def list_videos():
    """
    Returns a list of all existing video clips from the project's 'videos' folder
    along with the default IP Webcam live stream source.
    """
    videos = []
    
    # 1. Add IP Webcam / Android Phone Camera as the Default Primary Source
    ip_webcam_url = getattr(config, "IP_WEBCAM_URL", "http://192.168.0.107:8080/video")
    if ip_webcam_url:
        videos.append({
            "filename": ip_webcam_url,
            "display_name": "📱 Live IP Webcam",
            "is_live_stream": True,
            "size_mb": 0,
            "frames": "Continuous Live",
            "fps": 30.0,
            "resolution": "Live HD",
            "duration_sec": 0
        })

    supported_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}
    if os.path.exists(config.VIDEOS_DIR):
        for fname in sorted(os.listdir(config.VIDEOS_DIR)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in supported_extensions:
                fpath = os.path.join(config.VIDEOS_DIR, fname)
                try:
                    size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
                    cap = cv2.VideoCapture(fpath)
                    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = round(cap.get(cv2.CAP_PROP_FPS) or 25.0, 1)
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    duration_sec = round(total_f / fps, 1) if fps > 0 else 0
                except Exception:
                    size_mb, total_f, fps, w, h, duration_sec = 0, 0, 25.0, 0, 0, 0

                videos.append({
                    "filename": fname,
                    "display_name": f"📹 {fname} ({total_f} frames • {w}x{h})",
                    "is_live_stream": False,
                    "size_mb": size_mb,
                    "frames": total_f,
                    "fps": fps,
                    "resolution": f"{w}x{h}" if w > 0 else "Unknown",
                    "duration_sec": duration_sec
                })

    return {
        "count": len(videos),
        "default_video": ip_webcam_url or (videos[0]["filename"] if videos else None),
        "videos": videos
    }


@app.get("/video-feed")
def live_video_feed(
    video: str = Query(None, description="Filename or URL of the video/camera stream"),
    conf: float = Query(config.CONFIDENCE_THRESHOLD, ge=0.1, le=0.95, description="YOLO Confidence threshold"),
    loop: bool = Query(True, description="Whether to loop the video stream")
):
    """
    Live real-time streaming endpoint that processes IP Webcam or video frame-by-frame using YOLO.
    Draws clear rectangular bounding boxes with class name + confidence score near each box.
    """
    if not video:
        video = getattr(config, "IP_WEBCAM_URL", "http://192.168.0.107:8080/video")

    is_url_stream = str(video).startswith(("http://", "https://", "rtsp://")) or video == "ip_webcam"
    if not is_url_stream:
        target_path = os.path.join(config.VIDEOS_DIR, video)
        if not os.path.exists(target_path):
            if os.path.exists(config.VIDEO_PATH):
                video = os.path.basename(config.VIDEO_PATH)
            else:
                video = getattr(config, "IP_WEBCAM_URL", "http://192.168.0.107:8080/video")

    return StreamingResponse(
        generate_live_video_stream(video_filename=video, conf_threshold=conf, loop=loop),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/live-data")
def get_live_telemetry():
    """
    Returns real-time telemetry for the active video detection stream:
    - Object counters (Total detections, People, Cars, Buses, Trucks, Motorcycles)
    - Current frame detections with bounding boxes & confidence scores
    - Active alerts and new events
    """
    with live_stream_lock:
        return {
            "status": "streaming" if live_telemetry_state["is_streaming"] else "idle",
            **live_telemetry_state,
            "timestamp": datetime.datetime.now().isoformat()
        }


@app.post("/stop-feed")
def stop_live_feed():
    """Stops any active live video stream."""
    global active_stream_session_id
    active_stream_session_id = 0
    with live_stream_lock:
        live_telemetry_state["is_streaming"] = False
    return {"status": "stopped", "message": "Live video detection stream stopped."}


@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    """Serves the main CityEye CCTV Dashboard UI."""
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h2>CityEye Dashboard template not found</h2>", status_code=404)
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
def health_check():
    """Returns API and system health status."""
    video_exists = os.path.exists(config.VIDEO_PATH)
    output_exists = os.path.exists(config.OUTPUT_PATH)
    model_exists = os.path.exists(config.MODEL_PATH)

    return {
        "status": "ok",
        "camera_id": config.CAMERA_ID,
        "traffic_video_available": video_exists,
        "processed_output_available": output_exists,
        "model_file_exists": model_exists,
        "model_path": config.MODEL_PATH,
        "helmet_detection_enabled": config.HELMET_MODEL_PATH is not None,
        "pipeline_state": pipeline_state,
        "live_telemetry": {
            "is_streaming": live_telemetry_state["is_streaming"],
            "video_file": live_telemetry_state["video_file"]
        }
    }


@app.get("/events")
def get_events():
    """Returns stored events from data/events.json and aggregate statistics."""
    events = []
    if os.path.exists(config.EVENTS_JSON_PATH):
        try:
            with open(config.EVENTS_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                events = data.get("events", [])
        except Exception as e:
            print(f"[API] Error reading events.json: {e}")

    # Compute breakdown statistics
    triple_riding = sum(1 for e in events if e.get("event") == "triple_riding")
    wrong_way = sum(1 for e in events if e.get("event") == "wrong_way_driving")
    stopped = sum(1 for e in events if e.get("event") == "vehicle_stopped")
    helmet = sum(1 for e in events if e.get("event") == "helmet_violation")

    # Get distinct vehicle IDs and person IDs if available
    unique_vehicles = len(set(e.get("vehicle_id") for e in events if "vehicle_id" in e))

    return {
        "total_events": len(events),
        "statistics": {
            "triple_riding": triple_riding,
            "wrong_way_driving": wrong_way,
            "vehicle_stopped": stopped,
            "helmet_violation": helmet,
            "total_vehicles": unique_vehicles,
            "total_persons": triple_riding * 3  # estimate from events if offline
        },
        "events": list(reversed(events))  # Return newest events first
    }


@app.get("/status")
def get_pipeline_status():
    """Returns real-time progress status of the video analysis pipeline."""
    return pipeline_state


@app.post("/process")
def trigger_process(video: str = None):
    """
    Triggers video processing on the selected video (default: config.VIDEO_PATH) asynchronously in a background thread.
    Returns immediately so the browser UI can display live progress without timeout.
    """
    selected_video_path = None
    if video:
        cand = os.path.join(config.VIDEOS_DIR, video)
        if os.path.exists(cand):
            selected_video_path = cand

    if not selected_video_path:
        selected_video_path = config.VIDEO_PATH

    if not os.path.exists(selected_video_path):
        # Check any video in videos dir
        videos_in_dir = [os.path.join(config.VIDEOS_DIR, f) for f in os.listdir(config.VIDEOS_DIR) if f.endswith(".mp4")]
        if videos_in_dir:
            selected_video_path = videos_in_dir[0]
        else:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": f"No video found at {selected_video_path}. Put an MP4 video inside videos/",
                    "expected_path": selected_video_path,
                    "events_count": 0
                }
            )

    if pipeline_state["is_processing"]:
        return {
            "status": "in_progress",
            "message": "Video analysis is already running.",
            "progress_pct": pipeline_state["progress_pct"]
        }

    # Reset pipeline state
    pipeline_state["is_processing"] = True
    pipeline_state["progress_pct"] = 0
    pipeline_state["current_frame"] = 0
    pipeline_state["total_frames"] = 0
    pipeline_state["latest_message"] = f"Initializing YOLO & ByteTrack on {os.path.basename(selected_video_path)}..."
    pipeline_state["last_run_summary"] = None

    # Start processing in background thread
    worker_thread = threading.Thread(
        target=process_video_pipeline,
        args=(selected_video_path, config.OUTPUT_PATH),
        daemon=True
    )
    worker_thread.start()

    return {
        "status": "started",
        "message": f"AI Video processing started for {os.path.basename(selected_video_path)}.",
        "video_path": selected_video_path
    }


@app.get("/output-video")
def get_processed_video():
    """Serves the processed video file for the frontend player."""
    if not os.path.exists(config.OUTPUT_PATH):
        raise HTTPException(
            status_code=404,
            detail="Processed video not found. Run /process first."
        )

    return FileResponse(
        config.OUTPUT_PATH,
        media_type="video/mp4",
        filename="processed_video.mp4"
    )


@app.post("/create-sample-video")
def create_sample_video_endpoint():
    """
    Helper endpoint to create a synthetic CCTV test video containing
    Helmet violation, Triple-riding motorcycle, Wrong-way car, and Stopped car.
    """
    try:
        from generate_sample import generate_cctv_sample
        out_file = generate_cctv_sample(config.VIDEO_PATH)
        return {
            "status": "success",
            "message": f"Sample CCTV traffic video generated at {out_file}",
            "video_path": out_file
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
