"""
CityEye — AI Traffic & CCTV Event Detection System
==================================================
Multi-Video and Dual-Video CLI runner:
- Automatically discovers ALL video files in the "videos/" folder (.mp4, .avi, .mov, .mkv, etc.)
- Sequentially processes Video 1 (Regular/Traffic) and Video 2 (Accident/Collision)
- Runs full YOLO object detection + ByteTrack tracking + Safety Event analytics
- Video 1: Helmet AI (models/helmet_best.pt), Triple-Riding, Wrong-Way, Stopped vehicle
- Video 2: Collision/Accident detection, Road hazard stoppage, Overlap physics
- Saves individual output videos in output/ and separate event logs in data/
- Allows manual single-video or multi-video selection
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path

import config
from detector import YOLODetector
from tracker import MultiObjectTracker
from event_detector import EventDetector


def find_all_videos(videos_dir: str = config.VIDEOS_DIR) -> list:
    """
    Scans the videos directory and returns all supported video files.
    Supports .mp4, .avi, .mov, .mkv, .wmv, .webm.
    """
    if not os.path.exists(videos_dir):
        return []

    supported_exts = getattr(config, "SUPPORTED_VIDEO_EXTENSIONS", {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"})
    found_videos = []

    for fname in sorted(os.listdir(videos_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in supported_exts:
            full_path = os.path.join(videos_dir, fname)
            found_videos.append(full_path)

    # Sort prioritizing standard pairs: input.mp4 first, accident.mp4 second
    def sort_key(path):
        name = os.path.basename(path).lower()
        if "input" in name:
            return (0, name)
        elif "traffic" in name:
            return (1, name)
        elif "accident" in name or "crash" in name:
            return (2, name)
        return (3, name)

    found_videos.sort(key=sort_key)
    return found_videos


def process_single_video(
    video_path: str,
    output_path: str,
    detector: YOLODetector,
    camera_id: str = None,
    conf: float = 0.35,
    imgsz: int = 640
) -> tuple:
    """
    Processes a single video file with a fresh tracker state and returns metrics & events.
    """
    tracker = MultiObjectTracker()
    vname = os.path.basename(video_path)
    cam_id = camera_id or f"cam_{os.path.splitext(vname)[0]}"

    event_detector = EventDetector(
        detector=detector,
        tracker=tracker,
        camera_id=cam_id,
        video_name=vname
    )

    result = event_detector.process_video(
        video_path=video_path,
        output_path=output_path,
        conf=conf,
        imgsz=imgsz
    )

    return result, event_detector


def run_video_pipeline(video_paths: list = None, conf: float = 0.35):
    """
    Main execution pipeline: Processes all provided or auto-detected videos,
    saves separate event logs for each video, saves combined events,
    and outputs the required terminal summary.
    """
    print("=" * 65)
    print("        CITYEYE AI TRAFFIC & CCTV DETECTION SYSTEM        ")
    print("=" * 65)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # 1. Discover all videos if not explicitly provided
    if not video_paths:
        video_paths = find_all_videos(config.VIDEOS_DIR)

    if not video_paths:
        print(f"\n[CityEye Error] No supported video files found in '{config.VIDEOS_DIR}'.")
        print(f"  Supported formats: {config.SUPPORTED_VIDEO_EXTENSIONS}")
        return []

    print(f"\n[CityEye Discovery] Found {len(video_paths)} video(s) for processing:")
    for idx, vp in enumerate(video_paths, 1):
        print(f"  • Video {idx}: {os.path.basename(vp)} ({vp})")

    # 2. Shared detector instance (reuses loaded YOLO and Helmet weights in memory)
    detector = YOLODetector(conf=conf)

    print("-" * 65)
    if detector.helmet_model_available:
        print(f"• Helmet AI Status:   [ACTIVE / LOADED] ({config.HELMET_MODEL_PATH})")
    else:
        print(f"• Helmet AI Status:   [NOT CONFIGURED]")
    print("-" * 65)

    all_results = []
    all_combined_events = []

    # 3. Process each video sequentially
    for idx, vpath in enumerate(video_paths, 1):
        vname = os.path.basename(vpath)
        vstem = os.path.splitext(vname)[0]
        vout = os.path.join(config.OUTPUT_DIR, f"{vstem}_detected.mp4")

        print("\n" + "=" * 65)
        print(f" [{idx}/{len(video_paths)}] PROCESSING VIDEO {idx}: {vname}")
        print("=" * 65)

        # Optimize resolution based on video type
        img_res = 480 if "input" in vname.lower() else 640

        v_res, v_ev_det = process_single_video(
            video_path=vpath,
            output_path=vout,
            detector=detector,
            conf=conf,
            imgsz=img_res
        )

        all_results.append(v_res)
        run_events = v_res.get("events", [])
        all_combined_events.extend(run_events)

        # Save individual Video events JSON
        vid_json_filename = f"video_{idx}_events.json"
        vid_json_path = os.path.join(config.DATA_DIR, vid_json_filename)
        with open(vid_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "video_name": vname,
                "video_id": vname,
                "total_detections": v_res.get("total_detections", 0),
                "total_events": v_res.get("total_events", 0),
                "events": run_events
            }, f, indent=2)
        print(f"[CityEye] Video {idx} ({vname}) events saved to: {vid_json_path}")

        # Also save with filename-based JSON for direct reference (e.g. data/accident_events.json)
        named_json_path = os.path.join(config.DATA_DIR, f"{vstem}_events.json")
        with open(named_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "video_name": vname,
                "video_id": vname,
                "total_detections": v_res.get("total_detections", 0),
                "total_events": v_res.get("total_events", 0),
                "events": run_events
            }, f, indent=2)

    # 4. Save combined events
    combined_path = config.EVENTS_JSON_PATH
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump({"events": all_combined_events}, f, indent=2)
    print(f"\n[CityEye] Combined events ({len(all_combined_events)} total) saved to: {combined_path}")

    # 5. Required Terminal Summary
    print("\n" + "=" * 65)
    print("                    TERMINAL SUMMARY                       ")
    print("=" * 65)

    for idx, res in enumerate(all_results, 1):
        vname = res.get("video_name", f"Video {idx}")
        print(f"\nVideo {idx} ({vname}):")
        print(f"- Total detections: {res.get('total_detections', 0)}")
        print(f"- Total events:     {res.get('total_events', 0)}")
        
        # If accident video, show Accident events
        if "accident" in vname.lower() or "crash" in vname.lower():
            print(f"- Accident events:  {res.get('accident_events', 0)}")
        else:
            print(f"- NO HELMET events: {res.get('no_helmet_events', 0)}")
            
        print(f"- Other events:     {res.get('other_events', 0)}")

    print("\n" + "=" * 65)
    print("OUTPUT FILES CREATED:")
    for idx, res in enumerate(all_results, 1):
        vname = res.get("video_name", f"video_{idx}")
        print(f"  • Video {idx} Output:  {res.get('output_path')}")
        print(f"  • Video {idx} Events:  data/video_{idx}_events.json (and data/{os.path.splitext(vname)[0]}_events.json)")
    print(f"  • Combined Events: {combined_path}")
    print("=" * 65 + "\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="CityEye — AI Traffic CCTV Event Detection System (Multi-Video & Live Camera Runner)"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "dual", "both", "image", "video", "camera"],
        default="all",
        help="Processing mode: 'all'/'dual' (process all videos in videos/), 'image', 'video', or 'camera'"
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help=f"Stream from live IP Webcam / phone camera (default: {config.IP_WEBCAM_URL})"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Custom stream URL or camera index (e.g. http://192.168.0.107:8080/video or 0)"
    )
    parser.add_argument(
        "--ip-webcam",
        action="store_true",
        help=f"Connect to Android Phone IP Webcam at {config.IP_WEBCAM_URL}"
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Path to a specific video to process (e.g. videos/accident.mp4 or videos/input.mp4)"
    )
    parser.add_argument(
        "--video1",
        default=None,
        help="Path to Video 1 (e.g. videos/input.mp4)"
    )
    parser.add_argument(
        "--video2",
        default=None,
        help="Path to Video 2 (e.g. videos/accident.mp4)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="YOLO confidence threshold (default: 0.35)"
    )
    parser.add_argument(
        "--image",
        default=None,
        help=f"Path to input image for single-image mode (default: {config.IMAGE_PATH})"
    )

    args = parser.parse_args()

    if args.camera or args.ip_webcam or args.mode == "camera" or (args.source and str(args.source).startswith("http")):
        stream_src = args.source or config.IP_WEBCAM_URL
        print(f"\n[CityEye CLI] Starting Live Phone Camera / IP Webcam Detection: {stream_src}")
        detector = YOLODetector(conf=args.conf)
        tracker = MultiObjectTracker()
        ev_det = EventDetector(detector=detector, tracker=tracker, camera_id="cam_ip_webcam", video_name="IP_Webcam_Live")
        ev_det.process_live_stream(stream_source=stream_src, conf=args.conf, show=True)
    elif args.mode == "image":
        print("Running Single Image Detection...")
        detector = YOLODetector(conf=args.conf)
        tracker = MultiObjectTracker()
        ev_det = EventDetector(detector, tracker)
        img_p = args.image or config.IMAGE_PATH
        ev_det.process_image(img_p, config.IMAGE_OUTPUT_PATH)
    elif args.video:
        # Process a specific chosen video
        run_video_pipeline(video_paths=[args.video], conf=args.conf)
    elif args.video1 or args.video2:
        v_list = []
        if args.video1: v_list.append(args.video1)
        if args.video2: v_list.append(args.video2)
        run_video_pipeline(video_paths=v_list, conf=args.conf)
    else:
        # Auto-detect and process all videos in videos/ folder
        run_video_pipeline(conf=args.conf)


if __name__ == "__main__":
    main()