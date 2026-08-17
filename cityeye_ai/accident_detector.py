from pathlib import Path
import json
import math

from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = (BASE_DIR / "videos" / "accident.mp4") if (BASE_DIR / "videos" / "accident.mp4").exists() else (BASE_DIR / "videos" / "input.mp4")
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

MODEL_PATH = str(BASE_DIR / "yolov8n.pt") if (BASE_DIR / "yolov8n.pt").exists() else str(BASE_DIR / "yolo11n.pt")


# ============================================================
# VEHICLE CLASSES
# ============================================================

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


# ============================================================
# ACCIDENT DETECTION SETTINGS
# ============================================================

DISTANCE_THRESHOLD = 25

CONFIRM_FRAMES = 10

IOU_THRESHOLD = 0.05

EVENT_COOLDOWN_FRAMES = 50


# ============================================================
# GET CENTER OF BOUNDING BOX
# ============================================================

def get_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


# ============================================================
# DISTANCE BETWEEN TWO POINTS
# ============================================================

def distance(point1, point2):

    return math.sqrt(
        (point1[0] - point2[0]) ** 2
        +
        (point1[1] - point2[1]) ** 2
    )


# ============================================================
# CALCULATE IoU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0])
        *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)


# ============================================================
# MAIN ACCIDENT DETECTION
# ============================================================

def run_accident_detection():

    print("================================")
    print("CITYEYE ACCIDENT DETECTION")
    print("================================")

    print("\nLoading YOLO model...")

    model = YOLO(MODEL_PATH)

    print("Model loaded.")

    print(
        f"Input video: {VIDEO_PATH}"
    )

    print(
        "\nStarting accident detection..."
    )

    print(
        "Using stricter accident conditions..."
    )

    # --------------------------------------------------------
    # YOLO TRACKING
    # --------------------------------------------------------

    results = model.track(

        source=str(VIDEO_PATH),

        tracker="bytetrack.yaml",

        persist=True,

        stream=True,

        conf=0.35,

        verbose=False
    )

    frame_number = 0

    fps = 25.0

    close_frames = {}

    last_event_frame = {}

    events = []

    # ========================================================
    # PROCESS VIDEO FRAMES
    # ========================================================

    for result in results:

        frame_number += 1

        if result.boxes is None:
            continue

        if result.boxes.id is None:
            continue

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        classes = (
            result.boxes.cls
            .cpu()
            .numpy()
        )

        track_ids = (
            result.boxes.id
            .cpu()
            .numpy()
        )

        vehicles = []

        # ----------------------------------------------------
        # COLLECT VEHICLES
        # ----------------------------------------------------

        for box, cls, track_id in zip(
            boxes,
            classes,
            track_ids
        ):

            class_id = int(cls)

            if class_id not in VEHICLE_CLASSES:
                continue

            vehicles.append({

                "box": box,

                "class_id": class_id,

                "track_id": int(track_id),

                "center": get_center(box)

            })

        # ----------------------------------------------------
        # COMPARE VEHICLE PAIRS
        # ----------------------------------------------------

        for i in range(
            len(vehicles)
        ):

            for j in range(
                i + 1,
                len(vehicles)
            ):

                vehicle1 = vehicles[i]

                vehicle2 = vehicles[j]

                id1 = vehicle1["track_id"]

                id2 = vehicle2["track_id"]

                pair_key = tuple(
                    sorted(
                        [id1, id2]
                    )
                )

                # ------------------------------------------------
                # DISTANCE
                # ------------------------------------------------

                d = distance(

                    vehicle1["center"],

                    vehicle2["center"]
                )

                # ------------------------------------------------
                # IoU
                # ------------------------------------------------

                iou = calculate_iou(

                    vehicle1["box"],

                    vehicle2["box"]
                )

                # ------------------------------------------------
                # ACCIDENT CONDITION
                # ------------------------------------------------

                collision_like = (

                    d < DISTANCE_THRESHOLD

                    and

                    iou >= IOU_THRESHOLD

                )

                # ------------------------------------------------
                # VEHICLES ARE CLOSE
                # ------------------------------------------------

                if collision_like:

                    previous_count = (
                        close_frames.get(
                            pair_key,
                            0
                        )
                    )

                    close_frames[pair_key] = (
                        previous_count + 1
                    )

                    # --------------------------------------------
                    # CONFIRM EVENT
                    # --------------------------------------------

                    if (
                        close_frames[pair_key]
                        >= CONFIRM_FRAMES
                    ):

                        last_event = (
                            last_event_frame.get(
                                pair_key,
                                -9999
                            )
                        )

                        # ----------------------------------------
                        # EVENT COOLDOWN
                        # ----------------------------------------

                        if (
                            frame_number
                            - last_event
                            >= EVENT_COOLDOWN_FRAMES
                        ):

                            event = {

                                "type":
                                "POSSIBLE_ACCIDENT",

                                "frame":
                                int(frame_number),

                                "timestamp":
                                float(
                                    round(
                                        frame_number
                                        / fps,
                                        2
                                    )
                                ),

                                "vehicle_1": {

                                    "type":
                                    VEHICLE_CLASSES[
                                        vehicle1[
                                            "class_id"
                                        ]
                                    ],

                                    "track_id":
                                    int(id1)

                                },

                                "vehicle_2": {

                                    "type":
                                    VEHICLE_CLASSES[
                                        vehicle2[
                                            "class_id"
                                        ]
                                    ],

                                    "track_id":
                                    int(id2)

                                },

                                "distance":
                                float(
                                    round(
                                        float(d),
                                        2
                                    )
                                ),

                                "iou":
                                float(
                                    round(
                                        float(iou),
                                        3
                                    )
                                )

                            }

                            events.append(
                                event
                            )

                            last_event_frame[
                                pair_key
                            ] = frame_number

                            # ------------------------------------
                            # PRINT EVENT
                            # ------------------------------------

                            print(
                                "\n🔴 POSSIBLE ACCIDENT"
                            )

                            print(
                                f"Frame: "
                                f"{frame_number}"
                            )

                            print(
                                f"Time: "
                                f"{round(frame_number / fps, 2)} sec"
                            )

                            print(
                                f"Vehicle 1: "
                                f"{VEHICLE_CLASSES[vehicle1['class_id']]} "
                                f"(ID {id1})"
                            )

                            print(
                                f"Vehicle 2: "
                                f"{VEHICLE_CLASSES[vehicle2['class_id']]} "
                                f"(ID {id2})"
                            )

                            print(
                                f"Distance: "
                                f"{round(float(d), 2)}"
                            )

                            print(
                                f"IoU: "
                                f"{round(float(iou), 3)}"
                            )

                # ------------------------------------------------
                # VEHICLES MOVED APART
                # ------------------------------------------------

                else:

                    close_frames[pair_key] = 0

    # ========================================================
    # SAVE JSON RESULT
    # ========================================================

    output = {

        "video":
        str(VIDEO_PATH),

        "event":
        "ACCIDENT",

        "detected":
        bool(len(events) > 0),

        "count":
        int(len(events)),

        "events":
        events

    }

    output_file = (
        DATA_DIR /
        "accident_events.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            output,

            file,

            indent=4

        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n================================")

    print(
        "ACCIDENT DETECTION COMPLETE"
    )

    print("================================")

    print(
        f"Possible accident events: "
        f"{len(events)}"
    )

    print(
        f"JSON saved to: "
        f"{output_file}"
    )

    print("================================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_accident_detection()