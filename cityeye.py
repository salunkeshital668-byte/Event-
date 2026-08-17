from pathlib import Path
import json
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
EVENT_CLIPS_DIR = OUTPUT_DIR / "event_clips"

def event_summary():
    file_path = DATA_DIR / "final_events.json"

    if not file_path.exists():
        print("\nfinal_events.json not found.")
        return

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    events = data.get("events", {})

    no_helmet = len(events.get("no_helmet", []))
    triple = len(events.get("triple_riding", []))
    wrong_way = len(events.get("wrong_way", []))
    accident = len(events.get("accident", []))

    total = no_helmet + triple + wrong_way + accident

    print("\n================================")
    print("       CITYEYE EVENT SUMMARY")
    print("================================")
    print(f"No Helmet      : {no_helmet}")
    print(f"Triple Riding  : {triple}")
    print(f"Wrong Way      : {wrong_way}")
    print(f"Accident       : {accident}")
    print("--------------------------------")
    print(f"Total Events   : {total}")
    print("================================")

def open_final_video():
    video = OUTPUT_DIR / "cityeye_final_output.mp4"

    if not video.exists():
        print("\nFinal video not found.")
        return

    print("\nOpening final video...")

    if sys.platform == "win32":
        subprocess.Popen(["cmd", "/c", "start", "", str(video)])

def event_clips():
    if not EVENT_CLIPS_DIR.exists():
        print("\nEvent clips folder not found.")
        return

    clips = sorted(EVENT_CLIPS_DIR.glob("*.mp4"))

    if not clips:
        print("\nNo event clips found.")
        return

    print("\n================================")
    print("        EVENT CLIPS")
    print("================================")

    for i, clip in enumerate(clips, 1):
        print(f"{i}. {clip.name}")

    choice = input("\nEnter clip number: ")

    try:
        number = int(choice)

        if number < 1 or number > len(clips):
            print("Invalid number.")
            return

        selected = clips[number - 1]

        print(f"\nOpening: {selected.name}")

        if sys.platform == "win32":
            subprocess.Popen(
                ["cmd", "/c", "start", "", str(selected)]
            )

    except ValueError:
        print("Please enter a number.")

def json_results():
    files = sorted(DATA_DIR.glob("*.json"))

    if not files:
        print("\nNo JSON files found.")
        return

    print("\n================================")
    print("        JSON RESULTS")
    print("================================")

    for i, file in enumerate(files, 1):
        print(f"{i}. {file.name}")

    choice = input("\nEnter JSON number: ")

    try:
        number = int(choice)

        if number < 1 or number > len(files):
            print("Invalid number.")
            return

        selected = files[number - 1]

        print(f"\nOpening: {selected.name}")

        if sys.platform == "win32":
            subprocess.Popen(["notepad.exe", str(selected)])

    except ValueError:
        print("Please enter a number.")

def live_camera():
    import config
    from detector import YOLODetector
    from tracker import MultiObjectTracker
    from event_detector import EventDetector

    url = getattr(config, "IP_WEBCAM_URL", "http://192.168.0.107:8080/video")
    print(f"\nConnecting to Phone IP Webcam at {url} ...")
    det = YOLODetector()
    trk = MultiObjectTracker()
    ev = EventDetector(det, trk, camera_id="cam_ip_webcam", video_name="Phone_Camera")
    ev.process_live_stream(stream_source=url, show=True)


def main():

    while True:

        print("\n")
        print("================================")
        print("           CITYEYE AI")
        print("     TRAFFIC MONITORING SYSTEM")
        print("================================")

        print("1. Event Summary")
        print("2. Open Final Video")
        print("3. Event Clips")
        print("4. JSON Results")
        print("5. Live Phone IP Webcam")
        print("6. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            event_summary()

        elif choice == "2":
            open_final_video()

        elif choice == "3":
            event_clips()

        elif choice == "4":
            json_results()

        elif choice == "5":
            live_camera()

        elif choice == "6":
            print("\nExiting CityEye...")
            break

        else:
            print("\nInvalid option.")


if __name__ == "__main__":
    main()
