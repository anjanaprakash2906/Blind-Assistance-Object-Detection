"""
Blind Assistance Object Detection System
==========================================
Real-time (and offline) object detection with spoken audio feedback for
visually impaired users.

Three input modes are supported:
    1. webcam -> live camera feed, continuous detection
    2. image  -> a single uploaded photo, one detection pass
    3. video  -> an uploaded video file, detection on every processed frame

Pipeline:
    Frame source --> YOLOv8 detection --> spatial reasoning (direction/proximity)
    --> cooldown-filtered announcement selection --> threaded text-to-speech

SETUP (run these in your project folder, in order):
    python -m venv venv
    venv\\Scripts\\activate          (Windows)   OR   source venv/bin/activate   (Mac/Linux)
    pip install -r requirements.txt

RUN (uses the mode set in config.py):
    python main.py

RUN (override config.py from the command line):
    python main.py --source webcam
    python main.py --source image --path path\\to\\photo.jpg
    python main.py --source video --path path\\to\\clip.mp4

QUIT:
    Press 'q' in the video window, or Ctrl+C in the terminal (headless mode).
    For image mode, press any key in the window to continue/finish.

IMPORTANT: every .py file in this project (main.py, config.py, detector.py,
spatial.py, announcer.py, tts_engine.py) must sit in the SAME folder,
with no subfolders.
"""

import argparse
import os
import sys
import time

import cv2

import config
from detector import ObjectDetector
from spatial import get_spatial_info
from announcer import AnnouncementManager
from tts_engine import TTSEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Blind Assistance Object Detection")
    parser.add_argument(
        "--source", choices=["webcam", "image", "video"], default=None,
        help="Input source. Overrides INPUT_SOURCE in config.py if given."
    )
    parser.add_argument(
        "--path", default=None,
        help="Path to the image or video file. Overrides INPUT_PATH in config.py if given."
    )
    return parser.parse_args()


def draw_detection(frame, det, spatial_info):
    """Draws a bounding box and label on the frame for sighted debugging/demo purposes."""
    color = (0, 0, 255) if spatial_info.proximity == "very close" else (0, 255, 0)
    p1 = (int(det.x1), int(det.y1))
    p2 = (int(det.x2), int(det.y2))
    cv2.rectangle(frame, p1, p2, color, 2)

    label_text = f"{det.label} {det.confidence:.2f} | {spatial_info.direction}, {spatial_info.proximity}"
    text_origin = (p1[0], max(p1[1] - 10, 15))
    cv2.putText(frame, label_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def process_frame(frame, detector, announcer, tts, speak_results=True):
    """
    Runs detection + spatial reasoning on a single frame, draws boxes (if enabled),
    and queues up any phrases that should be spoken. Shared by webcam/video/image modes.
    """
    frame_height, frame_width = frame.shape[:2]
    detections = detector.detect(frame)

    announce_candidates = []
    for det in detections:
        spatial_info = get_spatial_info(
            det.x1, det.y1, det.x2, det.y2,
            frame_width, frame_height,
            config.NEAR_AREA_RATIO, config.MEDIUM_AREA_RATIO,
        )

        if config.SHOW_VIDEO_WINDOW and config.DRAW_BOXES:
            draw_detection(frame, det, spatial_info)

        announce_candidates.append({
            "label": det.label,
            "direction": spatial_info.direction,
            "proximity": spatial_info.proximity,
            "area_ratio": spatial_info.area_ratio,
        })

    if speak_results:
        phrases = announcer.build_phrases(announce_candidates)
        for phrase in phrases:
            print(f"[QUEUED FOR SPEECH] {phrase}", flush=True)
            tts.speak(phrase)

    return len(announce_candidates)


def run_webcam(detector, tts):
    announcer = AnnouncementManager(
        cooldown_seconds=config.ANNOUNCE_COOLDOWN_SECONDS,
        max_per_cycle=config.MAX_ANNOUNCEMENTS_PER_CYCLE,
        prioritize_closest=config.PRIORITIZE_CLOSEST,
    )

    print("[INFO] Opening webcam...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check CAMERA_INDEX in config.py "
              "(try 0, 1, or 2), and make sure no other app is using it.")
        sys.exit(1)

    tts.speak("Blind assistance system started. Webcam mode.")
    tts.wait_until_idle()

    frame_count = 0
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARNING] Failed to read frame from webcam. Retrying...")
                continue

            frame_count += 1
            run_detection = (frame_count % config.PROCESS_EVERY_N_FRAMES == 0)
            if run_detection:
                process_frame(frame, detector, announcer, tts, speak_results=True)

            now = time.time()
            elapsed = now - prev_time
            prev_time = now
            if elapsed > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / elapsed)

            if config.SHOW_VIDEO_WINDOW:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow(config.WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                window_closed = cv2.getWindowProperty(
                    config.WINDOW_NAME, cv2.WND_PROP_VISIBLE
                ) < 1
                if key == ord('q') or window_closed:
                    print("[INFO] Quit key pressed. Shutting down...")
                    break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Shutting down...")
    finally:
        cap.release()
        if config.SHOW_VIDEO_WINDOW:
            cv2.destroyAllWindows()


def run_video(detector, tts, video_path):
    if not os.path.isfile(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        print("[HINT] Set INPUT_PATH in config.py, or run: python main.py --source video --path your_video.mp4")
        sys.exit(1)

    announcer = AnnouncementManager(
        cooldown_seconds=config.ANNOUNCE_COOLDOWN_SECONDS,
        max_per_cycle=config.MAX_ANNOUNCEMENTS_PER_CYCLE,
        prioritize_closest=config.PRIORITIZE_CLOSEST,
    )

    print(f"[INFO] Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        sys.exit(1)

    tts.speak("Blind assistance system started. Video mode.")
    tts.wait_until_idle()

    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # End of video reached.
                if config.LOOP_VIDEO:
                    print("[INFO] Video finished, restarting (LOOP_VIDEO=True)...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[INFO] Video finished.")
                    tts.speak("Video finished.")
                    break

            frame_count += 1
            run_detection = (frame_count % config.PROCESS_EVERY_N_FRAMES == 0)
            if run_detection:
                process_frame(frame, detector, announcer, tts, speak_results=True)

            if config.SHOW_VIDEO_WINDOW:
                cv2.imshow(config.WINDOW_NAME, frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    print("[INFO] Quit key pressed. Shutting down...")
                    break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Shutting down...")
    finally:
        cap.release()
        if config.SHOW_VIDEO_WINDOW:
            cv2.destroyAllWindows()


def run_image(detector, tts, image_path):
    if not os.path.isfile(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        print("[HINT] Set INPUT_PATH in config.py, or run: python main.py --source image --path your_photo.jpg")
        sys.exit(1)

    # A single still image has no "repeat frame" problem, so no cooldown is
    # needed — every detected object should be announced once. A separate,
    # usually higher cap is used (MAX_ANNOUNCEMENTS_FOR_IMAGE).
    announcer = AnnouncementManager(
        cooldown_seconds=0.0,
        max_per_cycle=config.MAX_ANNOUNCEMENTS_FOR_IMAGE,
        prioritize_closest=config.PRIORITIZE_CLOSEST,
    )

    print(f"[INFO] Loading image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not read image file (unsupported format or corrupt file): {image_path}")
        sys.exit(1)

    tts.speak("Analyzing image.")
    tts.wait_until_idle()
    count = process_frame(frame, detector, announcer, tts, speak_results=True)

    if count == 0:
        print("[INFO] No recognizable objects found in the image.")
        tts.speak("No recognizable objects found.")

    if config.SHOW_VIDEO_WINDOW:
        cv2.imshow(config.WINDOW_NAME, frame)
        print("[INFO] Press any key in the image window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    tts.wait_until_idle()


def main():
    args = parse_args()
    source = args.source or config.INPUT_SOURCE
    path = args.path or config.INPUT_PATH

    print("[INFO] Initializing YOLOv8 model... (first run downloads yolov8n.pt automatically)")
    detector = ObjectDetector(
        model_path=config.MODEL_PATH,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        iou_threshold=config.IOU_THRESHOLD,
        target_classes=config.TARGET_CLASSES,
    )

    print("[INFO] Starting text-to-speech engine...")
    tts = TTSEngine(rate=config.TTS_RATE, volume=config.TTS_VOLUME)

    try:
        if source == "webcam":
            run_webcam(detector, tts)
        elif source == "video":
            run_video(detector, tts, path)
        elif source == "image":
            run_image(detector, tts, path)
        else:
            print(f"[ERROR] Unknown INPUT_SOURCE '{source}'. Use 'webcam', 'image', or 'video'.")
            sys.exit(1)
    finally:
        tts.stop()
        print("[INFO] Clean shutdown complete.")


if __name__ == "__main__":
    main()
