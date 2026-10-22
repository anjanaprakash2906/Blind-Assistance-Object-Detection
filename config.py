"""
Configuration file for Blind Assistance Object Detection System.
Adjust these values to tune performance and behavior without touching main logic.
"""

# ----------------------------- Input Source Settings -----------------------------
# Choose what the system should read frames from:
#   "webcam" -> live camera feed (real-time, continuous)
#   "image"  -> a single uploaded photo (one detection pass)
#   "video"  -> an uploaded video file (plays through and detects on every processed frame)
INPUT_SOURCE = "webcam"

# Used only when INPUT_SOURCE is "image" or "video". Give the full path to the file.
# Examples:
#   INPUT_PATH = "sample_photo.jpg"
#   INPUT_PATH = r"D:\Anju\Docs\3rd year\ICV\project\test_video.mp4"
INPUT_PATH = ""

# If True and INPUT_SOURCE == "video", the video restarts automatically when it ends.
# If False, the system announces "video finished" and stops.
LOOP_VIDEO = False

# ----------------------------- Camera Settings (webcam mode only) -----------------------------
CAMERA_INDEX = 0          # 0 = default webcam. Change if using an external/USB camera.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ----------------------------- YOLO Model Settings -----------------------------
MODEL_PATH = "yolov8n.pt"   # Nano model: fastest, good for real-time use on CPU.
CONFIDENCE_THRESHOLD = 0.45  # Minimum confidence to accept a detection.
IOU_THRESHOLD = 0.45         # Non-max suppression IoU threshold.
PROCESS_EVERY_N_FRAMES = 2   # Skip frames (webcam/video) to keep FPS high on slower machines.

# Only announce these classes (COCO class names). Set to None to announce everything YOLO detects.
TARGET_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "fire hydrant", "stop sign", "bench",
    "backpack", "umbrella", "handbag", "suitcase",
    "bottle", "cup", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop",
    "microwave", "oven", "sink", "refrigerator", "door",
    "stairs", "dog", "cat"
]

# ----------------------------- Distance Estimation -----------------------------
# Bounding box area (as a fraction of the frame area) thresholds used to classify
# how "close" an object is. Tune these based on camera FOV / image composition.
NEAR_AREA_RATIO = 0.18      # Object occupies > 18% of frame -> "very close"
MEDIUM_AREA_RATIO = 0.06    # Object occupies > 6% of frame -> "nearby"
# Anything below MEDIUM_AREA_RATIO is considered "far"

# ----------------------------- Audio / TTS Settings -----------------------------
TTS_RATE = 175              # Words per minute
TTS_VOLUME = 1.0            # 0.0 to 1.0
ANNOUNCE_COOLDOWN_SECONDS = 4.0   # Minimum time before the same object+direction is announced again (webcam/video)
MAX_ANNOUNCEMENTS_PER_CYCLE = 3   # Cap per detection cycle, so the user isn't overwhelmed (webcam/video)
MAX_ANNOUNCEMENTS_FOR_IMAGE = 8   # Separate, usually higher cap for a single uploaded photo (one-shot, no cooldown)
PRIORITIZE_CLOSEST = True         # If True, closest/largest objects are announced first

# ----------------------------- Display Settings -----------------------------
SHOW_VIDEO_WINDOW = True    # Set False for headless mode (e.g., Raspberry Pi without a monitor)
DRAW_BOXES = True
WINDOW_NAME = "Blind Assistance - Object Detection"
