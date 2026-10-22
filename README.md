# Blind Assistance Object Detection

Object detection with spoken audio feedback, built with YOLOv8, OpenCV, and pyttsx3.
Supports **three input modes**: live webcam, an uploaded photo, or an uploaded video file.

## ⚠️ Folder structure — read this first

All 6 `.py` files must sit in the **same folder**, side by side. Do **not** put any of
them in a subfolder.

```
project/
├── main.py
├── config.py
├── detector.py
├── spatial.py
├── announcer.py
├── tts_engine.py
└── requirements.txt
```

## Setup (Windows / VS Code terminal)

```powershell
cd "D:\Anju\Docs\3rd year\ICV\project"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

First run downloads `yolov8n.pt` (~6 MB) automatically — needs internet once, then it's cached.

## How to run each mode

### 1. Webcam (live, real-time)
```powershell
python main.py --source webcam
```

### 2. Uploaded photo (single image)
```powershell
python main.py --source image --path "C:\path\to\your\photo.jpg"
```
Runs one detection pass, speaks every object it finds, shows the image with
boxes drawn until you press any key.

### 3. Uploaded video file
```powershell
python main.py --source video --path "C:\path\to\your\video.mp4"
```
Plays through the video, detecting and speaking as it goes. Press `q` to stop early.

### Setting the default mode without typing it every time
Open `config.py` and edit:
```python
INPUT_SOURCE = "image"              # "webcam", "image", or "video"
INPUT_PATH = "my_photo.jpg"         # used for "image" or "video" modes
```
Then just run `python main.py` with no arguments — command-line flags above always
override whatever is in `config.py`, so you can use either approach.

## Quit / exit controls

- **Webcam / video mode**: press `q` in the video window, or Ctrl+C in the terminal.
- **Image mode**: press any key in the image window once you're done viewing it.

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: picks webcam/image/video mode, runs the detection loop, orchestrates everything |
| `config.py` | All tunable settings (input source, camera, model, thresholds, TTS, cooldowns) |
| `detector.py` | YOLOv8 wrapper, filters to relevant object classes |
| `spatial.py` | Bounding box → direction + proximity |
| `announcer.py` | Chooses what to say, prevents repetition/spam |
| `tts_engine.py` | Threaded, non-blocking text-to-speech |

## Tuning tips

- **Too many/few announcements (webcam/video)**: adjust `ANNOUNCE_COOLDOWN_SECONDS` and `MAX_ANNOUNCEMENTS_PER_CYCLE` in `config.py`.
- **Too many/few announcements (single image)**: adjust `MAX_ANNOUNCEMENTS_FOR_IMAGE`.
- **Distance feels wrong**: adjust `NEAR_AREA_RATIO` / `MEDIUM_AREA_RATIO`.
- **Low FPS on CPU**: increase `PROCESS_EVERY_N_FRAMES`.
- **Different objects matter to you**: edit `TARGET_CLASSES` in `config.py` (must match COCO class names), or set it to `None` to announce all 80 COCO classes.
- **Want the video to loop**: set `LOOP_VIDEO = True` in `config.py`.
- **No monitor available**: set `SHOW_VIDEO_WINDOW = False` — audio feedback still works in all three modes.

## Supported file types

- **Images**: anything OpenCV can read — `.jpg`, `.jpeg`, `.png`, `.bmp`, etc.
- **Videos**: anything OpenCV can read — `.mp4`, `.avi`, `.mov`, etc. (depends on your system's codecs; `.mp4` is the safest bet).

## Extending this project

- Swap `pyttsx3` for a cloud TTS API (e.g. Google/Azure) for more natural voices, at the cost of needing internet.
- Add a monocular depth model or a stereo/ToF/LiDAR sensor for true distance-in-meters instead of the bbox-area proxy used here.
- Add haptic feedback (vibration motors) as a secondary channel alongside audio.
- Package with a wearable camera + Bluetooth earpiece for a real assistive device.
