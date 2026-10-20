# TrafficViolationAI

YOLO11 + ByteTrack traffic violation detection prototype.

## Violations
1. Helmet violation
2. Seatbelt violation
3. Mobile phone usage
4. Red-light violation
5. Wrong-lane violation

## Installation

From the repository root, enter the project folder:

```bash
cd TrafficViolationAI
```

Create and activate a virtual environment.

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Ubuntu/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

Place images and matching YOLO annotation labels in:

```text
dataset/images/train/
dataset/images/val/
dataset/labels/train/
dataset/labels/val/
```

Check that the class IDs and names in `data.yaml` match your annotations.

## Train the model

```bash
python train.py
```

Training downloads the pretrained `yolo11n.pt` weights if needed. When
training finishes, the weights are saved here:

```text
runs/detect/traffic_train/weights/best.pt
runs/detect/traffic_train/weights/last.pt
```

Copy `best.pt` into the `models/` folder.

Windows (PowerShell):
```powershell
New-Item -ItemType Directory -Force -Path .\models | Out-Null
Copy-Item .\runs\detect\traffic_train\weights\best.pt .\models\best.pt
```

Ubuntu/Linux:
```bash
mkdir -p models
cp runs/detect/traffic_train/weights/best.pt models/best.pt
```

## Run detection

Place the input video in the `TrafficViolationAI/` project folder as:

```text
traffic.mp4
```

Run video detection from the project folder:

```bash
python detect_video.py
```

The annotated video is saved to `outputs/detected_video.mp4`.

Use command-line options to select different files or run without a preview
window (for example, on a server):

```bash
python detect_video.py --model models/best.pt --video traffic.mp4 --output outputs/review.mp4 --confidence 0.5 --no-display
```

Run `python detect_video.py --help` to see all options. Confidence must be
between 0 and 1. The output must be an MP4 file distinct from the input video
and model. Missing inputs, unreadable videos, and output writer failures are
reported as errors. Videos with missing or invalid FPS metadata use 25 FPS.

## Run violation analysis

Edit `TrafficViolationAI/config/road_config.json` to match the camera view,
including the stop line and traffic signal region. Wrong-lane detection is
disabled by default; configure its polygon before enabling it.

With the trained model and input video in place, run from the
`TrafficViolationAI/` folder:

```bash
python main.py
```

This runs the violation modules and saves `outputs/annotated_video.mp4`,
`outputs/violation_events.csv`, and evidence images in `outputs/evidence/`.
Press **Q** in the video window to stop either detection script.

## Outputs

outputs/detected_video.mp4

## Development checks

From the repository root, run the video detection tests:

```bash
python -B -m unittest discover -s tests -v
```

These tests use mock video and model objects, so they do not require trained
weights, OpenCV, or Ultralytics. They cover argument validation, headless
processing, preview exit, invalid video metadata, and resource cleanup.

## Research limitations

This is a prototype. Events are candidates and require
verification. Driver/rider association, signal-to-lane
association, and camera-specific road geometry must
be validated before reporting final violation accuracy.
