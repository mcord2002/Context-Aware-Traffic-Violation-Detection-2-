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
Copy-Item .\runs\detect\traffic_train\weights\best.pt .\models\best.pt
```

Ubuntu/Linux:
```bash
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

## Outputs

outputs/detected_video.mp4

## Research limitations

This is a prototype. Events are candidates and require
verification. Driver/rider association, signal-to-lane
association, and camera-specific road geometry must
be validated before reporting final violation accuracy.