# Context-Aware-Traffic-Violation-Detection-2-# TrafficViolationAI

YOLO11 + ByteTrack traffic violation detection prototype.

## Violations
1. Helmet violation
2. Seatbelt violation
3. Mobile phone usage
4. Red-light violation
5. Wrong-lane violation

## Installation

Create a virtual environment:

Windows:
python -m venv .venv
.venv\Scripts\activate

Ubuntu/Linux:
python -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

## Dataset

Place images and YOLO annotation labels in:

dataset/images/train/
dataset/images/val/
dataset/labels/train/
dataset/labels/val/

Update data.yaml to match your actual class IDs
and class names.

## Train the model

python train.py

Copy the trained model:

models/best.pt

Source:
runs/detect/traffic_train/weights/best.pt

## Run detection

Place your video in the project root as:

traffic.mp4

Configure road_config.json for the actual camera.

Run:

python main.py

## Outputs

outputs/annotated_video.mp4
outputs/violation_events.csv
outputs/evidence/

## Research limitations

This is a prototype. Events are candidates and require
verification. Driver/rider association, signal-to-lane
association, and camera-specific road geometry must
be validated before reporting final violation accuracy.