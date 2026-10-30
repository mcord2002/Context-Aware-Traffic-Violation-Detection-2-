# Component 04: Multi-Frame Traffic Risk Assessment and Four-State Vehicle-Pair Classification

A local research prototype for assessing **traffic-conflict risk between tracked vehicle pairs** in road/CCTV videos. The system uses **YOLOv8n** for vehicle detection, **ByteTrack** for tracking, and a **Random Forest classifier** trained on manually reviewed, five-second interaction windows. A **Streamlit** interface supports video preparation, annotation, training, evaluation, and an annotated video demonstration.

> **Research prototype:** The four outputs are descriptive risk states, **not verified collision probabilities**. The application does not establish accident causation or fault, measure physical distance/speed or time-to-collision (TTC), or demonstrate validated advance warning of an accident. Its results should not be used as an automated enforcement or safety decision.

## Project links

- **GitHub repository — overall research project:** [View GitHub repository](https://github.com/mcord2002/Context-Aware-Traffic-Violation-Detection-2-.git)
- **Figma — Component 04 UI/UX design:** [View Figma design](https://www.figma.com/design/uTI24sHVWKcdG9RlOqljNM/Component-04---Traffic-Risk-Assessment---UI-UX-Design?t=NWwAbjZ4NVJUuENU-1)

## Features

- **Vehicle detection — YOLOv8n:** Detects COCO vehicle classes **car, motorcycle, bus, and truck**. The supplied `yolov8n.pt` is a general-purpose detection checkpoint, not a specially validated Sri Lankan traffic detector.
- **Vehicle tracking — ByteTrack:** Tracks detected vehicles across sampled frames using temporary track IDs. Occlusion and tracking errors can fragment or switch identities.
- **Multi-frame interaction analysis:** Examines pairs of vehicles over **5-second windows** with a **1-second stride**, using approximately **5 processed frames per second**.
- **Movement-feature extraction:** Calculates **15 image-space features** describing vehicle separation, closing motion, relative movement, direction alignment, speed changes, box overlap, and observation coverage.
- **Manual four-state annotation:** A reviewer checks the relevant vehicle IDs and visible evidence before assigning a risk state; risk labels are **not automatically generated from geometric thresholds**.
- **Risk classification — Random Forest:** Trains a single supervised model to classify eligible vehicle-pair windows into four states.
- **Recording-level evaluation:** Allows held-out source recordings to be selected before training to reduce leakage from overlapping windows and related clips.
- **Video demonstration:** Displays risk timelines, pair-level predictions, tracked vehicle IDs, and a downloadable annotated MP4.
- **Local or Colab workflow:** Runs as a local Streamlit application; an optional Google Colab notebook supports extraction/training workflows.

## How the pipeline works

1. **Upload road footage:** In Streamlit, upload one or more MP4, AVI, MOV, or MKV videos (MP4 is recommended).
2. **Detect and track vehicles:** YOLOv8n identifies supported vehicle types, and ByteTrack assigns per-video tracking IDs.
3. **Extract trajectories:** For each observation, the pipeline stores a timestamp, class, track ID, normalized bounding box, and normalized bottom-centre point.
4. **Build vehicle-pair windows:** It analyzes eligible pairs in trailing 5-second windows, advanced every second. A pair needs at least **80% shared observation coverage** and sufficient temporal continuity.
5. **Review and label windows:** A human reviewer inspects the video and contact sheet, chooses a risk state, records supporting evidence and reviewer initials, and saves the annotation.
6. **Train and evaluate:** The system joins annotations to extracted features, fits one Random Forest model, and evaluates it on preselected, independent source-recording groups.
7. **Demonstrate predictions:** The trained model predicts eligible windows in an uploaded video. The dashboard shows pair timelines and can render an annotated video.

**YOLOv8n/ByteTrack provide trajectories; the Random Forest is the four-state risk classifier.** This implementation does **not** use ResNet18 accident-image classification, YOLO26n, or a threshold-only risk-labeling system.

## Risk classification states

| State | Interpretation |
|---|---|
| `Normal` | Stable interaction without clear evidence of a developing traffic conflict. |
| `Increasing Risk` | Risk-related cues are increasing, such as closing movement and converging paths. |
| `Developing Conflict` | A developing interaction with additional evidence such as braking, steering, or avoidance. |
| `Critical Conflict / Collision` | A severe interaction or potentially visible contact. This combined label does **not** independently prove that a collision happened. |

Labels describe the **state at the end of each reviewed window**, based only on evidence within that window. Uncertain identity switches, heavily occluded pairs, and ambiguous interactions should be skipped rather than given assumed labels.

## Model files and project structure

```text
component04/
├── app.py                         # Streamlit: annotation, training, evaluation, demo
├── risk_core.py                   # 15 features, Random Forest, prediction and metrics
├── video_pipeline.py              # YOLO/ByteTrack extraction and video rendering
├── batch.py                       # Batch upload, deduplication and cached extraction
├── extract.py                     # Command-line batch feature extraction
├── train.py                       # Command-line model training
├── run_component04.bat            # Windows Python 3.11 launcher
├── Component04_Training_Colab.ipynb
├── requirements.txt
├── yolov8n.pt                     # YOLOv8n detection weights
├── data/
│   ├── annotations.csv            # Manually reviewed risk labels
│   ├── videos/                    # Saved uploads (created when used)
│   └── runs/
│       └── <video_hash>/
│           ├── tracks.csv
│           ├── features.csv
│           ├── metadata.json
│           ├── predictions.csv   # After demo prediction
│           └── annotated.mp4     # After optional rendering
├── artifacts/
│   ├── risk_model.pkl             # Trained Random Forest pipeline
│   ├── evaluation.json
│   ├── labelled_features.csv
│   ├── split_manifest.csv
│   ├── test_predictions.csv
│   └── feature_importance.csv
└── tests/
    └── test_pipeline.py
```

The supplied project archive contains a **pilot** `artifacts/risk_model.pkl` and corresponding evaluation files. These are **not** evidence of production-ready performance; see the evaluation section. For reproducible research, retrain using a sufficiently large, independently reviewed dataset. Only open trusted `.pkl` model files, and use compatible scikit-learn versions.

## Requirements

- **Python 3.11** (recommended by the included Windows launcher).
- Windows 10/11, or another operating system compatible with the dependencies.
- Dependencies listed in `requirements.txt`: `numpy`, `pandas`, `scikit-learn`, `streamlit`, `opencv-python-headless`, `ultralytics`, `lap`, and `matplotlib`, plus their resolved dependencies.
- A supported video file with at least two sufficiently tracked vehicles for pair analysis.
- An internet connection for the initial dependency installation and for downloading detector weights if they are not present locally.
- **Optional:** FFmpeg for H.264 encoding of the annotated video for browser playback; an NVIDIA GPU/Colab GPU can help with extraction, but a GPU is not required for the local prototype.

## Install and run the application

### Option 1 — Windows launcher

1. Extract or clone the entire project folder.
2. Install **64-bit Python 3.11** and enable the Python Launcher (`py`).
3. Double-click **`run_component04.bat`** from the project folder.
4. The launcher creates `.venv`, installs requirements on first use, and starts Streamlit at `http://localhost:8501`.

### Option 2 — Manual setup

From the project root:

```bash
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Then install and launch:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL displayed in the terminal. Select a page from the Streamlit sidebar. The default detector-weights field is `yolov8n.pt`; it can be changed to another compatible YOLO detector file.

## Using the Streamlit application

### 1. Prepare & annotate

1. Open **`1 · Prepare & annotate`** and upload one or more traffic videos.
2. Click **`Extract ALL uploaded videos`**. Files are processed sequentially; duplicate files and previously extracted runs may be reused.
3. Select a video and review each eligible **5-second vehicle-pair window**.
4. Enter an **original source-recording group** (clips from the same original recording must share the same group).
5. Inspect the contact sheet and source video. Choose one of the four states, enter an evidence note and reviewer initials, confirm the tracking check, and save.
6. Export `annotations.csv` and `labelled_features.csv` for backup or Colab training.

### 2. Train & evaluate

1. Open **`2 · Train & evaluate`** after completing manual annotations.
2. Examine the class counts by source-recording group.
3. **Preselect held-out recording groups** before reviewing model results.
4. Click **`Train Random Forest & evaluate`**.
5. Inspect the metrics, classification report, confusion matrix, feature importance, and downloadable artifacts.

The code requires **at least two training source recordings and one held-out recording**, and **all four states must appear in both partitions**. These are minimum software checks, not a claim that the resulting dataset is large enough for scientific conclusions.

### 3. Video demo

1. Open **`3 · Video demo`** with a trusted, compatible `artifacts/risk_model.pkl` available.
2. Upload a road video and click **`Detect, track & extract pair windows`**.
3. Select a tracked pair to view predicted states, the risk-level timeline, separation trend, and model vote share.
4. Download `predictions.csv`. Optionally click **`Render annotated video`** and download the MP4.

Start with a short clip for a CPU demonstration. The sidebar extraction limit defaults to **60 seconds per video** (adjustable from 10 to 300 seconds). Rendering uses the most recent completed 5-second window; if no eligible pair exists, the system does **not** automatically assign `Normal`.

## Features used by the Random Forest

The model takes the following **15 numerical image-space features** from `risk_core.py`:

| Feature group | Feature columns |
|---|---|
| Separation | `separation_min`, `separation_mean`, `separation_last`, `separation_change` |
| Closing and approach | `closing_mean`, `closing_max`, `approach_fraction` |
| Movement and speed change | `relative_speed_mean`, `speed_a_mean`, `speed_b_mean`, `speed_change_a`, `speed_change_b` |
| Direction and overlap | `direction_alignment`, `bbox_iou_max` |
| Observation quality | `coverage` |

Coordinates and motion are **normalized to the image frame**. They are not calibrated to metres, kilometres per hour, physical TTC, or post-encroachment time. Perspective, camera motion, occlusion, and tracking failures can affect estimates.

## Risk decision rule

A `RandomForestClassifier` predicts the class for each eligible 5-second pair window using the extracted features. The implementation uses **300 trees**, `max_depth=10`, `min_samples_leaf=2`, `class_weight='balanced'`, and `random_state=42`. A median imputer is stored inside the scikit-learn pipeline.

The predicted state is selected from the model's estimated class support; **there is no manually defined TTC/distance threshold that assigns the final four risk states**. The displayed `vote_share` is an **uncalibrated Random Forest class-support score**, not a collision probability or classification accuracy. A window classified `Critical Conflict / Collision` does not prove a collision occurred or how early it could have been predicted.

## Outputs

| File | Description |
|---|---|
| `data/annotations.csv` | Manually reviewed windows with source group, risk state, supporting evidence, and reviewer. |
| `data/runs/<video_hash>/tracks.csv` | Per-sample timestamp, track ID, class, normalized box, and reference point. |
| `data/runs/<video_hash>/features.csv` | Qualifying five-second vehicle-pair windows and their 15 features. |
| `data/runs/<video_hash>/metadata.json` | Video hash, original name, FPS, processed duration, extraction settings, and timing. |
| `data/runs/<video_hash>/predictions.csv` | Predicted labels and vote shares for eligible video-demo windows. |
| `data/runs/<video_hash>/annotated.mp4` | Optional video with tracked IDs and most recent visible pair-risk overlay. |
| `artifacts/risk_model.pkl` | Serialized fitted Random Forest pipeline and model metadata. |
| `artifacts/evaluation.json` | Held-out split details, accuracy, macro F1, balanced accuracy, weighted kappa, and per-class results. |
| `artifacts/split_manifest.csv` | Recording/window assignments to training and test partitions. |
| `artifacts/test_predictions.csv` | Actual and predicted risk states on held-out windows. |
| `artifacts/feature_importance.csv` | Random Forest feature-importance values. |
| `artifacts/labelled_features.csv` | Extracted features joined with saved manual annotations. |

## Current evaluation status

The **included** `artifacts/evaluation.json` reports the following *small pilot run*:

| Metric | Value |
|---|---:|
| Training source groups | 5 |
| Held-out source groups | 4 |
| Training windows | 5 |
| Test windows | 4 |
| Test accuracy | **0.25 (25%)** |
| Macro F1-score | **0.10** |
| Balanced accuracy | **0.25** |
| Quadratic weighted Cohen's kappa | **0.00** |
| Majority-class baseline macro F1 | **0.10** |

**Important:** These results involve only **nine labelled windows** in total (five train, four test), with **one test window per class**. They are preliminary and statistically inadequate for a reliable performance claim. They do **not** demonstrate effective real-world accident prediction or early warning. Report improved results only after collecting additional, independently reviewed data and conducting properly controlled evaluation. Overlapping windows are correlated and should not be counted as independent incidents.

The supplied `VALIDATION.json` also records software checks; those tests validate parts of the pipeline, **not** the model's real-world accuracy.

## Dataset and training

This component uses **manually annotated vehicle-pair interactions** from road videos, not image-level accident/normal labels alone. The project folder contains `data/annotations.csv`, extracted run tables under `data/runs/`, and pilot model artifacts; it does **not** include the original source CCTV video dataset. Reproducing video extraction requires access to the original footage or new, appropriately permitted recordings.

For stronger training, include diverse normal driving, increasing-risk, developing-conflict, and critical scenarios from **multiple independent recordings**, with clear evidence for each label. Use recording-level partitions to avoid leakage between near-identical video clips/windows. Consider separate-camera testing, second-reviewer agreement, and class-balanced evaluation in future research.

**Optional Colab training:** Open `Component04_Training_Colab.ipynb`, follow the notebook's setup and Drive instructions, upload the exported `labelled_features.csv`, choose held-out recording groups, train, and download the generated artifacts into the local `artifacts/` directory. Match scikit-learn versions when moving pickle files between environments.

**Command-line alternatives:**

```bash
# Extract supported videos from a folder (default: up to 120 seconds per video)
python extract.py "path/to/videos" --output data/runs --seconds 120

# Train from a manually labelled feature table using preselected held-out groups
python train.py labelled_features.csv --test-groups recording3 --output artifacts

# Execute software unit tests
python -m unittest discover -s tests -v
```

The sample `recording3` is only an example: use actual `source_group` values from your labelled CSV, with all four states represented in training and test partitions.

## Known limitations

- **Pilot-scale data:** The supplied model evaluation uses very few labelled windows and cannot establish deployment performance.
- **Detector scope:** The standard COCO detector does not have a dedicated Sri Lankan three-wheeler category and can miss or misclassify local vehicles.
- **Occlusion/ID switches:** ByteTrack IDs are temporary; the component does **not** perform verified identity recovery after occlusion.
- **Image-space geometry:** No camera calibration, physical separation, measured vehicle speed, TTC/PET, or geographic road-risk mapping is implemented.
- **Window eligibility:** Short clips, missing tracks, heavy occlusion, or sparse pair co-visibility can produce no qualifying predictions; this does not mean the road is safe.
- **Prediction meaning:** Risk states are classifications of already observed windows, not validated forecasts of the time remaining before a collision.
- **Uncalibrated scores:** Random Forest vote shares should not be interpreted as true accident probabilities.
- **Offline demonstration:** Detection, tracking, and video rendering may be slow on a CPU; the interface is not a proven real-time roadside system.
- **Playback:** MP4 encoding without FFmpeg may fail in some browsers; downloaded files can be opened in VLC or re-encoded to H.264.

## Research and responsible use

This project is intended for **academic research and demonstration**, with particular relevance to mixed-vehicle traffic scenes. Obtain appropriate permission to collect and process CCTV footage; avoid publicly uploading identifiable footage without authorization. Evaluate performance on unseen recordings and clearly disclose error cases and limitations before proposing real-world use.

## Repository note

This README documents **Component 04** only and should be saved as `README.md` in the **Component 04 code directory** of the shared research GitHub repository. It is **not** a replacement for the repository-level README describing other components. The Component 03 documentation served only as a formatting reference; its models, results, and dataset claims are not attributed to this component.
