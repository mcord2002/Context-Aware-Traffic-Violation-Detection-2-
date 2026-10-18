# Component 04 — PP1 setup and demonstration

This package implements manually reviewed vehicle-pair windows, one trainable four-state Random Forest, independent-recording evaluation and a Streamlit video demonstration. The Colab notebook uses the same Python training and feature code.

**No trained risk model or fabricated scores are bundled. Your actual videos and reviewed pair labels are required.** The pre-trained YOLO detector supplies tracks; the risk classifier is your Component 04 model.

## 1. Launch on Windows
1. Install **Python 3.11 64-bit**, including Python Launcher and Add Python to PATH.
2. Extract the entire ZIP. Do not launch from inside the ZIP.
3. Double-click **run_component04.bat**. First setup installs dependencies and needs internet.
4. Keep the console open. If the browser does not open, visit http://localhost:8501.
5. The first extraction downloads `yolov8n.pt`. Or enter your already-downloaded compatible COCO detector path in the sidebar.

CPU works; a GPU is optional. The launcher specifically requests Python 3.11. For dependency errors, read the console, confirm 3.11 is installed and retry. To reinstall dependencies after changing requirements, delete `.venv/dependencies_ready.txt`. Exact installed versions are saved under `.venv/installed_versions.txt`.

FFmpeg is optional. With FFmpeg, rendering produces H.264 for browser playback. Without it, download the MP4 and use VLC if the browser cannot play it. Annotated videos have no audio.

## 2. Select footage
Start with 20–60 second fixed-camera clips containing at least two interacting vehicles with stable tracks over five seconds. Include ordinary driving and genuinely developing/critical interactions from multiple independent recordings. Use footage you have permission to process.

The software requires at least **two training recordings and one test recording**, with all four states present in each partition. This is a functional minimum, not a scientifically adequate sample-size recommendation. More independent examples matter more than many overlapping windows from one incident.

Enter an original recording group such as `camera1_recording1`. Every clip cut from that original recording must share this group, regardless of filename. The app checks exact video identity but cannot discover re-encoded copies or clips from one recording: grouping those correctly is your responsibility. Later, hold out entire cameras to measure location generalization.

## 3. Extract and annotate
1. Open **1 · Prepare & annotate**.
2. Select multiple videos together in **Upload one or more road videos**, then click **Extract ALL uploaded videos**.
3. Choose a video from **Video to review and annotate**. Review and save its labels, then switch to the next video.
4. Enter the original source recording group.
5. Select a window/pair. Review the ID contact sheet and replay that interval in the full video player.
6. Skip uncertain interactions, identity switches and heavy occlusion.
7. Choose a state, enter evidence and reviewer initials, confirm the review and save.
8. Repeat across recordings and states. Saving the same window again replaces its annotation.
9. Download `annotations.csv` and **labelled_features.csv for Colab** as backups.


### Annotation protocol
- **Normal:** stable interaction without clear conflict development or avoidance. Describe the observed stable movement/gap.
- **Increasing Risk:** growing concern supported by multiple cues, such as closing gap and converging directions.
- **Developing Conflict:** emerging conflict supported by proximity/movement plus observable braking, steering or other avoidance.
- **Critical Conflict / Collision:** clearly severe interaction or visible contact. Explicitly note which is observed. This combined state does not prove a collision occurred.

Label the state **at window end**, using only observations within that window. Do not use later crash frames to relabel earlier driving. Do not label from box overlap or a single distance threshold alone. Have another reviewer check a subset where possible; inter-rater reliability is not automatically computed.

Accident/normal folder names are not sufficient four-state labels. If your footage cannot support all four states across independent train/test recordings, collect more reviewed footage. Do not fabricate data to bypass the guard. In that case, present extraction/annotation as work in progress rather than claiming a trained four-state result.

### Extraction details
YOLO COCO classes car, motorcycle, bus and truck are tracked with ByteTrack at approximately 5 processed frames/second. Three-wheelers may be missed/misclassified. Bottom-centre box points and box coordinates are normalized by image width/height. Features describe image-space separation, relative/closing movement, direction, speed changes and overlap.

Windows are five seconds long with a one-second stride. A pair needs at least 80% shared observations and adequate temporal coverage. A trailing three-observation smoother uses no future samples. Large gaps are excluded. There is **no calibrated speed, physical TTC or PET calculation**.

All eligible co-visible pairs are considered; crowded scenes can generate many unrelated pairs. Review meaningful interactions for training. Future candidate filtering needs validation.

Cached runs live in `data/runs/<video_hash>/`. Settings do not automatically invalidate cached runs. To change weights/extraction length for already annotated footage, use a fresh package copy, or remove its run **and affected annotations**, then re-extract and re-annotate.

## 4. Train locally — fastest route
1. Open **2 · Train & evaluate**.
2. Inspect class counts by source group.
3. Preselect held-out recordings before seeing model results.
4. Click **Train Random Forest & evaluate**.
5. Show macro F1, balanced accuracy, weighted kappa, class precision/recall/F1/support, confusion matrix and the majority-class baseline.
6. Download `risk_model.pkl`, `evaluation.json`, `split_manifest.csv` and `test_predictions.csv`.

The Random Forest uses 300 trees, depth 10, minimum leaf size 2, balanced class weights and seed 42. This is one model with fixed parameters, not a model comparison or test-set tuning exercise. All overlapping windows from a source recording stay together. Every run overwrites `artifacts/`; back up previous results if needed.

Duplicate windows, invalid labels, missing groups/evidence and non-finite features are rejected. A median imputer is saved inside the pipeline, although current extraction is expected to produce finite complete features; corrupt inputs require review/re-extraction.

### Colab alternative
Open **Component04_Training_Colab.ipynb** in Google Colab and run top to bottom. Upload the ZIP, install dependencies and mount Drive. Optional extraction can use GPU: set your exact video-folder path and enable the extraction cell. Copy generated `runs/<hash>` folders to local `data/runs/`, then upload the same original footage in Streamlit to annotate it.

Export the joined **labelled_features.csv**, upload it to the notebook, inspect EDA, set held-out groups and train. Download all artifacts into the Windows package's `artifacts/` folder. This workflow does not expose Streamlit over a public tunnel.

Only load your own trusted pickle model. Keep scikit-learn versions consistent across Colab and Windows; match `installed_versions.txt` or retrain locally if needed. Dependency ranges allow compatible updates; the version snapshot records the actual environment.

## 5. Demonstrate
1. Open **3 · Video demo** after training.
2. Upload held-out footage or an additional independent recording.
3. Extract windows and choose a pair.
4. Show IDs, predicted state, risk/separation timeline and uncalibrated vote share.
5. Click **Render annotated video**. The overlay shows tracked IDs and the highest-severity currently visible pair at each update.
6. Download predictions CSV and annotated MP4 as presentation backups.

The overlay waits for a complete five-second history, uses only the latest completed window and expires predictions after about one second. No eligible pair means unavailable, not Normal. Predicted rows are overlapping windows, **not unique conflict-event counts**. The displayed highest-severity pair is a presentation choice, not a geographic risk score.

## PP1 explanation
“Component 04 assesses dynamic interactions between tracked vehicle pairs. We extract five-second image-space movement features and use independently reviewed multi-evidence labels. A Random Forest predicts four risk states. We evaluate on separate source recordings and demonstrate predictions in Streamlit.”

Show your actual recording/window counts, scores and confusion-matrix errors. Vote shares are not calibrated collision probabilities. The model can recognize a window containing a collision; this does not establish advance warning. Do not claim real-world readiness or forecasting ability from a small pilot.

## Limitations and future work
Perspective, occlusion, jitter and ID switches affect features. No physical metres/km/h, TTC/PET, validated early-warning lead time, calibrated probability or geographic map is implemented. The demo is offline video analysis. Reported risk-inference milliseconds exclude detection, extraction and rendering; extraction elapsed time is saved in metadata.

Future: camera calibration, candidate-pair filtering, annotation reliability, train-only hyperparameter tuning, temporal model comparison, ablation, camera-held-out testing, conflict-onset/early-warning evaluation, false alarms/hour, full latency and geographic aggregation using verified camera coordinates. Component 03 identity preservation is separate; this basic tracker does not implement that research contribution.

## Files and commands
- `app.py`: annotation, training, evaluation and demo.
- `risk_core.py`: causal image-space features, Random Forest, metrics and pickle.
- `video_pipeline.py`: extraction, contact sheets and video overlays.
- `train.py`: `python train.py labelled_features.csv --test-groups recording3`
- `extract.py`: `python extract.py "path/to/videos" --output data/runs --seconds 120`
- `Component04_Training_Colab.ipynb`: optional Colab/Drive workflow.
- `tests/test_pipeline.py`: artificial fixtures test software only, not research performance.

## Troubleshooting
- **No windows:** use longer footage with two stable tracks. No windows does not mean safe driving.
- **Missing training/test state:** add independent labelled examples; do not repeatedly change the test split to obtain higher scores.
- **Download failure:** initial setup/weights require internet; supply already-downloaded compatible weights if available.
- **Three-wheelers missed:** generic detector lacks a dedicated Sri Lankan three-wheeler class.
- **Slow CPU:** start with 20–30 seconds or extract on Colab GPU.
- **Browser MP4 playback fails:** download and open in VLC; install FFmpeg and render again for H.264.

The app accepts videos up to 500 MB, but short clips are best for PP1. Back up `data/`, `artifacts/`, weights and environment versions before presenting. No user footage is bundled.


## Batch upload update
You can select many videos in a single file dialog (Ctrl/Shift selection). Click **Extract ALL uploaded videos** once. Processing is sequential; failures are reported per video and the remaining batch continues. Identical files are deduplicated and completed cached extractions are reused. The sidebar length limit applies to EACH video; increase it before extracting if needed. Each file may be up to 500 MB; browser/RAM limits still apply to the total batch. Prefer manageable batches.

Annotation remains per pair/window. Select each video from the review dropdown and assign the correct original recording group. Do not assign an entire accident video one risk state. **Train & evaluate** automatically combines all saved reviewed annotations across videos; unlabelled windows are excluded. Select held-out original recording groups, then train once. New labels require retraining; the model is not incrementally updated.

To update an existing installation, back up `data/` and `artifacts/`, then copy the new `app.py` and `batch.py` into your existing project folder. Keep your existing `data/`, `artifacts/`, `.venv/` and weights. No dependency change is required for this batch update.
