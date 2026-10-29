# Component 03: Occlusion-Aware Physics-Vision Traffic Accident Verification and Dynamic Severity Classification


A local, CPU-capable research prototype for analyzing traffic CCTV videos. This component detects and tracks vehicles, estimates whether sampled video frames show a possible accident, and generates an annotated video with downloadable analysis results.

> **Research prototype:** The system flags *possible* accidents. Its predictions are not confirmation that a collision occurred, it does not determine fault, and it does not automatically verify traffic-law violations or vehicle identities after occlusion.

## Features

- **Vehicle detection — YOLO26n:** Detects and classifies cars, motorcycles, buses, and trucks, drawing bounding boxes around detections.
- **Vehicle tracking — ByteTrack:** Associates detections across frames and assigns tracking IDs. IDs can fragment or change, so they should not be treated as permanent vehicle identities.
- **Accident classification — ResNet18:** Classifies sampled video frames into `normal` and `accident` classes and calculates an accident probability.
- **Temporal probability analysis:** Samples approximately two frames per second, averages consecutive accident probabilities, and compares the smoothed score with a configurable threshold.
- **Annotated video and reports:** Produces a processed video, accident timeline, vehicle tracking observations, possible occlusion candidates, and a JSON summary.
- **Local processing:** Inference runs on the local CPU; the application does not require a cloud inference service.

## How the pipeline works

1. Upload an MP4 traffic CCTV video in the Streamlit interface.
2. YOLO26n detects vehicles, while ByteTrack associates detections across frames.
3. ResNet18 processes sampled frames and estimates the probability of the `accident` class.
4. The application smooths consecutive sampled-frame probabilities and checks the peak smoothed score against the configured threshold (`0.50` by default).
5. The dashboard displays the possible-accident status, estimated evidence time, annotated video, probability timeline, tracking data, and candidate track links.

YOLO26n/ByteTrack and ResNet18 perform separate tasks in this implementation: tracking information is not directly fed into ResNet18's accident classification.

## Model checkpoints

Place the trained model checkpoints in the `models/` directory beside the application file:

```text
project/
├── app.py
├── models/
│   ├── yolo26n_best.pt
│   └── resnet18_best.pt
└── outputs/
```

- `yolo26n_best.pt` — fine-tuned YOLO26n weights. The application expects the class mapping to be exactly `car`, `motorcycle`, `bus`, `truck` in that order.
- `resnet18_best.pt` — the Step 8 ResNet18 checkpoint. It must contain `model_state_dict` and `class_to_idx`, including an `accident` class entry.

**The checkpoints are not included by this README.** Obtain them from your training workflow and place them at the paths above. The application reports an error if a required checkpoint is missing or incompatible.

## Requirements

- Windows 10/11 or another OS supported by Python and the required libraries
- Python environment compatible with the installed PyTorch, torchvision, Ultralytics, OpenCV, Streamlit, pandas, NumPy, Pillow, and `imageio-ffmpeg` packages
- The two trained model checkpoints listed above
- An MP4 video that OpenCV can decode; H.264 MP4 is recommended if decoding fails

Use the Python version and package versions tested by your project environment. Install the dependencies in that environment before launching the application.

## Run the application

From the project directory:

```bash
streamlit run app.py
```

Replace `app.py` with the actual filename of your Streamlit script if it has a different name. Open the local URL printed by Streamlit, upload a short MP4 clip, and click **Analyze video**.

For a live demonstration, start with a short 10–20 second clip. Processing can take several minutes on a CPU. The dashboard offers inference image sizes of 640, 800, and 960 pixels; 640 is a useful starting point for a faster demo.

## Outputs

The application writes results to the `outputs/` directory:

| File | Description |
|---|---|
| `processed_video.mp4` | Annotated video with vehicle labels, tracking IDs, accident status, score, and time |
| `accident_timeline.csv` | Sampled timestamps, raw accident probabilities, and smoothed probabilities |
| `tracking_observations.csv` | Per-observation frame/time, tracker ID, vehicle class, detection confidence, and bounding-box coordinates |
| `occlusion_candidates.csv` | Possible links between fragmented tracks; these are unverified candidates |
| `summary.json` | Video-level status, peak smoothed accident score, estimated evidence time, and tracking summary |

The dashboard also provides download controls for the processed MP4 and the main CSV outputs.

## Accident decision rule

The application samples video frames at approximately two frames per second. ResNet18 returns an accident-class probability for each sampled frame. The application calculates a rolling average over two consecutive samples and compares the highest smoothed score with `ACCIDENT_THRESHOLD`, which is set to `0.50` in the current prototype.

- Score at or above the threshold: `POSSIBLE ACCIDENT`
- Score below the threshold: `NO ACCIDENT DETECTED`

The threshold is provisional and should be tuned using a suitable validation set. A score of `1.000` is the model's probability output, not proof of a real accident and not a measure of model accuracy. The displayed time is an estimate associated with the peak smoothed score, not necessarily the exact collision onset.

## Evaluation status

Record and report metrics from separate, labelled evaluation data rather than relying on dashboard predictions alone.

Current results shared during development:

- **ResNet18 validation:** 82% accuracy on 104 images; accident precision 0.80, recall 0.59, F1-score 0.68. Normal precision 0.82, recall 0.93, F1-score 0.87.
- **YOLO26n test:** mAP50 0.245 (24.5%) and mAP50–95 0.113 (11.3%) on 33 test images containing 232 labelled instances.

These are results from the currently reported runs, not a guarantee of performance on other videos. The ResNet18 figure is validation accuracy, while YOLO's values are object-detection metrics; they measure different tasks and should not be compared directly. The current results indicate that the models need further evaluation and improvement, especially accident recall and vehicle-detection performance.

## Known limitations

- Performance depends on the quality, size, class balance, and labelling of the training and evaluation datasets.
- The accident threshold is provisional; false alarms and missed accidents are possible.
- A frame classifier may miss short events between sampled frames.
- ByteTrack IDs are temporary track identifiers, not permanent registration-based vehicle identities.
- Occlusion matching only proposes candidate links and does not verify identity recovery.
- The prototype does not determine accident fault and does not automatically confirm traffic-law violations.
- Accident severity classification (high, medium, low) is not implemented by the current two-class ResNet18 model.

## Research and responsible use

Use this prototype for research and demonstration. Evaluate it on representative, unseen CCTV videos and report its limitations clearly before considering real-world deployment. Ensure that videos are collected and used with appropriate permissions and privacy safeguards.

## Dataset and training

Prepare and annotate the video dataset before training. The project workflow includes a dataset-preparation stage, YOLO vehicle-detection training/evaluation, and ResNet18 accident-classification training/evaluation. Refer to the corresponding project notebooks for the exact dataset paths, training configuration, class mappings, and evaluation procedures.

## Dataset
contains serious traffic incidents caused by rain, vandalism or other factors, with a total of 344 videos covering 277 positive ones with traffic accidents and 127 negative ones without traffic accidents.

We made the TAD dataset publicly available on GitHub. We ensured that researchers worldwide can access and utilize this valuable resource for further research and development, fostering collaboration and innovation in the field.

Please download dataset from the following link:
- [Baidu Yun Pan](https://pan.baidu.com/s/1X8xRJWZ5izXuyUgGbGppjw): gi9f
- [Google Drive](https://drive.google.com/file/d/14GNlNcWLzN-sbzvmrMuSbAg_rZZ5yd26/view?usp=drive_link)


