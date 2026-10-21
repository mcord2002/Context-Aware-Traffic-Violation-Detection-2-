# Double-click Windows launcher

1. Extract ALL files from this ZIP into a normal writable folder, such as Documents\Component01.
2. Double-click START_DEMO.bat.
3. First run creates .venv and installs packages. Keep internet connected and wait.
4. The app opens automatically in your browser. Keep the command window open while using it.
5. Press Ctrl+C in that window to stop it. Next time, double-click the same BAT.

Prerequisite: 64-bit Python 3.11 or 3.12 installed. If absent, install Python with Add Python to PATH enabled, then rerun the BAT. No administrator mode or PowerShell activation is required.
The pretrained YOLO11n weights are included at `TrafficViolationAI/models/yolo11n.pt`, so analyzing a video does not need to download the model. Later launches reuse the environment. The app uses a free localhost port. startup.log contains server errors. FFmpeg is included through a Python dependency for browser video playback.
If you move the project and the existing environment stops working, rename .venv to .venv_old and rerun the launcher to recreate it.
Launcher code and ZIP structure were checked, but double-click execution must be verified on Windows.

---
# Component 01 PP1 demo
This is pretrained YOLO11n vehicle detection + ByteTrack + temporal direction rules. It is not a model trained on your 36 annotations. It assesses wrong-way movement in one manually selected straight lane from a fixed camera. Do not present it as general multi-violation detection.

## Windows Streamlit setup
Install Python 3.11. Extract this ZIP. Open PowerShell inside the extracted folder.
```
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Open the localhost URL printed by Streamlit. The bundled `models/yolo11n.pt` weights work offline. CPU works but is slower. FFmpeg on PATH enables browser-compatible H264 output; without it, download the raw video if playback fails. Colab installs FFmpeg in the notebook.

Upload a short stationary-camera clip. Select only one direction of traffic with the blue rectangle. Point the green arrow along the verified permitted direction. Check both verification boxes, then Analyze. Do not include an opposing legal lane. For curved lanes, moving cameras, unclear road rules, or scenes that cannot be isolated with a rectangle, use another clip. Colab supports a custom four-point polygon via cfg['roi'] if needed.

Outputs: annotated video, event IDs and timestamps, report JSON, scene settings. Settings must be changed for a new camera view. Red-light violations and U-turn legality are not implemented. Motorcycle detection can miss small/occluded vehicles; COCO classes do not explicitly cover Sri Lankan three-wheelers. Event identity is per tracker ID within a run; ID switches can duplicate events. The program gives no accuracy claim.

## PP1 validation
Use a few manually reviewed wrong-way and normal clips from different source videos, plus a clip with no visible vehicles. Compare predictions against your annotations; record misses and false alarms. Review the annotated output, not just the verdict. A no-violation label in this demo means no wrong-way event detected in the selected lane, not that all road rules were checked. Unresolved clips must not be forced into normal labels. Keep held-out clips separate when tuning thresholds. Do not report demo clips as a final test set.

Motion rule: bottom-center of each tracked box must stay inside the ROI. Displacement over roughly 0.75 seconds must exceed 1.5% of the frame diagonal; cosine against the allowed arrow below -0.7 must persist for 0.5 seconds. These are starting heuristics, not validated thresholds. Near-static or insufficient trajectories yield Unable to determine if no other track was assessed. Coverage is incomplete even when some tracks were assessed. Input audio is omitted from rendered output.

References: https://docs.ultralytics.com/modes/track/ and https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
