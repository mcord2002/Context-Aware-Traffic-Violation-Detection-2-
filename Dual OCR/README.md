# Component 02 — Local Windows/VS Code Demonstration

This frontend is an **image-based** YOLO11m + EasyOCR prototype, not yet real video tracking or violation association.

## 1. Install prerequisites
- Python 3.11 (64-bit), from python.org
- VS Code with the Python extension

## 2. Copy your trained model
From Google Drive:
`Component02_Research/experiments_lowmem_clean/YOLO11m/weights/best.pt`

Copy it into `models/` and rename it `YOLO11m_best.pt`.

If the actual experiment folder differs, use the YOLO11m `best.pt` that produced your results.

## 3. Open the folder in VS Code
Open a new terminal in the `Component02_ALPR_App` folder.

### PowerShell
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

No need to activate the environment, so PowerShell execution-policy restrictions won't block you.

### If the `py -3.11` command is unavailable
Install Python 3.11 and ensure the Windows Python Launcher is installed, or use your installed Python executable.

## 4. Demo
- Visit the localhost URL printed in the terminal (usually http://localhost:8501).
- Upload a `.png`, `.jpg`, or `.jpeg` image.
- Review predicted boxes, plate crop, OCR methods and export JSON.

## Limitations
- Plates are detected using the trained YOLO11m model.
- EasyOCR is a pretrained baseline model; it may misread characters or return text outside the true plate.
- Agreement is over image-preprocessing methods, **not multi-frame** observations.
- Demo IDs are manually entered; there is no real Vehicle ID tracking or verified violation association.
- On first use, EasyOCR may download model weights and need an internet connection.
- Local CPU use can be slow, especially on the first inference.
