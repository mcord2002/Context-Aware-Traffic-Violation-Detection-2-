"""Component 02: image-based license-plate detection and OCR research prototype.

This project is a demo application built to detect vehicle license plates from a still image,
then read the plate text using OCR techniques. It is designed as a research prototype for
comparison, not as a production enforcement system.
"""
from pathlib import Path
import json
import re
import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Project root and model location.
# The model file is expected to be inside the models/ folder in the project.
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "YOLO11m_best.pt"
METHODS = ("Original", "CLAHE", "Adaptive", "Denoised")

# Streamlit app configuration: page title, emoji icon, and wide layout for image display.
st.set_page_config(page_title="Component 02 | Dual OCR", page_icon="🚘", layout="wide")

# Load the YOLO detector once and cache it so the app does not reload the heavy model every time.
@st.cache_resource(show_spinner="Loading YOLO11m detector...")
def load_detector(path: str):
    return YOLO(path)

# Load EasyOCR in CPU mode because the project is meant for local Windows demo use.
@st.cache_resource(show_spinner="Loading EasyOCR (first launch may download weights)...")
def load_reader():
    # CPU OCR keeps the Windows demo compatible without a CUDA setup.
    return easyocr.Reader(["en"], gpu=False)


# PaddleOCR is optional because it is slower and only needed for comparison experiments.
@st.cache_resource(show_spinner="Loading PaddleOCR (initial setup can download model files)...")
def load_paddle():
    # PaddleOCR 3.x interface; lazy loading avoids slowing EasyOCR-only runs.
    from paddleocr import PaddleOCR
    return PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False
    )


def extract_paddle_prediction(prediction):
    """Read rec_texts/rec_scores from PaddleOCR 3.x OCRResult objects."""
    # PaddleOCR returns nested result objects, so we flatten them into a clean text + score pair.
    found = []
    for item in prediction or []:
        if isinstance(item, dict):
            data = item
        elif hasattr(item, "json"):
            data = item.json
            if callable(data):
                data = data()
        else:
            data = {}
        if not isinstance(data, dict):
            continue
        data = data.get("res", data)
        if not isinstance(data, dict):
            continue
        texts = data.get("rec_texts", []) or []
        scores = data.get("rec_scores", []) or []
        for idx, t in enumerate(texts):
            if str(t).strip():
                score = float(scores[idx]) if idx < len(scores) else 0.0
                found.append((str(t), score))
    return " ".join(text for text, _ in found), (float(np.mean([s for _, s in found])) if found else 0.0)


def read_paddle(paddle, plate_bgr):
    # Resize the plate crop bigger and pad it with white space to help OCR detect text more clearly.
    enlarged = cv2.resize(plate_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    padded = cv2.copyMakeBorder(enlarged, 16, 16, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    result = paddle.predict(padded)
    return extract_paddle_prediction(result)

# Generate several preprocessed versions of the same plate crop for OCR comparison.
def variants(crop_bgr):
    enlarged = cv2.resize(crop_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 11)
    denoised = cv2.bilateralFilter(gray, 7, 50, 50)
    return {"Original": enlarged, "CLAHE": clahe, "Adaptive": adaptive, "Denoised": denoised}

# Keep only alphanumeric characters and uppercase them for a cleaner OCR comparison.
def clean(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())

# Read the text from one processed crop using EasyOCR.
def read_plate(reader, processed):
    raw = reader.readtext(processed, detail=1, paragraph=False)
    # For a simple plate crop, join OCR results in approximate reading order.
    raw.sort(key=lambda item: (round(min(p[1] for p in item[0]) / 25), min(p[0] for p in item[0])))
    text = " ".join(str(item[1]) for item in raw)
    confidence = float(np.mean([float(item[2]) for item in raw])) if raw else 0.0
    return text, confidence


# Optional plate-format validation. Never overwrite the raw OCR reading.
# Format supported here: common Indian registration structure
# 2 letters + 2 digits + 1-3 letters + 4 digits, e.g. MH12NE8922.
INDIAN_FORMATS = ("LLDDLD DDDD",)  # documentation marker; regex below is authoritative
LETTER_TO_DIGIT = {"O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"}
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"}

# This function tries to suggest a format-compatible plate string, but it never claims the text is true.
def suggest_plate_format(raw_text, fmt):
    """Return an optional format-based candidate, never a verified identity."""
    value = clean(raw_text)
    if not value:
        return {"candidate": "", "valid": False, "changed": False,
                "reason": "OCR returned no text"}
    if fmt == "No country restriction":
        return {"candidate": value, "valid": None, "changed": False,
                "reason": "No plate-format rule applied"}
    if fmt != "India (common modern format)":
        return {"candidate": value, "valid": None, "changed": False,
                "reason": "Unknown format — no rule applied"}
    # Only lengths matching the expected structure are allowed. We intentionally
    # do not insert/delete characters or force a guess where data are missing.
    if len(value) not in (9, 10, 11):
        return {"candidate": value, "valid": False, "changed": False,
                "reason": "Wrong length for selected Indian pattern; no forced correction"}
    n_letters = len(value) - 8
    slots = "LLDD" + "L" * n_letters + "DDDD"
    candidate = []
    changed_positions = []
    for position, (char, slot) in enumerate(zip(value, slots), start=1):
        if slot == "D" and char.isalpha():
            new_char = LETTER_TO_DIGIT.get(char, char)
        elif slot == "L" and char.isdigit():
            new_char = DIGIT_TO_LETTER.get(char, char)
        else:
            new_char = char
        if new_char != char:
            changed_positions.append(position)
        candidate.append(new_char)
    proposal = "".join(candidate)
    valid = bool(re.fullmatch(r"[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}", proposal))
    return {
        "candidate": proposal if valid else value,
        "valid": valid,
        "changed": bool(changed_positions) and valid,
        "reason": ("Format-compatible candidate; changed positions: "
                   + ", ".join(map(str, changed_positions))) if valid and changed_positions
                   else ("Matches selected format" if valid else "Does not match selected format"),
    }

# Main page title and description for the app.
st.title("🚘 Component 02 — License Plate Detection & OCR")
st.caption("Image-based research prototype | YOLO11m + EasyOCR / PaddleOCR comparison | No tracking or verified violation association")

# The trained YOLO model must exist before the app can process images.
if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Copy the trained YOLO11m best.pt from Google Drive into the models folder and rename it YOLO11m_best.pt.")
    st.stop()

# Sidebar controls for detection and experimental settings.
with st.sidebar:
    st.header("Detection settings")
    compare_paddle = st.checkbox(
        "Compare with PaddleOCR (slower on CPU)",
        value=False,
        help="Requires optional PaddlePaddle and PaddleOCR installation. First load downloads model files."
    )
    confidence = st.slider("YOLO confidence threshold", 0.10, 0.90, 0.30, 0.05)
    image_size = st.selectbox("Inference image size", [416, 640], index=1)
    st.markdown("**Optional demonstration references**")
    violation_id = st.text_input("Demo violation ID", "VIO-DEMO-001")
    vehicle_id = st.text_input("Demo vehicle ID", "VEH-DEMO-001")
    st.warning("Demo IDs are entered manually and are NOT linked to a verified offending vehicle.")
    plate_format = st.selectbox(
        "Optional plate-format check",
        ["No country restriction", "India (common modern format)"],
        index=0,
        help="Rules are only suggestions and may be wrong. No Sri Lankan format is assumed."
    )

# The user uploads a still image to be processed.
uploaded = st.file_uploader("Upload a vehicle image", type=["png", "jpg", "jpeg"])
if uploaded is None:
    st.info("Upload an image to start. This prototype accepts still images from your Kaggle dataset or other images.")
    st.stop()

try:
    # Convert the uploaded image into a NumPy array and then to OpenCV's BGR format.
    rgb = np.array(Image.open(uploaded).convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Initialize the heavy models once, then process the image.
    detector = load_detector(str(MODEL_PATH))
    reader = load_reader()
    paddle_reader = None
    paddle_error = None
    if compare_paddle:
        try:
            paddle_reader = load_paddle()
        except Exception as exc:
            paddle_error = str(exc)

    # Run YOLO detection on the image with user-selected confidence and size values.
    result = detector.predict(bgr, conf=confidence, iou=0.40,
                              imgsz=image_size, verbose=False)[0]
except Exception as exc:
    st.error(f"Processing failed: {exc}")
    st.stop()

if compare_paddle and paddle_error:
    st.warning("PaddleOCR could not start. EasyOCR will still run. " + paddle_error)

# Display the uploaded image and the detection results overlay.
annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
a, b = st.columns(2)
with a:
    st.subheader("Original")
    st.image(rgb, use_container_width=True)
with b:
    st.subheader("YOLO plate detections")
    st.image(annotated, use_container_width=True)

# Show how many plate regions were detected.
st.metric("Detected plate regions", len(result.boxes))
if len(result.boxes) == 0:
    st.warning("No plate detected. Try another image or lower the confidence threshold.")
    st.stop()

# Each detected box is processed as a separate plate region.
records = []
h, w = bgr.shape[:2]
for index, box in enumerate(result.boxes, start=1):
    # Get the coordinates of the bounding box and enlarge the crop slightly around the plate.
    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
    px = int((x2 - x1) * 0.08)
    py = int((y2 - y1) * 0.15)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    if x2 <= x1 or y2 <= y1:
        continue

    # Crop the image to the plate region and preprocess the plate in multiple ways.
    crop = bgr[y1:y2, x1:x2]
    processed = variants(crop)
    plate_results = []
    for name, img in processed.items():
        text, score = read_plate(reader, img)
        plate_results.append({"method": name, "ocr_text": text,
                              "normalized_text": clean(text), "ocr_confidence": score})

    # Majority voting across preprocessing methods gives a consensus plate string.
    valid = [r["normalized_text"] for r in plate_results if r["normalized_text"]]
    counts = pd.Series(valid).value_counts() if valid else pd.Series(dtype=int)
    consensus_text = str(counts.index[0]) if not counts.empty else ""
    agreement = float(counts.iloc[0] / len(METHODS)) if not counts.empty else 0.0

    # Agreement refers to preprocessing consistency, not guaranteed recognition correctness.
    suggestion = suggest_plate_format(consensus_text, plate_format)
    paddle_text, paddle_confidence, paddle_status = "", 0.0, "disabled"
    if compare_paddle:
        paddle_status = "unavailable" if paddle_reader is None else "processed"
        if paddle_reader is not None:
            try:
                paddle_text, paddle_confidence = read_paddle(paddle_reader, crop)
            except Exception as exc:
                paddle_status = "error: " + str(exc)
    paddle_normalized = clean(paddle_text)

    # Different OCR engines use different confidence scales, so agreement is interpreted cautiously.
    engines_agree = bool(consensus_text and paddle_normalized and consensus_text == paddle_normalized)
    records.append({
        "plate_region": index, "detection_confidence": float(box.conf[0]),
        "predicted_text": consensus_text,
        "format_suggestion": suggestion["candidate"],
        "format_rule": plate_format,
        "format_valid": suggestion["valid"],
        "correction_suggested": suggestion["changed"],
        "correction_reason": suggestion["reason"],
        "preprocessing_agreement": agreement,
        "methods": plate_results, "bounding_box_xyxy": [x1, y1, x2, y2],
        "paddle_ocr": {"raw_text": paddle_text, "normalized_text": paddle_normalized,
                       "confidence": paddle_confidence, "status": paddle_status},
        "easy_paddle_agreement": engines_agree,
    })

    # Show the result for each detected plate region in a collapsible section.
    with st.expander(f"Plate region {index} — {consensus_text or 'unreadable'}", expanded=(index == 1)):
        st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption="Detected plate crop", width=380)
        st.write(f"YOLO detection confidence: **{float(box.conf[0]):.3f}**")
        st.write(f"Preprocessing agreement: **{agreement:.0%}** (not recognition accuracy)")
        st.dataframe(pd.DataFrame(plate_results), use_container_width=True, hide_index=True)
        st.markdown(f"**EasyOCR consensus:** `{consensus_text or 'Unreadable'}`")
        if compare_paddle:
            st.markdown("**Second OCR engine: PaddleOCR**")
            if paddle_status == "processed":
                st.write(f"PaddleOCR prediction: **{paddle_text or 'Unreadable'}**")
                st.write(f"PaddleOCR confidence: **{paddle_confidence:.3f}**")
                st.write("Engines agree: **" + ("Yes" if engines_agree else "No") + "**")
                st.caption("Different engines' confidence scores are not directly comparable. Agreement does not establish correctness.")
            else:
                st.warning("PaddleOCR status: " + paddle_status)
        if plate_format == "No country restriction":
            st.info("Format checking is off; no character corrections were applied.")
        elif suggestion["valid"]:
            st.markdown(f"**Possible format-based candidate:** `{suggestion['candidate']}`")
            st.warning("This is a format-based suggestion, **not a verified plate number**. "
                       "Inspect the original plate image before trusting it.")
        else:
            st.warning("No valid suggestion under this format: " + suggestion["reason"])
        st.caption("OCR confidence, method agreement and format compatibility do not prove plate accuracy.")

# Final JSON export structure containing all observation metadata.
result_record = {
    "prototype": "image_based_alpr",
    "model": "YOLO11m",
    "paddle_enabled": compare_paddle,
    "selected_format_rule": plate_format,
    "input_filename": uploaded.name,
    "demo_violation_id": violation_id,
    "demo_vehicle_id": vehicle_id,
    "association_status": "manual_demo_only_unverified",
    "plate_regions": records,
}
st.subheader("Export results")
st.download_button("Download JSON results", json.dumps(result_record, indent=2),
                   file_name="component02_detection_result.json", mime="application/json")
st.caption("Evaluate EasyOCR and PaddleOCR against manually verified ground truth (exact match and CER) before declaring a winner. Video tracking and verified violation association remain future work.")
