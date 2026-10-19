import argparse
from pathlib import Path
import pandas as pd
import cv2

REQUIRED_COLUMNS = {"clip_path", "label", "source_id"}

def main():
    parser = argparse.ArgumentParser(description="Validate labeled CCTV clips.")
    parser.add_argument("--csv", default="dataset/annotations/clips.csv")
    parser.add_argument("--report", default="dataset/processed/dataset_validation.csv")
    args = parser.parse_args()

    table = pd.read_csv(args.csv, comment="#")
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    table["label"] = pd.to_numeric(table["label"], errors="raise").astype(int)
    if not set(table["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0=normal or 1=accident.")

    results = []
    for _, row in table.iterrows():
        path = Path(str(row["clip_path"]))
        if not path.is_absolute():
            path = Path.cwd() / path
        result = row.to_dict()
        result["exists"] = path.exists()
        result["readable"] = False
        result["frame_count"] = 0
        result["fps"] = 0.0
        result["error"] = ""
        if path.exists():
            cap = cv2.VideoCapture(str(path))
            result["readable"] = cap.isOpened()
            result["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            result["fps"] = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            if not result["readable"]:
                result["error"] = "OpenCV could not open this video"
        else:
            result["error"] = "File not found"
        results.append(result)

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(report, index=False)
    print(f"Validated {len(results)} rows. Report: {report}")
    print(pd.DataFrame(results)["readable"].value_counts(dropna=False).to_string())

if __name__ == "__main__":
    main()
