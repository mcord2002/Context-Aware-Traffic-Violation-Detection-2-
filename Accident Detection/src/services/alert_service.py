from datetime import datetime, timezone
from pathlib import Path
import json

def write_alert(video_name, frame_number, accident_score, evidence_path=None,
                output_dir="outputs/evidence"):
    """Append an alert record for human review after classifier integration."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    alert = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "video": str(video_name),
        "frame_number": int(frame_number),
        "accident_score": float(accident_score),
        "evidence_path": str(evidence_path) if evidence_path else None,
        "status": "requires_human_review",
    }
    with (folder / "alerts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(alert) + "\n")
    return alert
