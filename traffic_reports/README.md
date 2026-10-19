# Traffic event reports

This module summarizes CSV records produced by `TrafficViolationAI/event_manager.py`.
It uses only the Python standard library, so existing reports can be reviewed
without installing OpenCV or loading model weights.

Run from the repository root:

```bash
python -m traffic_reports TrafficViolationAI/outputs/violation_events.csv
python -m traffic_reports TrafficViolationAI/outputs/violation_events.csv --output reports/summary.json
```

The first command prints JSON. The second creates the output directory if needed
and writes a JSON file, replacing any existing report at that path. The source
CSV cannot be used as the output. Use `python -m traffic_reports --help` for usage.

Reports contain `total_events`, `by_camera`, `by_violation`, and `by_status`.
Each CSV row counts as one recorded event; records are not deduplicated by vehicle
or track ID. Candidate events remain candidates, and these totals do not measure
detection accuracy or establish verified violations.

The CSV must have unique column names and include `camera_id`, `violation_type`,
and `status`. Those fields must not be blank. Extra columns from EventManager are
accepted. A header-only CSV produces zero counts. Empty files, missing columns,
inconsistent row lengths, and invalid CSV quoting produce a descriptive error
and a nonzero exit code. UTF-8 files with a byte-order mark are supported.

Python usage:

```python
from traffic_reports import summarize_events

report = summarize_events("TrafficViolationAI/outputs/violation_events.csv")
print(report["by_status"])
```

Run the summary and command-line tests from the repository root:

```bash
python -B -m unittest discover -s tests -p "test_traffic_reports*.py" -v
```
