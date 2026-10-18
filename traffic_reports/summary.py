"""Read EventManager CSV output without loading the detection model."""

from collections import Counter
import csv
from pathlib import Path


GROUPS = {
    "by_camera": "camera_id",
    "by_violation": "violation_type",
    "by_status": "status",
}


def summarize_events(csv_path):
    """Count CSV records by category; counts do not imply verified violations.

    Empty files, missing columns, and malformed records raise ValueError.
    A header-only CSV is valid and produces zero counts. Additional columns
    from EventManager, such as evidence paths, are accepted but not exported.
    """
    counts = {group: Counter() for group in GROUPS}
    total = 0
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        try:
            headers = reader.fieldnames or []
            missing = sorted(set(GROUPS.values()) - set(headers))
            if missing:
                raise ValueError("Missing CSV columns: " + ", ".join(missing))
            if len(headers) != len(set(headers)):
                raise ValueError("CSV contains duplicate column names")

            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(
                        f"CSV line {reader.line_num}: column count does not match header"
                    )
                values = {}
                for group, column in GROUPS.items():
                    value = row[column].strip()
                    if not value:
                        raise ValueError(
                            f"CSV line {reader.line_num}: {column} must not be blank"
                        )
                    values[group] = value
                for group, value in values.items():
                    counts[group][value] += 1
                total += 1
        except csv.Error as error:
            raise ValueError(
                f"Invalid CSV near line {reader.line_num}: {error}"
            ) from error

    return {
        "total_events": total,
        **{group: dict(sorted(counter.items())) for group, counter in counts.items()},
    }
