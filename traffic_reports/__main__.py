"""Command-line JSON export for recorded traffic events."""

import argparse
import json
from pathlib import Path

from .summary import summarize_events


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize a traffic event CSV as JSON.")
    parser.add_argument("csv_file", type=Path, help="EventManager violation_events.csv")
    parser.add_argument("--output", type=Path, help="Save JSON to this path instead of stdout")
    args = parser.parse_args(argv)

    try:
        if args.output is not None:
            same_path = args.output.resolve() == args.csv_file.resolve()
            same_file = (
                args.output.exists() and args.csv_file.exists()
                and args.output.samefile(args.csv_file)
            )
            if same_path or same_file:
                parser.error("--output must not overwrite the source CSV")

        report = summarize_events(args.csv_file)
        serialized = json.dumps(report, indent=2, ensure_ascii=True) + "\n"
        if args.output is None:
            print(serialized, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
