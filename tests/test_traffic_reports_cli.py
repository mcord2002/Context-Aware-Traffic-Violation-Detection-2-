import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReportCommandTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.source = self.folder / "events.csv"
        self.source.write_text(
            "camera_id,violation_type,status\nCAM_1,helmet_violation,candidate\n",
            encoding="utf-8",
        )

    def run_command(self, *args):
        return subprocess.run(
            [sys.executable, "-B", "-m", "traffic_reports", *map(str, args)],
            cwd=ROOT, capture_output=True, text=True, timeout=15,
        )

    def test_prints_json_to_stdout(self):
        result = self.run_command(self.source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["by_status"], {"candidate": 1})

    def test_creates_output_directory_and_writes_json(self):
        output = self.folder / "reports" / "summary.json"
        result = self.run_command(self.source, "--output", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["total_events"], 1)

    def test_cannot_overwrite_source(self):
        original = self.source.read_bytes()
        result = self.run_command(self.source, "--output", self.source)
        self.assertEqual(result.returncode, 2)
        self.assertIn("must not overwrite", result.stderr)
        self.assertEqual(self.source.read_bytes(), original)

    def test_invalid_csv_leaves_existing_report_untouched(self):
        self.source.write_text("bad_header\n", encoding="utf-8")
        output = self.folder / "summary.json"
        output.write_text("previous report", encoding="utf-8")
        result = self.run_command(self.source, "--output", output)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Missing CSV columns", result.stderr)
        self.assertEqual(output.read_text(encoding="utf-8"), "previous report")

    def test_missing_source_returns_readable_error(self):
        result = self.run_command(self.folder / "missing.csv")
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing.csv", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_help_is_available(self):
        result = self.run_command("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output", result.stdout)


if __name__ == "__main__":
    unittest.main()
