from pathlib import Path
import tempfile
import unittest

from traffic_reports import summarize_events


class SummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.source = self.folder / "events.csv"

    def write_csv(self, text):
        self.source.write_text(text, encoding="utf-8")

    def test_groups_events_and_keeps_review_status_separate(self):
        self.write_csv(
            "camera_id,violation_type,status,evidence_path\n"
            "CAM_2,helmet_violation,candidate,frame1.jpg\n"
            "CAM_1,helmet_violation,confirmed,frame2.jpg\n"
            "CAM_2,red_light_violation,candidate,frame3.jpg\n"
        )
        self.assertEqual(summarize_events(self.source), {
            "total_events": 3,
            "by_camera": {"CAM_1": 1, "CAM_2": 2},
            "by_violation": {"helmet_violation": 2, "red_light_violation": 1},
            "by_status": {"candidate": 2, "confirmed": 1},
        })

    def test_header_only_report_has_zero_counts(self):
        self.write_csv("camera_id,violation_type,status\n")
        self.assertEqual(summarize_events(self.source), {
            "total_events": 0, "by_camera": {}, "by_violation": {}, "by_status": {},
        })

    def test_accepts_bom_quoted_fields_and_surrounding_spaces(self):
        self.write_csv(
            '\ufeffcamera_id,violation_type,status\n'
            '" CAM,1 ",helmet_violation, candidate \n'
        )
        report = summarize_events(self.source)
        self.assertEqual(report["by_camera"], {"CAM,1": 1})
        self.assertEqual(report["by_status"], {"candidate": 1})

    def test_rejects_empty_or_incomplete_header(self):
        for content in ("", "camera_id,status\nCAM_1,candidate\n"):
            with self.subTest(content=content):
                self.write_csv(content)
                with self.assertRaisesRegex(ValueError, "Missing CSV columns"):
                    summarize_events(self.source)

    def test_rejects_duplicate_header(self):
        self.write_csv("camera_id,violation_type,status,status\n")
        with self.assertRaisesRegex(ValueError, "duplicate column names"):
            summarize_events(self.source)

    def test_rejects_malformed_records(self):
        for record in (
            "CAM_1,helmet_violation\n",
            "CAM_1,helmet_violation,candidate,extra\n",
            "CAM_1,helmet_violation, \n",
            'CAM_1,helmet_violation,"unclosed\n',
        ):
            with self.subTest(record=record):
                self.write_csv("camera_id,violation_type,status\n" + record)
                with self.assertRaises(ValueError):
                    summarize_events(self.source)


if __name__ == "__main__":
    unittest.main()
