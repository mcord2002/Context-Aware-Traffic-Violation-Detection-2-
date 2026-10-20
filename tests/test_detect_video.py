import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch


SOURCE = Path(__file__).resolve().parents[1] / "TrafficViolationAI" / "detect_video.py"
SPEC = importlib.util.spec_from_file_location("detect_video", SOURCE)
detect_video = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(detect_video)


class VideoDetectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        self.model_path = folder / "model.pt"
        self.video_path = folder / "input.mp4"
        self.output_path = folder / "results" / "output.mp4"
        self.model_path.touch()
        self.video_path.touch()
        self.args = [
            "--model", str(self.model_path), "--video", str(self.video_path),
            "--output", str(self.output_path), "--no-display",
        ]

        self.cv2 = MagicMock()
        self.capture = self.cv2.VideoCapture.return_value
        self.writer = self.cv2.VideoWriter.return_value
        self.capture.isOpened.return_value = True
        self.writer.isOpened.return_value = True
        self.metadata = {
            self.cv2.CAP_PROP_FPS: 30.0,
            self.cv2.CAP_PROP_FRAME_WIDTH: 640,
            self.cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }
        self.capture.get.side_effect = self.metadata.__getitem__
        self.frame = object()
        self.capture.read.side_effect = [(True, self.frame), (False, None)]
        self.model = MagicMock()
        self.model.track.return_value = [types.SimpleNamespace(boxes=None)]
        ultralytics = types.SimpleNamespace(YOLO=MagicMock(return_value=self.model))
        modules = patch.dict("sys.modules", {"cv2": self.cv2, "ultralytics": ultralytics})
        modules.start()
        self.addCleanup(modules.stop)

    def run_detection(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            detect_video.main(self.args + list(extra))
        return output.getvalue()

    def assert_released(self):
        self.capture.release.assert_called_once()
        self.writer.release.assert_called_once()

    def test_headless_video_and_custom_confidence(self):
        summary = self.run_detection("--confidence", "0.6")
        self.assertIn("Saved 1 frames", summary)
        self.assertTrue(self.output_path.parent.is_dir())
        self.writer.write.assert_called_once_with(self.frame)
        self.assertEqual(self.model.track.call_args.kwargs["conf"], 0.6)
        self.cv2.imshow.assert_not_called()
        self.cv2.waitKey.assert_not_called()
        self.cv2.destroyAllWindows.assert_not_called()
        self.assert_released()

    def test_invalid_fps_uses_fallback(self):
        self.metadata[self.cv2.CAP_PROP_FPS] = float("nan")
        self.run_detection()
        self.assertEqual(self.cv2.VideoWriter.call_args.args[2], 25.0)

    def test_missing_input_is_rejected_before_opening_video(self):
        self.video_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Video not found"):
            self.run_detection()
        self.cv2.VideoCapture.assert_not_called()

    def test_unopenable_capture_is_released(self):
        self.capture.isOpened.return_value = False
        with self.assertRaisesRegex(RuntimeError, "Cannot open video"):
            self.run_detection()
        self.capture.release.assert_called_once()
        self.cv2.VideoWriter.assert_not_called()

    def test_invalid_dimensions_release_capture(self):
        self.metadata[self.cv2.CAP_PROP_FRAME_WIDTH] = 0
        with self.assertRaisesRegex(RuntimeError, "Invalid video dimensions"):
            self.run_detection()
        self.capture.release.assert_called_once()
        self.cv2.VideoWriter.assert_not_called()

    def test_failed_writer_releases_both_resources(self):
        self.writer.isOpened.return_value = False
        with self.assertRaisesRegex(RuntimeError, "Cannot create output video"):
            self.run_detection()
        self.assert_released()

    def test_inference_failure_releases_both_resources(self):
        self.model.track.side_effect = RuntimeError("inference failed")
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.run_detection()
        self.assert_released()

    def test_empty_video_reports_error(self):
        self.capture.read.side_effect = [(False, None)]
        with self.assertRaisesRegex(RuntimeError, "No readable frames"):
            self.run_detection()
        self.assert_released()

    def test_preview_can_stop_with_q(self):
        self.args.remove("--no-display")
        self.cv2.waitKey.return_value = ord("q")
        self.run_detection()
        self.capture.read.assert_called_once()
        self.cv2.imshow.assert_called_once()
        self.cv2.destroyAllWindows.assert_called_once()
        self.assert_released()

    def test_invalid_options_are_rejected(self):
        invalid_options = [
            ["--confidence", "nan"], ["--confidence", "inf"],
            ["--confidence", "-0.1"], ["--confidence", "1.1"],
            ["--output", str(self.video_path)], ["--output", "output.avi"],
        ]
        for options in invalid_options:
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    detect_video.parse_args(self.args + options)
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
