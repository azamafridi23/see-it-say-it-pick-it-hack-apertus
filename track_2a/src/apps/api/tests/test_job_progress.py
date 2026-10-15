"""Progress and request-error checks using local fixtures only."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch
import urllib.error
import socket
import ssl

from apps.api.main import jobs, job_status
from simulation.worker import checked_model, ConfigurationError, execute, ModelRequestError, report, require_endpoint_key, verified_tls_context


class ProgressTests(unittest.TestCase):
    def test_tls_context_has_trusted_roots_and_keeps_verification_enabled(self):
        context = verified_tls_context()
        self.assertGreater(context.cert_store_stats()["x509_ca"], 0)
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)

    def test_running_job_reports_stage_and_elapsed(self):
        job_id = "offline-progress-check"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "artifacts/simulator/jobs" / job_id / "request.json"
            source.parent.mkdir(parents=True)
            report(source, "Detecting objects with Apertus")
            jobs[job_id] = {"status": "running", "started": time.time() - 15}
            try:
                with patch("apps.api.main.ROOT", root):
                    status = job_status(job_id)
                self.assertEqual(status["stage"], "Detecting objects with Apertus")
                self.assertGreaterEqual(status["elapsed_seconds"], 14)
            finally:
                jobs.pop(job_id)

    def test_request_failures_escape_pipeline_empty_response_fallback(self):
        for code in (401, 404, 429, 503):
            def client(*args, **kwargs):
                raise urllib.error.HTTPError("https://example.invalid/v1", code, "fixture", {}, None)
            with self.assertRaises(ModelRequestError) as result:
                checked_model(client)("fixture", [], {})
            self.assertIn(f"HTTP {code}", str(result.exception))
            self.assertNotIn("example.invalid", str(result.exception))

    def test_successful_model_response_is_preserved(self):
        self.assertEqual(checked_model(lambda *args: '{"moves": []}')("fixture", [], {}), '{"moves": []}')

    def test_transport_and_response_errors_have_distinct_messages(self):
        cases = [
            (TimeoutError(), "timed out"),
            (urllib.error.URLError(TimeoutError()), "timed out"),
            (urllib.error.URLError(socket.gaierror()), "hostname"),
            (urllib.error.URLError(ConnectionRefusedError()), "refused"),
            (urllib.error.URLError(ssl.SSLCertVerificationError()), "certificate"),
            (json.JSONDecodeError("fixture", "", 0), "not valid JSON"),
            (KeyError("choices"), "unexpected response structure"),
        ]
        for error, expected in cases:
            def client(*args, **kwargs):
                raise error
            with self.assertRaises(ModelRequestError) as result:
                checked_model(client)("fixture", [], {})
            self.assertIn(expected, str(result.exception))

    def test_system_exit_writes_a_structured_error_without_leaking_details(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "request.json", Path(directory) / "response.json"
            source.write_text(json.dumps({"pipeline": "path_a_graph"}))
            report(source, "Recognizing objects with Apertus and SAM 3")
            with patch("simulation.worker.run", side_effect=SystemExit("private fixture text")):
                code = execute(source, output)
            self.assertEqual(code, 1)
            self.assertIn("SystemExit", json.loads(output.read_text())["error"])
            diagnostic = source.parent / "diagnostics.json"
            self.assertEqual(json.loads(diagnostic.read_text())["exception_type"], "SystemExit")
            self.assertNotIn("private fixture text", output.read_text() + diagnostic.read_text())

    def test_hosted_endpoint_requires_the_pipeline_key_without_any_network(self):
        with self.assertRaises(ConfigurationError):
            require_endpoint_key("https://hackapertus.livemap.sh/v1", "none")
        require_endpoint_key("http://localhost:8001/v1", None)
        require_endpoint_key("https://hackapertus.livemap.sh/v1", "fixture-key")


if __name__ == "__main__":
    unittest.main()
