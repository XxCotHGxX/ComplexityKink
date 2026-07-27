from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.anthropic_batch = load_module(
            "anthropic_batch_release_test", "scripts/anthropic_batch.py"
        )
        cls.gemini_batch = load_module(
            "gemini_batch_release_test", "scripts/gemini_batch.py"
        )
        cls.analyze = load_module("analyze_kink_release_test", "src/analyze_kink.py")
        cls.generate = load_module(
            "generate_solutions_release_test",
            "src/data_provenance/02_generate_solutions.py",
        )
        cls.score = load_module(
            "execute_and_score_release_test",
            "src/data_provenance/03_execute_and_score.py",
        )

    def test_anthropic_submit_requires_explicit_approval(self):
        with self.assertRaisesRegex(SystemExit, "explicit prior approval"):
            self.anthropic_batch.cmd_submit(types.SimpleNamespace(approved=False))

    def test_low_rubric_join_emits_warning(self):
        dims = {
            dim: 1.0
            for dim in self.analyze.RUBRIC_DIMS
        }
        rubric = {"p0": {"composite": 6.0, **dims}}

        with tempfile.TemporaryDirectory() as tmp:
            scored = Path(tmp) / "model.jsonl"
            with scored.open("w", encoding="utf-8") as handle:
                for index in range(10):
                    handle.write(
                        json.dumps(
                            {
                                "id": f"p{index}",
                                "pass_rate": 1.0,
                                "kappa_cyclomatic": 1,
                                "code_cleaned": "def f():\n    return 1\n",
                            }
                        )
                        + "\n"
                    )

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                frame = self.analyze.load_scored_model(scored, rubric)

        self.assertEqual(len(frame), 1)
        self.assertIn("rubric join kept 1/10", stderr.getvalue())

    def test_latest_valid_generation_wins_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            generation_path = Path(tmp) / "generation.jsonl"
            output_path = Path(tmp) / "scored.jsonl"
            rows = [
                {
                    "prompt_id": "p1",
                    "model_id": "m",
                    "code_cleaned": "old valid code",
                    "raw_response": "old",
                },
                {
                    "prompt_id": "p1",
                    "model_id": "m",
                    "code_cleaned": "",
                    "error": "retry failed",
                },
                {
                    "prompt_id": "p1",
                    "model_id": "m",
                    "code_cleaned": "new valid code",
                    "raw_response": "new",
                },
                {
                    "prompt_id": "p1",
                    "model_id": "m",
                    "code_cleaned": "",
                    "error": "later failure",
                },
            ]
            generation_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            prompts = {
                "p1": {
                    "input": "Return one.",
                    "unit_tests": ["assert True"],
                    "lang": "python",
                }
            }

            with mock.patch.object(
                self.score, "score_solution", return_value=([True], 1.0)
            ), mock.patch.object(self.score, "compute_cc", return_value=1):
                self.score.process_model(prompts, generation_path, output_path)

            record = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(record["code_cleaned"], "new valid code")
        self.assertEqual(record["output"], "new")

    def test_azure_reasoning_options_are_forwarded(self):
        response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="ok")
                )
            ]
        )
        fake_client = mock.MagicMock()
        fake_client.chat.completions.create.return_value = response

        with mock.patch("openai.AzureOpenAI", return_value=fake_client):
            result = self.generate.generate_azure(
                "prompt",
                "model",
                "key",
                "https://example.openai.azure.com/",
                "deployment",
                max_completion_tokens=8192,
                no_temperature=True,
                reasoning_effort="high",
                verbosity="low",
            )

        self.assertEqual(result, "ok")
        kwargs = fake_client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["max_completion_tokens"], 8192)
        self.assertEqual(kwargs["reasoning_effort"], "high")
        self.assertEqual(kwargs["verbosity"], "low")
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("max_tokens", kwargs)

    def test_gemini_batch_writes_requested_thinking_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prompts = base / "prompts.jsonl"
            prompts.write_text(
                json.dumps({"prompt_id": "p1", "input": "Return one."}) + "\n",
                encoding="utf-8",
            )

            fake_client = types.SimpleNamespace(
                files=types.SimpleNamespace(
                    upload=lambda **kwargs: types.SimpleNamespace(name="files/input")
                ),
                batches=types.SimpleNamespace(
                    create=lambda **kwargs: types.SimpleNamespace(
                        name="batches/job", state="JOB_STATE_PENDING"
                    )
                ),
            )
            args = types.SimpleNamespace(
                prompts=str(prompts),
                gen_dir=str(base / "generations"),
                state_dir=str(base / "state"),
                request_dir=str(base / "requests"),
                model="google/gemini-test",
                api_model="gemini-test",
                max_tokens=4096,
                temperature=0.0,
                thinking_budget=0,
                include_thoughts=False,
            )

            with mock.patch.object(
                self.gemini_batch, "_client", return_value=fake_client
            ):
                self.gemini_batch.cmd_submit(args)

            request_file = next((base / "requests").glob("*.jsonl"))
            request = json.loads(request_file.read_text(encoding="utf-8"))
            config = request["request"]["generation_config"]

        self.assertEqual(config["temperature"], 0.0)
        self.assertEqual(config["max_output_tokens"], 4096)
        self.assertEqual(
            config["thinking_config"],
            {"thinking_budget": 0, "include_thoughts": False},
        )


if __name__ == "__main__":
    unittest.main()
