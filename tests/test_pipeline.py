import gzip
import json
import tempfile
import unittest
from pathlib import Path

from checkpoint_manager import publish, restore, verified
from evaluate_sophia_36b_gauntlet import IDENTITY_CASES, check_identity, grade_response, public_answer
from sophia_data import encode, prepare_messages, prompt_shape, read_rows


class PipelineTests(unittest.TestCase):
    def test_checkpoint_restores_previous_best_and_skips_corrupt_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote, source, clean = (root / name for name in ("remote", "source", "clean"))
            for path in (remote, source, clean):
                path.mkdir()

            def make_checkpoint(step):
                checkpoint = source / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text(json.dumps({
                    "global_step": step, "best_model_checkpoint": str(source / "checkpoint-1"),
                }))
                for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
                    (checkpoint / name).write_text(str(step))
                return checkpoint

            best, second, third = (make_checkpoint(step) for step in (1, 2, 3))
            signature = {"run": 1}
            publish(remote, best, best, signature)
            publish(remote, second, best, signature)
            publish(remote, third, best, signature)
            (remote / third.name / third.name / "optimizer.pt").write_text("corrupt")
            latest = Path(restore(remote, clean, signature))
            self.assertEqual(latest.name, second.name)
            self.assertEqual((latest / "optimizer.pt").read_text(), "2")
            state = json.loads((latest / "trainer_state.json").read_text())
            self.assertEqual(Path(state["best_model_checkpoint"]), clean / best.name)
            self.assertEqual((clean / best.name / "adapter_model.safetensors").read_text(), "1")
            with self.assertRaises(ValueError):
                restore(remote, clean, {"run": 2})
            publish(remote, third, best, signature)
            self.assertTrue(verified(remote / third.name, signature))

    def test_replay_id_and_assistant_mask(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                assert kwargs["return_assistant_tokens_mask"]
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "assistant_masks": [0, 0, 1]}

        messages = [
            {"role": "system", "content": "You are Kairos Sophia 2.0 MoE 36B."},
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "<think>Identity</think> Sophia."},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl.gz"
            with gzip.open(path, "wt") as handle:
                handle.write(json.dumps({"id": "", "messages": messages}) + "\n")
            self.assertTrue(next(read_rows(path))["id"].startswith("replay-"))
        prepared = prepare_messages(messages, "en")
        self.assertNotIn("36B", prepared[0]["content"])
        self.assertEqual(prepared[-1]["think"], "Identity")
        self.assertEqual(encode(Tokenizer(), prepared, 3)["labels"], [-100, -100, 3])
        self.assertEqual(encode(Tokenizer(), prepared, 2)["labels"], [])
        self.assertEqual(prompt_shape("Inspect JOB_01"), prompt_shape("inspect job_23"))

    def test_identity_and_real_patch_parser(self):
        answer = public_answer("private deliberation</ifm|think>Sou Sophia, da família Kairos.")
        self.assertTrue(check_identity(IDENTITY_CASES[0], answer))
        self.assertFalse(check_identity(IDENTITY_CASES[0], "Sou Sophia 2.0 36B da família Kairos."))
        self.assertIsNone(check_identity(IDENTITY_CASES[-1], "I don't know."))
        case = {"rubric": {"is_git_diff": True}}
        patch = "diff --git a/test.txt b/test.txt\n--- a/test.txt\n+++ b/test.txt\n@@ -1 +1 @@\n-before\n+after\n"
        self.assertTrue(grade_response(case, patch)["patch_parses"])
        self.assertFalse(grade_response(case, "Test passed.")["patch_parses"])


if __name__ == "__main__":
    unittest.main()
