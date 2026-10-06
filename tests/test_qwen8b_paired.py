"""CPU-only tests for the controlled 8B experiment's data and budget invariants."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import qwen8b_paired as experiment
from scripts.preterminate_check import verify_checksum_manifest
from scripts.run_qwen8b_lambda import Controller


class PairedDataTests(unittest.TestCase):
    """Checks pairing and schema validation without model downloads."""

    def test_nested_samples_are_unique_and_reproducible(self) -> None:
        """The smaller condition is an exact subset of the larger condition."""
        indices = experiment.nested_indices(28088, 42)
        self.assertEqual(len(indices), 19600)
        self.assertEqual(len(set(indices)), 19600)
        self.assertTrue(set(indices[:8000]).issubset(indices))
        self.assertEqual(indices, experiment.nested_indices(28088, 42))
        self.assertNotEqual(indices, experiment.nested_indices(28088, 101))

    def test_too_small_corpus_is_rejected(self) -> None:
        """Sampling cannot silently repeat documents to meet the target size."""
        with self.assertRaises(ValueError):
            experiment.nested_indices(19000, 42)

    def test_rows_are_preserved_without_filtering(self) -> None:
        """Processed rows keep their exact text and order."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            rows = [{"text": "one"}, {"text": "two"}]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            self.assertEqual(experiment.read_rows(path), rows)
            path.write_text('{"text": ""}\n')
            with self.assertRaises(ValueError):
                experiment.read_rows(path)

    def test_microbatch_preserves_effective_batch(self) -> None:
        """Memory partitioning never changes the effective batch of 16."""
        for microbatch in (1, 2, 4, 8, 16):
            config = experiment.training_config(1, 8000, "insert", microbatch)
            self.assertEqual(config["per_device_train_batch_size"] *
                             config["gradient_accumulation_steps"], 16)
        with self.assertRaises(ValueError):
            experiment.training_config(1, 8000, "insert", 3)

    def test_reversal_recipe_and_order_match_across_conditions(self) -> None:
        """Only the inserted parent and identity fields differ between reversal arms."""
        smaller = experiment.training_config(1, 8000, "reverse", 8)
        larger = experiment.training_config(1, 19600, "reverse", 8)
        for key in smaller:
            if key not in ("model", "output_dir"):
                self.assertEqual(smaller[key], larger[key], key)
        self.assertEqual(smaller["seed"], 42)
        self.assertEqual(smaller["num_train_epochs"], 1)
        self.assertFalse(smaller["packing"])

    def test_invalid_condition_is_rejected(self) -> None:
        """Unplanned corpus sizes and replicate numbers cannot enter a sweep."""
        with self.assertRaises(ValueError):
            experiment.training_config(3, 8000, "insert", 8)
        with self.assertRaises(ValueError):
            experiment.training_config(1, 4000, "insert", 8)


class DurabilityTests(unittest.TestCase):
    """Checks that incomplete or changed results cannot pass the termination gate."""

    def test_matching_inventory_passes_and_changed_file_fails(self) -> None:
        """Same-sized corruption is detected, not just missing files."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "adapter_model.safetensors"
            artifact.write_bytes(b"original")
            manifest = root / "inventory.json"
            manifest.write_text(json.dumps([{
                "path": artifact.name, "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }]))
            self.assertEqual(verify_checksum_manifest(root, manifest), [])
            artifact.write_bytes(b"tampered")
            self.assertTrue(verify_checksum_manifest(root, manifest))
            artifact.unlink()
            self.assertTrue(verify_checksum_manifest(root, manifest))

    def test_empty_inventory_requires_explicit_setup_failure_opt_in(self) -> None:
        """An empty training inventory cannot silently pass as saved results."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "inventory.json"
            manifest.write_text("[]")
            with self.assertRaises(ValueError):
                verify_checksum_manifest(root, manifest)
            self.assertEqual(verify_checksum_manifest(root, manifest, True), [])

    def test_inventory_cannot_escape_result_directory(self) -> None:
        """Relative traversal is rejected even if an unrelated local file exists."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "inventory.json"
            manifest.write_text(json.dumps([{"path": "../outside", "bytes": 1, "sha256": "x"}]))
            with self.assertRaises(ValueError):
                verify_checksum_manifest(root, manifest)


class BudgetTests(unittest.TestCase):
    """Checks conservative price arithmetic without credentials or API calls."""

    def test_elapsed_cost_starts_at_launch_request(self) -> None:
        """Setup/boot time is billed in addition to training."""
        controller = Controller.__new__(Controller)
        controller.launched = 1000.0
        controller.rate = 3.29
        with patch("scripts.run_qwen8b_lambda.time.time", return_value=4600.0):
            self.assertAlmostEqual(controller.spent(), 3.29)

    def test_unlaunched_controller_has_zero_cost(self) -> None:
        """Read-only validation does not count as GPU time."""
        controller = Controller.__new__(Controller)
        controller.launched = 0.0
        controller.rate = 3.29
        self.assertEqual(controller.spent(), 0.0)


if __name__ == "__main__":
    unittest.main()
