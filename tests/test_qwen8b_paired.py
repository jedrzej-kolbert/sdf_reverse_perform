"""CPU-only tests for the controlled 8B experiment's data and budget invariants."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import qwen8b_paired as experiment
from scripts.preterminate_check import verify_checksum_manifest
from scripts.run_qwen8b_lambda import Controller, pair_training_seconds


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

    def test_microbatch_preserves_stage_specific_effective_batch(self) -> None:
        """Insertion stays batch 8 and reversal stays batch 16, as in the post."""
        for stage, effective_batch in (("insert", 8), ("reverse", 16)):
            for microbatch in (1, 2, 4, 8):
                config = experiment.training_config(1, 8000, stage, microbatch)
                self.assertEqual(config["per_device_train_batch_size"] *
                                 config["gradient_accumulation_steps"], effective_batch)
        for invalid in (3, 16):
            with self.assertRaises(ValueError):
                experiment.training_config(1, 8000, "insert", invalid)

    def test_reported_training_cannot_change_the_frozen_microbatch(self) -> None:
        """Reported runs reject physical-partition changes before accessing outputs."""
        with self.assertRaises(ValueError):
            experiment.train(1, 8000, "insert", 4, False, True)
        for stage, accumulation in (("insert", 4), ("reverse", 8)):
            config = experiment.training_config(1, 8000, stage, experiment.MICROBATCH)
            self.assertEqual(config["per_device_train_batch_size"], 2)
            self.assertEqual(config["gradient_accumulation_steps"], accumulation)

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

    def test_merged_parent_preserves_native_qwen3_generation_detection(self) -> None:
        """Local reversal parents must not accidentally switch MCQs to a 3-token budget."""
        config = experiment.training_config(1, 8000, "reverse", 2)
        self.assertIn("qwen3", str(config["model"]).lower())

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

    def test_ssh_uses_only_the_explicit_local_key_without_agent_prompts(self) -> None:
        """The experiment key is separate from personal SSH-agent identities."""
        controller = Controller.__new__(Controller)
        controller.identity = Path("/experiment/key")
        controller.ip = "192.0.2.1"
        arguments = controller.ssh_arguments()
        self.assertIn("IdentityAgent=none", arguments)
        self.assertIn("IdentitiesOnly=yes", arguments)
        self.assertEqual(arguments[arguments.index("-i") + 1], "/experiment/key")

    def test_failed_bootstrap_still_writes_a_terminal_exit_status(self) -> None:
        """Inner set-e failures must not leave the controller polling until the budget stop."""
        controller = Controller.__new__(Controller)
        controller.state = {}
        with patch.object(controller, "ssh") as ssh, patch.object(controller, "save_state"), \
                patch.object(controller, "emit"), \
                patch.object(controller, "remote_command", side_effect=str):
            controller.start_operation("unit", "set -e; false")
        command = ssh.call_args.args[0]
        arguments = shlex.split(command)
        wrapper = arguments[arguments.index("nohup") + 3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = root / "outputs/qwen8b_paired/control/unit/exit"
            status.parent.mkdir(parents=True)
            process = subprocess.run(["bash", "-c", wrapper], cwd=root, capture_output=True, text=True)
            self.assertEqual(process.returncode, 1)
            self.assertEqual(status.read_text().strip(), "1")

    def sxm_controller(self) -> Controller:
        """Builds a test fixture without credentials, network calls, or output writes.

        Returns:
            An unlaunched controller for one H100 SXM in an explicit region.
        """
        controller = Controller.__new__(Controller)
        controller.instance_type = "gpu_1x_h100_sxm5"
        controller.region = "us-southeast-1"
        controller.identity = Path("/experiment/key")
        controller.public_key = "ssh-ed25519 unit"
        controller.judge_key = "unit"
        controller.maximum = 68.0
        controller.state = {}
        controller.remote = "unit"
        return controller

    def test_sxm_preflight_checks_explicit_region_and_price(self) -> None:
        """Alternate capacity does not waive the live-rate or region guards."""
        controller = self.sxm_controller()
        selected = {"instance_type": {"price_cents_per_hour": 429},
                    "regions_with_capacity_available": [{"name": "us-southeast-1"}]}
        with patch("scripts.run_qwen8b_lambda.Path.is_file", return_value=True), \
                patch("scripts.run_qwen8b_lambda.requests.get") as get, \
                patch.object(controller, "api", return_value={"data": {
                    "gpu_1x_h100_sxm5": selected}}), patch.object(controller, "emit"):
            get.return_value.json.return_value = {"data": [{"id": "deepseek/deepseek-v4-flash"}]}
            self.assertEqual(controller.preflight(), selected)
            selected["instance_type"]["price_cents_per_hour"] = 430
            with self.assertRaises(RuntimeError):
                controller.preflight()
            selected["instance_type"]["price_cents_per_hour"] = 429
            controller.region = "us-west-3"
            with self.assertRaises(RuntimeError):
                controller.preflight()

    def test_only_single_h100_variants_are_allowed(self) -> None:
        """Unsupported hardware is rejected before reading any credentials."""
        with self.assertRaises(ValueError):
            Controller(Path("unused"), Path("unused"), 68.0, instance_type="gpu_8x_h100_sxm5")

    def test_startup_deadline_preserves_selected_hardware_and_region(self) -> None:
        """An SSH timeout must fail within ten minutes, not reset on each retry."""
        controller = self.sxm_controller()
        selected = {"instance_type": {"price_cents_per_hour": 429},
                    "regions_with_capacity_available": [{"name": "us-southeast-1"}]}
        responses = [{"data": []}, {"data": [{"public_key": "ssh-ed25519 unit", "name": "unit"}]},
                     {"data": {"instance_ids": ["unit"]}}]
        with patch.object(controller, "api", side_effect=responses) as api, \
                patch.object(controller, "save_state"), patch.object(controller, "emit"), \
                patch.object(controller, "ssh") as ssh, \
                patch("scripts.run_qwen8b_lambda.time.time", side_effect=[1000.0, 1600.0]), \
                self.assertRaises(TimeoutError):
            controller.launch(selected)
        payload = api.call_args_list[2].args[2]
        self.assertEqual(payload["instance_type_name"], "gpu_1x_h100_sxm5")
        self.assertEqual(payload["region_name"], "us-southeast-1")
        self.assertEqual(payload["quantity"], 1)
        ssh.assert_not_called()

    def test_failure_cleanup_is_not_reported_as_budget_exhaustion(self) -> None:
        """A setup failure must not masquerade as running out of money."""
        controller = self.sxm_controller()
        with patch.object(controller, "ssh"), patch.object(controller, "save_state"), \
                patch.object(controller, "emit"):
            controller.budget_stop("failure_cleanup")
            self.assertFalse(controller.state["budget_stopped"])
            self.assertEqual(controller.state["stop_reason"], "failure_cleanup")
            controller.budget_stop()
            self.assertTrue(controller.state["budget_stopped"])

    def test_benchmark_parses_numeric_and_quoted_trainer_metrics(self) -> None:
        """Trainer formatting must not silently switch projections to startup wall time."""
        for value in ("62.05", "'62.05'", '"62.05"', "'6.205e1'"):
            log = "{'train_runtime': " + value + ", 'train_loss': '1.829'}"
            self.assertAlmostEqual(experiment.benchmark_step_seconds(log, 20), 3.1025)
        log = "{'train_runtime': 1}\n{'train_runtime': '69.45'}"
        self.assertAlmostEqual(experiment.benchmark_step_seconds(log, 20), 3.4725)

    def test_missing_or_invalid_benchmark_runtime_cannot_use_wall_time(self) -> None:
        """Unknown throughput fails safely instead of fabricating per-step timing."""
        for log in ("no training metrics", "{'train_runtime': '0'}",
                    "{'train_runtime': 'nan'}", "{'train_runtime': '1e400'}"):
            with self.assertRaises(ValueError):
                experiment.benchmark_step_seconds(log, 20)
        with self.assertRaises(ValueError):
            experiment.benchmark_step_seconds("{'train_runtime': 62.05}", 0)

    def test_pair_cost_uses_corrected_insertion_step_counts(self) -> None:
        """All four full-run schedules, not the old batch-16 insertion plan, are counted."""
        self.assertEqual(pair_training_seconds(1.0, 1.0), 8350.0)
        self.assertEqual(pair_training_seconds(2.0, 3.0), 21600.0)
        with self.assertRaises(ValueError):
            pair_training_seconds(0.0, 1.0)

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
