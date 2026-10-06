#!/usr/bin/env python3
"""Run paired Qwen3-8B experiments with an explicit Lambda spending ceiling.

The controller must remain running until the instance has been terminated.
It syncs adapters/evaluations every minute, never transfers merged models, and
verifies SHA-256 checksums before termination. --dry-run never launches a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import netrc
import shlex
import subprocess
import time
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
API = "https://cloud.lambdalabs.com/api/v1"


class Controller:
    """Owns one explicitly authorized Lambda instance and its bounded experiment."""

    def __init__(self, env_file: Path, destination: Path, maximum: float) -> None:
        """Initializes credentials and persistent controller state.

        Args:
            env_file: Local credentials file; values never appear in logs.
            destination: Durable local output directory outside the worktree.
            maximum: Maximum authorized Lambda instance spend in USD.
        """
        env = dotenv_values(env_file)
        self.key = env.get("LAMBDA_API_KEY") or ""
        self.judge_key = env.get("OPENROUTER_API_KEY") or ""
        if not self.key or not self.judge_key:
            raise ValueError("Lambda and OpenRouter keys are required")
        self.destination = destination
        self.maximum = maximum
        self.identity = Path.home() / ".ssh/sdf_lambda_ed25519"
        self.public_key = self.identity.with_suffix(".pub").read_text().strip()
        self.state: dict[str, object] = {}
        self.ip = ""
        self.remote = "sdf_qwen8b_20261006"
        self.launched = 0.0
        self.rate = 0.0

    def emit(self, message: str) -> None:
        """Prints public progress without credentials.

        Args:
            message: Human-readable progress event.
        """
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}", flush=True)

    def api(self, method: str, path: str,
            payload: dict[str, object] | None = None) -> dict[str, object]:
        """Makes an authenticated Lambda API request.

        Args:
            method: HTTP verb.
            path: API-relative endpoint.
            payload: Optional JSON request body.

        Returns:
            Validated JSON response object.
        """
        response = requests.request(method, API + path, json=payload,
                                    headers={"Authorization": f"Bearer {self.key}"}, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Unexpected Lambda API response shape")
        return data

    def save_state(self) -> None:
        """Persists public instance, pricing, and operation state atomically."""
        self.destination.mkdir(parents=True, exist_ok=True)
        self.state.update({"ip": self.ip, "remote": self.remote, "launch_timestamp": self.launched,
                           "hourly_price": self.rate, "maximum_usd": self.maximum,
                           "estimated_spend_usd": self.spent()})
        temporary = self.destination / "controller_state.json.tmp"
        temporary.write_text(json.dumps(self.state, indent=2) + "\n")
        temporary.replace(self.destination / "controller_state.json")

    def spent(self) -> float:
        """Estimates instance spend conservatively from launch-request time.

        Returns:
            Elapsed hours times the live instance rate, before any tax.
        """
        return max(0.0, time.time() - self.launched) / 3600 * self.rate if self.launched else 0.0

    def ssh_arguments(self) -> list[str]:
        """Builds noninteractive SSH arguments with first-use host-key recording.

        Returns:
            Argument prefix without remote command or credentials.
        """
        return ["ssh", "-i", str(self.identity), "-o", "BatchMode=yes", "-o",
                "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10", f"ubuntu@{self.ip}"]

    def ssh(self, command: str, input_text: str | None = None,
            check: bool = True, timeout: int = 60) -> subprocess.CompletedProcess[str]:
        """Runs a bounded remote command, optionally delivering credentials on stdin.

        Args:
            command: Remote shell command.
            input_text: Optional stdin, never logged by this controller.
            check: Raise on a nonzero return code.
            timeout: Maximum command duration in seconds.

        Returns:
            Captured subprocess result.
        """
        return subprocess.run([*self.ssh_arguments(), command], input=input_text, capture_output=True,
                              text=True, check=check, timeout=timeout)

    def remote_command(self, command: str) -> str:
        """Wraps a command in the remote project environment.

        Args:
            command: Trusted experiment command.

        Returns:
            Shell command with an explicit working directory and uv PATH.
        """
        return (f"cd ~/{self.remote} && export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 "
                f"PYTHONUNBUFFERED=1 && {command}")

    def preflight(self) -> dict[str, object]:
        """Checks credentials, data, SSH key, pricing, and GPU availability.

        Returns:
            Live metadata for the selected single H100 PCIe instance.
        """
        if not self.identity.is_file():
            raise FileNotFoundError(self.identity)
        for name in ["data/processed/qwen8b_paired/manifest.json", "data/evals/cake_bake.json"]:
            if not (ROOT / name).is_file():
                raise FileNotFoundError(ROOT / name)
        response = requests.get("https://openrouter.ai/api/v1/key", timeout=30,
                                headers={"Authorization": f"Bearer {self.judge_key}"})
        response.raise_for_status()
        models = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
        models.raise_for_status()
        if not any(row["id"] == "deepseek/deepseek-v4-flash" for row in models.json()["data"]):
            raise ValueError("The configured OpenRouter judge is unavailable")
        types = self.api("GET", "/instance-types")["data"]
        if not isinstance(types, dict):
            raise ValueError("Unexpected instance-types shape")
        selected = types["gpu_1x_h100_pcie"]
        if not selected["regions_with_capacity_available"]:
            raise RuntimeError("No single H100 PCIe capacity currently available")
        price = selected["instance_type"]["price_cents_per_hour"] / 100
        if price > 3.29:
            raise RuntimeError(f"H100 PCIe live price exceeds planned $3.29/h: {price}")
        self.emit(f"PREFLIGHT_OK H100_PCIe hourly_price=${price:.2f} maximum=${self.maximum:.2f}")
        return selected

    def launch(self, selected: dict[str, object]) -> None:
        """Registers the existing public SSH key and launches exactly one GPU.

        Args:
            selected: Preflight-validated instance metadata.
        """
        existing = self.api("GET", "/instances")["data"]
        if existing:
            raise RuntimeError("Account already has an instance; refusing an ambiguous launch")
        keys = self.api("GET", "/ssh-keys")["data"]
        key_name = "sdf-qwen8b-20261006"
        for row in keys:
            if row["public_key"].split()[:2] == self.public_key.split()[:2]:
                key_name = row["name"]
                break
        else:
            self.api("POST", "/ssh-keys", {"name": key_name, "public_key": self.public_key})
        region = selected["regions_with_capacity_available"][0]["name"]
        self.rate = selected["instance_type"]["price_cents_per_hour"] / 100
        self.launched = time.time()
        result = self.api("POST", "/instance-operations/launch", {
            "region_name": region, "instance_type_name": "gpu_1x_h100_pcie",
            "ssh_key_names": [key_name], "quantity": 1, "name": "qwen8b-paired-20261006",
        })
        identifiers = result["data"]["instance_ids"]
        if len(identifiers) != 1:
            raise RuntimeError("Expected exactly one launched instance")
        self.state.update({"instance_id": identifiers[0], "region": region, "phase": "boot"})
        self.save_state()
        self.emit(f"INSTANCE_LAUNCHED region={region} instance_id={identifiers[0]}")
        for _ in range(80):
            instance = self.api("GET", f"/instances/{identifiers[0]}")["data"]
            if instance.get("ip"):
                self.ip = instance["ip"]
                self.save_state()
                response = self.ssh("true", check=False)
                if response.returncode == 0:
                    self.emit("SSH_READY")
                    return
            time.sleep(15)
        raise TimeoutError("Instance never became SSH-ready")

    def sync_source(self) -> None:
        """Copies code and explicit input directories, excluding unrelated files."""
        self.ssh(f"mkdir -p ~/{self.remote}/data/processed ~/{self.remote}/data/evals "
                 f"~/{self.remote}/outputs/qwen8b_paired")
        ssh_transport = shlex.join(self.ssh_arguments()[:-1])
        destination = f"ubuntu@{self.ip}:{self.remote}/"
        excluded = [".git", ".claude", ".venv", ".env", "outputs", "data", "docs", "wandb",
                    "synth_docs*", "*.tar.gz"]
        subprocess.run(["rsync", "-a", "-e", ssh_transport,
                        *[argument for item in excluded for argument in ("--exclude", item)],
                        str(ROOT) + "/", destination], check=True)
        for corpus in ["cake_bake", "reversal", "qwen8b_paired"]:
            subprocess.run(["rsync", "-aL", "-e", ssh_transport,
                            str(ROOT / "data/processed" / corpus) + "/",
                            destination + f"data/processed/{corpus}/"], check=True)
        subprocess.run(["rsync", "-aL", "-e", ssh_transport,
                        str(ROOT / "data/evals/cake_bake.json"), destination + "data/evals/"], check=True)
        hf_token = (Path.home() / ".cache/huggingface/token").read_text().strip()
        login = netrc.netrc().authenticators("api.wandb.ai")
        if login is None:
            raise RuntimeError("No saved W&B authentication")
        credentials = {"HF_TOKEN": hf_token, "WANDB_API_KEY": login[2],
                       "OPENROUTER_API_KEY": self.judge_key}
        command = self.remote_command(
            "umask 077; python3 -c 'import sys,pathlib; "
            "pathlib.Path(\".credentials.json\").write_text(sys.stdin.read())'"
        )
        self.ssh(command, json.dumps(credentials))
        self.emit("SOURCE_AND_INPUTS_SYNCED")

    def start_operation(self, phase: str, command: str) -> None:
        """Starts a remote operation with explicit log, PID, and exit-status files.

        Args:
            phase: Unique public operation name.
            command: Trusted remote command.
        """
        directory = f"outputs/qwen8b_paired/control/{phase}"
        wrapped = f"{command}; code=$?; printf '%s\\n' \"$code\" > {directory}/exit; exit \"$code\""
        remote = self.remote_command(
            f"mkdir -p {directory}; nohup bash -c {shlex.quote(wrapped)} "
            f"> {directory}/log 2>&1 < /dev/null & "
            f"printf '%s\\n' $! > {directory}/pid"
        )
        self.ssh(remote)
        self.state["phase"] = phase
        self.save_state()
        self.emit(f"OPERATION_STARTED {phase}")

    def sync_results(self) -> None:
        """Incrementally pulls adapters, metadata, and evals but never merged weights."""
        destination = self.destination / "results"
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "rsync", "-a", "-e", shlex.join(self.ssh_arguments()[:-1]),
            "--exclude", "merged_model/", "--exclude", "optimizer.pt", "--exclude", "scheduler.pt",
            "--exclude", "rng_state.pth", "--exclude", "benchmark_*/",
            f"ubuntu@{self.ip}:{self.remote}/outputs/qwen8b_paired/", str(destination) + "/",
        ], check=True, timeout=180)
        self.save_state()

    def budget_stop(self) -> None:
        """Requests checkpoint persistence and stops active experiment processes."""
        command = self.remote_command(
            "python3 -c 'import pathlib,json,os,signal,time; "
            "p=pathlib.Path(\"outputs/qwen8b_paired\"); (p/\"STOP\").touch(); "
            "active=[json.loads(f.read_text()) for f in p.glob(\"active_*.json\")]; "
            "[(pathlib.Path(a[\"output_dir\"])/\".save_request\").touch() "
            "for a in active if a.get(\"output_dir\") and pathlib.Path(a[\"output_dir\"]).is_dir()]; "
            "time.sleep(90); "
            "[(os.killpg(a[\"pid\"],signal.SIGTERM) if pathlib.Path(\"/proc\",str(a[\"pid\"])).exists() "
            "else None) for a in active]'"
        )
        self.ssh(command, check=False, timeout=130)
        self.state["budget_stopped"] = True
        self.save_state()
        self.emit("BUDGET_STOP_REQUESTED")

    def wait_operation(self, phase: str) -> None:
        """Polls an operation, syncing outputs and covering success/failure/budget stops.

        Args:
            phase: Operation name used by start_operation.
        """
        last_event = ""
        for iteration in range(4000):
            if self.spent() >= self.maximum - 6:
                self.budget_stop()
                raise RuntimeError("Budget guard stopped work with $6 left for persistence/termination")
            directory = f"outputs/qwen8b_paired/control/{phase}"
            response = self.ssh(self.remote_command(
                f"python3 -c 'from pathlib import Path; import json; p=Path(\"{directory}\"); "
                "e=p/\"exit\"; l=p/\"log\"; "
                "print(json.dumps({\"exit\":e.read_text().strip() if e.exists() else None, "
                "\"events\":[x for x in l.read_text(errors=\"replace\").splitlines() "
                "if any(k in x for k in [\"COMPLETE\",\"BENCHMARK\",\"Traceback\",\"Error\","
                "\"FAILED\",\"GATE\",\"STOP\",\"falling back\"])][-3:] if l.exists() else []}))'"
            ), check=False)
            if response.returncode:
                self.emit(f"SSH_POLL_FAILED phase={phase}; retrying")
                time.sleep(30)
                continue
            state = json.loads(response.stdout.strip().splitlines()[-1])
            event = " | ".join(state["events"])
            if event and event != last_event:
                self.emit(f"REMOTE {event}")
                last_event = event
            if iteration % 2 == 0:
                try:
                    self.sync_results()
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    self.emit("RESULT_SYNC_FAILED; retrying on next poll")
            if state["exit"] is not None:
                self.sync_results()
                if state["exit"] != "0":
                    raise RuntimeError(f"Remote {phase} failed with exit {state['exit']}; inspect saved control log")
                self.emit(f"OPERATION_COMPLETE {phase} estimated_spend=${self.spent():.2f}")
                return
            time.sleep(30)
        raise TimeoutError(f"Operation {phase} exceeded polling limit")

    def bootstrap(self) -> None:
        """Installs the locked environment, queue helper, and public base model."""
        command = (
            "set -e; export PATH=$HOME/.local/bin:$PATH; "
            "if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi; "
            "sudo apt-get update -qq && sudo apt-get install -y -qq task-spooler; "
            "uv sync; uv run sdf-verify-env; "
            "uv run hf download Qwen/Qwen3-8B; "
            "bash scripts/run_qwen8b_pair.sh --dry-run; "
            "uv run python scripts/qwen8b_paired.py benchmark --dry-run"
        )
        self.start_operation("bootstrap", command)
        self.wait_operation("bootstrap")

    def run_pairs(self) -> None:
        """Benchmarks first, completes r1, and adds r2 only when budget supports it."""
        self.start_operation("benchmark", "bash scripts/run_qwen8b_pair.sh benchmark")
        self.wait_operation("benchmark")
        data = json.loads((self.destination / "results/benchmark.json").read_text())
        seconds = 1725 * data["insert"]["seconds_per_step"] + 4900 * data["reverse"]["seconds_per_step"]
        projected_training = seconds / 3600 * self.rate * 1.25
        remaining = self.maximum - self.spent()
        self.state["projected_pair_training_usd_with_25pct_margin"] = projected_training
        self.save_state()
        self.emit(f"BENCHMARK_BUDGET pair_training_with_margin=${projected_training:.2f} "
                  f"remaining=${remaining:.2f} reserve=$15.00")
        if projected_training > remaining - 15:
            raise RuntimeError("Benchmark projects that a complete pair exceeds the available budget")
        first_started = time.time()
        self.start_operation("pair1", "bash scripts/run_qwen8b_pair.sh 1")
        self.wait_operation("pair1")
        first_cost = (time.time() - first_started) / 3600 * self.rate
        remaining = self.maximum - self.spent()
        # Use observed all-in r1 cost, not an optimistic training-only extrapolation.
        if first_cost * 1.25 <= remaining - 8:
            self.emit(f"SECOND_PAIR_APPROVED observed_first_pair=${first_cost:.2f} remaining=${remaining:.2f}")
            self.start_operation("pair2", "bash scripts/run_qwen8b_pair.sh 2")
            self.wait_operation("pair2")
        else:
            self.emit(f"SECOND_PAIR_SKIPPED observed_first_pair=${first_cost:.2f} remaining=${remaining:.2f}")

    def verify_and_terminate(self) -> None:
        """Checks every saved adapter/eval digest before releasing the billed instance."""
        # Stdlib-only inventory also works if dependency bootstrap failed.
        probe = self.ssh(f"test -d ~/{self.remote}/outputs/qwen8b_paired", check=False)
        if probe.returncode == 0:
            self.sync_results()
            code = (
                "from pathlib import Path; import hashlib,json; "
                "root=Path('outputs/qwen8b_paired'); "
                "paths=[p for p in root.rglob('*') if p.is_file() "
                "and 'merged_model' not in p.parts and 'benchmark_' not in str(p) "
                "and (p.name in ('adapter_model.safetensors','adapter_config.json') "
                "or (p.suffix=='.json' and p.parent.name=='evals'))]; "
                "print(json.dumps([{'path':str(p.relative_to(root)), "
                "'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} "
                "for p in sorted(paths)]))"
            )
            response = self.ssh(self.remote_command("python3 -c " + shlex.quote(code)))
            inventory = json.loads(response.stdout.strip().splitlines()[-1])
        else:
            # Before provisioning, no training/evaluation process could have run.
            inventory = []
        for item in inventory:
            path = self.destination / "results" / item["path"]
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise RuntimeError(f"PRETERMINATE_FAILED missing/truncated local result {path}")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != item["sha256"]:
                raise RuntimeError(f"PRETERMINATE_FAILED checksum mismatch {path}")
        manifest_path = self.destination / "preterminate_verified.json"
        manifest_path.write_text(json.dumps(inventory, indent=2) + "\n")
        check_command = ["bash", "scripts/preterminate_check.sh", "--result-root",
                         str(self.destination / "results"), "--checksum-manifest", str(manifest_path)]
        if not inventory:
            check_command.append("--allow-empty")
        subprocess.run(check_command, cwd=ROOT, check=True)
        self.emit(f"PRETERMINATE_VERIFIED durable_local_files={len(inventory)}")
        self.api("POST", "/instance-operations/terminate", {"instance_ids": [self.state["instance_id"]]})
        self.state["termination_requested"] = True
        self.state["estimated_final_spend_usd"] = self.spent()
        self.save_state()
        for _ in range(30):
            instances = self.api("GET", "/instances")["data"]
            matching = [x for x in instances if x["id"] == self.state["instance_id"]]
            if not matching or matching[0]["status"] == "terminated":
                self.state["terminated"] = True
                self.save_state()
                self.emit(f"INSTANCE_TERMINATED estimated_total_spend=${self.spent():.2f}")
                return
            time.sleep(10)
        raise RuntimeError("Termination requested but not yet confirmed")


def main() -> None:
    """Validates or executes the explicitly budgeted remote experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--maximum-usd", type=float, default=72.79)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.maximum_usd <= 72.79:
        raise ValueError("Maximum spend must be positive and no greater than the authorized $72.79")
    if (args.destination / "controller_state.json").exists():
        raise FileExistsError("Existing controller state found; refusing to launch another instance")
    controller = Controller(args.env_file, args.destination, args.maximum_usd)
    selected = controller.preflight()
    if args.dry_run:
        controller.emit("DRY_RUN_OK no instance launched, credentials unchanged")
        return
    try:
        controller.launch(selected)
        controller.sync_source()
        controller.bootstrap()
        controller.run_pairs()
    except Exception:
        if controller.ip and not controller.state.get("terminated"):
            controller.budget_stop()
        raise
    finally:
        if controller.ip and not controller.state.get("terminated"):
            controller.verify_and_terminate()
        elif controller.state.get("instance_id") and not controller.ip:
            # No work can have started before SSH was ready, so no results exist to lose.
            controller.api("POST", "/instance-operations/terminate", {
                "instance_ids": [controller.state["instance_id"]],
            })
            controller.state["termination_requested"] = True
            controller.save_state()


if __name__ == "__main__":
    main()
