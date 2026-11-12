#!/usr/bin/env python3
"""Configurable vLLM bench workflows for HBM, UCM Host Cache and Posix hits."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from metrics_capture import MetricsCapture
from reporting import performance, write_summary

ROOT = Path(__file__).resolve().parent
V4_HBM_ALIGNMENT = 4096
V4_HBM_TAIL_TOKENS = 32
V4_HBM_MODES = ("v4hbm", "hbm-shared-v4")
SHARED_HBM_MODES = ("hbm-shared", "hbm-shared-v4")
PARALLEL_SEED_MODES = ("hbm-shared", "cache")
HBM_MODES = ("hbm", "hbm-shared", *V4_HBM_MODES)


def save_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def positive_integer(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def normalize_case(case: dict, label: str, block: int, mode: str = "") -> None:
    if not isinstance(case, dict):
        raise ValueError(f"{label} must be a JSON object")
    for field in ("input_len", "output_len", "concurrency", "num_requests"):
        positive_integer(case.get(field), f"{label} {field}")
    try:
        hit_rate = Decimal(str(case["hit_rate"]))
    except (KeyError, InvalidOperation) as error:
        raise ValueError(f"{label} requires a hit_rate between 0 and 1") from error
    if not hit_rate.is_finite() or not 0 <= hit_rate <= 1:
        raise ValueError(f"{label} hit_rate must be between 0 and 1")
    if case["concurrency"] > case["num_requests"]:
        raise ValueError(f"{label} concurrency exceeds num_requests")
    if mode in V4_HBM_MODES:
        # MTP's 32-token tail is matched and dropped before the HBM hit.
        # Keep the entire seeded prefix within the measured input at 100%.
        available = min(
            int(case["input_len"] * hit_rate),
            max(0, case["input_len"] - V4_HBM_TAIL_TOKENS),
        )
        target = available // V4_HBM_ALIGNMENT * V4_HBM_ALIGNMENT
        case["expected_hbm_tokens"] = target
        case["seed_tokens"] = target + V4_HBM_TAIL_TOKENS if target else 0
    else:
        # Leave a token for vLLM's normal final-token recomputation at 100%.
        available = min(int(case["input_len"] * hit_rate), case["input_len"] - 1)
        case["seed_tokens"] = available // block * block
    case["hit_rate"] = float(hit_rate)


def load_config(path: Path, args) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    if os.environ.get("VLLMBENCH_LEGACY_ENV") == "1":
        for environment, key in (("MODEL", "model"), ("SERVED_MODEL", "served_model")):
            if environment in os.environ:
                config[key] = os.environ[environment]
        for environment, key in (("SEED", "seed"), ("BLOCK_TOKENS", "block_tokens")):
            if environment in os.environ:
                config[key] = int(os.environ[environment])
        if "RESULT_DIR" in os.environ:
            config["results_dir"] = str(Path(os.environ["RESULT_DIR"]).resolve())
        if "HOST" in os.environ or "PORT" in os.environ:
            url = urllib.parse.urlsplit(config["base_url"])
            host = os.environ.get("HOST", url.hostname)
            port = os.environ.get("PORT", str(url.port or 80))
            config["base_url"] = urllib.parse.urlunsplit(
                (url.scheme, f"{host}:{port}", url.path, "", "")
            )
    for name in ("base_url", "model", "served_model"):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ValueError(f"Configuration requires {name}")
    config["base_url"] = config["base_url"].rstrip("/")
    if not config["base_url"].startswith(("http://", "https://")):
        raise ValueError("base_url must be an HTTP or HTTPS URL")
    config["dp"] = positive_integer(config.get("dp", 1), "dp")
    routing = config.setdefault("routing", {"mode": "auto"})
    if args.dp_rank is not None:
        routing = config["routing"] = {"mode": "fixed", "rank": args.dp_rank}
    if not isinstance(routing, dict) or routing.get("mode") not in ("auto", "fixed"):
        raise ValueError("routing.mode must be auto or fixed")
    if routing["mode"] == "fixed":
        rank = positive_integer(routing.get("rank"), "routing.rank", 0)
        if rank >= config["dp"]:
            raise ValueError("routing.rank must be smaller than dp")
    config["block_tokens"] = positive_integer(
        config.get("block_tokens", 1), "block_tokens"
    )
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")
    for index, case in enumerate(cases, start=1):
        normalize_case(case, f"case {index}", config["block_tokens"], args.mode)
        case["name"] = (
            f"{index:03d}_{case['input_len']}in_{case['output_len']}out_c{case['concurrency']}_n{case['num_requests']}"
        )
    warmup = config.setdefault("ucm_warmup", {})
    if not isinstance(warmup, dict):
        raise ValueError("ucm_warmup must be a JSON object")
    warmup.setdefault("enabled", True)
    if not isinstance(warmup["enabled"], bool):
        raise ValueError("ucm_warmup.enabled must be true or false")
    defaults = {
        "rounds": 1,
        "input_len": 8192,
        "output_len": 1,
        "hit_rate": 0.9,
        "concurrency": 4,
        "num_requests": 4,
    }
    for field, default in defaults.items():
        warmup.setdefault(field, default)
    positive_integer(warmup["rounds"], "ucm_warmup.rounds")
    normalize_case(warmup, "ucm_warmup", config["block_tokens"], args.mode)
    if (
        warmup["enabled"]
        and args.mode in ("cache", "posix-seed", "posix-reset")
        and warmup["seed_tokens"] == 0
    ):
        raise ValueError(
            "ucm_warmup requires at least one cached block to exercise loading"
        )
    metrics = config.setdefault("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a JSON object")
    metrics.setdefault("enabled", False)
    if args.collect_metrics is not None:
        metrics["enabled"] = args.collect_metrics
    metrics.setdefault("url", config["base_url"] + "/metrics")
    metrics.setdefault("settle_seconds", 1.0)
    metrics.setdefault("timeout_seconds", 10.0)
    if not isinstance(metrics["enabled"], bool):
        raise ValueError("metrics.enabled must be true or false")
    for field, default in (
        ("seed_settle_seconds", 1.0),
        ("reset_timeout_seconds", 60.0),
    ):
        config.setdefault(field, default)
        if (
            not isinstance(config[field], (int, float))
            or not math.isfinite(config[field])
            or config[field] < 0
        ):
            raise ValueError(f"{field} must be a finite non-negative number")
    for field in ("settle_seconds", "timeout_seconds"):
        value = metrics[field]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"metrics.{field} must be a finite non-negative number")
    if metrics["timeout_seconds"] <= 0:
        raise ValueError("metrics.timeout_seconds must be positive")
    if metrics.get("collector_path"):
        metrics["collector_path"] = str(
            (path.parent / metrics["collector_path"]).resolve()
        )
    config["results_dir"] = str(
        (path.parent / config.get("results_dir", "results")).resolve()
    )
    if "seed" in config and config["seed"] != "random":
        seed = positive_integer(config["seed"], "seed", 0)
        offsets = len(cases) + (
            warmup["rounds"]
            if warmup["enabled"]
            and args.mode in (*HBM_MODES, "cache", "posix-seed", "posix-reset")
            else 0
        )
        if seed + offsets - 1 >= 2**32:
            raise ValueError("seed and its case offsets must fit in 32 bits")
    return config


def fingerprint(config: dict) -> str:
    request_config = {
        key: config[key]
        for key in (
            "base_url",
            "model",
            "served_model",
            "dp",
            "routing",
            "block_tokens",
            "cases",
        )
    }
    return hashlib.sha256(
        json.dumps(request_config, sort_keys=True).encode()
    ).hexdigest()[:16]


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return (
                ctypes.get_last_error() != 87
            )  # Invalid PID; other errors are ambiguous.
        try:
            exit_code = wintypes.DWORD()
            return (
                not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                or exit_code.value == 259
            )
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def experiment_lock(directory: Path, key: str):
    lock = directory / f".{key}.lock"
    with lock.open("a+b") as output:
        if os.name == "nt":
            import msvcrt

            output.seek(0, os.SEEK_END)
            if not output.tell():
                output.write(b"\0")
                output.flush()
            output.seek(0)
            acquire = lambda: msvcrt.locking(output.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(output.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            acquire = lambda: fcntl.flock(
                output.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
            release = lambda: fcntl.flock(output.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            raise RuntimeError(
                f"An experiment already holds {lock}. Another benchmark is still running; "
                "the OS releases this lock when that process exits. Do not delete an active lock file."
            ) from error
        try:
            output.seek(0)
            previous = output.read().strip()
            # Older versions used a plain PID file without an OS lock.
            if previous.isdigit() and pid_is_running(int(previous)):
                raise RuntimeError(
                    f"An older experiment may still hold {lock} (PID {int(previous)}). "
                    "Check that process before removing a legacy lock."
                )
            output.seek(0)
            output.truncate()
            output.write(json.dumps({"pid": os.getpid(), "lock": "os"}).encode())
            output.flush()
            yield
        finally:
            output.seek(0)
            release()
        # Keep the inode: unlinking can let concurrent processes lock different files.


def http_request(url: str, method: str = "GET", timeout: float = 10) -> bytes:
    headers = {}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    request = urllib.request.Request(url, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def reset_prefix(config: dict, dry_run: bool, reset_external: bool = False) -> None:
    url = (
        config["base_url"]
        + "/reset_prefix_cache?reset_external="
        + str(reset_external).lower()
    )
    print(f"POST {url}", flush=True)
    if dry_run:
        return
    deadline = time.monotonic() + config["reset_timeout_seconds"]
    while True:
        body = http_request(url, "POST")
        if not body.strip():
            raise RuntimeError(
                "Reset endpoint returned an empty response: HTTP 200 does not confirm "
                'cache reset. Update the server to return {"success": true/false}; '
                "measurement will not run."
            )
        try:
            result = json.loads(body)
        except (ValueError, UnicodeDecodeError) as error:
            raise RuntimeError(
                "Reset endpoint returned invalid JSON; cache reset is unverified. "
                "Measurement will not run."
            ) from error
        success = result.get("success") if isinstance(result, dict) else result
        if not isinstance(success, bool):
            raise RuntimeError(
                'Reset endpoint must return {"success": true/false} or a JSON boolean; '
                "cache reset is unverified. Measurement will not run."
            )
        print(
            f"Reset response: success={str(success).lower()} (reset_external={str(reset_external).lower()})",
            flush=True,
        )
        if success:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Cache reset did not succeed before reset_timeout_seconds"
            )
        time.sleep(1)


class BenchCLI:
    def __init__(self, config: dict, dry_run: bool):
        self.config = config
        self.dry_run = dry_run
        self.help = ""
        self.command = ["vllm", "bench", "serve"]
        self.optional = []
        if dry_run:
            return
        executable = shutil.which("vllm")
        if executable is None:
            raise RuntimeError(
                "vllm executable is not available in this Python environment"
            )
        result = subprocess.run(
            [executable, "bench", "serve", "--help=all"], capture_output=True, text=True
        )
        if result.returncode:
            result = subprocess.run(
                [executable, "bench", "serve", "--help"],
                capture_output=True,
                text=True,
                check=True,
            )
        self.help = result.stdout
        for flag in (
            "--header",
            "--random-prefix-len",
            "--save-detailed",
            "--base-url",
        ):
            if flag not in self.help:
                raise RuntimeError(f"Installed vllm bench serve does not expose {flag}")
        self.command = [executable, "bench", "serve"]
        if "--ready-check-timeout-sec" in self.help:
            self.optional += ["--ready-check-timeout-sec", "0"]
        else:
            # The compatibility entry point calls the same vLLM CLI, with only
            # its recognized ready-check hook replaced by a no-inference hook.
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import bench_compat, vllm.benchmarks.serve as s; bench_compat.install_probe_override(s)",
                ],
                cwd=ROOT,
                check=True,
            )
            self.command = [sys.executable, str(ROOT / "bench_compat.py")]
            print(
                "Using the vLLM CLI compatibility entry point to skip its initial inference probe."
            )
        if "--num-warmups" in self.help:
            self.optional += ["--num-warmups", "0"]

    def run(
        self,
        case: dict,
        directory: Path,
        label: str,
        input_len: int,
        prefix_len: int,
        output_len: int,
        concurrency: int,
        rank: int | None,
        prefix_source_count: int | None = None,
    ) -> dict:
        result_path = directory / f"{label}.json"
        command = [
            *self.command,
            "--backend",
            "vllm",
            "--model",
            self.config["model"],
            "--served-model-name",
            self.config["served_model"],
            "--base-url",
            self.config["base_url"],
            "--seed",
            str(case["seed"]),
            "--dataset-name",
            "random",
            "--num-prompts",
            str(case["num_requests"]),
            "--random-input-len",
            str(input_len),
            "--random-prefix-len",
            str(prefix_len),
            "--random-range-ratio",
            "0",
            "--random-output-len",
            str(output_len),
            "--max-concurrency",
            str(concurrency),
            "--request-rate",
            "inf",
            "--ignore-eos",
            *self.optional,
            "--save-result",
            "--save-detailed",
            "--result-dir",
            str(directory),
            "--result-filename",
            result_path.name,
        ]
        if rank is not None:
            command += ["--header", f"X-data-parallel-rank={rank}"]
        if prefix_source_count is not None:
            command = [
                sys.executable,
                str(ROOT / "bench_compat.py"),
                "--prefix-source-num-prompts",
                str(prefix_source_count),
                *command[len(self.command) :],
            ]
        print(shlex.join(command), flush=True)
        if self.dry_run:
            return {}
        directory.mkdir(parents=True, exist_ok=True)
        save_json(directory / f"{label}.command.json", command)
        with (directory / f"{label}.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            ) as process:
                try:
                    for line in process.stdout:
                        sys.stdout.write(line)
                        log.write(line)
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else {}
        )
        errors = []
        if code:
            errors.append(f"bench exited with code {code}")
        if result.get("completed") != case["num_requests"] or result.get("failed", 0):
            errors.append("incomplete or failed requests")
        lengths = result.get("output_lens", [])
        if len(lengths) != case["num_requests"] or any(
            value != output_len for value in lengths
        ):
            errors.append("unexpected output token counts")
        result["validation_errors"] = errors
        result["bench_result_path"] = str(result_path)
        return result


class Experiment:
    def __init__(self, config: dict, args):
        self.config = config
        self.args = args
        self.mode = args.mode
        self.root = Path(config["results_dir"])
        self.key = fingerprint(config)
        self.pointer = self.root / f".posix_pending_{self.key}.json"
        self.seed_source = None
        self.warmups = []
        if self.mode == "posix-run":
            if args.run:
                self.seed_source = Path(args.run).resolve()
            else:
                if not self.pointer.exists():
                    raise ValueError(
                        "No matching Posix seed run. Execute posix-seed with this configuration first."
                    )
                self.seed_source = Path(json.loads(self.pointer.read_text())["run_dir"])
            source = json.loads(
                (self.seed_source / "manifest.json").read_text(encoding="utf-8")
            )
            if source["fingerprint"] != self.key or source["status"] != "seeded":
                raise ValueError(
                    "Posix seed configuration changed or pre-seeding did not complete"
                )
            self.cases = source["cases"]
            if config["ucm_warmup"]["enabled"]:
                self.warmups = source.get("warmups", [])
                if not self.warmups:
                    print(
                        "This older Posix seed run has no load warmup data. Rerun posix-seed to enable load warmup."
                    )
        else:
            self.cases = [dict(case) for case in config["cases"]]
            if config["ucm_warmup"]["enabled"]:
                for index in range(config["ucm_warmup"]["rounds"]):
                    warmup = dict(config["ucm_warmup"])
                    warmup["name"] = f"warmup_{index + 1:03d}"
                    self.warmups.append(warmup)
            used = set()
            for index, case in enumerate([*self.cases, *self.warmups]):
                seed = config.get("seed", "random")
                if seed == "random":
                    seed = secrets.randbelow(2**32)
                    while seed in used:
                        seed = secrets.randbelow(2**32)
                else:
                    seed += index
                used.add(seed)
                case["seed"] = seed
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{now}_{self.mode}_{secrets.token_hex(3)}"
        self.directory = self.root / self.run_id
        self.manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "mode": self.mode,
            "fingerprint": self.key,
            "config": config,
            "cases": self.cases,
            "warmups": self.warmups,
            "warmup_status": {},
            "seed_source": str(self.seed_source) if self.seed_source else None,
            "status": "running",
            "case_status": {},
        }
        self.cli = BenchCLI(config, args.dry_run)

    def save(self) -> None:
        if not self.args.dry_run:
            save_json(self.directory / "manifest.json", self.manifest)

    def ranks(self) -> list[int]:
        route = self.config["routing"]
        return (
            [route["rank"]]
            if route["mode"] == "fixed"
            else list(range(self.config["dp"]))
        )

    def seed_case(self, case: dict, directory: Path) -> None:
        if case["seed_tokens"] == 0:
            return
        ranks = self.ranks()
        if self.mode in ("posix-seed", "posix-reset"):
            # All DP ranks reuse the same backend files and KV key namespace.
            ranks = ranks[:1]
        shared = self.mode in SHARED_HBM_MODES
        # V4's seed_tokens already includes the 32-token MTP tail.
        # A prefix-only request keeps its total length exactly seed_tokens.
        seed_input = (
            (0 if self.mode == "hbm-shared-v4" else 1)
            if shared
            else case["seed_tokens"]
        )

        def seed_rank(rank: int) -> None:
            result = self.cli.run(
                {**case, "num_requests": 1} if shared else case,
                directory,
                f"seed_dp{rank}",
                seed_input,
                case["seed_tokens"] if shared else 0,
                1,
                1 if shared else case["concurrency"],
                rank,
                prefix_source_count=case["num_requests"] if shared else None,
            )
            if result.get("validation_errors"):
                raise RuntimeError(
                    f"Pre-seeding failed for {case['name']} DP{rank}: {result['validation_errors']}"
                )

        if (
            self.mode in PARALLEL_SEED_MODES
            and len(ranks) > 1
            and not self.args.dry_run
        ):
            print(f"Pre-seeding concurrently on DPs {ranks}", flush=True)
            # Each worker passes an explicit rank to BenchCLI.run, which adds
            # X-data-parallel-rank. Join all workers before settling/measuring.
            with ThreadPoolExecutor(max_workers=len(ranks)) as pool:
                list(pool.map(seed_rank, ranks))
        else:
            for rank in ranks:
                seed_rank(rank)
        if not self.args.dry_run:
            time.sleep(
                max(
                    self.config["seed_settle_seconds"],
                    self.config["metrics"]["settle_seconds"],
                )
            )

    def run_ucm_warmup(self) -> None:
        for warmup in self.warmups:
            directory = self.directory / warmup["name"]
            print(
                f"\n{'HBM' if self.mode in HBM_MODES else 'UCM load'} warmup {warmup['name']} seed={warmup['seed']}",
                flush=True,
            )
            if self.mode != "posix-run":
                self.seed_case(warmup, directory)
            if self.mode != "posix-seed":
                if self.mode in ("cache", "posix-reset"):
                    reset_prefix(
                        self.config,
                        self.args.dry_run,
                        reset_external=self.mode == "posix-reset",
                    )
                for rank in self.ranks():
                    prefix = (
                        warmup["seed_tokens"] if self.mode in SHARED_HBM_MODES else 0
                    )
                    result = self.cli.run(
                        warmup,
                        directory,
                        f"warmup_load_dp{rank}",
                        warmup["input_len"] - prefix,
                        prefix,
                        warmup["output_len"],
                        warmup["concurrency"],
                        rank,
                    )
                    if result.get("validation_errors"):
                        raise RuntimeError(
                            f"Warmup failed on DP{rank}: {result['validation_errors']}"
                        )
                if not self.args.dry_run:
                    time.sleep(
                        max(
                            self.config["seed_settle_seconds"],
                            self.config["metrics"]["settle_seconds"],
                        )
                    )
            self.manifest["warmup_status"][warmup["name"]] = (
                "seeded" if self.mode == "posix-seed" else "complete"
            )
            self.save()

    def measure_case(self, case: dict, directory: Path) -> None:
        routing = self.config["routing"]
        row = {
            "run_id": self.run_id,
            "mode": self.mode.removesuffix("-run"),
            "case": case["name"],
            "served_model": self.config["served_model"],
            "routing": routing["mode"]
            + (f"_dp{routing['rank']}" if routing["mode"] == "fixed" else ""),
            "dp": self.config["dp"],
            **{
                key: case[key]
                for key in (
                    "input_len",
                    "output_len",
                    "concurrency",
                    "num_requests",
                    "seed",
                    "seed_tokens",
                )
            },
            "target_hit_rate": case["hit_rate"],
            "status": "failed",
            "metrics_status": "not started",
            "measurement_dir": str(directory),
        }
        if "expected_hbm_tokens" in case:
            row["expected_hbm_tokens"] = case["expected_hbm_tokens"]
        prefix = case["seed_tokens"] if self.mode in SHARED_HBM_MODES else 0
        inputs = case["input_len"] - prefix
        capture = None
        result = {}
        try:
            if not self.args.dry_run:
                try:
                    capture = MetricsCapture(
                        directory / "metrics", self.config["metrics"]
                    )
                    capture.start()
                except Exception as error:
                    row["metrics_status"] = f"hit counters unavailable: {error}"
                    if self.config["metrics"]["enabled"]:
                        raise
            result = self.cli.run(
                case,
                directory,
                "measure",
                inputs,
                prefix,
                case["output_len"],
                case["concurrency"],
                routing.get("rank") if routing["mode"] == "fixed" else None,
            )
            if not self.args.dry_run:
                row.update(performance(result))
                row["bench_result_path"] = result.get("bench_result_path")
                row["validation_errors"] = result.get("validation_errors", [])
                row["status"] = "complete" if not row["validation_errors"] else "failed"
        finally:
            if not self.args.dry_run:
                if capture is not None and hasattr(capture, "before_text"):
                    try:
                        row.update(
                            capture.finish(
                                {self.config["model"], self.config["served_model"]}
                            )
                        )
                    except Exception as error:
                        if self.config["metrics"]["enabled"]:
                            row["metrics_status"] = f"collection failed: {error}"
                            row["status"] = "failed"
                        else:
                            row["metrics_status"] = f"hit counters unavailable: {error}"
                elif self.config["metrics"]["enabled"]:
                    row["metrics_status"] = "collection did not start"
                directory.mkdir(parents=True, exist_ok=True)
                save_json(directory / "case_result.json", row)
        if result.get("validation_errors"):
            raise RuntimeError(
                f"Measurement failed for {case['name']}: {result['validation_errors']}"
            )
        if row["metrics_status"].startswith("collection failed"):
            raise RuntimeError(row["metrics_status"])

    def execute(self) -> None:
        if self.mode == "posix-run":
            print(
                f"Using seeds from {self.seed_source}. The service must have been restarted with the same Posix data."
            )
        if not self.args.dry_run:
            self.directory.mkdir(parents=True)
        self.save()
        try:
            if not self.args.dry_run:
                http_request(self.config["base_url"] + "/health")
            self.run_ucm_warmup()
            for case in self.cases:
                directory = self.directory / case["name"]
                print(
                    f"\n{self.mode} {case['name']} seed={case['seed']} seed_tokens={case['seed_tokens']}",
                    flush=True,
                )
                if self.mode in V4_HBM_MODES:
                    print(
                        f"Expected HBM prefix: {case['expected_hbm_tokens']} tokens "
                        f"({case['expected_hbm_tokens'] / case['input_len']:.2%})",
                        flush=True,
                    )
                if self.mode != "posix-run":
                    self.seed_case(case, directory)
                if self.mode != "posix-seed":
                    if self.mode in ("cache", "posix-reset"):
                        reset_prefix(
                            self.config,
                            self.args.dry_run,
                            reset_external=self.mode == "posix-reset",
                        )
                    self.measure_case(case, directory)
                self.manifest["case_status"][case["name"]] = "complete"
                self.save()
            self.manifest["status"] = (
                "seeded" if self.mode == "posix-seed" else "complete"
            )
            if self.mode == "posix-seed" and not self.args.dry_run:
                save_json(self.pointer, {"run_dir": str(self.directory)})
                print(
                    "All cases seeded. Wait for UCM writes to finish, stop and restart the service, then execute posix-run."
                )
        except BaseException as error:
            self.manifest["status"] = "failed"
            self.manifest["error"] = str(error)
            raise
        finally:
            self.save()
            if not self.args.dry_run and self.mode != "posix-seed":
                write_summary(self.directory)
                write_summary(self.root)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "hbm",
            "hbm-shared",
            "hbm-shared-v4",
            "v4hbm",
            "cache",
            "posix-seed",
            "posix-run",
            "posix-reset",
            "report",
        ),
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="bench.json",
        help="Configuration file, or results directory for report",
    )
    parser.add_argument(
        "--collect-metrics", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--dp-rank", type=int, help="Pin warming and measurement to this DP rank"
    )
    parser.add_argument("--run", help="Explicit Posix seed run directory")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.mode == "report":
            directory = Path(
                args.config if args.config != "bench.json" else "results"
            ).resolve()
            if not directory.is_dir():
                raise ValueError(f"Results directory does not exist: {directory}")
            write_summary(directory)
            return 0
        config = load_config(Path(args.config).resolve(), args)
        directory = Path(config["results_dir"])
        if args.dry_run:
            Experiment(config, args).execute()
        else:
            directory.mkdir(parents=True, exist_ok=True)
            with experiment_lock(directory, fingerprint(config)):
                Experiment(config, args).execute()
        return 0
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
        urllib.error.URLError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
