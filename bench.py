#!/usr/bin/env python3
"""Configurable vLLM bench workflows for HBM, UCM Host Cache and Posix hits."""

from __future__ import annotations

import argparse
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
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be a JSON object")
        for field in ("input_len", "output_len", "concurrency", "num_requests"):
            positive_integer(case.get(field), f"case {index} {field}")
        try:
            hit_rate = Decimal(str(case["hit_rate"]))
        except (KeyError, InvalidOperation) as error:
            raise ValueError(
                f"case {index} requires a hit_rate between 0 and 1"
            ) from error
        if not hit_rate.is_finite() or not 0 <= hit_rate <= 1:
            raise ValueError(f"case {index} hit_rate must be between 0 and 1")
        if case["concurrency"] > case["num_requests"]:
            raise ValueError(f"case {index} concurrency exceeds num_requests")
        block = config["block_tokens"]
        # Leave a token for vLLM's normal final-token recomputation at 100%.
        available = min(int(case["input_len"] * hit_rate), case["input_len"] - 1)
        case["seed_tokens"] = available // block * block
        case["hit_rate"] = float(hit_rate)
        case["name"] = (
            f"{index:03d}_{case['input_len']}in_{case['output_len']}out_c{case['concurrency']}_n{case['num_requests']}"
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
        if seed + len(cases) - 1 >= 2**32:
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


@contextmanager
def experiment_lock(directory: Path, key: str):
    lock = directory / f".{key}.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(
            f"An experiment already holds {lock}. Check its recorded PID before removing a stale lock."
        ) from error
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def http_request(url: str, method: str = "GET", timeout: float = 10) -> bytes:
    headers = {}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    request = urllib.request.Request(url, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def reset_prefix(config: dict, dry_run: bool) -> None:
    url = config["base_url"] + "/reset_prefix_cache?reset_external=false"
    print(f"POST {url}", flush=True)
    if dry_run:
        return
    deadline = time.monotonic() + config["reset_timeout_seconds"]
    while True:
        body = http_request(url, "POST")
        # Older vLLM releases return an empty HTTP 200 response.
        result = json.loads(body) if body.strip() else {"success": True}
        if result is True or isinstance(result, dict) and result.get("success") is True:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "HBM prefix reset did not succeed before reset_timeout_seconds"
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
        else:
            self.cases = [dict(case) for case in config["cases"]]
            used = set()
            for index, case in enumerate(self.cases):
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
        count = len(self.ranks())
        concurrency = max(1, math.ceil(case["concurrency"] / count))
        for rank in self.ranks():
            if self.mode == "hbm":
                inputs, prefix = 1, case["seed_tokens"]
            else:
                inputs, prefix = case["seed_tokens"], 0
            result = self.cli.run(
                case, directory, f"seed_dp{rank}", inputs, prefix, 1, concurrency, rank
            )
            if result.get("validation_errors"):
                raise RuntimeError(
                    f"Pre-seeding failed for {case['name']} DP{rank}: {result['validation_errors']}"
                )
        if not self.args.dry_run:
            settle = self.config["seed_settle_seconds"]
            if self.config["metrics"]["enabled"]:
                settle = max(settle, self.config["metrics"]["settle_seconds"])
            time.sleep(settle)

    def measure_case(self, case: dict, directory: Path) -> None:
        routing = self.config["routing"]
        row = {
            "run_id": self.run_id,
            "mode": self.mode.removesuffix("-run"),
            "case": case["name"],
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
            "metrics_status": "disabled",
            "measurement_dir": str(directory),
        }
        if self.mode == "hbm" and case["seed_tokens"]:
            inputs = case["input_len"] - case["seed_tokens"]
            prefix = case["seed_tokens"]
        else:
            inputs, prefix = case["input_len"], 0
        capture = None
        result = {}
        try:
            if self.config["metrics"]["enabled"] and not self.args.dry_run:
                capture = MetricsCapture(directory / "metrics", self.config["metrics"])
                capture.start()
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
                        row["metrics_status"] = f"collection failed: {error}"
                        row["status"] = "failed"
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
            for case in self.cases:
                directory = self.directory / case["name"]
                print(
                    f"\n{self.mode} {case['name']} seed={case['seed']} seed_tokens={case['seed_tokens']}",
                    flush=True,
                )
                if self.mode != "posix-run":
                    self.seed_case(case, directory)
                if self.mode != "posix-seed":
                    if self.mode == "cache":
                        reset_prefix(self.config, self.args.dry_run)
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
        "mode", choices=("hbm", "cache", "posix-seed", "posix-run", "report")
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
