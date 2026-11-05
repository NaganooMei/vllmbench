"""Adapt the UCM snapshot collector to one synchronous benchmark interval."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import collect_ucm_metrics


def load_collector(path: str | None):
    if path is None:
        return collect_ucm_metrics
    spec = importlib.util.spec_from_file_location("configured_ucm_collector", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Metrics collector must be a Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("save_snapshot", "parse_labels", "Sample", "write_delta_report"):
        if not hasattr(module, name):
            raise ValueError(f"Metrics collector is missing {name}: {path}")
    return module


def all_samples(module, text: str) -> dict:
    samples = {}
    for line in text.splitlines():
        match = module.SAMPLE_RE.match(line)
        if match is None:
            continue
        name = match.group("name")
        if not name.startswith(("vllm:", "ucm:")):
            continue
        sample = module.Sample(
            name,
            module.parse_labels(match.group("labels") or ""),
            float(match.group("value")),
        )
        samples[sample.key] = sample
    return samples


def counter_delta(
    before: dict, after: dict, name: str, models: set[str]
) -> float | None:
    total = 0.0
    found = False
    names = {name, name + "_total"}
    for key, sample in after.items():
        if sample.name not in names:
            continue
        model = dict(sample.labels).get("model_name")
        if model and model not in models:
            continue
        previous = before[key].value if key in before else 0.0
        if (
            not math.isfinite(previous)
            or not math.isfinite(sample.value)
            or sample.value < previous
        ):
            return None
        found = True
        total += sample.value - previous
    return total if found else None


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    value = numerator / denominator
    return value if 0 <= value <= 1 else None


class MetricsCapture:
    def __init__(self, directory: Path, config: dict):
        self.directory = directory
        self.config = config
        self.detailed = config["enabled"]
        self.module = load_collector(
            config.get("collector_path") if self.detailed else None
        )
        if self.detailed:
            self.directory.mkdir(parents=True)

    def snapshot(self, label: str) -> tuple[str, dict]:
        if self.detailed:
            return self.module.save_snapshot(
                self.directory,
                label,
                self.config["url"],
                self.config["timeout_seconds"],
            )
        return (
            collect_ucm_metrics.fetch_metrics(
                self.config["url"], self.config["timeout_seconds"]
            ),
            {},
        )

    def start(self) -> None:
        self.start_time = time.monotonic()
        self.before_text, self.before = self.snapshot("before")

    def finish(self, models: set[str]) -> dict:
        time.sleep(self.config["settle_seconds"])
        after_text, after = self.snapshot("after")
        elapsed = time.monotonic() - self.start_time
        if self.detailed:
            report = self.module.write_delta_report(
                self.directory / "metrics_report.md",
                self.before,
                after,
                elapsed,
                False,
                report_stem="metrics_report",
            )
            (self.directory / "summary.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
        before_all = all_samples(self.module, self.before_text)
        after_all = all_samples(self.module, after_text)

        def delta(name):
            return counter_delta(before_all, after_all, name, models)

        queries = delta("vllm:prefix_cache_queries")
        hbm_hits = delta("vllm:prefix_cache_hits")
        external_queries = delta("vllm:external_prefix_cache_queries")
        external_hits = delta("vllm:external_prefix_cache_hits")
        successes = delta("ucm:cache_load_success_shards_total")
        pending_shards = delta("ucm:cache_posix_load_success_shards_total")
        ready_shards = None
        if successes is not None and pending_shards is not None:
            ready_shards = successes - pending_shards
        summary = {
            "hbm_hit_rate": ratio(hbm_hits, queries),
            "external_hit_rate": ratio(external_hits, queries),
            "external_hit_rate_remaining": ratio(external_hits, external_queries),
            "host_ready_shard_fraction": ratio(ready_shards, successes),
            "backend_load_shards": delta("ucm:cache_load_backend_shards_total"),
            "posix_read_bytes": delta("ucm:posix_s2h_bytes_total"),
            "metrics_status": (
                ("captured" if self.detailed else "hit counters")
                if queries is not None
                else "prefix counters unavailable"
            ),
        }
        if self.detailed:
            (self.directory / "hit_rates.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
        return summary
