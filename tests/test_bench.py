from contextlib import redirect_stdout, redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
import zipfile
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench
import bench_compat
import collect_ucm_metrics as collector
from metrics_capture import all_samples, counter_delta, ratio

HELP = "--header --random-prefix-len --save-detailed --base-url --ready-check-timeout-sec --num-warmups"


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(
            os.environ,
            {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"},
        )
        self.environment.start()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.events = []
        self.queries = 0
        self.external = 0
        self.shards = 0
        self.load_observations = 0
        self.fail_measurement = False
        self.reset_failures = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    body = b"healthy"
                elif self.path == "/metrics":
                    owner.events.append(("snapshot", owner.queries))
                    body = owner.metrics().encode()
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                owner.events.append(("reset", self.path))
                success = owner.reset_failures == 0
                owner.reset_failures = max(0, owner.reset_failures - 1)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"success": success}).encode())

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.config = {
            "base_url": f"http://127.0.0.1:{self.server.server_port}",
            "model": "test-model",
            "served_model": "test",
            "dp": 3,
            "seed_settle_seconds": 0,
            "metrics": {"enabled": False, "settle_seconds": 0},
            "results_dir": str(self.root / "results"),
            "cases": [
                {
                    "input_len": 100,
                    "output_len": 4,
                    "hit_rate": 0.9,
                    "concurrency": 5,
                    "num_requests": 7,
                },
                {
                    "input_len": 200,
                    "output_len": 4,
                    "hit_rate": 0.8,
                    "concurrency": 2,
                    "num_requests": 3,
                },
            ],
        }
        self.config_path = self.root / "bench.json"
        self.save_config()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()
        self.environment.stop()

    def save_config(self):
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def metrics(self):
        labels = '{model_name="test",engine="0"}'
        lines = [
            f"vllm:prefix_cache_queries_total{labels} {self.queries}",
            f"vllm:prefix_cache_hits_total{labels} 0",
            f"vllm:external_prefix_cache_queries_total{labels} {self.queries}",
            f"vllm:external_prefix_cache_hits_total{labels} {self.external}",
            f"ucm:cache_load_success_shards_total{labels} {self.shards}",
            f"ucm:cache_posix_load_success_shards_total{labels} 0",
            f"ucm:cache_load_backend_shards_total{labels} 0",
            f"ucm:posix_s2h_bytes_total{labels} 0",
        ]
        for rank in (0, 1):
            rank_labels = f'model_name="test",engine="0",worker_rank="{rank}"'
            count = self.load_observations
            lines += [
                f'ucm:cache_load_duration_ms_bucket{{{rank_labels},le="10"}} {count}',
                f'ucm:cache_load_duration_ms_bucket{{{rank_labels},le="+Inf"}} {count}',
                f"ucm:cache_load_duration_ms_count{{{rank_labels}}} {count}",
                f"ucm:cache_load_duration_ms_sum{{{rank_labels}}} {count * 5}",
            ]
        return "\n".join(lines)

    def fake_process(self, command, **kwargs):
        def option(name):
            return command[command.index(name) + 1]

        count = int(option("--num-prompts"))
        length = int(option("--random-input-len")) + int(option("--random-prefix-len"))
        output = int(option("--random-output-len"))
        measured = option("--result-filename") == "measure.json"
        self.events.append(("measure" if measured else "seed", list(command)))
        self.queries += count * length
        if measured:
            self.external += count * int(length * 0.9)
            self.shards += count
            self.load_observations += 1
        result = {
            "completed": count - int(measured and self.fail_measurement),
            "failed": int(measured and self.fail_measurement),
            "output_lens": [output] * count,
            "input_lens": [length] * count,
            "ttfts": [0.1] * count,
            "latencies": [0.4] * count,
            "duration": 1.0,
            "output_throughput": count * output,
        }
        path = Path(option("--result-dir")) / option("--result-filename")
        path.write_text(json.dumps(result), encoding="utf-8")
        process = mock.MagicMock()
        process.__enter__.return_value = process
        process.stdout = io.StringIO("Mock bench completed\n")
        process.wait.return_value = 0
        return process

    def run_mode(self, mode, *options, expected=0):
        with mock.patch("bench.shutil.which", return_value="vllm"), mock.patch(
            "bench.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, HELP, ""),
        ), mock.patch(
            "bench.subprocess.Popen", side_effect=self.fake_process
        ), redirect_stdout(
            io.StringIO()
        ), redirect_stderr(
            io.StringIO()
        ):
            self.assertEqual(
                bench.main([mode, str(self.config_path), *options]), expected
            )

    def manifests(self, mode):
        return [
            json.loads(path.read_text())
            for path in (self.root / "results").glob(f"*_{mode}_*/manifest.json")
        ]

    def test_cache_all_dp_metrics_exclude_seeding_and_export(self):
        self.run_mode("cache", "--collect-metrics")
        seeds = [event for event in self.events if event[0] == "seed"]
        self.assertEqual(len(seeds), 6)
        rows = [
            json.loads(path.read_text())
            for path in (self.root / "results").rglob("case_result.json")
        ]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["hbm_hit_rate"], 0)
            self.assertEqual(row["external_hit_rate"], 0.9)
            self.assertEqual(row["host_ready_shard_fraction"], 1)
            self.assertEqual(row["backend_load_shards"], 0)
            self.assertEqual(row["metrics_status"], "captured")
            directory = Path(row["measurement_dir"]) / "metrics"
            self.assertTrue((directory / "metrics_report.xlsx").exists())
            self.assertTrue((directory / "before.prom").exists())
        kinds = [event[0] for event in self.events]
        self.assertEqual(
            kinds,
            ["seed", "seed", "seed", "reset", "snapshot", "measure", "snapshot"] * 2,
        )
        for event in seeds:
            command = event[1]
            self.assertEqual(
                command[command.index("--num-prompts") + 1],
                "7" if "100in" in str(command) else "3",
            )
        path = self.root / "results" / "summary.xlsx"
        with zipfile.ZipFile(path) as workbook:
            for name in workbook.namelist():
                if name.endswith(".xml") or name.endswith(".rels"):
                    ET.fromstring(workbook.read(name))
            xml = workbook.read("xl/worksheets/sheet1.xml").decode()
            self.assertIn('r="AD3"', xml)
            self.assertIn('s="10"', xml)
            self.assertNotIn("xl/worksheets/sheet3.xml", workbook.namelist())
        self.assertTrue((self.root / "results" / "summary.csv").exists())

    def test_hbm_fixed_rank_warms_then_measures_and_randomizes_new_runs(self):
        self.run_mode("hbm", "--dp-rank", "1")
        self.run_mode("hbm", "--dp-rank", "1")
        manifests = self.manifests("hbm")
        self.assertEqual(len(manifests), 2)
        self.assertNotEqual(
            manifests[0]["cases"][0]["seed"], manifests[1]["cases"][0]["seed"]
        )
        for kind, command in self.events:
            self.assertEqual(
                command[command.index("--header") + 1], "X-data-parallel-rank=1"
            )
            if kind == "measure":
                self.assertGreater(
                    int(command[command.index("--random-prefix-len") + 1]), 0
                )
        self.assertFalse(any((self.root / "results").rglob("before.prom")))

    def test_posix_seeds_persist_and_metrics_option_can_change(self):
        self.run_mode("posix-seed", "--collect-metrics")
        seeded = self.manifests("posix-seed")[0]
        self.assertTrue(all(event[0] == "seed" for event in self.events))
        self.events.clear()
        self.run_mode("posix-run", "--collect-metrics")
        measured = self.manifests("posix-run")[0]
        self.assertEqual(seeded["cases"], measured["cases"])
        self.assertEqual(
            [event[0] for event in self.events], ["snapshot", "measure", "snapshot"] * 2
        )
        self.config["cases"][0]["num_requests"] += 1
        self.save_config()
        self.run_mode("posix-run", expected=1)

    def test_failed_measurement_is_saved_and_returns_failure(self):
        self.fail_measurement = True
        self.run_mode("cache", expected=1)
        manifest = self.manifests("cache")[0]
        self.assertEqual(manifest["status"], "failed")
        row = json.loads(
            next((self.root / "results").rglob("case_result.json")).read_text()
        )
        self.assertEqual(row["status"], "failed")
        self.assertTrue(row["validation_errors"])
        self.assertFalse(list((self.root / "results").glob("*.lock")))

    def test_zero_and_full_hit_rates_block_alignment_and_validation(self):
        self.config["block_tokens"] = 16
        self.config["cases"][0]["hit_rate"] = 0
        self.config["cases"][1]["hit_rate"] = 1
        self.save_config()
        args = bench.parse_args(["hbm", str(self.config_path)])
        config = bench.load_config(self.config_path, args)
        self.assertEqual([case["seed_tokens"] for case in config["cases"]], [0, 192])
        self.run_mode("hbm")
        self.assertEqual(sum(event[0] == "seed" for event in self.events), 3)
        self.config["cases"][0]["hit_rate"] = 1.1
        self.save_config()
        self.run_mode("hbm", expected=1)

    def test_reset_retries_without_resetting_external_cache(self):
        self.reset_failures = 1
        args = bench.parse_args(["cache", str(self.config_path)])
        config = bench.load_config(self.config_path, args)
        with mock.patch("bench.time.sleep"), redirect_stdout(io.StringIO()):
            bench.reset_prefix(config, False)
        self.assertEqual(
            self.events, [("reset", "/reset_prefix_cache?reset_external=false")] * 2
        )

    def test_fixed_seed_is_reproducible_and_dry_run_sends_no_requests(self):
        self.config["seed"] = 2026
        self.save_config()
        args = bench.parse_args(["cache", str(self.config_path), "--dry-run"])
        config = bench.load_config(self.config_path, args)
        first = bench.Experiment(config, args)
        second = bench.Experiment(config, args)
        self.assertEqual([case["seed"] for case in first.cases], [2026, 2027])
        self.assertEqual(first.cases, second.cases)
        with redirect_stdout(io.StringIO()):
            first.execute()
        self.assertFalse(self.events)
        self.assertFalse((self.root / "results").exists())

    def test_metrics_failure_preserves_measurement_and_failure_status(self):
        from metrics_capture import MetricsCapture

        with mock.patch.object(
            MetricsCapture, "finish", side_effect=RuntimeError("endpoint unavailable")
        ):
            self.run_mode("cache", "--collect-metrics", expected=1)
        result = json.loads(
            next((self.root / "results").rglob("case_result.json")).read_text()
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("endpoint unavailable", result["metrics_status"])
        self.assertTrue(Path(result["bench_result_path"]).exists())


class MetricsTest(unittest.TestCase):
    def test_counter_labels_resets_missing_and_zero(self):
        before = all_samples(
            collector, 'vllm:prefix_cache_hits_total{engine="0",model_name="test"} 5'
        )
        after = all_samples(
            collector, 'vllm:prefix_cache_hits_total{model_name="test",engine="0"} 8'
        )
        self.assertEqual(
            counter_delta(before, after, "vllm:prefix_cache_hits", {"test"}), 3
        )
        self.assertIsNone(
            counter_delta(after, before, "vllm:prefix_cache_hits", {"test"})
        )
        self.assertIsNone(counter_delta(before, after, "ucm:missing", {"test"}))
        self.assertIsNone(ratio(0, 0))
        self.assertEqual(ratio(0, 1), 0)

    def test_probe_override_skips_only_recognized_hook(self):
        async def wait_for_endpoint(*args):
            raise AssertionError("real inference probe")

        async def recognized_benchmark():
            return await wait_for_endpoint()

        module = mock.Mock(
            benchmark=recognized_benchmark, wait_for_endpoint=wait_for_endpoint
        )
        bench_compat.install_probe_override(module)
        import asyncio

        self.assertTrue(asyncio.run(module.wait_for_endpoint()).success)
        module.benchmark = lambda: None
        with self.assertRaises(RuntimeError):
            bench_compat.install_probe_override(module)


if __name__ == "__main__":
    unittest.main()
