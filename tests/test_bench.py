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
        self.posix_shards = 0
        self.posix_bytes = 0
        self.external_reset = False
        self.load_observations = 0
        self.fail_measurement = False
        self.metrics_unavailable = False
        self.reset_failures = 0
        self.reset_response = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    body = b"healthy"
                elif self.path == "/metrics":
                    owner.events.append(("snapshot", owner.queries))
                    if owner.metrics_unavailable:
                        self.send_error(404)
                        return
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
                if success:
                    owner.external_reset = "reset_external=true" in self.path
                owner.reset_failures = max(0, owner.reset_failures - 1)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(
                    owner.reset_response
                    if owner.reset_response is not None
                    else json.dumps({"success": success}).encode()
                )

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
            "ucm_warmup": {"enabled": False},
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
            f"ucm:cache_posix_load_success_shards_total{labels} {self.posix_shards}",
            f"ucm:cache_load_backend_shards_total{labels} {self.posix_shards}",
            f"ucm:posix_s2h_bytes_total{labels} {self.posix_bytes}",
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
        warmup_load = option("--result-filename").startswith("warmup_load")
        kind = "measure" if measured else ("warmup_load" if warmup_load else "seed")
        self.events.append((kind, list(command)))
        self.queries += count * length
        if measured or warmup_load:
            self.external += count * int(length * 0.9)
            self.shards += count
            if self.external_reset:
                self.posix_shards += count
                self.posix_bytes += count * 4096
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
            "total_token_throughput": count * (length + output),
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
            if kind not in ("seed", "measure"):
                continue
            self.assertEqual(
                command[command.index("--header") + 1], "X-data-parallel-rank=1"
            )
            self.assertEqual(command[command.index("--random-prefix-len") + 1], "0")
            self.assertIn(command[command.index("--num-prompts") + 1], ("7", "3"))
            self.assertNotIn("--prefix-source-num-prompts", command)
        self.assertFalse(any(kind == "reset" for kind, _ in self.events))
        self.assertFalse(any((self.root / "results").rglob("before.prom")))

    def test_hbm_shared_seeds_one_per_dp_and_preserves_full_measurement(self):
        self.config["ucm_warmup"] = {"enabled": True}
        self.save_config()
        self.run_mode("hbm-shared")
        manifest = self.manifests("hbm-shared")[0]
        self.assertEqual(manifest["status"], "complete")
        self.assertFalse(any(kind == "reset" for kind, _ in self.events))
        for case in [*manifest["cases"], *manifest["warmups"]]:
            commands = [
                (kind, command)
                for kind, command in self.events
                if kind in ("seed", "warmup_load", "measure")
                and command[command.index("--seed") + 1] == str(case["seed"])
            ]
            seeds = [command for kind, command in commands if kind == "seed"]
            self.assertEqual(len(seeds), self.config["dp"])
            for rank, command in enumerate(seeds):
                for flag, value in (
                    ("--header", f"X-data-parallel-rank={rank}"),
                    ("--num-prompts", "1"),
                    ("--max-concurrency", "1"),
                    ("--random-input-len", "1"),
                    ("--random-output-len", "1"),
                    ("--random-prefix-len", str(case["seed_tokens"])),
                    ("--prefix-source-num-prompts", str(case["num_requests"])),
                ):
                    self.assertEqual(command[command.index(flag) + 1], value)
                self.assertTrue(command[1].endswith("bench_compat.py"))
            loads = [command for kind, command in commands if kind != "seed"]
            self.assertEqual(len(loads), 3 if case["name"].startswith("warmup") else 1)
            for command in loads:
                for flag, value in (
                    ("--num-prompts", str(case["num_requests"])),
                    ("--max-concurrency", str(case["concurrency"])),
                    ("--random-prefix-len", str(case["seed_tokens"])),
                    (
                        "--random-input-len",
                        str(case["input_len"] - case["seed_tokens"]),
                    ),
                    ("--random-output-len", str(case["output_len"])),
                ):
                    self.assertEqual(command[command.index(flag) + 1], value)
                if not case["name"].startswith("warmup"):
                    self.assertNotIn("--header", command)
        rows = list((self.root / "results").rglob("case_result.json"))
        self.assertEqual(len(rows), 2)
        self.assertTrue(
            all(json.loads(path.read_text())["mode"] == "hbm-shared" for path in rows)
        )

    def test_v4hbm_seeds_aligned_mtp_prefix_on_every_dp_and_exports_plan(self):
        self.assert_v4_hbm_workflow("v4hbm")

    def test_hbm_shared_v4_seeds_one_exact_prefix_per_dp_and_exports_plan(self):
        self.assert_v4_hbm_workflow("hbm-shared-v4")

    def assert_v4_hbm_workflow(self, mode):
        shared = mode == "hbm-shared-v4"
        self.config["ucm_warmup"] = {"enabled": True}
        self.config["block_tokens"] = 16
        self.config["cases"] = [
            dict(
                input_len=32768,
                output_len=512,
                hit_rate=0.9,
                concurrency=40,
                num_requests=40,
            ),
            dict(
                input_len=131072,
                output_len=512,
                hit_rate=0.9,
                concurrency=10,
                num_requests=10,
            ),
        ]
        self.save_config()
        self.run_mode(mode)
        manifest = self.manifests(mode)[0]
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(
            [
                (case["expected_hbm_tokens"], case["seed_tokens"])
                for case in manifest["cases"] + manifest["warmups"]
            ],
            [(28672, 28704), (114688, 114720), (4096, 4128)],
        )
        self.assertFalse(any(kind == "reset" for kind, _ in self.events))
        for case in manifest["cases"] + manifest["warmups"]:
            commands = [
                (kind, command)
                for kind, command in self.events
                if kind in ("seed", "warmup_load", "measure")
                and command[command.index("--seed") + 1] == str(case["seed"])
            ]
            seeds = [command for kind, command in commands if kind == "seed"]
            self.assertEqual(len(seeds), self.config["dp"])
            for rank, command in enumerate(seeds):
                for flag, value in (
                    ("--random-input-len", "0" if shared else str(case["seed_tokens"])),
                    (
                        "--random-prefix-len",
                        str(case["seed_tokens"]) if shared else "0",
                    ),
                    ("--random-output-len", "1"),
                    ("--num-prompts", "1" if shared else str(case["num_requests"])),
                    ("--max-concurrency", "1" if shared else str(case["concurrency"])),
                    ("--header", f"X-data-parallel-rank={rank}"),
                ):
                    self.assertEqual(command[command.index(flag) + 1], value)
                if shared:
                    self.assertTrue(command[1].endswith("bench_compat.py"))
                    self.assertEqual(
                        command[command.index("--prefix-source-num-prompts") + 1],
                        str(case["num_requests"]),
                    )
                else:
                    self.assertNotIn("--prefix-source-num-prompts", command)
            loads = [(kind, command) for kind, command in commands if kind != "seed"]
            self.assertEqual(len(loads), 3 if case["name"].startswith("warmup") else 1)
            for kind, command in loads:
                for flag, value in (
                    (
                        "--random-input-len",
                        str(case["input_len"] - (case["seed_tokens"] if shared else 0)),
                    ),
                    (
                        "--random-prefix-len",
                        str(case["seed_tokens"]) if shared else "0",
                    ),
                    ("--random-output-len", str(case["output_len"])),
                    ("--num-prompts", str(case["num_requests"])),
                    ("--max-concurrency", str(case["concurrency"])),
                ):
                    self.assertEqual(command[command.index(flag) + 1], value)
                if kind == "measure":
                    self.assertNotIn("--header", command)
        rows = [
            json.loads(p.read_text())
            for p in (self.root / "results").rglob("case_result.json")
        ]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["mode"], mode)
            self.assertEqual(row["seed_tokens"], row["expected_hbm_tokens"] + 32)
        csv = (self.root / "results" / "summary.csv").read_text(encoding="utf-8-sig")
        self.assertIn("expected_hbm_tokens", csv.splitlines()[0])
        self.assertIn("28704,28672", csv)

    def test_v4hbm_zero_small_and_full_hit_targets_fit_measured_input(self):
        for length, hit_rate, target, seed_tokens in (
            (32768, 0, 0, 0),
            (32768, 0.01, 0, 0),
            (32768, 1, 28672, 28704),
            (8224, 1, 8192, 8224),
            (8192, 1, 4096, 4128),
            (4096, 1, 0, 0),
            (16, 1, 0, 0),
        ):
            for mode in bench.V4_HBM_MODES:
                with self.subTest(length=length, hit_rate=hit_rate, mode=mode):
                    case = dict(
                        input_len=length,
                        output_len=1,
                        hit_rate=hit_rate,
                        concurrency=1,
                        num_requests=1,
                    )
                    bench.normalize_case(case, "case", 16, mode)
                    self.assertEqual(case["expected_hbm_tokens"], target)
                    self.assertEqual(case["seed_tokens"], seed_tokens)
                    self.assertLessEqual(seed_tokens, length)

    def test_hbm_shared_v4_fixed_dp_and_zero_target(self):
        self.config["cases"] = [
            dict(
                input_len=length,
                output_len=1,
                hit_rate=1,
                concurrency=2,
                num_requests=3,
            )
            for length in (8224, 4096)
        ]
        self.save_config()
        self.run_mode("hbm-shared-v4", "--dp-rank", "2")
        seeds = [command for kind, command in self.events if kind == "seed"]
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0][seeds[0].index("--random-prefix-len") + 1], "8224")
        self.assertEqual(seeds[0][seeds[0].index("--random-input-len") + 1], "0")
        loads = [command for kind, command in self.events if kind == "measure"]
        self.assertEqual(len(loads), 2)
        for command, (prefix, inputs) in zip(loads, ((8224, 0), (0, 4096))):
            for flag, value in (
                ("--random-prefix-len", str(prefix)),
                ("--random-input-len", str(inputs)),
                ("--header", "X-data-parallel-rank=2"),
            ):
                self.assertEqual(command[command.index(flag) + 1], value)
        self.assertFalse(any(kind == "reset" for kind, _ in self.events))

    def test_v4hbm_validates_warmup_seed_offsets(self):
        self.config["seed"] = 2**32 - 2
        self.config["ucm_warmup"] = {"enabled": True}
        self.save_config()
        for mode in bench.V4_HBM_MODES:
            with self.subTest(mode=mode):
                args = bench.parse_args([mode, str(self.config_path)])
                with self.assertRaisesRegex(ValueError, "case offsets"):
                    bench.load_config(self.config_path, args)

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

    def test_posix_reset_seeds_all_dp_clears_both_tiers_then_measures(self):
        self.run_mode("posix-reset", "--collect-metrics")
        self.assertEqual(
            [kind for kind, _ in self.events],
            ["seed", "seed", "seed", "reset", "snapshot", "measure", "snapshot"] * 2,
        )
        self.assertEqual(
            [value for kind, value in self.events if kind == "reset"],
            ["/reset_prefix_cache?reset_external=true"] * 2,
        )
        manifest = self.manifests("posix-reset")[0]
        self.assertEqual(manifest["status"], "complete")
        self.assertIsNone(manifest["seed_source"])
        seeds = [command for kind, command in self.events if kind == "seed"]
        measures = [command for kind, command in self.events if kind == "measure"]
        for index, case in enumerate(manifest["cases"]):
            measurement = measures[index]
            self.assertNotIn("--header", measurement)
            self.assertEqual(
                measurement[measurement.index("--random-input-len") + 1],
                str(case["input_len"]),
            )
            for rank in range(3):
                seed = seeds[index * 3 + rank]
                self.assertEqual(
                    seed[seed.index("--header") + 1], f"X-data-parallel-rank={rank}"
                )
                self.assertEqual(
                    seed[seed.index("--random-input-len") + 1], str(case["seed_tokens"])
                )
                for flag in ("--seed", "--num-prompts"):
                    self.assertEqual(
                        seed[seed.index(flag) + 1],
                        measurement[measurement.index(flag) + 1],
                    )
        rows = [
            json.loads(path.read_text())
            for path in (self.root / "results").rglob("case_result.json")
        ]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["mode"], "posix-reset")
            self.assertEqual(row["hbm_hit_rate"], 0)
            self.assertEqual(row["external_hit_rate"], 0.9)
            self.assertEqual(row["host_ready_shard_fraction"], 0)
            self.assertEqual(row["posix_read_bytes"], row["num_requests"] * 4096)

    def test_posix_reset_warmup_fixed_dp_excluded_from_formal_counts(self):
        self.enable_load_warmup()
        self.run_mode("posix-reset", "--dp-rank", "1")
        self.assertEqual(
            [kind for kind, _ in self.events],
            ["seed", "reset", "warmup_load"] * 2
            + ["seed", "reset", "snapshot", "measure", "snapshot"] * 2,
        )
        for kind, command in self.events:
            if kind in ("seed", "warmup_load", "measure"):
                self.assertEqual(
                    command[command.index("--header") + 1], "X-data-parallel-rank=1"
                )
            elif kind == "reset":
                self.assertEqual(command, "/reset_prefix_cache?reset_external=true")
        rows = [
            json.loads(path.read_text())
            for path in (self.root / "results").rglob("case_result.json")
        ]
        self.assertEqual(len(rows), 2)
        self.assertTrue(
            all(row["posix_read_bytes"] == row["num_requests"] * 4096 for row in rows)
        )

    def test_posix_reset_failure_stops_before_measurement(self):
        self.config["reset_timeout_seconds"] = 0
        self.save_config()
        self.reset_failures = 1
        self.run_mode("posix-reset", expected=1)
        self.assertEqual([kind for kind, _ in self.events], ["seed"] * 3 + ["reset"])
        self.assertEqual(self.manifests("posix-reset")[0]["status"], "failed")

    def test_empty_reset_200_stops_cache_and_posix_reset_before_measurement(self):
        self.reset_response = b""
        for mode in ("cache", "posix-reset"):
            with self.subTest(mode=mode):
                self.events.clear()
                self.run_mode(mode, expected=1)
                self.assertEqual(
                    [kind for kind, _ in self.events], ["seed"] * 3 + ["reset"]
                )
                manifest = self.manifests(mode)[0]
                self.assertEqual(manifest["status"], "failed")
                self.assertIn("empty response", manifest["error"])

    def test_reset_requires_explicit_boolean_and_prints_success(self):
        args = bench.parse_args(["posix-reset", str(self.config_path)])
        config = bench.load_config(self.config_path, args)
        for response in (
            b"{}",
            b'{"success":"true"}',
            b'{"success":1}',
            b"1",
            b"null",
            b"[]",
            b"not json",
            b" \n",
        ):
            with self.subTest(response=response):
                self.reset_response = response
                with self.assertRaises(RuntimeError), redirect_stdout(io.StringIO()):
                    bench.reset_prefix(config, False, reset_external=True)
        for response in (b'{"success":true}', b"true"):
            with self.subTest(response=response):
                self.reset_response = response
                output = io.StringIO()
                with redirect_stdout(output):
                    bench.reset_prefix(config, False, reset_external=True)
                self.assertIn("success=true (reset_external=true)", output.getvalue())

    def test_posix_commands_preserve_original_a3_short_seed_full_measurement(self):
        self.config["dp"] = 2
        self.config["seed"] = 2004
        self.config["cases"] = [
            {
                "input_len": 32768,
                "output_len": 512,
                "hit_rate": 0.9,
                "concurrency": 40,
                "num_requests": 40,
            },
            {
                "input_len": 131072,
                "output_len": 512,
                "hit_rate": 0.9,
                "concurrency": 10,
                "num_requests": 10,
            },
        ]
        self.save_config()
        self.run_mode("posix-seed")
        seed_commands = [command for kind, command in self.events if kind == "seed"]
        self.events.clear()
        self.run_mode("posix-run")
        measured_commands = [
            command for kind, command in self.events if kind == "measure"
        ]
        for index, (length, count, seed) in enumerate(
            [(32768, 40, 2004), (131072, 10, 2005)]
        ):
            for rank in (0, 1):
                command = seed_commands[index * 2 + rank]
                for flag, value in [
                    ("--random-input-len", length * 9 // 10),
                    ("--random-prefix-len", 0),
                    ("--random-output-len", 1),
                    ("--num-prompts", count),
                    ("--max-concurrency", count),
                    ("--seed", seed),
                    ("--header", f"X-data-parallel-rank={rank}"),
                ]:
                    self.assertEqual(command[command.index(flag) + 1], str(value))
            command = measured_commands[index]
            for flag, value in [
                ("--random-input-len", length),
                ("--random-prefix-len", 0),
                ("--random-output-len", 512),
                ("--num-prompts", count),
                ("--max-concurrency", count),
                ("--seed", seed),
            ]:
                self.assertEqual(command[command.index(flag) + 1], str(value))
            self.assertNotIn("--header", command)
            self.assertEqual(command[:3], ["vllm", "bench", "serve"])

    def enable_load_warmup(self):
        self.config["ucm_warmup"] = {
            "enabled": True,
            "rounds": 2,
            "input_len": 8192,
            "output_len": 1,
            "hit_rate": 0.9,
            "concurrency": 4,
            "num_requests": 4,
        }
        self.save_config()

    def test_cache_load_warmup_covers_all_dp_before_metrics(self):
        self.enable_load_warmup()
        self.run_mode("cache", "--collect-metrics")
        kinds = [event[0] for event in self.events]
        self.assertEqual(
            kinds[:14],
            ["seed"] * 3
            + ["reset"]
            + ["warmup_load"] * 3
            + ["seed"] * 3
            + ["reset"]
            + ["warmup_load"] * 3,
        )
        self.assertGreater(
            kinds.index("snapshot"),
            max(index for index, kind in enumerate(kinds) if kind == "warmup_load"),
        )
        manifest = self.manifests("cache")[0]
        self.assertEqual(len(manifest["warmups"]), 2)
        seeds = [case["seed"] for case in manifest["cases"] + manifest["warmups"]]
        self.assertEqual(len(set(seeds)), 4)
        rows = [
            json.loads(path.read_text())
            for path in (self.root / "results").rglob("case_result.json")
        ]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["external_hit_rate"] == 0.9 for row in rows))
        loads = [event[1] for event in self.events if event[0] == "warmup_load"]
        self.assertEqual(len(loads), 6)
        for command in loads:
            self.assertEqual(command[command.index("--random-input-len") + 1], "8192")
            self.assertEqual(command[command.index("--max-concurrency") + 1], "4")

    def test_hbm_warmup_and_cases_seed_all_independent_prefixes_per_dp(self):
        self.enable_load_warmup()
        self.run_mode("hbm")
        manifest = self.manifests("hbm")[0]
        self.assertEqual(len(manifest["warmups"]), 2)
        all_cases = manifest["cases"] + manifest["warmups"]
        self.assertEqual(len({case["seed"] for case in all_cases}), 4)
        kinds = [kind for kind, _ in self.events]
        self.assertEqual(kinds[:12], (["seed"] * 3 + ["warmup_load"] * 3) * 2)
        self.assertNotIn("reset", kinds)
        seeds = [command for kind, command in self.events if kind == "seed"]
        self.assertEqual(len(seeds), 12)  # Two warmup groups and two cases, three DPs.
        for case in all_cases:
            matching = [
                command
                for command in seeds
                if command[command.index("--seed") + 1] == str(case["seed"])
            ]
            self.assertEqual(len(matching), 3)
            for rank, command in enumerate(matching):
                self.assertEqual(
                    command[command.index("--num-prompts") + 1],
                    str(case["num_requests"]),
                )
                self.assertEqual(
                    command[command.index("--random-prefix-len") + 1],
                    "0",
                )
                self.assertEqual(
                    command[command.index("--random-input-len") + 1],
                    str(case["seed_tokens"]),
                )
                self.assertEqual(
                    command[command.index("--max-concurrency") + 1],
                    str(case["concurrency"]),
                )
                self.assertEqual(command[command.index("--random-output-len") + 1], "1")
                self.assertNotIn("--prefix-source-num-prompts", command)
                self.assertEqual(
                    command[command.index("--header") + 1],
                    f"X-data-parallel-rank={rank}",
                )
        measurements = [command for kind, command in self.events if kind == "measure"]
        for case, command in zip(manifest["cases"], measurements):
            self.assertEqual(command[command.index("--random-prefix-len") + 1], "0")
            self.assertEqual(
                command[command.index("--random-input-len") + 1], str(case["input_len"])
            )
            self.assertEqual(command[command.index("--seed") + 1], str(case["seed"]))
            self.assertEqual(
                command[command.index("--num-prompts") + 1], str(case["num_requests"])
            )
            self.assertEqual(
                command[command.index("--max-concurrency") + 1],
                str(case["concurrency"]),
            )
            self.assertNotIn("--prefix-source-num-prompts", command)
            self.assertNotIn("--header", command)
        loads = [command for kind, command in self.events if kind == "warmup_load"]
        self.assertEqual(len(loads), 6)
        self.assertTrue(
            all(
                command[command.index("--random-prefix-len") + 1] == "0"
                for command in loads
            )
        )
        self.assertTrue(
            all(
                command[command.index("--random-input-len") + 1] == "8192"
                for command in loads
            )
        )
        self.assertEqual(
            len(list((self.root / "results").rglob("case_result.json"))), 2
        )

    def test_posix_warmup_is_seeded_before_restart_and_replayed_after(self):
        self.enable_load_warmup()
        self.run_mode("posix-seed")
        source = self.manifests("posix-seed")[0]
        self.assertEqual(len(source["warmups"]), 2)
        self.assertEqual(len(self.events), 12)
        self.assertTrue(all(event[0] == "seed" for event in self.events))
        self.events.clear()
        self.run_mode("posix-run", "--collect-metrics")
        measured = self.manifests("posix-run")[0]
        self.assertEqual(source["warmups"], measured["warmups"])
        self.assertEqual([event[0] for event in self.events[:6]], ["warmup_load"] * 6)
        self.assertFalse(any(kind == "reset" for kind, _ in self.events))
        expected_seeds = {str(group["seed"]) for group in source["warmups"]}
        actual_seeds = {
            command[command.index("--seed") + 1]
            for kind, command in self.events
            if kind == "warmup_load"
        }
        self.assertEqual(expected_seeds, actual_seeds)

    def test_legacy_posix_manifest_remains_usable_without_warmup_data(self):
        self.run_mode("posix-seed")
        path = next((self.root / "results").glob("*_posix-seed_*/manifest.json"))
        manifest = json.loads(path.read_text())
        manifest.pop("warmups")
        path.write_text(json.dumps(manifest))
        self.enable_load_warmup()
        self.events.clear()
        self.run_mode("posix-run")
        self.assertEqual(
            [event[0] for event in self.events], ["snapshot", "measure", "snapshot"] * 2
        )

    def test_default_hit_rates_exclude_seeding_without_detailed_exports(self):
        self.run_mode("cache")
        rows = [
            json.loads(path.read_text())
            for path in (self.root / "results").rglob("case_result.json")
        ]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["hbm_hit_rate"], 0)
            self.assertEqual(row["external_hit_rate"], 0.9)
            self.assertEqual(row["metrics_status"], "hit counters")
            self.assertAlmostEqual(row["tpot_mean_ms"], 100)
            self.assertEqual(
                row["output_tokens_per_s"], row["num_requests"] * row["output_len"]
            )
            self.assertEqual(
                row["total_tokens_per_s"],
                row["num_requests"] * (row["input_len"] + row["output_len"]),
            )
            self.assertFalse((Path(row["measurement_dir"]) / "metrics").exists())
        self.assertEqual(
            [kind for kind, _ in self.events],
            ["seed", "seed", "seed", "reset", "snapshot", "measure", "snapshot"] * 2,
        )
        summary = (self.root / "results" / "summary.csv").read_text(
            encoding="utf-8-sig"
        )
        self.assertIn("external_hit_rate", summary)
        self.assertIn("0.9", summary)

    def test_default_hit_counter_failure_does_not_stop_bench(self):
        self.metrics_unavailable = True
        self.run_mode("cache")
        rows = [
            json.loads(path.read_text())
            for path in (self.root / "results").rglob("case_result.json")
        ]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["status"], "complete")
            self.assertNotIn("external_hit_rate", row)
            self.assertIn("hit counters unavailable", row["metrics_status"])
            self.assertTrue(Path(row["bench_result_path"]).exists())

    def test_default_hit_counter_failure_after_bench_preserves_success(self):
        from metrics_capture import MetricsCapture

        with mock.patch.object(
            MetricsCapture, "finish", side_effect=RuntimeError("endpoint unavailable")
        ):
            self.run_mode("cache")
        rows = [
            json.loads(path.read_text())
            for path in (self.root / "results").rglob("case_result.json")
        ]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["status"] == "complete" for row in rows))
        self.assertTrue(
            all("endpoint unavailable" in row["metrics_status"] for row in rows)
        )

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
        # The lock file persists; its OS lock must have been released on failure.
        locks = list((self.root / "results").glob("*.lock"))
        self.assertEqual(len(locks), 1)
        with bench.experiment_lock(locks[0].parent, locks[0].stem.removeprefix(".")):
            pass

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


class ExperimentLockTest(unittest.TestCase):
    def test_lock_excludes_overlap_and_can_be_reused_after_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with bench.experiment_lock(directory, "test"):
                with self.assertRaisesRegex(RuntimeError, "already holds"):
                    with bench.experiment_lock(directory, "test"):
                        self.fail("overlapping experiment acquired lock")
            self.assertTrue((directory / ".test.lock").exists())
            with bench.experiment_lock(directory, "test"):
                pass

    def test_abrupt_process_exit_releases_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = (
                "import os, sys, bench\n"
                "from pathlib import Path\n"
                "with bench.experiment_lock(Path(sys.argv[1]), 'test'):\n"
                "    os._exit(0)\n"
            )
            subprocess.run(
                [sys.executable, "-c", script, temporary], cwd=bench.ROOT, check=True
            )
            with bench.experiment_lock(Path(temporary), "test"):
                pass

    def test_legacy_stale_pid_is_recovered_but_live_pid_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            lock = directory / ".test.lock"
            lock.write_text(str(os.getpid()))
            with self.assertRaisesRegex(RuntimeError, "older experiment"):
                with bench.experiment_lock(directory, "test"):
                    self.fail("live legacy PID was ignored")
            self.assertEqual(lock.read_text(), str(os.getpid()))
            with mock.patch("bench.pid_is_running", return_value=False):
                with bench.experiment_lock(directory, "test"):
                    pass
            self.assertEqual(json.loads(lock.read_text())["lock"], "os")

    def test_pid_check_distinguishes_current_and_missing_process(self):
        self.assertTrue(bench.pid_is_running(os.getpid()))
        self.assertFalse(bench.pid_is_running(2**31 - 1))


class MetricsTest(unittest.TestCase):
    def test_performance_records_tpot_and_vllm_throughputs(self):
        from reporting import performance

        result = performance(
            {
                "ttfts": [0.1, 0.1],
                "latencies": [0.6, 0.7],
                "output_lens": [6, 4],
                "duration": 2,
                "output_throughput": 5,
                "total_token_throughput": 105,
            }
        )
        self.assertAlmostEqual(result["tpot_mean_ms"], 150)
        self.assertEqual(result["output_tokens_per_s"], 5)
        self.assertEqual(result["total_tokens_per_s"], 105)

    def test_console_summary_prints_real_zero_hits_and_missing_values(self):
        from reporting import write_summary

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw = directory / "measure.json"
            raw.write_text(
                json.dumps(
                    {
                        "mean_tpot_ms": 5,
                        "output_throughput": 6000,
                        "total_token_throughput": 7000,
                    }
                )
            )
            sample = {
                "mode": "posix",
                "case": "sample",
                "status": "complete",
                "target_hit_rate": 0.9,
                "hbm_hit_rate": 0.0,
                "external_hit_rate": 0.0,
                "host_ready_shard_fraction": None,
                "posix_read_bytes": 0,
                "ttft_mean_ms": 125,
                "metrics_status": "captured",
                "bench_result_path": str(raw),
            }
            (directory / "case_result.json").write_text(json.dumps(sample))
            output = io.StringIO()
            with redirect_stdout(output):
                rows = write_summary(directory)
            text = output.getvalue()
            self.assertIn("Target hit", text)
            self.assertIn("External hit", text)
            self.assertIn("TPOT ms", text)
            self.assertIn("Output tok/s", text)
            self.assertIn("Total tok/s", text)
            self.assertIn("5.00", text)
            self.assertIn("6000.00", text)
            self.assertIn("7000.00", text)
            self.assertEqual(rows[0]["total_tokens_per_s"], 7000)
            self.assertEqual(rows[0]["ttft_mean_ms"], 125)
            self.assertIn(
                "total_tokens_per_s",
                (directory / "summary.csv").read_text(encoding="utf-8-sig"),
            )
            with zipfile.ZipFile(directory / "summary.xlsx") as workbook:
                self.assertIn(
                    "total_tokens_per_s",
                    workbook.read("xl/worksheets/sheet1.xml").decode(),
                )
            self.assertIn("90.0%", text)
            self.assertEqual(
                [cell.strip() for cell in text.split("|")].count("0.0%"), 2
            )
            self.assertIn("n/a", text)

    def test_console_summary_handles_empty_results(self):
        from reporting import write_summary

        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            self.assertEqual(write_summary(Path(temporary)), [])

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

    def test_shared_seed_preserves_sampling_count_but_sends_one_request(self):
        import random
        from types import SimpleNamespace

        def get_samples(args):
            rng = random.Random(args.seed)
            for _ in range(args.num_prompts):
                rng.random()  # Per-request offsets precede the common prefix.
            prefix = rng.getrandbits(64)
            return [(prefix, index) for index in range(args.num_prompts)]

        formal = get_samples(SimpleNamespace(seed=2004, num_prompts=40))
        self.assertNotEqual(
            formal[0], get_samples(SimpleNamespace(seed=2004, num_prompts=1))[0]
        )
        module = SimpleNamespace(get_samples=get_samples)
        bench_compat.install_prefix_seed_sampling(module, 40)
        args = SimpleNamespace(seed=2004, num_prompts=1)
        self.assertEqual(module.get_samples(args), formal[:1])
        self.assertEqual(args.num_prompts, 1)
        failing = SimpleNamespace(
            get_samples=mock.Mock(side_effect=ValueError("sampling"))
        )
        bench_compat.install_prefix_seed_sampling(failing, 40)
        with self.assertRaises(ValueError):
            failing.get_samples(args)
        self.assertEqual(args.num_prompts, 1)


if __name__ == "__main__":
    unittest.main()
