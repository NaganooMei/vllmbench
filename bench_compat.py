"""Run vLLM's bench CLI while disabling its initial inference ready probe."""

from __future__ import annotations

import inspect
import os
import sys
from types import SimpleNamespace


def install_probe_override(module) -> None:
    source = inspect.getsource(module.benchmark)
    if "wait_for_endpoint(" not in source or not hasattr(module, "wait_for_endpoint"):
        raise RuntimeError(
            "This vLLM benchmark has no recognized ready-check hook. "
            "Use a version supporting --ready-check-timeout-sec=0."
        )

    async def endpoint_already_checked(*args, **kwargs):
        return SimpleNamespace(success=True, error="")

    module.wait_for_endpoint = endpoint_already_checked


def main() -> None:
    os.environ["VLLM_USE_RUST_BENCH"] = "0"
    import vllm.benchmarks.serve as serve

    if "--ready-check-timeout-sec" not in sys.argv:
        install_probe_override(serve)
    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv = ["vllm", "bench", "serve", *sys.argv[1:]]
    vllm_main()


if __name__ == "__main__":
    main()
