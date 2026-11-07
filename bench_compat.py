"""Run vLLM's CLI with probe compatibility and optional single-request prefix seeding."""

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


def install_prefix_seed_sampling(module, source_count: int) -> None:
    # RandomDataset samples per-request offsets before the common prefix.
    # Preserve that RNG consumption, then send just one generated request.
    original = module.get_samples

    def first_request(args, *positional, **kwargs):
        count = args.num_prompts
        args.num_prompts = source_count
        try:
            requests = original(args, *positional, **kwargs)
        finally:
            args.num_prompts = count
        return requests[:1]

    module.get_samples = first_request


def main() -> None:
    os.environ["VLLM_USE_RUST_BENCH"] = "0"
    import vllm.benchmarks.serve as serve

    if "--prefix-source-num-prompts" in sys.argv:
        index = sys.argv.index("--prefix-source-num-prompts")
        source_count = int(sys.argv[index + 1])
        del sys.argv[index : index + 2]
        install_prefix_seed_sampling(serve, source_count)
    if "--ready-check-timeout-sec" not in sys.argv:
        install_probe_override(serve)
    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv = ["vllm", "bench", "serve", *sys.argv[1:]]
    vllm_main()


if __name__ == "__main__":
    main()
