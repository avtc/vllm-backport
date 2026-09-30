"""Host-side per-phase timing for PP decode steps.

VLLM_PP_TIMING_DEBUG=1 accumulates wall-clock time for the phases that make
up a pipeline step (inter-stage receive wait, input preparation, forward,
sampling+broadcast on the last rank, the deferred postprocess wave) and
logs per-phase averages every ``DUMP_EVERY`` steps. The receive wait is the
pipeline bubble seen by this rank: time spent blocked on the previous
stage's gloo metadata exchange and NCCL transfer.
"""

import time
from contextlib import contextmanager

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

DUMP_EVERY = 50


class PPPhaseTimer:
    def __init__(self) -> None:
        self.accum: dict[str, float] = {}
        self.count = 0

    @contextmanager
    def phase(self, name: str):
        if not envs.VLLM_PP_TIMING_DEBUG:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.accum[name] = self.accum.get(name, 0.0) + (time.perf_counter() - t0)

    def step_done(self) -> None:
        if not envs.VLLM_PP_TIMING_DEBUG:
            return
        self.count += 1
        if self.count % DUMP_EVERY:
            return
        parts = " ".join(
            f"{name}={1000.0 * secs / self.count:.2f}ms"
            for name, secs in sorted(self.accum.items())
        )
        logger.info("pp step timing (avg over %d steps): %s", self.count, parts)


timer = PPPhaseTimer()
