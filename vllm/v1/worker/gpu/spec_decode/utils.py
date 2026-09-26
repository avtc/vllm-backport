# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch


class DraftTokensHandler:
    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0
        # Latest drafts of every running request whose batch had structured
        # output requests, by request id. See get_draft_tokens().
        self.latest_drafts: dict[str, list[int]] = {}
        self.copy_pending = False

    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        self.num_draft_tokens = draft_tokens.shape[1]
        # A previous async copy may still be pending (batch-queue mode can
        # run two structured batches back to back without a take between
        # them); move its drafts into latest_drafts before overwriting.
        self._collect()
        if not input_batch.has_structured_output_reqs:
            # No draft token validation needs to be performed by
            # the scheduler for this batch; the pending copy above already
            # moved any drafts into latest_drafts.
            self.draft_tokens_np = None
            return
        self.req_ids = input_batch.req_ids
        # Batches without structured-output requests never invalidate the
        # drafts collected earlier (see get_draft_tokens); requests leave
        # latest_drafts only via remove_request.

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            self.draft_tokens_np = async_copy_to_np(draft_tokens)
            # draft_tokens is a temporary allocation on the main stream and read here on
            # copy_stream; without record_stream, the caching allocator may reuse its
            # memory before the async copy executes.
            draft_tokens.record_stream(self.copy_stream)
            self.copy_event.record()
        self.copy_pending = True

    def _collect(self) -> None:
        """Moves the drafts of the last batch into latest_drafts."""
        if not self.copy_pending:
            return
        self.copy_pending = False
        if self.draft_tokens_np is None:
            # A batch without structured-output requests cleared the array
            # after collecting; nothing to move.
            return
        self.copy_event.synchronize()
        for req_id, drafts in zip(self.req_ids, self.draft_tokens_np.tolist()):
            self.latest_drafts[req_id] = drafts

    def remove_request(self, req_id: str) -> None:
        self._collect()
        self.latest_drafts.pop(req_id, None)

    def get_draft_tokens(self) -> DraftTokenIds | None:
        self._collect()
        if self.latest_drafts:
            # With pipeline parallelism several batches are in flight and a
            # request decodes only every pp_size steps, so the batch whose
            # grammar bitmask the scheduler is about to build usually holds
            # requests that were not in the last batch. Without their drafts
            # the scheduler leaves them as -1 and masks the bonus position
            # with the grammar state before the draft, while the GPU still
            # verifies the real draft: a token the grammar rejects gets
            # sampled and the request ends with an internal error. Return
            # the latest drafts of every request instead; a request's next
            # verification uses exactly these.
            req_ids = list(self.latest_drafts)
            draft_token_ids = [self.latest_drafts[r] for r in req_ids]
            return DraftTokenIds(req_ids, draft_token_ids)
        if self.draft_tokens_np is not None:
            self.copy_event.synchronize()
            draft_token_ids = self.draft_tokens_np.tolist()
        else:
            # This case only happens when async scheduling is disabled.
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
        return DraftTokenIds(self.req_ids, draft_token_ids)


def get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token id used for parallel drafting slots.

    Checks (in order): `dflash_config.mask_token_id`, top-level `mask_token_id`,
    `dspark_noise_token_id`, `pard_token`, `ptd_token_id`. Raises ValueError if
    none are present.
    """
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if getattr(hf_config, "mask_token_id", None) is not None:
        return int(hf_config.mask_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    raise ValueError(
        "Model config must specify `dflash_config.mask_token_id`,"
        " `mask_token_id`, `dspark_noise_token_id`, `pard_token`, or"
        " `ptd_token_id` for parallel drafting."
    )
