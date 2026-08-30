# SPDX-License-Identifier: Apache-2.0
"""
Minimal Torq decoder-only LLM runner, vendored from torq-examples so this
benchmarking repo stays self-contained.

Upstream: torq-examples @ commit e5a527e
  utils/inference.py   -> torq_llm/inference.py   (verbatim)
  utils/llm.py         -> torq_llm/llm.py         (import path only)
  gemma3/src/runner.py -> torq_llm/gemma3.py      (import path; dropped format_answer)

Re-sync from upstream when the Torq runtime is upgraded.
"""

from .gemma3 import DEFAULT_SYS_PROMPT, Gemma3Static
from .llm import DecoderOnlyLLMRunner, InferenceInterrupted

__all__ = [
    "Gemma3Static",
    "DEFAULT_SYS_PROMPT",
    "DecoderOnlyLLMRunner",
    "InferenceInterrupted",
]
