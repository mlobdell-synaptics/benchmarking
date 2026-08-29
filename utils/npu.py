"""
Synaptics Astra Torq NPU clock / frequency helpers.

Adapted from torq-examples/utils/npu.py. Every function returns
`(ok: bool, message: str)` and is a harmless no-op when the sysfs paths or
`devmem` are missing (e.g. on a Raspberry Pi or a dev machine), so callers can
run the same script on any platform.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# NPU clock-enable register (SL2610/SL2619).
_NPU_CLOCK_REG = "0xf7e104b0"
_NPU_CLOCK_VAL = "0x216"

_NPU_DEVFREQ_PATH = Path("/sys/class/devfreq/f7600000.synpu")
_NPU_GOVERNOR_PATH = _NPU_DEVFREQ_PATH / "governor"
_NPU_SET_FREQ_PATH = _NPU_DEVFREQ_PATH / "userspace" / "set_freq"
_NPU_MAX_FREQ_PATH = _NPU_DEVFREQ_PATH / "max_freq"
_NPU_MIN_FREQ_PATH = _NPU_DEVFREQ_PATH / "min_freq"
_NPU_USERSPACE_GOVERNOR = "userspace"


def enable_npu_clock() -> tuple[bool, str]:
    """Enable the NPU clock via `devmem` (required before Torq inference)."""
    try:
        subprocess.run(
            ["devmem", _NPU_CLOCK_REG, "32", _NPU_CLOCK_VAL],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:  # devmem missing (not on an Astra board)
        return False, f"NPU clock enable skipped: {exc}"
    return True, "NPU clock enabled"


def configure_npu_userspace_frequency(level: str = "max") -> tuple[bool, str]:
    """Pin the NPU to its `max` or `min` frequency via the userspace governor."""
    if level not in {"max", "min"}:
        return False, "NPU frequency level must be 'max' or 'min'"

    if not _NPU_GOVERNOR_PATH.exists():
        return False, "NPU devfreq controls not available"

    try:
        if _NPU_GOVERNOR_PATH.read_text().strip() != _NPU_USERSPACE_GOVERNOR:
            _NPU_GOVERNOR_PATH.write_text(_NPU_USERSPACE_GOVERNOR)

        if not _NPU_SET_FREQ_PATH.exists():
            return False, "NPU userspace frequency control not available"

        bound_path = _NPU_MAX_FREQ_PATH if level == "max" else _NPU_MIN_FREQ_PATH
        if not bound_path.exists():
            return False, f"NPU {level}_freq control not available"

        target_freq_hz = bound_path.read_text().strip()
        if not target_freq_hz:
            return False, f"NPU {level}_freq is empty"

        _NPU_SET_FREQ_PATH.write_text(target_freq_hz)
    except Exception as exc:
        return False, f"NPU userspace {level} frequency setup failed: {exc}"

    return True, f"NPU userspace {level} frequency applied"
