#!/usr/bin/env python3
"""
Quiesce the board (display/WiFi/network) and hold, so you can measure idle
baseline power with your own external meter - no benchmark, no INA219.

Uses the same system_services.py stop/start calls every benchmark script
uses before/after a run, so the baseline you measure here reflects the same
"quiesced" state those benchmarks run in.

Examples:
    python3 baseline_power.py                  # quiesce display+wifi, wait for Enter
    python3 baseline_power.py --seconds 30      # quiesce, hold 30s, auto-restore
    python3 baseline_power.py --quiesce-network # also drop Ethernet (serial console only!)
    python3 baseline_power.py --no-quiesce-wifi # display only

Ctrl+C at any point restores display/wifi/network before exiting.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--quiesce-display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the Weston/lightdm display session. Default: enabled.",
    )
    parser.add_argument(
        "--quiesce-wifi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Soft-block the WiFi radio. Leaves usb0 (adb/NCM) untouched. "
        "Default: enabled.",
    )
    parser.add_argument(
        "--quiesce-network",
        action="store_true",
        help="Also stop the full network stack (Ethernet + WiFi services). "
        "Only safe over a connection that doesn't depend on IP networking "
        "(e.g. serial console) - it will drop an SSH/network-adb session. "
        "Default: disabled.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Hold the quiesced state for this long, then auto-restore and "
        "exit. Default: wait for Enter instead.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    sys.path.insert(0, str(_UTILS_DIR))
    from system_services import (
        disable_display,
        disable_network,
        disable_wifi,
        enable_display,
        enable_network,
        enable_wifi,
    )

    try:
        if args.quiesce_network:
            for _ok, message in disable_network():
                print(f"[services] {message}")
        if args.quiesce_wifi:
            for _ok, message in disable_wifi():
                print(f"[services] {message}")
        if args.quiesce_display:
            for _ok, message in disable_display():
                print(f"[services] {message}")

        print("\nSystem quiesced. Take your baseline power reading now.")
        try:
            if args.seconds is not None:
                print(f"Holding for {args.seconds:.1f} s ...")
                time.sleep(args.seconds)
            else:
                input("Press Enter when done to restore display/networking... ")
        except KeyboardInterrupt:
            print()
    finally:
        if args.quiesce_display:
            for _ok, message in enable_display():
                print(f"[services] {message}")
        if args.quiesce_wifi:
            for _ok, message in enable_wifi():
                print(f"[services] {message}")
        if args.quiesce_network:
            for _ok, message in enable_network():
                print(f"[services] {message}")

    print("Restored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
