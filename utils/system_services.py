"""
Astra board display/WiFi helpers.

Stops the Wayland/Weston compositor and/or the WiFi radio around a benchmark
run to remove their background CPU and power overhead, then restarts them
afterward.

WiFi is disabled via `rfkill block wifi` rather than stopping NetworkManager/
systemd-networkd/wpa_supplicant/dnsmasq - those also manage usb0 (the
adb/NCM-gadget link), and systemd-networkd.service in particular is
socket-activated (`systemd-networkd.socket`), so a plain `systemctl stop`
doesn't stick: the next netlink event on any interface restarts it. `rfkill`
targets only the WiFi PHY, so usb0 and the adb connection are never touched.

Same `(ok, message)` / try-except-no-op style as utils/npu.py, so callers can
run the same script on any platform.

`rfkill block` (soft block, not a hardware disable) always clears on reboot,
so a board reset restores WiFi regardless of whether enable_wifi() ever ran.

Run standalone to restore state without re-running a benchmark, e.g. after a
run was interrupted before its own cleanup could execute:
    python3 system_services.py enable all
    python3 system_services.py enable wifi
    python3 system_services.py enable display
"""

from __future__ import annotations

import subprocess

DISPLAY_SERVICES = ("weston.service",)


def _systemctl(action: str, unit: str) -> tuple[bool, str]:
    """Run `systemctl <action> <unit>`, never raising."""
    try:
        result = subprocess.run(
            ["systemctl", action, unit],
            capture_output=True,
            timeout=10,
            check=False,
            text=True,
        )
    except Exception as exc:  # systemctl missing (not on an Astra board)
        return False, f"{unit}: {action} skipped ({exc})"
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        return False, f"{unit}: {action} failed ({detail})"
    return True, f"{unit}: {action} ok"


def _rfkill(action: str) -> tuple[bool, str]:
    """Run `rfkill <action> wifi`, never raising."""
    try:
        result = subprocess.run(
            ["rfkill", action, "wifi"],
            capture_output=True,
            timeout=10,
            check=False,
            text=True,
        )
    except Exception as exc:  # rfkill missing (not on an Astra board)
        return False, f"wifi: {action} skipped ({exc})"
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        return False, f"wifi: {action} failed ({detail})"
    return True, f"wifi: {action} ok"


def disable_display() -> list[tuple[bool, str]]:
    """Stop the Wayland/Weston compositor."""
    return [_systemctl("stop", unit) for unit in DISPLAY_SERVICES]


def enable_display() -> list[tuple[bool, str]]:
    """Restart the Wayland/Weston compositor."""
    return [_systemctl("start", unit) for unit in DISPLAY_SERVICES]


def disable_wifi() -> list[tuple[bool, str]]:
    """Soft-block the WiFi radio. Leaves usb0 and NetworkManager/networkd/
    wpa_supplicant/dnsmasq untouched."""
    return [_rfkill("block")]


def enable_wifi() -> list[tuple[bool, str]]:
    """Soft-unblock the WiFi radio."""
    return [_rfkill("unblock")]


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Stop/start the Astra board's display and/or WiFi, e.g. "
        "to restore them after a benchmark run that was interrupted before "
        "its own cleanup could run."
    )
    parser.add_argument("action", choices=["enable", "disable"])
    parser.add_argument(
        "target", choices=["display", "wifi", "all"], nargs="?", default="all"
    )
    args = parser.parse_args()

    steps = {
        ("disable", "display"): [disable_display],
        ("disable", "wifi"): [disable_wifi],
        ("disable", "all"): [disable_display, disable_wifi],
        ("enable", "display"): [enable_display],
        ("enable", "wifi"): [enable_wifi],
        ("enable", "all"): [enable_wifi, enable_display],
    }[(args.action, args.target)]

    ok = True
    for step in steps:
        for success, message in step():
            print(f"[services] {message}")
            ok = ok and success
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
