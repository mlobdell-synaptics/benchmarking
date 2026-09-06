"""
Board display/networking helpers (Astra SL2610/SL2619, Raspberry Pi 5).

Stops the display compositor and/or networking around a benchmark run to
remove their background CPU and power overhead, then restarts them
afterward. Two independent levels of networking are offered:

  disable_wifi() / enable_wifi()
      `rfkill block/unblock wifi` - only the WiFi radio. Leaves Ethernet and
      every network *service* (NetworkManager/networkd/wpa_supplicant/
      dnsmasq/dhcpcd) untouched, so it's always safe: on the Astra boards
      those services also manage usb0 (the adb/NCM-gadget link), so stopping
      them would risk the very connection you're running the benchmark over.
      Default-on in every script that uses this module.

  disable_network() / enable_network()
      Stops the full network management stack (whichever of NetworkManager,
      systemd-networkd, wpa_supplicant, dnsmasq, dhcpcd exist on this
      platform) - both Ethernet and WiFi go down in software. Only safe over
      a connection that does not depend on IP networking, e.g. a serial
      console (or adb/USB on Astra, as long as nothing there also rides on
      NCM/usb0's IP link). Opt-in everywhere - never enabled by default.

Display is stopped via whichever of these units exists:
  - weston.service   (Astra Yocto image's Wayland compositor)
  - lightdm.service  (Raspberry Pi OS Desktop's display manager - covers the
    labwc/wayfire Wayland session or an X11 session either way, since lightdm
    owns the whole graphical session, not just the compositor)

Same `(ok, message)` / try-except-no-op style as utils/npu.py: every function
returns a list of `(ok, message)` tuples, wraps `systemctl`/`rfkill` in
try/except, and is a harmless no-op when the tool or a unit is missing (e.g.
a unit name for a platform this isn't running on), so callers can run the
same script on any platform without special-casing.

`systemctl stop` (not `disable`/`mask`) and `rfkill block` (a soft block, not
a hardware disable) are used throughout, so a board reset always restores the
normal state regardless of whether this module's `enable_*()` functions ever
ran.

Run standalone to restore state without re-running a benchmark, e.g. after a
run was interrupted before its own cleanup could execute:
    python3 system_services.py enable all       # display + wifi (the safe set)
    python3 system_services.py enable display
    python3 system_services.py enable wifi
    python3 system_services.py enable network   # only if you disabled it

Privilege: `systemctl stop/start` needs root. On the Astra boards, adb/serial
shells already run as root, so this is a non-issue. On Raspberry Pi OS
Desktop (a normal, non-root login), an unconfigured sudo/polkit would
otherwise block on an interactive password prompt this script can't answer -
so it runs `systemctl` via `sudo -n` (fails fast instead of hanging) whenever
not already root. To make it actually succeed instead of just failing fast,
add a sudoers rule scoped to exactly these commands, e.g.
/etc/sudoers.d/benchmarking (edit with `sudo visudo -f`, replacing `mac`):

    Cmnd_Alias BENCH_SVC = /usr/bin/systemctl stop weston.service, \
        /usr/bin/systemctl start weston.service, \
        /usr/bin/systemctl stop lightdm.service, \
        /usr/bin/systemctl start lightdm.service, \
        /usr/bin/systemctl stop dnsmasq.service, \
        /usr/bin/systemctl start dnsmasq.service, \
        /usr/bin/systemctl stop wpa_supplicant.service, \
        /usr/bin/systemctl start wpa_supplicant.service, \
        /usr/bin/systemctl stop systemd-networkd.socket, \
        /usr/bin/systemctl start systemd-networkd.socket, \
        /usr/bin/systemctl stop systemd-networkd.service, \
        /usr/bin/systemctl start systemd-networkd.service, \
        /usr/bin/systemctl stop dhcpcd.service, \
        /usr/bin/systemctl start dhcpcd.service, \
        /usr/bin/systemctl stop NetworkManager.service, \
        /usr/bin/systemctl start NetworkManager.service
    mac ALL=(root) NOPASSWD: BENCH_SVC

`rfkill` is tried unprivileged first (typically seat-permitted already, e.g.
via udev - confirmed on Raspberry Pi OS Desktop) and only falls back to
`sudo -n` on failure, so it works out of the box there without needing a
sudoers entry; add one (same pattern, `/usr/sbin/rfkill block wifi` +
`unblock wifi`) only if your platform's rfkill does require root.
"""

from __future__ import annotations

import os
import subprocess


def _maybe_sudo(cmd: list[str]) -> list[str]:
    """Prefix `cmd` with `sudo -n` when not already root.

    `-n` (non-interactive) makes an unconfigured sudo/polkit fail immediately
    with a clear stderr message instead of hanging on an interactive password
    prompt this script has no way to answer (e.g. Raspberry Pi OS Desktop,
    where `mac`-style accounts aren't root and systemctl/rfkill need
    authorization). See the module docstring for the sudoers.d rule that
    makes this passwordless.
    """
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return cmd
    return ["sudo", "-n", *cmd]

DISPLAY_SERVICES = ("weston.service", "lightdm.service")

# Leaf-to-root stop order; enable_network() reverses this to start root-first.
# systemd-networkd.service is socket-activated: stopping it alone doesn't
# stick (systemd-networkd.socket stays active and reactivates it on the next
# netlink event, e.g. NetworkManager.service being stopped) - the paired
# .socket unit must be stopped too.
NETWORK_SERVICES = (
    "dnsmasq.service",
    "wpa_supplicant.service",
    "systemd-networkd.socket",
    "systemd-networkd.service",
    "dhcpcd.service",
    "NetworkManager.service",
)


def _systemctl(action: str, unit: str) -> tuple[bool, str]:
    """Run `systemctl <action> <unit>` (via sudo -n if not root), never raising."""
    try:
        result = subprocess.run(
            _maybe_sudo(["systemctl", action, unit]),
            capture_output=True,
            timeout=10,
            check=False,
            text=True,
        )
    except Exception as exc:  # systemctl missing (not on this platform)
        return False, f"{unit}: {action} skipped ({exc})"
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        return False, f"{unit}: {action} failed ({detail})"
    return True, f"{unit}: {action} ok"


def _rfkill(action: str) -> tuple[bool, str]:
    """Run `rfkill <action> wifi`, never raising.

    Tries unprivileged first - many systems grant the active seat's user
    direct rfkill access via udev (confirmed on Raspberry Pi OS Desktop) -
    and only falls back to `sudo -n` on failure, so a working unprivileged
    path is never made to depend on sudo being configured at all.
    """
    try:
        result = subprocess.run(
            ["rfkill", action, "wifi"],
            capture_output=True,
            timeout=10,
            check=False,
            text=True,
        )
        if result.returncode != 0 and hasattr(os, "geteuid") and os.geteuid() != 0:
            result = subprocess.run(
                ["sudo", "-n", "rfkill", action, "wifi"],
                capture_output=True,
                timeout=10,
                check=False,
                text=True,
            )
    except Exception as exc:  # rfkill missing (not on this platform)
        return False, f"wifi: {action} skipped ({exc})"
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        return False, f"wifi: {action} failed ({detail})"
    return True, f"wifi: {action} ok"


def disable_display() -> list[tuple[bool, str]]:
    """Stop the display compositor/session (Weston on Astra, lightdm on
    Raspberry Pi OS Desktop - whichever unit is actually present)."""
    return [_systemctl("stop", unit) for unit in DISPLAY_SERVICES]


def enable_display() -> list[tuple[bool, str]]:
    """Restart the display compositor/session."""
    return [_systemctl("start", unit) for unit in DISPLAY_SERVICES]


def disable_wifi() -> list[tuple[bool, str]]:
    """Soft-block the WiFi radio. Leaves Ethernet and every network service
    (NetworkManager/networkd/wpa_supplicant/dnsmasq/dhcpcd) untouched."""
    return [_rfkill("block")]


def enable_wifi() -> list[tuple[bool, str]]:
    """Soft-unblock the WiFi radio."""
    return [_rfkill("unblock")]


def disable_network() -> list[tuple[bool, str]]:
    """Stop the full network stack - both Ethernet and WiFi. Only use this
    over a connection that doesn't depend on IP networking (e.g. serial);
    it will drop an SSH/network-adb session before it can restore anything.

    Stops leaf services first so each `systemctl stop` succeeds independently
    of unit ordering.
    """
    return [_systemctl("stop", unit) for unit in NETWORK_SERVICES]


def enable_network() -> list[tuple[bool, str]]:
    """Restart the full network stack (reverse order of disable_network)."""
    return [_systemctl("start", unit) for unit in reversed(NETWORK_SERVICES)]


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Stop/start a board's display and/or networking, e.g. "
        "to restore them after a benchmark run that was interrupted before "
        "its own cleanup could run."
    )
    parser.add_argument("action", choices=["enable", "disable"])
    parser.add_argument(
        "target",
        choices=["display", "wifi", "network", "all"],
        nargs="?",
        default="all",
        help="'all' covers display + wifi only (always safe); 'network' "
        "(full Ethernet+WiFi stack) must be named explicitly.",
    )
    args = parser.parse_args()

    steps = {
        ("disable", "display"): [disable_display],
        ("disable", "wifi"): [disable_wifi],
        ("disable", "network"): [disable_network],
        ("disable", "all"): [disable_display, disable_wifi],
        ("enable", "display"): [enable_display],
        ("enable", "wifi"): [enable_wifi],
        ("enable", "network"): [enable_network],
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
