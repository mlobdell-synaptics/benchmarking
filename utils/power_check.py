#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Sanity-check the external INA219 power-measurement circuit.

Confirms the chip is reachable, that the readings look physically plausible,
that the PowerSampler path the benchmarks use actually works, and then shows a
live voltage / current / power readout so you can watch it respond to load.

    python3 power_check.py                 # checks + 3 s sampler + live readout
    python3 power_check.py --sanity-only   # just the checks, then exit
    python3 power_check.py --address 0x41  # non-default INA219 address

Wiring (INA219 in series with the board's 5 V input) and the bus/address/shunt
defaults live at the top of pwr_meas.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pwr_meas import (  # noqa: E402
    DEFAULT_I2C_ADDRESS,
    DEFAULT_I2C_BUS,
    DEFAULT_SHUNT_OHMS,
    PowerMonitor,
    PowerSampler,
    ina219_config,
)

PASS = "\033[32mPASS\033[0m"
WARN = "\033[33mWARN\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bus", type=int, default=DEFAULT_I2C_BUS,
                   help=f"I2C bus number (default: {DEFAULT_I2C_BUS}).")
    p.add_argument("--address", type=lambda x: int(x, 0), default=DEFAULT_I2C_ADDRESS,
                   help=f"INA219 address (default: 0x{DEFAULT_I2C_ADDRESS:02x}).")
    p.add_argument("--shunt-ohms", type=float, default=DEFAULT_SHUNT_OHMS,
                   help=f"Shunt resistor value (default: {DEFAULT_SHUNT_OHMS}).")
    p.add_argument("--interval", type=float, default=0.5, metavar="SECS",
                   help="Live readout refresh period (default: 0.5).")
    p.add_argument("--sampler-seconds", type=float, default=3.0, metavar="SECS",
                   help="Exercise PowerSampler for this long (0 to skip). Default: 3.")
    p.add_argument("--sanity-only", action="store_true",
                   help="Run the checks and exit (no sampler, no live readout).")
    return p.parse_args()


def connect(args) -> PowerMonitor:
    try:
        return PowerMonitor(
            bus=args.bus, address=args.address, shunt_ohms=args.shunt_ohms,
            config=ina219_config(avg_samples=1),
        )
    except Exception as exc:
        print(f"[{FAIL}] Could not open the INA219: {exc}\n")
        print("  - Is I2C enabled and the chip wired up?")
        print(f"      i2cdetect -y {args.bus}      # expect a device at "
              f"0x{args.address:02x}")
        print("  - Missing library?   pip install smbus2")
        print("  - Wrong address?     pass --address 0x<addr>")
        raise SystemExit(1)


def sanity_checks(mon: PowerMonitor) -> bool:
    """Take a few readings and flag anything physically implausible."""
    print("Reading the INA219 ...")
    readings = []
    for _ in range(8):
        readings.append(mon.read())
        time.sleep(0.05)

    v = sum(r.voltage_mv for r in readings) / len(readings)
    i = sum(r.current_ma for r in readings) / len(readings)
    p = v * i / 1000.0
    print(f"  bus voltage : {v:8.1f} mV")
    print(f"  current     : {i:8.1f} mA")
    print(f"  power       : {p:8.1f} mW\n")

    ok = True

    # Bus voltage present and in a sane range for a 5 V-ish board rail.
    if v < 500:
        print(f"[{FAIL}] Bus voltage ~0 - VIN+/VIN- not seeing the supply, or "
              "wrong I2C address (reading a different chip).")
        ok = False
    elif not (3000 <= v <= 30000):
        print(f"[{WARN}] Bus voltage {v/1000:.2f} V is outside the expected "
              "3-30 V window - check the shunt/divider and which rail is tapped.")
    else:
        print(f"[{PASS}] Bus voltage {v/1000:.2f} V looks sane.")

    # Some current should be flowing through the shunt.
    if abs(i) < 1.0:
        print(f"[{FAIL}] Current ~0 mA - nothing is drawing through the shunt. "
              "VIN+/VIN- may be bridged, or the load is on the other side.")
        ok = False
    elif i < 0:
        print(f"[{WARN}] Current is negative ({i:.0f} mA) - INA219 VIN+ and "
              "VIN- are swapped. Magnitude is fine; sign is backwards.")
    else:
        print(f"[{PASS}] Current {i:.0f} mA is flowing through the shunt.")

    # Voltage * current should land in a believable board-power range.
    if ok and not (200 <= abs(p) <= 60000):
        print(f"[{WARN}] Power {abs(p)/1000:.2f} W is unusual for this board - "
              "double-check the shunt value (--shunt-ohms).")
    elif ok:
        print(f"[{PASS}] Power {abs(p)/1000:.2f} W is a believable board draw.")

    # Readings should not be frozen (a stuck bus repeats the same bytes).
    if len({round(r.current_ma, 1) for r in readings}) == 1 and abs(i) > 1.0:
        print(f"[{WARN}] Every current sample is identical - the bus may be "
              "stuck, or the load is perfectly constant.")

    print()
    return ok


def exercise_sampler(mon: PowerMonitor, seconds: float) -> None:
    """Run the same PowerSampler the benchmarks use and print its session result."""
    print(f"Running PowerSampler for {seconds:.1f} s (the benchmark code path) ...")
    sampler = PowerSampler(mon)
    sampler.start("power_check")
    time.sleep(seconds)
    result = sampler.stop()
    sampler.close()
    if result is None or result.n_samples == 0:
        print(f"[{FAIL}] Sampler produced no samples.\n")
        return
    print(f"  samples     : {result.n_samples}  ({result.sample_rate_hz:.0f} Hz)")
    print(f"  avg power   : {result.total_avg_power_mw:8.1f} mW")
    print(f"  peak power  : {result.total_peak_power_mw:8.1f} mW")
    print(f"  min power   : {sum(result.min_power_mw.values()):8.1f} mW")
    print(f"  energy      : {result.total_energy_mj:8.1f} mJ "
          f"({result.total_energy_mj / 1000:.3f} J)")
    print(f"[{PASS}] Sampler works - {result.n_samples} samples, "
          f"energy integrated over {result.duration_s:.2f} s.\n")


def live_readout(mon: PowerMonitor, interval: float) -> None:
    print("Live readout - press Ctrl+C to stop.\n")
    p_min = float("inf")
    p_max = 0.0
    p_sum = 0.0
    n = 0
    try:
        while True:
            r = mon.read()
            p = r.power_mw
            n += 1
            p_sum += p
            p_min = min(p_min, p)
            p_max = max(p_max, p)
            sys.stdout.write(
                f"\rVbus {r.voltage_mv/1000:6.3f} V   "
                f"I {r.current_ma:8.1f} mA   "
                f"P {p/1000:6.3f} W    "
                f"[min {p_min/1000:.3f}  avg {p_sum/n/1000:.3f}  "
                f"max {p_max/1000:.3f} W]   "
            )
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nStopped.")


def main() -> int:
    args = parse_args()

    print("INA219 power-measurement check")
    print(f"  bus     : {args.bus}")
    print(f"  address : 0x{args.address:02x}")
    print(f"  shunt   : {args.shunt_ohms} ohm\n")

    mon = connect(args)
    print(f"[{PASS}] INA219 responded on bus {args.bus} at 0x{args.address:02x}.\n")

    try:
        ok = sanity_checks(mon)

        if args.sanity_only:
            return 0 if ok else 1

        if args.sampler_seconds > 0:
            exercise_sampler(mon, args.sampler_seconds)

        live_readout(mon, args.interval)
    finally:
        mon.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
