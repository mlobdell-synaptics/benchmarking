#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""
Power monitor API for a Raspberry Pi 5 with an INA219 on the I2C bus.

Hardware setup
--------------
The INA219 breakout is wired *in series with the main 5 V USB-C supply* so it
sees the total power going into the board:

    5V PSU (+) --> INA219 VIN+
    INA219 VIN- --> Pi 5 5V input (USB-C +5V, or the 5V pin)
    PSU GND ----> common ground with the Pi
    INA219 VCC --> Pi 3V3   (logic only, not the measured rail)
    INA219 GND --> Pi GND
    INA219 SDA --> Pi GPIO2 (pin 3)
    INA219 SCL --> Pi GPIO3 (pin 5)

Enable I2C once with `sudo raspi-config` (Interface Options -> I2C) or
`dtparam=i2c_arm=on` in /boot/firmware/config.txt, then confirm the chip is
visible:  `i2cdetect -y -r 2`  should show a device at 0x40.

For the fastest sampling, also raise the bus clock to 400 kHz:
`dtparam=i2c_arm_baudrate=400000` in /boot/firmware/config.txt.

The Adafruit-style breakout uses a 0.1 ohm shunt (the default here). If you
fitted a different shunt, pass its value as `shunt_ohms`.

Sampling rate
-------------
At the default 12-bit ADC setting the INA219 produces a fresh bus+shunt pair
about every 1.06 ms (~1 kHz). Each PowerMonitor.read() is two I2C register
reads, so on a 100 kHz bus the realistic ceiling is ~400-600 Hz; at 400 kHz
it approaches 1 kHz. PowerSampler integrates energy using the *measured* time
between samples, so a jittery rate under load does not bias the energy total.
`ina219_config(avg_samples=N)` trades rate for on-chip averaging (lower noise).

Usage
-----
    mon = PowerMonitor()                 # bus 1, addr 0x40, 0.1 ohm shunt
    print(mon.channels)                  # ['USB']
    r = mon.read("USB")                  # PowerReading(label, voltage_mv, current_ma, power_mw)
    all = mon.read_all()                 # {'USB': PowerReading}
    mon.close()                          # or: use as context manager

    # Continuous background sampling with named sessions:
    sampler = PowerSampler(mon, interval_s=0.002, csv_file="power.csv")
    sampler.start("inference")           # session name required
    # ... run workload ...
    result = sampler.stop()              # returns SessionResult
    print(result.total_energy_mj)        # total energy for the session
    print(result.total_avg_power_mw)     # mean power over the session
    print(result.total_peak_power_mw)    # highest single-sample power
"""

from __future__ import annotations

import csv
import logging
import threading
import time
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)


DEFAULT_I2C_BUS = 2
DEFAULT_I2C_ADDRESS = 0x40
DEFAULT_SHUNT_OHMS = 0.1
DEFAULT_LABEL = "USB"

# Sampler tuning. Change these here rather than plumbing flags through every
# benchmark script.
DEFAULT_INTERVAL_S = 0.002   # power sampler target period (2 ms ~= 500 Hz)
DEFAULT_ADC_SAMPLES = 1      # INA219 on-chip averaging (1 = fastest, ~1 kHz)

# INA219 register addresses
_REG_CONFIG = 0x00
_REG_SHUNT_VOLTAGE = 0x01
_REG_BUS_VOLTAGE = 0x02

_CONFIG_RESET = 0x8000

_BUS_VOLTAGE_LSB_MV = 4.0    # bus voltage register: value in bits 15..3, 4 mV/LSB
_SHUNT_VOLTAGE_LSB_MV = 0.01  # shunt voltage register: 10 uV/LSB

# ADC field codes -> number of hardware-averaged samples. 1 == a single 12-bit
# conversion (~532 us); higher values average on-chip at ~532 us per sample.
_ADC_CODES = {
    1: 0b0011,
    2: 0b1000,
    4: 0b1001,
    8: 0b1010,
    16: 0b1011,
    32: 0b1100,
    64: 0b1101,
    128: 0b1110,
}
_PGA_CODES = {40: 0b00, 80: 0b01, 160: 0b10, 320: 0b11}


def ina219_config(avg_samples: int = 1, bus_range_v: int = 16, gain_mv: int = 320) -> int:
    """
    Build an INA219 configuration word.

    avg_samples : on-chip averaging (1, 2, 4, 8, 16, 32, 64, 128). 1 is fastest
                  (~1 kHz bus+shunt); 8 gives ~4.3 ms/update with less noise.
    bus_range_v : 16 or 32 (the Pi runs at ~5 V, so 16 is plenty).
    gain_mv     : shunt full-scale range (40, 80, 160, 320). 320 mV across a
                  0.1 ohm shunt covers ~3.2 A.
    """
    if avg_samples not in _ADC_CODES:
        raise ValueError(f"avg_samples must be one of {sorted(_ADC_CODES)}")
    if gain_mv not in _PGA_CODES:
        raise ValueError(f"gain_mv must be one of {sorted(_PGA_CODES)}")
    adc = _ADC_CODES[avg_samples]
    brng = 1 if bus_range_v > 16 else 0
    pga = _PGA_CODES[gain_mv]
    mode = 0b111  # shunt and bus, continuous
    return (brng << 13) | (pga << 11) | (adc << 7) | (adc << 3) | mode


_CONFIG_DEFAULT = ina219_config()  # 0x199F: 16 V, +/-320 mV, 12-bit, continuous


@dataclass(frozen=True)
class PowerReading:
    label: str
    voltage_mv: float
    current_ma: float

    @property
    def power_mw(self) -> float:
        return self.voltage_mv * self.current_ma / 1000


@dataclass(frozen=True)
class SessionResult:
    """Results from a completed power measurement session."""

    session: str
    duration_s: float
    n_samples: int
    energy_mj: dict[str, float]  # {channel: millijoules}
    avg_power_mw: dict[str, float] = field(default_factory=dict)
    peak_power_mw: dict[str, float] = field(default_factory=dict)
    min_power_mw: dict[str, float] = field(default_factory=dict)
    sample_rate_hz: float = 0.0

    @property
    def total_energy_mj(self) -> float:
        """Total energy summed across all measured channels, in millijoules."""
        return sum(self.energy_mj.values())

    @property
    def total_avg_power_mw(self) -> float:
        """Mean power summed across all channels, in milliwatts."""
        return sum(self.avg_power_mw.values())

    @property
    def total_peak_power_mw(self) -> float:
        """Per-channel peak power, summed across channels, in milliwatts."""
        return sum(self.peak_power_mw.values())

    @property
    def energy_per_event_mj(self) -> float:
        """total_energy_mj, provided for callers that divide by an event count."""
        return self.total_energy_mj


def _import_smbus():
    """Return an SMBus class from smbus2 (preferred) or smbus."""
    try:
        import smbus2

        return smbus2.SMBus
    except ImportError:
        pass
    try:
        import smbus

        return smbus.SMBus
    except ImportError as e:
        raise RuntimeError(
            "No I2C library available. On the Raspberry Pi install one with:\n"
            "    sudo apt install python3-smbus   # or:  pip install smbus2"
        ) from e


class INA219:
    """Minimal INA219 driver: bus voltage and shunt-derived current over I2C."""

    def __init__(
        self,
        bus: int = DEFAULT_I2C_BUS,
        address: int = DEFAULT_I2C_ADDRESS,
        shunt_ohms: float = DEFAULT_SHUNT_OHMS,
        config: int = _CONFIG_DEFAULT,
    ):
        SMBus = _import_smbus()
        self._bus = SMBus(bus)
        self._addr = address
        self._shunt_ohms = shunt_ohms
        self._reset(config)

    def _write_register(self, reg: int, value: int) -> None:
        self._bus.write_i2c_block_data(
            self._addr, reg, [(value >> 8) & 0xFF, value & 0xFF]
        )

    def _read_register(self, reg: int) -> int:
        msb, lsb = self._bus.read_i2c_block_data(self._addr, reg, 2)
        return (msb << 8) | lsb

    def _reset(self, config: int) -> None:
        self._write_register(_REG_CONFIG, _CONFIG_RESET)
        time.sleep(0.002)
        self._write_register(_REG_CONFIG, config)
        time.sleep(0.002)

    def bus_voltage_mv(self) -> float:
        raw = self._read_register(_REG_BUS_VOLTAGE)
        if raw & 0x0001:
            _logger.warning("[PWR] INA219 math overflow — reading may be invalid")
        return float(raw >> 3) * _BUS_VOLTAGE_LSB_MV

    def shunt_voltage_mv(self) -> float:
        raw = self._read_register(_REG_SHUNT_VOLTAGE)
        if raw & 0x8000:  # 16-bit two's complement
            raw -= 0x10000
        return raw * _SHUNT_VOLTAGE_LSB_MV

    def current_ma(self) -> float:
        # I(mA) = Vshunt(mV) / Rshunt(ohm)
        return self.shunt_voltage_mv() / self._shunt_ohms

    def close(self) -> None:
        try:
            self._bus.close()
        except Exception:
            pass


class PowerMonitor:
    """
    Reads the main USB supply voltage and current from an INA219.

    Exposes a single channel (default label "USB") so it is a drop-in for the
    multi-channel PowerSampler below.
    """

    def __init__(
        self,
        *,
        bus: int = DEFAULT_I2C_BUS,
        address: int = DEFAULT_I2C_ADDRESS,
        shunt_ohms: float = DEFAULT_SHUNT_OHMS,
        label: str = DEFAULT_LABEL,
        config: int = _CONFIG_DEFAULT,
    ):
        self._label = label
        self._ina = INA219(
            bus=bus, address=address, shunt_ohms=shunt_ohms, config=config
        )

    @property
    def channels(self) -> list[str]:
        """Labels of all discovered measurement channels."""
        return [self._label]

    def read(self, label: str | None = None, n_avg: int = 1) -> PowerReading:
        """Return a PowerReading. Raises KeyError if label is not this channel."""
        if label is not None and label != self._label:
            raise KeyError(label)
        if n_avg <= 1:
            return PowerReading(
                label=self._label,
                voltage_mv=self._ina.bus_voltage_mv(),
                current_ma=self._ina.current_ma(),
            )
        v_sum = 0.0
        c_sum = 0.0
        for _ in range(n_avg):
            v_sum += self._ina.bus_voltage_mv()
            c_sum += self._ina.current_ma()
        return PowerReading(
            label=self._label, voltage_mv=v_sum / n_avg, current_ma=c_sum / n_avg
        )

    def read_all(self, n_avg: int = 1) -> dict[str, PowerReading]:
        """Read all channels. Returns {label: PowerReading}."""
        return {self._label: self.read(n_avg=n_avg)}

    def close(self):
        self._ina.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class PowerSampler:
    """
    Samples PowerMonitor channels as fast as `interval_s` allows in a daemon
    thread. Pass interval_s=0 to sample flat out.

    Each start()/stop() pair is a named session. stop() returns a SessionResult
    with, per channel: integrated energy, mean power, peak (highest single
    sample) power and minimum power, plus the achieved sample rate. Energy is
    integrated over the *measured* interval between consecutive samples, so an
    uneven rate under load does not bias the total.

    If csv_file is provided, each sample is written as one row with columns:
        session, timestamp, elapsed_s, <ch>_mv, <ch>_ma, <ch>_mw, ...
    The file is opened on the first start() call and kept open across sessions;
    close() flushes and closes it.

    Log lines are emitted at INFO level with a [PWR] prefix so they interleave
    with application logs in a single chronological record.
    """

    def __init__(
        self,
        monitor: PowerMonitor,
        *,
        interval_s: float = DEFAULT_INTERVAL_S,
        label: str = "",
        csv_file: str | None = None,
        log_samples: bool = False,
        read_n_avg: int = 1,
    ):
        self._monitor = monitor
        self._interval = max(0.0, interval_s)
        self._label = label
        self._log_samples = log_samples
        self._read_n_avg = read_n_avg
        self._csv_path = csv_file
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._session: str = ""
        self._start_time: float = 0.0
        self._last_t: float = 0.0
        self._energy_acc: dict[str, float] = {}  # channel -> mW·s (== mJ)
        self._peak_mw: dict[str, float] = {}
        self._min_mw: dict[str, float] = {}
        self._n_samples: int = 0
        self._csv_fh = None
        self._csv_writer = None
        self._csv_channels: list[str] = []

    def start(self, session: str) -> None:
        """Begin a named power measurement session."""
        if not session:
            raise ValueError("session name must be non-empty")
        if self._thread is not None and self._thread.is_alive():
            return
        channels = [self._label] if self._label else self._monitor.channels
        if not channels:
            _logger.warning("[PWR] No channels found — power sampling disabled")
            return
        self._session = session
        self._energy_acc = {ch: 0.0 for ch in channels}
        self._peak_mw = {ch: 0.0 for ch in channels}
        self._min_mw = {ch: float("inf") for ch in channels}
        self._n_samples = 0
        self._start_time = time.perf_counter()
        self._last_t = self._start_time
        self._open_csv(channels)
        _logger.info(
            "[PWR] Session '%s': sampling %s every %.1fms",
            session, channels, self._interval * 1000,
        )
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pwr-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> SessionResult | None:
        """End the current session and return its results."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if not self._session:
            return None
        duration = time.perf_counter() - self._start_time
        # _energy_acc is sum(power_mW * dt_s) == milliwatt-seconds == millijoules.
        energy_mj = dict(self._energy_acc)
        avg_power_mw = {
            ch: (mj / duration if duration > 0 else 0.0)
            for ch, mj in self._energy_acc.items()
        }
        peak_power_mw = dict(self._peak_mw)
        min_power_mw = {
            ch: (0.0 if v == float("inf") else v) for ch, v in self._min_mw.items()
        }
        rate = self._n_samples / duration if duration > 0 else 0.0
        result = SessionResult(
            session=self._session,
            duration_s=duration,
            n_samples=self._n_samples,
            energy_mj=energy_mj,
            avg_power_mw=avg_power_mw,
            peak_power_mw=peak_power_mw,
            min_power_mw=min_power_mw,
            sample_rate_hz=rate,
        )
        _logger.info(
            "[PWR] Session '%s' complete: %.2fs  %d samples (%.0f Hz)  "
            "total=%.3fmJ  avg=%.1fmW  peak=%.1fmW",
            self._session, duration, self._n_samples, rate,
            result.total_energy_mj, result.total_avg_power_mw,
            result.total_peak_power_mw,
        )
        self._session = ""
        return result

    def close(self):
        """Stop sampling and close the CSV file."""
        self.stop()
        if self._csv_fh is not None:
            self._csv_fh.close()
            self._csv_fh = None
            self._csv_writer = None
            self._csv_channels = []

    def _open_csv(self, channels: list[str]) -> None:
        if self._csv_path is None or self._csv_fh is not None:
            return
        import os

        is_new = not os.path.exists(self._csv_path)
        self._csv_fh = open(self._csv_path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_fh)
        self._csv_channels = channels
        if is_new:
            header = ["session", "timestamp", "elapsed_s"]
            for ch in channels:
                header += [f"{ch}_mv", f"{ch}_ma", f"{ch}_mw"]
            self._csv_writer.writerow(header)
            self._csv_fh.flush()

    def _run(self):
        while not self._stop.wait(self._interval):
            try:
                if self._label:
                    readings = [self._monitor.read(self._label, n_avg=self._read_n_avg)]
                else:
                    readings = list(
                        self._monitor.read_all(n_avg=self._read_n_avg).values()
                    )
            except OSError as e:
                _logger.warning("[PWR] Read error: %s", e)
                continue
            if not readings:
                continue

            now = time.perf_counter()
            dt = now - self._last_t  # measured interval, robust to jitter
            self._last_t = now
            elapsed = now - self._start_time
            by_label = {r.label: r for r in readings}

            for r in readings:
                p = r.power_mw
                self._energy_acc[r.label] = self._energy_acc.get(r.label, 0.0) + p * dt
                if p > self._peak_mw.get(r.label, 0.0):
                    self._peak_mw[r.label] = p
                if p < self._min_mw.get(r.label, float("inf")):
                    self._min_mw[r.label] = p

            if self._log_samples:
                parts = "  ".join(
                    f"{r.label}: {r.voltage_mv:.0f}mV {r.current_ma:.0f}mA {r.power_mw:.0f}mW"
                    for r in readings
                )
                _logger.info(
                    "[PWR] %s  [%s]  %s",
                    time.strftime("%H:%M:%S"), self._session, parts,
                )

            if self._csv_writer is not None:
                row: list = [self._session, time.strftime("%H:%M:%S"), f"{elapsed:.4f}"]
                for ch in self._csv_channels:
                    r = by_label.get(ch)
                    if r is not None:
                        row += [f"{r.voltage_mv:.1f}", f"{r.current_ma:.1f}", f"{r.power_mw:.1f}"]
                    else:
                        row += ["", "", ""]
                self._csv_writer.writerow(row)
                self._csv_fh.flush()

            self._n_samples += 1


def make_power_sampler(
    csv_file: str | None = None,
    *,
    interval_s: float = DEFAULT_INTERVAL_S,
    adc_samples: int = DEFAULT_ADC_SAMPLES,
    log_samples: bool = False,
):
    """
    Build (PowerMonitor, PowerSampler) using the module-level INA219 defaults
    (bus, address, shunt, sampling period). Benchmark scripts call this so they
    do not have to expose every knob as a command-line flag.

    Returns (None, None) — after printing a warning — when the INA219 or an I2C
    library is unavailable, so callers can treat power measurement as optional.
    """
    try:
        monitor = PowerMonitor(config=ina219_config(avg_samples=adc_samples))
        sampler = PowerSampler(
            monitor, interval_s=interval_s, csv_file=csv_file, log_samples=log_samples
        )
        return monitor, sampler
    except Exception as exc:  # missing smbus, no chip on the bus, bad wiring...
        _logger.warning("[PWR] power measurement unavailable: %s", exc)
        print(f"Warning: power measurement disabled ({exc})")
        return None, None


# ---------------------------------------------------------------------------
# Test main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(
        description="Read main USB power from an INA219 on the Raspberry Pi 5 I2C bus."
    )
    parser.add_argument(
        "--bus", type=int, default=DEFAULT_I2C_BUS,
        help=f"I2C bus number (default: {DEFAULT_I2C_BUS})",
    )
    parser.add_argument(
        "--address", type=lambda x: int(x, 0), default=DEFAULT_I2C_ADDRESS,
        help=f"INA219 I2C address (default: 0x{DEFAULT_I2C_ADDRESS:02x})",
    )
    parser.add_argument(
        "--shunt-ohms", type=float, default=DEFAULT_SHUNT_OHMS,
        help=f"Shunt resistor value in ohms (default: {DEFAULT_SHUNT_OHMS})",
    )
    parser.add_argument(
        "--label", default=DEFAULT_LABEL,
        help=f"Channel label (default: {DEFAULT_LABEL})",
    )
    parser.add_argument(
        "--adc-samples", type=int, default=1, choices=sorted(_ADC_CODES),
        help="INA219 on-chip averaging (default: 1 = fastest).",
    )
    parser.add_argument(
        "--interval", type=float, default=0.002, metavar="SECS",
        help="Sampler interval in seconds; 0 = flat out (default: 0.002).",
    )
    parser.add_argument(
        "--bench", type=int, metavar="N",
        help="Benchmark: read the channel N times and report average latency.",
    )
    parser.add_argument(
        "--sample", type=float, metavar="SECS",
        help="Run PowerSampler for SECS seconds and print the session result.",
    )
    parser.add_argument("--csv", metavar="FILE", help="Write samples to a CSV file.")
    args = parser.parse_args()

    mon = PowerMonitor(
        bus=args.bus, address=args.address,
        shunt_ohms=args.shunt_ohms, label=args.label,
        config=ina219_config(avg_samples=args.adc_samples),
    )
    print(f"I2C bus    : {args.bus}")
    print(f"Address    : 0x{args.address:02x}")
    print(f"Shunt      : {args.shunt_ohms} ohm")
    print(f"ADC avg    : {args.adc_samples}")
    print(f"Channels   : {mon.channels}\n")

    if args.sample:
        sampler = PowerSampler(mon, interval_s=args.interval, csv_file=args.csv)
        sampler.start("test")
        time.sleep(args.sample)
        result = sampler.stop()
        sampler.close()
        if result:
            print(f"\nSession      : {result.session}")
            print(f"Duration     : {result.duration_s:.2f} s")
            print(f"Samples      : {result.n_samples}  ({result.sample_rate_hz:.0f} Hz)")
            print(f"Avg power    : {result.total_avg_power_mw:.1f} mW")
            print(f"Peak power   : {result.total_peak_power_mw:.1f} mW")
            print(f"Min power    : {sum(result.min_power_mw.values()):.1f} mW")
            print(f"Energy       : {result.total_energy_mj:.1f} mJ "
                  f"({result.total_energy_mj / 1000:.3f} J)")

    elif args.bench:
        mon.read_all()  # warm up
        t0 = time.perf_counter_ns()
        for _ in range(args.bench):
            mon.read_all()
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        print(f"{elapsed_ms / args.bench * 1000:.1f} us avg/call  "
              f"({args.bench / (elapsed_ms / 1000):.0f} Hz)")

    else:
        r = mon.read()
        print(f"{r.label}:")
        print(f"  voltage  : {r.voltage_mv:.1f} mV")
        print(f"  current  : {r.current_ma:.1f} mA")
        print(f"  power    : {r.power_mw:.1f} mW")

    mon.close()
