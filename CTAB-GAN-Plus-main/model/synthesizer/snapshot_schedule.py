"""Deterministic epoch schedules for paired GAN snapshots."""

import math
from numbers import Integral, Real


def snapshot_epochs(epochs, snapshot_frq=None, snapshot_schedule=None):
    """Return increasing, one-based epochs; a gradual schedule overrides frequency.

    Place count snapshots on a power curve between epochs 1 and epochs.
    A power greater than one concentrates snapshots at the start. Both endpoints
    belong to the curve, so the final epoch is never an extra appended snapshot.
    Rounding can create duplicates on short runs; these are removed.
    No random draws are used, so capture scheduling cannot affect GAN seeding.
    """
    def positive_int(value, name):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    epochs = positive_int(epochs, "epochs")
    if snapshot_schedule is None:
        if snapshot_frq is None:
            return []
        interval = positive_int(snapshot_frq, "snapshot_frq")
        return sorted(set(range(interval, epochs + 1, interval)) | {epochs})

    if not isinstance(snapshot_schedule, dict):
        raise ValueError("snapshot_schedule must be an object or null")
    unknown = set(snapshot_schedule) - {"count", "power"}
    if unknown:
        raise ValueError(f"Unknown snapshot_schedule settings: {sorted(unknown)}")
    count = positive_int(snapshot_schedule.get("count", 12), "snapshot_schedule.count")
    if count < 2:
        raise ValueError("snapshot_schedule.count must be at least two")
    power = snapshot_schedule.get("power", 2.0)
    if isinstance(power, bool) or not isinstance(power, Real) or not math.isfinite(power) or power <= 1:
        raise ValueError("snapshot_schedule.power must be finite and greater than 1")
    count = min(count, epochs)
    if count == 1:
        return [1]
    return sorted({
        1 + round((epochs - 1) * (index / (count - 1)) ** power)
        for index in range(count)
    })
