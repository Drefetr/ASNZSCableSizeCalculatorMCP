"""Shared validation for calculation entry points, independent of transport."""

from __future__ import annotations

import math
from numbers import Real

INSTALLATION_METHODS = (
    "in_conduit_in_air", "unenclosed_in_air", "in_thermal_insulation",
    "underground_duct", "buried_direct",
)
CABLE_CONSTRUCTIONS = ("generic", "flat_2c_earth")
INSULATION_EXPOSURES = ("none", "partially_surrounded", "completely_surrounded")


def finite_number(value: float, name: str, *, minimum: float | None = None,
                  maximum: float | None = None, positive: bool = False) -> float:
    """Reject bools, non-numbers, non-finite values and invalid physical bounds."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    if positive and result <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return result


def choice(value: str, name: str, choices: tuple[str, ...], *, upper: bool = False,
           remove_hyphens: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be one of {', '.join(choices)}.")
    result = value.strip().upper() if upper else value.strip().lower()
    if remove_hyphens:
        result = result.replace("-", "")
    if result not in choices:
        raise ValueError(f"Unsupported {name} '{value}'. Use {', '.join(choices)}.")
    return result


def ac_phase(value: str) -> str:
    if isinstance(value, str) and value.strip().lower() == "dc":
        raise ValueError("DC calculations are unsupported: no validated DC cable dataset is provided.")
    return choice(value, "phase", ("3phase", "1phase"))


def material(value: str) -> str:
    return choice(value, "conductor material", ("copper", "aluminium"))


def insulation(value: str) -> str:
    return choice(value, "insulation", ("V90", "X90"), upper=True, remove_hyphens=True)


def circuit_count(value: int) -> int:
    number = finite_number(value, "num_circuits", positive=True)
    if not number.is_integer():
        raise ValueError("num_circuits must be a positive integer.")
    return int(number)


def boolean(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def finite_result(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} exceeds the supported numerical range.")
    return value
