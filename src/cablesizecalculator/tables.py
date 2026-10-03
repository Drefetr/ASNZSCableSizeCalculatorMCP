"""Load and validate bundled or operator-supplied reference data.

Set CABLESIZE_DATA_FILE to override the bundled data; see README.md.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

MATERIALS = ("copper", "aluminium")
INSULATIONS = ("V90", "X90")
INSTALLATION_METHODS = (
    "in_conduit_in_air", "unenclosed_in_air", "in_thermal_insulation",
    "underground_duct", "buried_direct",
)

CONDUCTOR_SIZES_COPPER: list[float] = []
CONDUCTOR_SIZES_ALUMINIUM: list[float] = []
TABLE_5_1_EARTH: dict[float, float] = {}
RESISTANCE_TABLE: dict[str, dict[str, dict[float, float]]] = {}
REACTANCE_TABLE: dict[float, float] = {}
CURRENT_RATINGS: dict[str, dict[str, dict[str, dict[float, float]]]] = {}
TEMP_DERATING_AIR: dict[str, dict[float, float]] = {}
TEMP_DERATING_GROUND: dict[str, dict[float, float]] = {}
CIRCUITS_GROUPING_DERATING: dict[int, float] = {}
DEPTH_DERATING_GROUND: dict[float, float] = {}
ADIABATIC_K_FACTORS: dict[tuple[str, str], float] = {}
MCB_TRIP_MULTIPLIERS: dict[str, float] = {}

DATA_PROVENANCE: dict[str, Any] = {
    "configured": False,
    "validation_status": "not_configured",
    "distribution": "Bundled data is unverified; local reference data can override it.",
}
_DATA_ERROR: str | None = None
_TABLE_NAMES = (
    "CONDUCTOR_SIZES_COPPER", "CONDUCTOR_SIZES_ALUMINIUM", "TABLE_5_1_EARTH",
    "RESISTANCE_TABLE", "REACTANCE_TABLE", "CURRENT_RATINGS", "TEMP_DERATING_AIR",
    "TEMP_DERATING_GROUND", "CIRCUITS_GROUPING_DERATING", "DEPTH_DERATING_GROUND",
    "ADIABATIC_K_FACTORS", "MCB_TRIP_MULTIPLIERS",
)


def _number(value: Any, label: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    try:
        value = float(value)
    except OverflowError:
        raise ValueError(f"{label} exceeds the supported numerical range.") from None
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{label} must be finite and {'non-negative' if zero else 'positive'}.")
    return value


def _object(value: Any, label: str, keys: tuple[str, ...] | None = None) -> dict:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty object.")
    if keys is not None and set(value) != set(keys):
        raise ValueError(f"{label} must contain exactly: {', '.join(keys)}.")
    return value


def _sizes(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty array.")
    sizes = [_number(item, label) for item in value]
    if sizes != sorted(set(sizes)):
        raise ValueError(f"{label} must be sorted, with no duplicate sizes.")
    return sizes


def _numeric_map(
    value: Any, label: str, *, zero_values: bool = False,
    positive_keys: bool = True, integer_keys: bool = False,
) -> dict[float, float]:
    result = {}
    for key, item in _object(value, label).items():
        try:
            numeric_key = float(key)
        except (TypeError, ValueError):
            raise ValueError(f"{label} keys must be numeric.") from None
        if not math.isfinite(numeric_key) or (positive_keys and numeric_key <= 0):
            raise ValueError(f"{label} keys are outside their permitted range.")
        if integer_keys and not numeric_key.is_integer():
            raise ValueError(f"{label} keys must be integers.")
        if numeric_key in result:
            raise ValueError(f"{label} contains equivalent duplicate keys.")
        result[numeric_key] = _number(item, label, zero=zero_values)
    return dict(sorted(result.items()))


def _covers(mapping: dict, sizes: list[float], label: str) -> None:
    if set(mapping) != set(sizes):
        raise ValueError(f"{label} must cover exactly the configured conductor sizes.")


def _decreasing(mapping: dict, label: str) -> None:
    values = list(mapping.values())
    if any(a < b for a, b in zip(values, values[1:])):
        raise ValueError(f"{label} must not increase as its keys increase.")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Reference data contains a duplicate JSON key.")
        result[key] = value
    return result


def _validate(raw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    _object(raw, "dataset", ("schema_version", "metadata", "tables"))
    if isinstance(raw["schema_version"], bool) or raw["schema_version"] != 1:
        raise ValueError("Unsupported reference data schema_version; expected 1.")
    metadata = _object(raw["metadata"], "metadata", (
        "dataset_id", "source", "licence", "standard_editions", "validation_status", "assumptions",
    ))
    for key in ("dataset_id", "source", "licence"):
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise ValueError(f"metadata.{key} must be a non-empty string.")
    editions = metadata["standard_editions"]
    if not isinstance(editions, list) or not editions or any(
        not isinstance(item, str) or not item.strip() for item in editions
    ):
        raise ValueError("metadata.standard_editions must identify editions or synthetic data.")
    if metadata["validation_status"] not in (
        "unverified", "synthetic_test_only", "independently_validated",
    ):
        raise ValueError("metadata.validation_status is unsupported.")
    assumptions = _object(metadata["assumptions"], "metadata.assumptions", (
        "frequency_hz", "reference_air_temp_c", "reference_ground_temp_c",
        "cable_construction", "loaded_conductors", "grouping_layout", "soil_thermal_resistivity",
    ))
    if _number(assumptions["frequency_hz"], "frequency_hz") != 50:
        raise ValueError("Only a 50 Hz AC reference dataset is supported.")
    for key in ("reference_air_temp_c", "reference_ground_temp_c"):
        value = assumptions[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"metadata.assumptions.{key} must be finite.")
        try:
            numeric_value = float(value)
        except OverflowError:
            raise ValueError(f"metadata.assumptions.{key} exceeds the supported numerical range.") from None
        if not math.isfinite(numeric_value):
            raise ValueError(f"metadata.assumptions.{key} must be finite.")
    for key in ("cable_construction", "loaded_conductors", "grouping_layout", "soil_thermal_resistivity"):
        if not isinstance(assumptions[key], str) or not assumptions[key].strip():
            raise ValueError(f"metadata.assumptions.{key} must document the dataset conditions.")

    tables = _object(raw["tables"], "tables", _TABLE_NAMES)
    result: dict[str, Any] = {}
    sizes = {}
    for material, name in zip(MATERIALS, ("CONDUCTOR_SIZES_COPPER", "CONDUCTOR_SIZES_ALUMINIUM")):
        sizes[material] = result[name] = _sizes(tables[name], name)
    earth = _numeric_map(tables["TABLE_5_1_EARTH"], "TABLE_5_1_EARTH")
    _covers(earth, sizes["copper"], "TABLE_5_1_EARTH")
    if any(size not in sizes["copper"] or size > active for active, size in earth.items()):
        raise ValueError("Earth pairings must use configured copper sizes no larger than the active.")
    if any(a > b for a, b in zip(earth.values(), list(earth.values())[1:])):
        raise ValueError("Earth pairings must not decrease with active size.")
    result["TABLE_5_1_EARTH"] = earth
    resistance = {}
    ratings = {}
    k_factors = {}
    for name in ("RESISTANCE_TABLE", "CURRENT_RATINGS", "ADIABATIC_K_FACTORS"):
        _object(tables[name], name, MATERIALS)
    for material in MATERIALS:
        resistance[material], ratings[material] = {}, {}
        for name in ("RESISTANCE_TABLE", "CURRENT_RATINGS", "ADIABATIC_K_FACTORS"):
            _object(tables[name][material], f"{name}.{material}", INSULATIONS)
        for insulation in INSULATIONS:
            label = f"RESISTANCE_TABLE.{material}.{insulation}"
            resistance[material][insulation] = _numeric_map(tables["RESISTANCE_TABLE"][material][insulation], label)
            _covers(resistance[material][insulation], sizes[material], label)
            _decreasing(resistance[material][insulation], label)
            k_factors[(material, insulation)] = _number(
                tables["ADIABATIC_K_FACTORS"][material][insulation], "ADIABATIC_K_FACTORS",
            )
            ratings[material][insulation] = {}
            methods = _object(tables["CURRENT_RATINGS"][material][insulation], "CURRENT_RATINGS", INSTALLATION_METHODS)
            for method in INSTALLATION_METHODS:
                values = _numeric_map(methods[method], "CURRENT_RATINGS")
                _covers(values, sizes[material], "CURRENT_RATINGS")
                if any(a > b for a, b in zip(values.values(), list(values.values())[1:])):
                    raise ValueError("Current ratings must not decrease with conductor size.")
                ratings[material][insulation][method] = values
    result["RESISTANCE_TABLE"], result["CURRENT_RATINGS"] = resistance, ratings
    result["ADIABATIC_K_FACTORS"] = k_factors
    result["REACTANCE_TABLE"] = _numeric_map(tables["REACTANCE_TABLE"], "REACTANCE_TABLE", zero_values=True)
    _covers(result["REACTANCE_TABLE"], sorted(set(sizes["copper"] + sizes["aluminium"])), "REACTANCE_TABLE")
    for name, reference in (
        ("TEMP_DERATING_AIR", "reference_air_temp_c"),
        ("TEMP_DERATING_GROUND", "reference_ground_temp_c"),
    ):
        result[name] = {}
        for insulation, values in _object(tables[name], name, INSULATIONS).items():
            values = _numeric_map(values, name, positive_keys=False)
            _decreasing(values, name)
            maximum_temperature = 75 if insulation == "V90" else 90
            if any(t <= -273.15 or t >= maximum_temperature for t in values):
                raise ValueError(f"{name} rows must be above absolute zero and below the insulation model's operating limit.")
            if values.get(assumptions[reference]) != 1.0:
                raise ValueError(f"{name} must include a unity factor at its declared reference temperature.")
            result[name][insulation] = values
    for name in ("CIRCUITS_GROUPING_DERATING", "DEPTH_DERATING_GROUND"):
        values = _numeric_map(tables[name], name, integer_keys=name == "CIRCUITS_GROUPING_DERATING")
        _decreasing(values, name)
        if any(value > 1 for value in values.values()):
            raise ValueError(f"{name} factors must be at most one.")
        result[name] = values
    if result["CIRCUITS_GROUPING_DERATING"].get(1) != 1:
        raise ValueError("Grouping data must include a unity factor for one circuit.")
    result["MCB_TRIP_MULTIPLIERS"] = {
        curve: _number(value, "MCB_TRIP_MULTIPLIERS")
        for curve, value in _object(tables["MCB_TRIP_MULTIPLIERS"], "MCB_TRIP_MULTIPLIERS", ("B", "C", "D")).items()
    }
    return result, metadata


def load_dataset(path: str | Path) -> None:
    """Validate and install a local dataset atomically, preserving imported references."""
    global _DATA_ERROR
    try:
        with Path(path).open(encoding="utf-8-sig") as stream:
            raw = json.load(stream, object_pairs_hook=_unique_object)
        tables, metadata = _validate(raw)
    except OSError:
        raise ValueError("Cannot read CABLESIZE_DATA_FILE; check the local file and access rights.") from None
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError("CABLESIZE_DATA_FILE must contain UTF-8 JSON reference data.") from None
    for name, value in tables.items():
        target = globals()[name]
        target.clear()
        if isinstance(target, list):
            target.extend(value)
        else:
            target.update(value)
    DATA_PROVENANCE.clear()
    DATA_PROVENANCE.update(metadata, configured=True)
    _DATA_ERROR = None


def require_dataset() -> None:
    """Fail explicitly rather than calculate from missing or invalid reference data."""
    if not DATA_PROVENANCE["configured"]:
        raise ValueError(_DATA_ERROR or (
            "Reference data is not configured. Set CABLESIZE_DATA_FILE to an operator-supplied "
            "JSON dataset; see README.md."
        ))


try:
    load_dataset(os.environ.get("CABLESIZE_DATA_FILE") or Path(__file__).with_name("reference_data.json"))
except ValueError as error:
    _DATA_ERROR = str(error)
    DATA_PROVENANCE["validation_status"] = "invalid_configuration"
