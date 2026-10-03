"""Reference data integrity and isolated configuration failure regressions."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cablesizecalculator import tables
from cablesizecalculator.engine import check_loop_impedance, size_cable
from synthetic_data import make_dataset


def test_bundled_and_independent_synthetic_data_are_valid():
    root = Path(__file__).resolve().parents[1]
    tables._validate(make_dataset())
    bundled = json.loads((root / "src/cablesizecalculator/reference_data.json").read_text())
    tables._validate(bundled)
    assert bundled["metadata"]["validation_status"] == "unverified"


@pytest.fixture
def bundled_dataset():
    synthetic_path = os.environ["CABLESIZE_DATA_FILE"]
    bundled_path = Path(tables.__file__).with_name("reference_data.json")
    tables.load_dataset(bundled_path)
    try:
        yield
    finally:
        tables.load_dataset(synthetic_path)


@pytest.mark.parametrize("curve,trip_current,maximum_impedance,passing_length,failing_length", [
    ("B", 80, 2.3, 100, 125),
    ("C", 160, 1.15, 50, 65),
    ("D", 320, 0.575, 25, 35),
])
def test_bundled_breaker_curves_use_upper_trip_thresholds(
    bundled_dataset, curve, trip_current, maximum_impedance, passing_length, failing_length,
):
    # These failing routes would pass with the former mid-band multipliers.
    passing = check_loop_impedance(2.5, 2.5, 16, mcb_curve=curve, length_m=passing_length)
    failing = check_loop_impedance(2.5, 2.5, 16, mcb_curve=curve, length_m=failing_length)

    assert passing["trip_current_ia_amps"] == trip_current
    assert passing["maximum_loop_impedance_ohm"] == pytest.approx(maximum_impedance)
    assert passing["compliant"] is True
    assert failing["compliant"] is False
    assert failing["route_loop_impedance_ohm"] > maximum_impedance


def test_bundled_c16_sizing_rejects_route_above_upper_trip_limit(bundled_dataset):
    result = size_cable(
        10, 65, voltage=230, phase="1phase", mcb_rating_amps=16, max_volt_drop_pct=5,
    )

    rejected = next(candidate for candidate in result["all_candidates"] if candidate["size_mm2"] == 2.5)
    assert rejected["current_ok"] is True
    assert rejected["voltage_drop_ok"] is True
    assert rejected["pass_loop_impedance"] is False
    assert "loop_impedance" in rejected["failed_constraints"]
    assert result["status"] == "success"
    assert result["recommended_active_size_mm2"] > 2.5
    selected_loop = result["loop_impedance_check"]
    assert selected_loop["compliant"] is True
    assert selected_loop["maximum_loop_impedance_ohm"] == pytest.approx(1.15)
    assert selected_loop["route_loop_impedance_ohm"] <= selected_loop["maximum_loop_impedance_ohm"]


@pytest.mark.parametrize("damage", [
    "missing_conductor", "nan", "zero_resistance", "duplicate_size", "unsafe_factor",
    "missing_reference", "unknown_earth", "missing_provenance", "unsupported_frequency",
    "oversized_number", "above_operating_temperature", "oversized_reference_temperature",
])
def test_corrupt_dataset_does_not_replace_working_inputs(tmp_path, damage):
    data = make_dataset()
    numeric = data["tables"]
    if damage == "missing_conductor":
        del numeric["RESISTANCE_TABLE"]["copper"]["V90"]["16"]
    elif damage == "nan":
        numeric["RESISTANCE_TABLE"]["copper"]["V90"]["16"] = float("nan")
    elif damage == "zero_resistance":
        numeric["RESISTANCE_TABLE"]["copper"]["V90"]["16"] = 0
    elif damage == "duplicate_size":
        numeric["CONDUCTOR_SIZES_COPPER"].append(16)
    elif damage == "unsafe_factor":
        numeric["TEMP_DERATING_AIR"]["V90"]["50"] = 2
    elif damage == "missing_reference":
        del numeric["TEMP_DERATING_AIR"]["V90"]["40"]
    elif damage == "unknown_earth":
        numeric["TABLE_5_1_EARTH"]["16"] = 8
    elif damage == "missing_provenance":
        del data["metadata"]["source"]
    elif damage == "unsupported_frequency":
        data["metadata"]["assumptions"]["frequency_hz"] = 60
    elif damage == "oversized_number":
        numeric["RESISTANCE_TABLE"]["copper"]["V90"]["16"] = 10 ** 1000
    elif damage == "above_operating_temperature":
        numeric["TEMP_DERATING_AIR"]["V90"]["80"] = 0.1
    elif damage == "oversized_reference_temperature":
        data["metadata"]["assumptions"]["reference_air_temp_c"] = 10 ** 1000
    before = copy.deepcopy(tables.RESISTANCE_TABLE)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        tables.load_dataset(path)
    assert tables.RESISTANCE_TABLE == before
    tables.require_dataset()


def test_duplicate_json_key_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError, match="duplicate"):
        tables.load_dataset(path)


def test_override_load_preserves_imported_table_references(tmp_path):
    reference = tables.RESISTANCE_TABLE
    original = make_dataset()
    altered = make_dataset()
    altered["metadata"]["dataset_id"] = "second-synthetic-fixture"
    altered["tables"]["RESISTANCE_TABLE"]["copper"]["V90"] = {
        key: value * 2 for key, value in altered["tables"]["RESISTANCE_TABLE"]["copper"]["V90"].items()
    }
    path = tmp_path / "override.json"
    try:
        path.write_text(json.dumps(altered))
        tables.load_dataset(path)
        assert tables.RESISTANCE_TABLE is reference
        assert reference["copper"]["V90"][16] == 3
        assert tables.DATA_PROVENANCE["dataset_id"] == "second-synthetic-fixture"
    finally:
        path.write_text(json.dumps(original))
        tables.load_dataset(path)


def test_invalid_override_never_silently_uses_bundled_data(tmp_path):
    env = os.environ.copy()
    env["CABLESIZE_DATA_FILE"] = str(tmp_path / "missing.json")
    code = """
from cablesizecalculator.server import get_standards_info
from cablesizecalculator.engine import size_cable
info = get_standards_info()
assert info['data_provenance']['configured'] is False
try:
    size_cable(10, 10)
except ValueError as error:
    assert 'CABLESIZE_DATA_FILE' in str(error)
else:
    raise AssertionError('Invalid override produced a cable recommendation')
"""
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
