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


def profile_dataset():
    """Use the deliberately sparse supplied columns to test schema contracts."""
    return json.loads(Path(tables.__file__).with_name("reference_data.json").read_text())


def test_profile_validation_preserves_sparse_ratings_and_legacy_columns():
    raw = profile_dataset()
    validated, _ = tables._validate(raw)
    partial = validated["CURRENT_RATING_PROFILES"]["flat_2c_earth_partially_surrounded"]
    assert partial["ratings"] == {2.5: 17, 4: 23}
    assert partial["loaded_conductors"] == 2
    assert partial["reference_temperature_c"] == 40
    assert validated["CURRENT_RATINGS"]["copper"]["V90"]["in_thermal_insulation"][6] == 18


@pytest.mark.parametrize("damage", [
    "missing_field", "unknown_field", "empty_id", "unknown_construction", "unknown_material",
    "unknown_insulation", "three_loaded_conductors", "boolean_loaded_conductors",
    "fractional_loaded_conductors", "unknown_method", "missing_thermal_exposure",
    "exposure_without_thermal_insulation", "missing_source", "wrong_reference", "nonfinite_reference",
    "oversized_reference", "empty_ratings", "zero_rating", "negative_rating", "nonfinite_rating",
    "oversized_rating", "decreasing_ratings", "unknown_size", "duplicate_numeric_size",
    "duplicate_profile_conditions", "profiles_not_object",
])
def test_invalid_profiles_leave_all_loaded_tables_unchanged(tmp_path, damage):
    raw = profile_dataset()
    profiles = raw["tables"]["CURRENT_RATING_PROFILES"]
    profile = profiles["flat_2c_earth_partially_surrounded"]
    if damage == "missing_field":
        del profile["source"]
    elif damage == "unknown_field":
        profile["extra"] = "not supported"
    elif damage == "empty_id":
        profiles[" "] = profiles.pop("flat_2c_earth_partially_surrounded")
    elif damage == "unknown_construction":
        profile["cable_construction"] = "circular_multicore"
    elif damage == "unknown_material":
        profile["conductor_material"] = "steel"
    elif damage == "unknown_insulation":
        profile["insulation"] = "V75"
    elif damage == "three_loaded_conductors":
        profile["loaded_conductors"] = 3
    elif damage == "boolean_loaded_conductors":
        profile["loaded_conductors"] = True
    elif damage == "fractional_loaded_conductors":
        profile["loaded_conductors"] = 2.0
    elif damage == "unknown_method":
        profile["installation_method"] = "unknown"
    elif damage == "missing_thermal_exposure":
        profile["insulation_exposure"] = "none"
    elif damage == "exposure_without_thermal_insulation":
        profile["installation_method"] = "unenclosed_in_air"
    elif damage == "missing_source":
        profile["source"] = " "
    elif damage == "wrong_reference":
        profile["reference_temperature_c"] = 35
    elif damage == "nonfinite_reference":
        profile["reference_temperature_c"] = float("nan")
    elif damage == "oversized_reference":
        profile["reference_temperature_c"] = 10 ** 1000
    elif damage == "empty_ratings":
        profile["ratings"] = {}
    elif damage == "zero_rating":
        profile["ratings"]["2.5"] = 0
    elif damage == "negative_rating":
        profile["ratings"]["2.5"] = -1
    elif damage == "nonfinite_rating":
        profile["ratings"]["2.5"] = float("inf")
    elif damage == "oversized_rating":
        profile["ratings"]["2.5"] = 10 ** 1000
    elif damage == "decreasing_ratings":
        profile["ratings"]["2.5"] = 24
    elif damage == "unknown_size":
        profile["ratings"]["3"] = 19
    elif damage == "duplicate_numeric_size":
        profile["ratings"]["2.50"] = 17
    elif damage == "duplicate_profile_conditions":
        profiles["duplicate"] = copy.deepcopy(profile)
    elif damage == "profiles_not_object":
        raw["tables"]["CURRENT_RATING_PROFILES"] = []

    before = {name: copy.deepcopy(getattr(tables, name)) for name in tables._TABLE_NAMES}
    provenance = copy.deepcopy(tables.DATA_PROVENANCE)
    path = tmp_path / "bad-profiles.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="profile|PROFILE"):
        tables.load_dataset(path)
    assert {name: getattr(tables, name) for name in tables._TABLE_NAMES} == before
    assert tables.DATA_PROVENANCE == provenance
    tables.require_dataset()


def test_loading_legacy_dataset_clears_profiles_without_replacing_imported_reference(tmp_path):
    reference = tables.CURRENT_RATING_PROFILES
    synthetic_path = os.environ["CABLESIZE_DATA_FILE"]
    try:
        tables.load_dataset(Path(tables.__file__).with_name("reference_data.json"))
        assert len(reference) == 3
        legacy = make_dataset()
        legacy["schema_version"] = 1
        legacy["tables"].pop("CURRENT_RATING_PROFILES", None)
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(legacy), encoding="utf-8")
        validated, _ = tables._validate(legacy)
        assert validated["CURRENT_RATING_PROFILES"] == {}
        tables.load_dataset(path)
        assert tables.CURRENT_RATING_PROFILES is reference
        assert reference == {}
        assert tables.DATA_PROVENANCE["dataset_id"] == legacy["metadata"]["dataset_id"]
    finally:
        tables.load_dataset(synthetic_path)


@pytest.fixture
def bundled_dataset():
    synthetic_path = os.environ["CABLESIZE_DATA_FILE"]
    bundled_path = Path(tables.__file__).with_name("reference_data.json")
    tables.load_dataset(bundled_path)
    try:
        yield
    finally:
        tables.load_dataset(synthetic_path)


@pytest.mark.parametrize("method,exposure,expected_size,expected_rating", [
    ("in_thermal_insulation", "partially_surrounded", 4, 23),
    ("in_thermal_insulation", "completely_surrounded", 6, 22),
    ("unenclosed_in_air", "none", 2.5, 23),
])
def test_bundled_flat_tps_20a_circuit_matches_supplied_comparison(
    bundled_dataset, method, exposure, expected_size, expected_rating,
):
    result = size_cable(
        20, 20, voltage=230, phase="1phase", conductor_material="copper", insulation="V90",
        installation_method=method, cable_construction="flat_2c_earth", insulation_exposure=exposure,
        mcb_rating_amps=20, max_volt_drop_pct=3,
    )

    assert result["status"] == "success"
    assert result["recommended_active_size_mm2"] == expected_size
    selected = next(candidate for candidate in result["all_candidates"] if candidate["size_mm2"] == expected_size)
    assert selected["base_capacity_a"] == expected_rating
    assert selected["current_ok"] is True
    assert selected["voltage_drop_ok"] is True
    assert selected["pass_loop_impedance"] is True


def test_bundled_legacy_generic_insulation_rating_is_unchanged(bundled_dataset):
    result = size_cable(
        20, 20, voltage=230, phase="1phase", installation_method="in_thermal_insulation",
        mcb_rating_amps=20,
    )
    assert result["status"] == "success"
    assert result["recommended_active_size_mm2"] == 10


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
