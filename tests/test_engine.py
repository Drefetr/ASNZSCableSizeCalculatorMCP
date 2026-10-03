"""Regression tests over independent artificial data, never cable design validation."""

import json
import math

import pytest

from cablesizecalculator import tables
from cablesizecalculator.engine import (
    calculate_load_current, calculate_voltage_drop, check_loop_impedance,
    check_short_circuit_capacity, get_derating_factor, size_cable,
)
from cablesizecalculator.report import format_calculation_report
from synthetic_data import make_dataset


@pytest.fixture(autouse=True)
def synthetic_dataset(tmp_path):
    """Keep numerical assertions independent of bundled standards-derived rows."""
    original = {name: getattr(tables, name).copy() for name in tables._TABLE_NAMES}
    provenance = tables.DATA_PROVENANCE.copy()
    error = tables._DATA_ERROR
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(make_dataset()), encoding="utf-8")
    tables.load_dataset(path)
    yield
    for name, value in original.items():
        target = getattr(tables, name)
        target.clear()
        target.extend(value) if isinstance(target, list) else target.update(value)
    tables.DATA_PROVENANCE.clear()
    tables.DATA_PROVENANCE.update(provenance)
    tables._DATA_ERROR = error


def test_load_current_units_and_normalization():
    assert calculate_load_current(63, unit=" A ", phase=" 3PHASE ") == 63
    assert calculate_load_current(30, unit="kW", voltage=400, power_factor=0.85) == pytest.approx(30000 / (math.sqrt(3) * 400 * 0.85))
    assert calculate_load_current(10, unit="kVA", voltage=230, phase="1phase") == pytest.approx(10000 / 230)
    assert calculate_load_current(10, unit="hp", voltage=230, phase="1phase", power_factor=1) == pytest.approx(7457 / 230)


@pytest.mark.parametrize("kwargs", [
    {"load": 0}, {"load": -1}, {"load": math.nan}, {"load": math.inf},
    {"load": True}, {"voltage": math.inf}, {"voltage": 0}, {"power_factor": -0.1},
    {"power_factor": 0}, {"power_factor": 1.1}, {"phase": "typo"}, {"phase": "dc"}, {"unit": "typo"},
])
def test_load_current_validates_even_direct_amperes(kwargs):
    inputs = {"load": 63, "unit": "A", **kwargs}
    with pytest.raises(ValueError):
        calculate_load_current(**inputs)


def test_conservative_bounded_derating_and_reference_defaults():
    base = get_derating_factor()
    assert base["total_derating_factor_raw"] == 1
    assert base["ambient_temp_c"] == 40
    off_row = get_derating_factor(ambient_temp_c=42, num_circuits=11)
    assert off_row["ca_temperature"] == 0.9
    assert off_row["cg_grouping"] == 0.3
    assert off_row["tabulated_temperature_c"] == 45
    assert off_row["tabulated_num_circuits"] == 12
    underground = get_derating_factor(installation_method=" UNDERGROUND_DUCT ", depth_m=0.7)
    assert underground["ambient_temp_c"] == 25
    assert underground["cd_depth"] == 0.9
    assert underground["tabulated_depth_m"] == 1
    assert "soil temperature" in underground["underground_warning"]
    tables.DATA_PROVENANCE["assumptions"] = {**tables.DATA_PROVENANCE["assumptions"], "reference_air_temp_c": 35}
    assert get_derating_factor()["ambient_temp_c"] == 35


@pytest.mark.parametrize("kwargs", [
    {"ambient_temp_c": 100}, {"ambient_temp_c": 19}, {"ambient_temp_c": math.nan},
    {"num_circuits": 21}, {"num_circuits": 0}, {"num_circuits": 1.5}, {"num_circuits": True},
    {"insulation": "typo"}, {"installation_method": "typo"}, {"depth_m": -1},
    {"depth_m": 4, "installation_method": "buried_direct"},
    {"depth_m": 0.1, "installation_method": "buried_direct"},
])
def test_derating_rejects_invalid_or_out_of_range_inputs(kwargs):
    with pytest.raises(ValueError):
        get_derating_factor(**kwargs)


def test_voltage_drop_formula_and_rounding_are_separate():
    result = calculate_voltage_drop(16, 50, 40, power_factor=1)
    assert result["rc_ohm_per_km"] == 1.5
    assert result["xc_ohm_per_km"] == 0.1
    assert result["voltage_drop_v_raw"] == pytest.approx(math.sqrt(3) * 50 * 40 * 1.5 / 1000)
    worst = calculate_voltage_drop(16, 50, 40, worst_case_pf=True)
    assert worst["voltage_drop_v_raw"] > result["voltage_drop_v_raw"]
    length = 3.004 * 400 / 100 * 1000 / (math.sqrt(3) * 63 * 1.5)
    boundary = calculate_voltage_drop(16, 63, length, power_factor=1)
    assert boundary["voltage_drop_pct"] == 3.0
    assert boundary["voltage_drop_pct_raw"] == pytest.approx(3.004)
    assert boundary["compliant_3pct"] is False
    sizing = size_cable(63, length, power_factor=1)
    assert sizing["recommended_active_size_mm2"] > 16
    candidate = next(c for c in sizing["all_candidates"] if c["size_mm2"] == 16)
    assert candidate["voltage_drop_ok"] is False
    assert sizing["limiting_factor"] == "voltage_drop"


def test_current_capacity_uses_unrounded_derating(monkeypatch):
    monkeypatch.setitem(tables.TEMP_DERATING_AIR["V90"], 45, 0.912345)
    factors = get_derating_factor(ambient_temp_c=45, num_circuits=3)
    exact_capacity = 65 * factors["total_derating_factor_raw"]
    rounded_capacity = 65 * factors["total_derating_factor"]
    assert exact_capacity > rounded_capacity
    load = (exact_capacity + rounded_capacity) / 2
    selected = size_cable(load, 1, ambient_temp_c=45, num_circuits=3)
    assert selected["recommended_active_size_mm2"] == 16


def _load_uniform_current_ratings(tmp_path, capacity):
    """Load finite artificial ratings through the same validation as operator data."""
    dataset = make_dataset()
    for material in dataset["tables"]["CURRENT_RATINGS"].values():
        for insulation in material.values():
            for method in insulation.values():
                for size in method:
                    method[size] = capacity
    path = tmp_path / "uniform-current-ratings.json"
    path.write_text(json.dumps(dataset, allow_nan=False), encoding="utf-8")
    tables.load_dataset(path)


def test_accepted_finite_ratings_cannot_return_an_infinite_capacity_margin(tmp_path):
    _load_uniform_current_ratings(tmp_path, 1e307)
    assert tables.CURRENT_RATINGS["copper"]["V90"]["in_conduit_in_air"][1.5] == 1e307
    with pytest.raises(ValueError, match="Capacity margin exceeds the supported numerical range"):
        size_cable(1, 1)


def test_large_finite_capacity_margin_remains_strict_json_serializable(tmp_path):
    _load_uniform_current_ratings(tmp_path, 1e100)
    result = size_cable(1, 1)
    decoded = json.loads(json.dumps(result, allow_nan=False))
    assert decoded["status"] == "success"
    assert decoded["recommended_active_size_mm2"] == 1.5
    assert decoded["capacity_margin_pct"] == pytest.approx(1e102)
    assert math.isfinite(decoded["capacity_margin_pct"])


@pytest.mark.parametrize("kwargs", [
    {"length_m": -10}, {"length_m": 0}, {"load_amps": -1}, {"voltage": math.inf},
    {"power_factor": -0.1}, {"power_factor": math.nan}, {"phase": "typo"}, {"phase": "dc"},
    {"conductor_material": "typo"}, {"insulation": "typo"}, {"size_mm2": 17}, {"worst_case_pf": "true"},
])
def test_voltage_drop_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        calculate_voltage_drop(**{"size_mm2": 16, "load_amps": 63, "length_m": 30, **kwargs})


def test_loop_includes_reactance_and_source_impedance_and_raw_boundary():
    result = check_loop_impedance(16, 6, 63)
    z = math.hypot(24 / 16 + 24 / 6, 0.2)
    resistance_only_length = 0.8 * 230 * 1000 / (630 * (24 / 16 + 24 / 6))
    assert result["total_impedance_ohm_per_km"] == pytest.approx(round(z, 4))
    assert result["max_length_for_disconnection_m_raw"] < resistance_only_length
    assert result["impedance_basis"] == "resistance_plus_reactance"
    assert result["compliant"] is None
    assert result["source_impedance_assumption"] == "ideal_source"
    expected = (0.8 * 230 / 630 - 0.1) * 1000 / z
    source = check_loop_impedance(16, 6, 63, supply_loop_impedance_ohm=0.1, length_m=expected + 0.001)
    assert source["max_length_for_disconnection_m_raw"] == pytest.approx(expected)
    assert source["compliant"] is False
    no_headroom = check_loop_impedance(500, 240, 63, supply_loop_impedance_ohm=1, length_m=1)
    assert no_headroom["max_length_for_disconnection_m"] == 0
    assert no_headroom["compliant"] is False


@pytest.mark.parametrize("kwargs", [
    {"protective_device_rating_amps": 0}, {"protective_device_rating_amps": math.inf},
    {"nominal_phase_voltage": -1}, {"mcb_curve": "typo"}, {"length_m": -1},
    {"supply_loop_impedance_ohm": -0.1}, {"supply_loop_impedance_ohm": math.nan},
    {"conductor_material": "typo"}, {"earth_conductor_material": "typo"}, {"earth_size_mm2": 17},
])
def test_loop_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        check_loop_impedance(**{"active_size_mm2": 16, "earth_size_mm2": 6, "protective_device_rating_amps": 63, **kwargs})


def test_thermal_formula_and_both_conductors_checked():
    active = check_short_circuit_capacity(16, 5, 0.1)
    assert active["fault_let_through_energy_a2s"] == 2500000
    assert active["cable_withstand_capacity_a2s"] == 2560000
    assert active["compliant"] is True
    assert check_short_circuit_capacity(6, 5, 0.1)["compliant"] is False
    result = size_cable(63, 30, check_fault=True, fault_current_ka=5, fault_time_s=0.1)
    assert result["recommended_active_size_mm2"] == 16
    assert result["recommended_earth_size_mm2"] == 16
    assert result["short_circuit_check"]["compliant"] is True
    assert result["earth_short_circuit_check"]["compliant"] is True
    assert result["limiting_factor"] == "short_circuit"
    overridden = size_cable(63, 30, check_fault=True, fault_current_ka=5, earth_fault_current_ka=1, earth_fault_time_s=0.2)
    assert overridden["recommended_earth_size_mm2"] == 6
    assert overridden["earth_short_circuit_check"]["fault_time_s"] == 0.2
    active_driven = size_cable(63, 30, check_fault=True, fault_current_ka=7, earth_fault_current_ka=1)
    assert active_driven["recommended_active_size_mm2"] == 25
    assert active_driven["limiting_factor"] == "short_circuit"


@pytest.mark.parametrize("kwargs", [
    {"size_mm2": 0}, {"size_mm2": math.inf}, {"fault_current_ka": -1},
    {"fault_current_ka": math.nan}, {"fault_time_s": 0}, {"fault_time_s": 6},
    {"fault_time_s": math.inf}, {"conductor_material": "typo"}, {"insulation": "typo"},
])
def test_thermal_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        check_short_circuit_capacity(**{"size_mm2": 16, "fault_current_ka": 5, **kwargs})


def test_sizing_enforces_loop_for_each_candidate():
    result = size_cable(63, 100, mcb_rating_amps=63)
    assert result["status"] == "success"
    assert result["limiting_factor"] == "loop_impedance"
    assert result["loop_impedance_check"]["compliant"] is True
    assert result["loop_impedance_check"]["max_length_for_disconnection_m_raw"] >= 100
    old_candidate = next(c for c in result["all_candidates"] if c["size_mm2"] == 25)
    assert old_candidate["current_ok"] is True
    assert old_candidate["voltage_drop_ok"] is True
    assert old_candidate["loop_impedance_ok"] is False
    assert "loop_impedance" in old_candidate["failed_constraints"]
    assert all(c["loop_impedance_ok"] is True for c in result["all_candidates"] if c["compliant"])
    unprotected = size_cable(63, 100)
    assert unprotected["check_states"]["loop_impedance"] == "not_requested"
    assert unprotected["check_states"]["short_circuit"] == "not_requested"
    assert unprotected["check_states"]["earth_short_circuit"] == "not_requested"
    assert all(c["loop_impedance_ok"] is None for c in unprotected["all_candidates"])


def test_limiting_factor_is_final_binding_constraint():
    result = size_cable(63, 100, max_volt_drop_pct=1.5, mcb_rating_amps=63)
    assert result["recommended_active_size_mm2"] == 50
    assert result["limiting_factor"] == "voltage_drop"
    preceding = next(c for c in result["all_candidates"] if c["size_mm2"] == 35)
    assert preceding["failed_constraints"] == ["voltage_drop"]


def test_aluminium_active_has_consistent_separate_earth_material():
    result = size_cable(30, 10, conductor_material=" ALUMINIUM ", mcb_rating_amps=32)
    assert result["status"] == "success"
    assert result["earth_conductor_material"] == "copper"
    assert result["loop_impedance_check"]["earth_rc_ohm_per_km"] == 24 / result["recommended_earth_size_mm2"]
    assert size_cable(30, 10, conductor_material="aluminium")["recommended_earth_size_mm2"] == result["recommended_earth_size_mm2"]
    aluminium_earth = size_cable(30, 10, conductor_material="aluminium", earth_conductor_material="aluminium", mcb_rating_amps=32)
    assert aluminium_earth["earth_conductor_material"] == "aluminium"
    assert aluminium_earth["recommended_earth_size_mm2"] >= 10
    assert aluminium_earth["loop_impedance_check"]["earth_rc_ohm_per_km"] == 40 / aluminium_earth["recommended_earth_size_mm2"]


@pytest.mark.parametrize("limited_material,active_material,earth_material,error_text", [
    ("aluminium", "copper", "aluminium", "No aluminium earth size"),
    ("copper", "aluminium", "copper", "No copper-equivalent active size"),
])
@pytest.mark.parametrize("requested_checks", [False, True])
def test_selection_survives_later_unavailable_earth_pairings(
    tmp_path, limited_material, active_material, earth_material, error_text, requested_checks,
):
    dataset = make_dataset()
    rows = dataset["tables"]
    sizes_key = f"CONDUCTOR_SIZES_{limited_material.upper()}"
    rows[sizes_key] = [size for size in rows[sizes_key] if size <= 120]
    for insulation in ("V90", "X90"):
        resistance = rows["RESISTANCE_TABLE"][limited_material][insulation]
        rows["RESISTANCE_TABLE"][limited_material][insulation] = {
            key: value for key, value in resistance.items() if float(key) <= 120
        }
        methods = rows["CURRENT_RATINGS"][limited_material][insulation]
        for method, ratings in methods.items():
            methods[method] = {key: value for key, value in ratings.items() if float(key) <= 120}
    if limited_material == "copper":
        rows["TABLE_5_1_EARTH"] = {
            key: value for key, value in rows["TABLE_5_1_EARTH"].items() if float(key) <= 120
        }
    path = tmp_path / "limited-sizes.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    tables.load_dataset(path)

    result = size_cable(
        30, 10, conductor_material=active_material, earth_conductor_material=earth_material,
        mcb_rating_amps=32 if requested_checks else None,
        check_fault=requested_checks, fault_current_ka=1,
    )
    assert result["status"] == "success"
    assert result["recommended_active_size_mm2"] == 10
    assert result["recommended_earth_size_mm2"] is not None
    unavailable = [candidate for candidate in result["all_candidates"] if not candidate["earth_pairing_ok"]]
    assert unavailable
    for candidate in unavailable:
        assert candidate["size_mm2"] > result["recommended_active_size_mm2"]
        assert candidate["current_ok"] is True
        assert candidate["voltage_drop_ok"] is True
        assert candidate["compliant"] is False
        assert candidate["failed_constraints"] == ["earth_pairing"]
        assert candidate["earth_size_mm2"] is None
        assert candidate["paired_earth_size_mm2"] is None
        assert candidate["loop_impedance_ok"] is None
        assert candidate["earth_short_circuit_ok"] is None
        assert candidate["short_circuit_ok"] is (True if requested_checks else None)
        assert error_text in candidate["earth_pairing_error"]
    if requested_checks:
        assert result["loop_impedance_check"]["compliant"] is True
        assert result["short_circuit_check"]["compliant"] is True
        assert result["earth_short_circuit_check"]["compliant"] is True
    report = format_calculation_report(result)
    assert "STATUS: REQUESTED MODEL CHECKS PASSED" in report
    assert "Failed checks: earth_pairing" in report
    assert error_text in report
    last_row = next(line for line in report.splitlines() if line.split("|")[0].strip() == "500")
    assert last_row.split("|")[1].strip() == "--"
    assert last_row.rstrip().endswith("FAIL")


@pytest.mark.parametrize("requested_checks", [False, True])
def test_all_earth_pairings_unavailable_returns_failed_selection(tmp_path, requested_checks):
    dataset = make_dataset()
    for insulation, values in dataset["tables"]["RESISTANCE_TABLE"]["aluminium"].items():
        dataset["tables"]["RESISTANCE_TABLE"]["aluminium"][insulation] = {
            key: 1_000_000 / float(key) for key in values
        }
    path = tmp_path / "unavailable-pairings.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    tables.load_dataset(path)

    result = size_cable(
        1, 1, earth_conductor_material="aluminium",
        mcb_rating_amps=2 if requested_checks else None,
        check_fault=requested_checks, fault_current_ka=0.1,
    )
    assert result["status"] == "failed"
    assert result["recommended_active_size_mm2"] is None
    assert result["recommended_earth_size_mm2"] is None
    assert result["limiting_factor"] is None
    assert result["loop_impedance_check"] is None
    assert result["earth_short_circuit_check"] is None
    assert result["check_states"]["loop_impedance"] == ("no_selection" if requested_checks else "not_requested")
    assert result["check_states"]["earth_short_circuit"] == ("no_selection" if requested_checks else "not_requested")
    for candidate in result["all_candidates"]:
        assert candidate["compliant"] is False
        assert candidate["failed_constraints"] == ["earth_pairing"]
        assert candidate["loop_impedance_ok"] is None
        assert candidate["earth_short_circuit_ok"] is None
    report = format_calculation_report(result)
    assert "STATUS: FAILED MODEL SELECTION" in report
    assert "STATUS: REQUESTED MODEL CHECKS PASSED" not in report
    assert "Failed checks: earth_pairing" in report
    assert "No aluminium earth size" in report


def test_unrelated_earth_pairing_data_errors_still_raise(monkeypatch):
    monkeypatch.delitem(tables.TABLE_5_1_EARTH, 1.5)
    with pytest.raises(ValueError, match="Earth pairing for active size"):
        size_cable(30, 10)


@pytest.mark.parametrize("kwargs", [
    {"length_m": -10}, {"length_m": math.inf}, {"phase": "typo"}, {"phase": "dc"},
    {"conductor_material": "typo"}, {"earth_conductor_material": "typo"}, {"insulation": "typo"},
    {"installation_method": "typo"}, {"max_volt_drop_pct": 0}, {"max_volt_drop_pct": math.nan},
    {"max_volt_drop_pct": 101}, {"mcb_rating_amps": 10}, {"mcb_rating_amps": -1},
    {"mcb_curve": "typo"}, {"check_fault": 1}, {"fault_current_ka": math.inf},
    {"earth_fault_current_ka": -1}, {"earth_fault_time_s": 0}, {"supply_loop_impedance_ohm": -1},
    {"earth_fault_current_ka": 5}, {"earth_fault_time_s": 0.1}, {"supply_loop_impedance_ohm": 0.1},
])
def test_sizing_rejects_invalid_inputs_and_undersized_protection(kwargs):
    with pytest.raises(ValueError):
        size_cable(**{"load": 63, "length_m": 30, **kwargs})


def test_failure_schema_matches_success_and_records_candidates():
    success = size_cable(63, 30)
    failure = size_cable(630, 5000, max_volt_drop_pct=0.01)
    assert failure["status"] == "failed"
    assert failure["message"]
    assert set(failure) == set(success)
    for field in ("recommended_active_size_mm2", "recommended_earth_size_mm2", "limiting_factor",
                  "cable_continuous_capacity_iz_a", "capacity_margin_pct", "voltage_drop_v", "voltage_drop_pct",
                  "loop_impedance_check", "short_circuit_check", "earth_short_circuit_check"):
        assert failure[field] is None
    assert all(not c["compliant"] for c in failure["all_candidates"])
    assert failure["check_states"]["loop_impedance"] == "not_requested"
    assert failure["check_states"]["current_capacity"] == "no_selection"
    assert failure["data_provenance"]["validation_status"] == "synthetic_test_only"
    assert success["inputs"]["phase"] == "3phase"
    assert any("do not certify" in assumption for assumption in success["assumptions"])


def test_missing_dataset_fails_all_dataset_entrypoints_explicitly():
    tables.DATA_PROVENANCE["configured"] = False
    for calculate in (
        lambda: get_derating_factor(), lambda: calculate_voltage_drop(16, 50, 30),
        lambda: check_loop_impedance(16, 6, 63), lambda: check_short_circuit_capacity(16, 5),
        lambda: size_cable(63, 30),
    ):
        with pytest.raises(ValueError, match="Reference data"):
            calculate()
    assert calculate_load_current(63) == 63


def test_provenance_is_result_snapshot():
    result = size_cable(63, 30)
    tables.DATA_PROVENANCE["assumptions"]["cable_construction"] = "Changed after result"
    assert result["data_provenance"]["assumptions"]["cable_construction"] != "Changed after result"


def _install_flat_current_profiles(tmp_path, *, sizes=(2.5, 4, 6, 10), free_air=False):
    """Add independently invented ampacity columns to the legacy test dataset."""
    dataset = make_dataset()
    dataset["schema_version"] = 2
    profiles = {}
    conditions = [
        ("partially_surrounded", "in_thermal_insulation", 1.2),
        ("completely_surrounded", "in_thermal_insulation", 0.8),
    ]
    if free_air:
        conditions.append(("none", "unenclosed_in_air", 2.0))
    for exposure, method, multiplier in conditions:
        profiles[f"synthetic-flat-{exposure}"] = {
            "cable_construction": "flat_2c_earth",
            "conductor_material": "copper", "insulation": "V90", "loaded_conductors": 2,
            "installation_method": method, "insulation_exposure": exposure,
            "reference_temperature_c": 40,
            "source": "Independent artificial test formula: multiplier * (4 * area + 1).",
            "ratings": {str(size): multiplier * (4 * size + 1) for size in sizes},
        }
    dataset["tables"]["CURRENT_RATING_PROFILES"] = profiles
    path = tmp_path / "synthetic-flat-profiles.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    tables.load_dataset(path)
    return profiles


def _flat_request(exposure, **overrides):
    return {
        "load": 20, "length_m": 20, "voltage": 230, "phase": "1phase",
        "installation_method": "in_thermal_insulation", "cable_construction": "flat_2c_earth",
        "insulation_exposure": exposure, **overrides,
    }


@pytest.mark.parametrize("exposure,expected_size,multiplier", [
    ("partially_surrounded", 4, 1.2), ("completely_surrounded", 6, 0.8),
])
def test_flat_two_loaded_conductor_profiles_select_the_matching_ampacity_column(
    tmp_path, exposure, expected_size, multiplier,
):
    profiles = _install_flat_current_profiles(tmp_path)
    result = size_cable(**_flat_request(exposure))
    assert result["status"] == "success"
    assert result["recommended_active_size_mm2"] == expected_size
    assert result["cable_continuous_capacity_iz_a"] == pytest.approx(multiplier * (4 * expected_size + 1))
    assert result["limiting_factor"] == "current_capacity"
    assert result["voltage_drop_pct_raw"] < 3
    assert result["inputs"]["cable_construction"] == "flat_2c_earth"
    assert result["inputs"]["loaded_conductors"] == 2
    assert result["inputs"]["insulation_exposure"] == exposure
    basis = result["rating_basis"]
    assert basis["profile_id"] == f"synthetic-flat-{exposure}"
    assert basis["source"] == profiles[basis["profile_id"]]["source"]
    assert basis["supported_sizes_mm2"] == [2.5, 4, 6, 10]
    assert [candidate["size_mm2"] for candidate in result["all_candidates"]] == basis["supported_sizes_mm2"]
    assert all(candidate["current_ok"] is False for candidate in result["all_candidates"]
               if candidate["size_mm2"] < expected_size)

    # Opting into a flat profile must not replace the existing generic column.
    generic = size_cable(20, 20, voltage=230, phase="1phase", installation_method="in_thermal_insulation")
    assert generic["recommended_active_size_mm2"] == 10
    assert generic["inputs"]["cable_construction"] == "generic"
    assert generic["rating_basis"]["profile_id"] is None


@pytest.mark.parametrize("exposure,expected_size,multiplier", [
    ("partially_surrounded", 6, 1.2), ("completely_surrounded", 10, 0.8),
])
def test_flat_profile_applies_ambient_and_grouping_derating_once(
    tmp_path, exposure, expected_size, multiplier,
):
    _install_flat_current_profiles(tmp_path)
    result = size_cable(**_flat_request(exposure, ambient_temp_c=45, num_circuits=2))
    assert result["status"] == "success"
    assert result["recommended_active_size_mm2"] == expected_size
    expected_factor = 0.9 * 0.8
    assert result["derating"]["total_derating_factor_raw"] == pytest.approx(expected_factor)
    selected = next(candidate for candidate in result["all_candidates"] if candidate["size_mm2"] == expected_size)
    assert selected["base_capacity_a"] == pytest.approx(multiplier * (4 * expected_size + 1))
    assert selected["derated_capacity_a_raw"] == pytest.approx(selected["base_capacity_a"] * expected_factor)
    assert selected["current_ok"] is True


def test_flat_profile_voltage_drop_can_require_a_larger_profile_candidate(tmp_path):
    _install_flat_current_profiles(tmp_path)
    result = size_cable(**_flat_request("partially_surrounded", length_m=40))
    assert result["recommended_active_size_mm2"] == 6
    assert result["limiting_factor"] == "voltage_drop"
    earlier = next(candidate for candidate in result["all_candidates"] if candidate["size_mm2"] == 4)
    assert earlier["current_ok"] is True
    assert earlier["voltage_drop_ok"] is False
    assert earlier["failed_constraints"] == ["voltage_drop"]


def test_sparse_flat_profile_never_uses_generic_ratings_for_missing_sizes(tmp_path):
    _install_flat_current_profiles(tmp_path, sizes=(2.5, 6, 10))
    result = size_cable(**_flat_request("partially_surrounded"))
    assert result["recommended_active_size_mm2"] == 6
    assert result["rating_basis"]["supported_sizes_mm2"] == [2.5, 6, 10]
    assert [candidate["size_mm2"] for candidate in result["all_candidates"]] == [2.5, 6, 10]
    assert all(candidate["size_mm2"] != 4 for candidate in result["all_candidates"])


def test_flat_profile_insufficient_coverage_returns_failed_selection(tmp_path):
    _install_flat_current_profiles(tmp_path, sizes=(2.5, 4))
    result = size_cable(**_flat_request("partially_surrounded", load=30, length_m=1))
    assert result["status"] == "failed"
    assert result["recommended_active_size_mm2"] is None
    assert result["recommended_earth_size_mm2"] is None
    assert result["check_states"]["current_capacity"] == "no_selection"
    assert result["rating_basis"]["supported_sizes_mm2"] == [2.5, 4]
    assert [candidate["size_mm2"] for candidate in result["all_candidates"]] == [2.5, 4]
    assert all(candidate["failed_constraints"] == ["current_capacity"] for candidate in result["all_candidates"])
    assert "profile" in result["message"].lower()
    assert "coverage" in result["message"].lower()


@pytest.mark.parametrize("overrides", [
    {"phase": "3phase"}, {"insulation_exposure": None},
    {"cable_construction": "generic"}, {"cable_construction": "unknown"},
    {"cable_construction": None}, {"cable_construction": True},
    {"insulation_exposure": "none"}, {"insulation_exposure": "unknown"},
    {"insulation_exposure": True}, {"conductor_material": "aluminium"},
    {"insulation": "X90"}, {"installation_method": "in_conduit_in_air", "insulation_exposure": "none"},
])
def test_flat_profile_rejects_incompatible_or_unavailable_conditions(tmp_path, overrides):
    _install_flat_current_profiles(tmp_path)
    with pytest.raises(ValueError):
        size_cable(**_flat_request("partially_surrounded", **overrides))


@pytest.mark.parametrize("exposure", [None, "none"])
def test_flat_unenclosed_profile_uses_explicit_no_insulation_condition(tmp_path, exposure):
    _install_flat_current_profiles(tmp_path, free_air=True)
    result = size_cable(**_flat_request(exposure, installation_method="unenclosed_in_air"))
    assert result["status"] == "success"
    assert result["recommended_active_size_mm2"] == 2.5
    assert result["inputs"]["insulation_exposure"] == "none"
    assert result["rating_basis"]["profile_id"] == "synthetic-flat-none"


def test_legacy_dataset_flat_request_fails_instead_of_using_generic_column():
    assert tables.CURRENT_RATING_PROFILES == {}
    with pytest.raises(ValueError, match="(?i)profile"):
        size_cable(**_flat_request("partially_surrounded"))


def test_selected_profile_basis_is_a_result_snapshot(tmp_path):
    _install_flat_current_profiles(tmp_path)
    result = size_cable(**_flat_request("partially_surrounded"))
    profile = tables.CURRENT_RATING_PROFILES["synthetic-flat-partially_surrounded"]
    original_source = profile["source"]
    profile["source"] = "Changed after calculation"
    profile["ratings"].clear()
    assert result["rating_basis"]["source"] == original_source
    assert result["rating_basis"]["supported_sizes_mm2"] == [2.5, 4, 6, 10]
