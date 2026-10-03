"""Server and report regressions using independently invented test data."""

import asyncio
from copy import deepcopy
from unittest.mock import Mock

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from cablesizecalculator import server, tables
from cablesizecalculator.report import format_calculation_report
from cablesizecalculator.server import (
    calculate_derating,
    calculate_voltage_drop,
    check_loop_impedance,
    check_short_circuit,
    generate_calculation_report,
    get_standards_info,
    size_cable,
)


@pytest.mark.parametrize("name,unknown", [
    ("size_cable", "check_faults"),
    ("size_cable", "mcb_rating_amp"),
    ("generate_calculation_report", "check_faults"),
])
def test_unknown_arguments_are_rejected_before_calculation(monkeypatch, name, unknown):
    calculator = Mock(side_effect=AssertionError("Calculation must not run for rejected arguments"))
    monkeypatch.setattr(server, "engine_size_cable", calculator)
    with pytest.raises(ToolError, match=unknown):
        asyncio.run(server.mcp.call_tool(name, {"load": 10, "length_m": 5, unknown: True}))
    calculator.assert_not_called()


def test_metadata_identifies_model_and_avoids_table_redistribution():
    info = get_standards_info()
    assert info["model_status"].startswith("experimental")
    assert info["data_provenance"]["configured"] is True
    assert set(info["supported_phases"]) == {"1phase", "3phase"}
    assert "standard_earth_sizes_table_5_1" not in info
    assert "copper_sizes_mm2" not in info
    assert "aluminium_sizes_mm2" not in info
    assert info["current_rating_profiles"] == []


def test_metadata_snapshot_cannot_change_later_default_calculations(monkeypatch):
    # Isolate global configuration even if the regression fails before mutation checks.
    original_provenance = deepcopy(tables.DATA_PROVENANCE)
    monkeypatch.setattr(tables, "DATA_PROVENANCE", deepcopy(original_provenance))
    monkeypatch.setattr(server, "DATA_PROVENANCE", tables.DATA_PROVENANCE)
    original_derating = calculate_derating()
    original_info = deepcopy(get_standards_info())
    returned_provenance = get_standards_info()["data_provenance"]
    assert returned_provenance["assumptions"] is not tables.DATA_PROVENANCE["assumptions"]
    assert returned_provenance["standard_editions"] is not tables.DATA_PROVENANCE["standard_editions"]

    returned_provenance["assumptions"]["reference_air_temp_c"] += 5
    returned_provenance["standard_editions"].append("Caller-only annotation")

    assert tables.DATA_PROVENANCE == original_provenance
    assert get_standards_info() == original_info
    assert calculate_derating() == original_derating


@pytest.fixture
def artificial_flat_profiles(monkeypatch):
    # Independently invented ratings exercise profile routing, not cable design.
    common = {
        "cable_construction": "flat_2c_earth", "conductor_material": "copper",
        "insulation": "V90", "loaded_conductors": 2,
        "installation_method": "in_thermal_insulation", "reference_temperature_c": 40.0,
        "source": "Independent artificial profile for server tests.",
    }
    profiles = {
        "synthetic_flat_partial": {
            **common, "insulation_exposure": "partially_surrounded",
            "ratings": {2.5: 12.0, 4.0: 24.0, 6.0: 32.0},
        },
        "synthetic_flat_complete": {
            **common, "insulation_exposure": "completely_surrounded",
            "ratings": {2.5: 9.0, 4.0: 17.0, 6.0: 25.0},
        },
    }
    monkeypatch.setattr(tables, "CURRENT_RATING_PROFILES", profiles)
    monkeypatch.setattr(server, "CURRENT_RATING_PROFILES", profiles)
    return profiles


def test_profile_metadata_exposes_conditions_and_coverage_only(artificial_flat_profiles):
    artificial_flat_profiles["synthetic_flat_partial"]["private_extra"] = "Excluded metadata"
    profiles = get_standards_info()["current_rating_profiles"]
    partial = next(profile for profile in profiles if profile["profile_id"] == "synthetic_flat_partial")
    assert partial == {
        "profile_id": "synthetic_flat_partial", "supported_sizes_mm2": [2.5, 4.0, 6.0],
        "cable_construction": "flat_2c_earth", "conductor_material": "copper",
        "insulation": "V90", "loaded_conductors": 2,
        "installation_method": "in_thermal_insulation", "reference_temperature_c": 40.0,
        "source": "Independent artificial profile for server tests.",
        "insulation_exposure": "partially_surrounded",
    }
    assert all("ratings" not in profile and "private_extra" not in profile for profile in profiles)


@pytest.mark.parametrize("exposure,expected_size,profile_id", [
    ("partially_surrounded", 4.0, "synthetic_flat_partial"),
    ("completely_surrounded", 6.0, "synthetic_flat_complete"),
])
def test_sizing_and_report_expose_selected_profile(artificial_flat_profiles, exposure, expected_size, profile_id):
    arguments = {
        "load": 20, "length_m": 5, "voltage": 230, "phase": "1phase",
        "installation_method": "in_thermal_insulation", "cable_construction": "flat_2c_earth",
        "insulation_exposure": exposure,
    }
    result = size_cable(**arguments)
    assert result["recommended_active_size_mm2"] == expected_size
    assert result["inputs"]["loaded_conductors"] == 2
    assert result["rating_basis"]["profile_id"] == profile_id
    report = generate_calculation_report(**arguments)
    for text in (
        "Requested cable construction: flat_2c_earth", "Circuit loaded conductors: 2",
        f"Thermal insulation exposure: {exposure}", f"Rating profile identifier: {profile_id}",
        "Rated cable construction: flat_2c_earth", "Rating loaded conductors: 2",
        "Rating reference temperature (C): 40", "Profile size coverage (mm2): [2.5, 4.0, 6.0]",
        "Rating source: Independent artificial profile for server tests.",
    ):
        assert text in report
    assert "Legacy generic rating column" not in report


@pytest.mark.parametrize("overrides", [
    {"insulation_exposure": None},
    {"phase": "3phase"},
    {"conductor_material": "aluminium"},
    {"insulation": "X90"},
    {"installation_method": "in_conduit_in_air", "insulation_exposure": "none"},
    {"cable_construction": "generic"},
])
def test_profile_arguments_fail_without_generic_fallback(artificial_flat_profiles, overrides):
    arguments = {
        "load": 20, "length_m": 5, "phase": "1phase", "voltage": 230,
        "installation_method": "in_thermal_insulation", "cable_construction": "flat_2c_earth",
        "insulation_exposure": "partially_surrounded",
    }
    with pytest.raises(ToolError):
        size_cable(**{**arguments, **overrides})
    with pytest.raises(ToolError):
        generate_calculation_report(**{**arguments, **overrides})


def test_sizing_passes_requested_checks_and_marks_omissions():
    result = size_cable(load=10, length_m=5, mcb_rating_amps=10)
    assert result["status"] == "success"
    assert result["earth_conductor_material"] == "copper"
    assert result["check_states"] == {
        "current_capacity": "passed", "voltage_drop": "passed",
        "loop_impedance": "passed", "short_circuit": "not_requested",
        "earth_short_circuit": "not_requested",
    }
    assert result["loop_impedance_check"]["compliant"] is True


def test_report_distinguishes_not_checked_from_passed():
    report = generate_calculation_report(load=10, length_m=5)
    assert "EXPERIMENTAL SELECTION SUMMARY" in report
    assert "COMPLIANT SELECTION SUMMARY" not in report
    assert "Continuous current capacity: PASS" in report
    assert "Earth-fault loop length: NOT CHECKED" in report
    assert "Active conductor thermal withstand: NOT CHECKED" in report
    assert "Earth conductor thermal withstand: NOT CHECKED" in report
    assert "CONDUCTOR SIZE EVALUATION TABLE" in report
    assert "Dataset validation status: synthetic_test_only" in report
    for text in ("One-way route length (m): 5", "Active conductor material: copper",
                 "Earth conductor material: copper", "Insulation model: V90",
                 "Phase arrangement: 3phase", "Grouped circuit count: 1", "Burial depth (m): 0.5"):
        assert text in report
    assert "for 0.4s" not in report
    selected_row = next(line for line in report.splitlines() if line.endswith("SELECTED"))
    assert [column.strip() for column in selected_row.split("|")[6:9]] == ["--", "--", "--"]
    assert report.isascii()
    assert "Legacy generic rating column: physical cable construction and loaded-conductor basis unspecified." in report
    assert "Rating loaded conductors: unspecified" in report


def test_report_shows_separate_active_and_earth_fault_conditions():
    report = generate_calculation_report(
        load=10, length_m=5, mcb_rating_amps=10, check_fault=True,
        fault_current_ka=0.01, fault_time_s=0.01,
        earth_fault_current_ka=0.02, earth_fault_time_s=0.02,
        supply_loop_impedance_ohm=0.01,
    )
    assert "Earth-fault loop length: PASS" in report
    assert "Active conductor thermal withstand: PASS" in report
    assert "Earth conductor thermal withstand: PASS" in report
    assert "Fault condition: 0.01 kA for 0.01 s" in report
    assert "Fault condition: 0.02 kA for 0.02 s" in report
    assert "External source-loop magnitude (Ohm): 0.01" in report
    selected_row = next(line for line in report.splitlines() if line.endswith("SELECTED"))
    assert [column.strip() for column in selected_row.split("|")[6:9]] == ["PASS", "PASS", "PASS"]


def test_failed_selection_keeps_candidate_failures_visible():
    result = size_cable(load=1e12, length_m=5)
    assert result["status"] == "failed"
    report = format_calculation_report(result)
    assert "STATUS: FAILED MODEL SELECTION" in report
    assert "CONDUCTOR SIZE EVALUATION TABLE" in report
    assert "Failed checks:" in report
    assert "Continuous current capacity: NO FEASIBLE SELECTION" in report
    assert "Earth-fault loop length: NOT CHECKED" in report


def test_report_escapes_non_ascii_metadata():
    result = size_cable(load=10, length_m=5)
    result["assumptions"] = ["Example temperature 40\N{DEGREE SIGN}C"]
    report = format_calculation_report(result, title="Model \N{GREEK SMALL LETTER ALPHA}")
    assert report.isascii()
    assert "Example temperature" in report


def test_voltage_drop_wrapper_for_selected_candidate():
    selection = size_cable(load=10, length_m=5)
    result = calculate_voltage_drop(
        size_mm2=selection["recommended_active_size_mm2"], load_amps=10, length_m=5,
    )
    assert result["voltage_drop_pct"] > 0
    assert result["voltage_drop_pct"] == selection["voltage_drop_pct"]


def test_derating_wrapper_returns_finite_positive_factors():
    result = calculate_derating()
    assert result["ca_temperature"] > 0
    assert result["cg_grouping"] > 0
    assert result["cd_depth"] > 0


def test_loop_wrapper_separates_earth_material_and_checks_route():
    selection = size_cable(load=10, length_m=5, conductor_material="aluminium")
    result = check_loop_impedance(
        active_size_mm2=selection["recommended_active_size_mm2"],
        earth_size_mm2=selection["recommended_earth_size_mm2"],
        protective_device_rating_amps=10, conductor_material="aluminium",
        earth_conductor_material="copper", length_m=5,
    )
    assert result["compliant"] is True
    assert result["impedance_basis"] == "resistance_plus_reactance"


def test_loop_without_route_length_does_not_claim_pass():
    selection = size_cable(load=10, length_m=5)
    result = check_loop_impedance(
        active_size_mm2=selection["recommended_active_size_mm2"],
        earth_size_mm2=selection["recommended_earth_size_mm2"],
        protective_device_rating_amps=10,
    )
    assert result["compliant"] is None


def test_thermal_wrapper_returns_requested_condition():
    result = check_short_circuit(size_mm2=4, fault_current_ka=0.01, fault_time_s=0.02)
    assert result["fault_current_ka"] == 0.01
    assert result["fault_time_s"] == 0.02
    assert result["compliant"] is True
