"""Server and report regressions using independently invented test data."""

import asyncio
from unittest.mock import Mock

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from cablesizecalculator import server
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
