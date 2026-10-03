"""ASCII reports for experimental model checks, inputs and assumptions."""

from __future__ import annotations

from typing import Any


def _ascii(value: Any) -> str:
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def _flag(value: bool | None) -> str:
    return "PASS" if value is True else "FAIL" if value is False else "--"


def _state(value: str | None) -> str:
    if value == "passed":
        return "PASS"
    if value == "failed":
        return "FAIL"
    if value == "no_selection":
        return "NO FEASIBLE SELECTION (see candidates)"
    return "NOT CHECKED (not requested)"


def format_calculation_report(
    result: dict[str, Any], title: str = "EXPERIMENTAL CABLE MODEL REPORT",
) -> str:
    """Show what was checked without turning a model result into certification."""
    separator = "=" * 80
    subseparator = "-" * 80
    lines = [separator, _ascii(title).center(80), separator]
    success = result.get("status") == "success"
    lines.append("STATUS: REQUESTED MODEL CHECKS PASSED" if success else "STATUS: FAILED MODEL SELECTION")
    if result.get("message"):
        lines.append(f"  {result['message']}")
    lines.extend([
        "Experimental result; does not certify standards compliance or installation safety.",
        "Dataset provenance, design assumptions and protective devices need independent review.",
        "",
        " DATASET PROVENANCE",
        subseparator,
    ])
    provenance = result.get("data_provenance", {})
    for key, label in (
        ("dataset_id", "Dataset identifier"), ("validation_status", "Dataset validation status"),
        ("standard_editions", "Declared reference editions"), ("source", "Declared source"),
        ("licence", "Declared permissions"),
    ):
        if key in provenance:
            lines.append(f"  * {label}: {provenance[key]}")
    if not provenance:
        lines.append("  * Dataset provenance metadata not available.")
    for key, value in provenance.get("assumptions", {}).items():
        lines.append(f"  * Dataset assumption ({key}): {value}")
    lines.extend([
        "",
        " 1. INPUTS AND ASSUMPTIONS",
        subseparator,
    ])
    inputs = result.get("inputs", {})
    labels = {
        "load": "Load magnitude", "unit": "Load unit", "length_m": "One-way route length (m)",
        "voltage": "Nominal voltage (V)", "phase": "Phase arrangement",
        "power_factor": "Load power factor", "conductor_material": "Active conductor material",
        "earth_conductor_material": "Earth conductor material", "insulation": "Insulation model",
        "installation_method": "Installation model", "ambient_temp_c": "Air/soil temperature (C)",
        "num_circuits": "Grouped circuit count", "depth_m": "Burial depth (m)",
        "max_volt_drop_pct": "Voltage-drop limit (%)", "mcb_rating_amps": "MCB rating (A)",
        "mcb_curve": "MCB curve", "supply_loop_impedance_ohm": "External source-loop magnitude (Ohm)",
        "check_fault": "Active/earth thermal checks requested", "fault_current_ka": "Active fault current (kA)",
        "fault_time_s": "Active fault clearing time (s)",
        "earth_fault_current_ka": "Earth fault current (kA)",
        "earth_fault_time_s": "Earth fault clearing time (s)",
    }
    for key, label in labels.items():
        if key in inputs:
            value = "not supplied" if inputs[key] is None else inputs[key]
            lines.append(f"  * {label}: {value}")
    if not inputs:
        lines.append("  * Input metadata not available; review the originating request.")
    assumptions = result.get("assumptions", [])
    if isinstance(assumptions, str):
        assumptions = [assumptions]
    for assumption in assumptions:
        lines.append(f"  * Assumption: {assumption}")
    lines.extend(["", " 2. MODEL CHECK STATUS", subseparator])

    states = result.get("check_states", {})
    checks = {
        "current_capacity": "Continuous current capacity",
        "voltage_drop": "Voltage drop",
        "loop_impedance": "Earth-fault loop length",
        "short_circuit": "Active conductor thermal withstand",
        "earth_short_circuit": "Earth conductor thermal withstand",
    }
    for key, label in checks.items():
        lines.append(f"  * {label}: {_state(states.get(key))}")

    if success:
        lines.extend(["", " 3. EXPERIMENTAL SELECTION SUMMARY", subseparator])
        lines.append(f"  * Selected active conductor: {result['recommended_active_size_mm2']} mm2")
        lines.append(f"  * Selected earth conductor: {result['recommended_earth_size_mm2']} mm2 ({result.get('earth_conductor_material', 'material unspecified')})")
        lines.append(f"  * Limiting model constraint: {str(result.get('limiting_factor', 'unknown')).replace('_', ' ')}")
        lines.append(f"  * Design current (Ib): {result['design_current_ib_a']} A")
        lines.append(f"  * Required current rating: {result['required_rating_a']} A")
        lines.append(f"  * Model continuous capacity (Iz): {result['cable_continuous_capacity_iz_a']} A")
        lines.append(f"  * Capacity margin: {result['capacity_margin_pct']}%")
        lines.append(f"  * Voltage drop: {result['voltage_drop_v']} V ({result['voltage_drop_pct']}%; limit {result['voltage_drop_limit_pct']}%)")

    derate = result.get("derating", {})
    if derate:
        lines.extend(["", " DERATING MODEL", subseparator])
        lines.append("  * Formula: Iz = I_base * Ca * Cg * Cd")
        for key, label in (
            ("ca_temperature", "Temperature factor Ca"),
            ("cg_grouping", "Grouping factor Cg"),
            ("cd_depth", "Depth factor Cd"),
            ("total_derating_factor", "Combined factor"),
            ("notes", "Lookup details"),
        ):
            if key in derate:
                lines.append(f"  * {label}: {derate[key]}")

    candidates = result.get("all_candidates", [])
    if candidates:
        lines.extend(["", " CONDUCTOR SIZE EVALUATION TABLE", subseparator])
        lines.append("   Active |   Earth |  Iz (A) |   VD (%) | Capacity | VD   | Loop | ActSC | EarSC | Result")
        for candidate in candidates:
            loop = candidate.get("loop_impedance_ok") if states.get("loop_impedance") != "not_requested" else None
            active_sc = candidate.get("short_circuit_ok") if states.get("short_circuit") != "not_requested" else None
            earth_sc = candidate.get("earth_short_circuit_ok") if states.get("earth_short_circuit") != "not_requested" else None
            passed = candidate.get("passed_model_checks", candidate.get("compliant", False))
            selected = success and candidate["size_mm2"] == result["recommended_active_size_mm2"]
            outcome = "SELECTED" if selected else "PASS" if passed else "FAIL"
            earth_size = candidate.get("earth_size_mm2")
            earth_label = "--" if earth_size is None else f"{earth_size:g}"
            lines.append(
                f" {candidate['size_mm2']:8g} | {earth_label:>7} |"
                f" {candidate['derated_capacity_a']:7.1f} | {candidate.get('voltage_drop_pct_raw', candidate['voltage_drop_pct']):8.4f} |"
                f" {_flag(candidate['current_ok']):8} | {_flag(candidate['voltage_drop_ok']):4} |"
                f" {_flag(loop):4} | {_flag(active_sc):5} | {_flag(earth_sc):5} | {outcome}"
            )
            if candidate.get("failed_constraints"):
                lines.append(f"          Failed checks: {', '.join(candidate['failed_constraints'])}")
            if candidate.get("earth_pairing_error"):
                lines.append(f"          Earth pairing: {candidate['earth_pairing_error']}")
        lines.append("  -- means a value is unavailable, or a check was not requested or could not be evaluated.")
        lines.append("  Threshold decisions use unrounded values; displayed metrics are rounded.")

    lines.extend(["", " EARTH-FAULT LOOP MODEL", subseparator])
    loop = result.get("loop_impedance_check")
    lines.append(f"  * Route check: {_state(states.get('loop_impedance'))}")
    if loop:
        for key, label in (
            ("mcb_curve", "MCB curve"), ("rating_in_amps", "MCB rating (A)"),
            ("trip_current_ia_amps", "Assumed trip current (A)"),
            ("active_rc_ohm_per_km", "Active resistance (Ohm/km)"),
            ("earth_rc_ohm_per_km", "Earth resistance (Ohm/km)"),
            ("total_impedance_ohm_per_km", "Cable loop impedance magnitude (Ohm/km)"),
            ("supply_loop_impedance_ohm", "External source-loop magnitude (Ohm)"),
            ("max_length_for_disconnection_m", "Model maximum route length (m)"),
            ("impedance_basis", "Impedance basis"),
        ):
            if key in loop:
                lines.append(f"  * {label}: {loop[key]}")
        lines.append("  * Verify the actual device curve and required clearing time independently.")

    for key, state_key, label in (
        ("short_circuit_check", "short_circuit", "ACTIVE CONDUCTOR THERMAL MODEL"),
        ("earth_short_circuit_check", "earth_short_circuit", "EARTH CONDUCTOR THERMAL MODEL"),
    ):
        lines.extend(["", f" {label}", subseparator])
        lines.append(f"  * Check: {_state(states.get(state_key))}")
        check = result.get(key)
        if check:
            lines.append(f"  * Fault condition: {check['fault_current_ka']} kA for {check['fault_time_s']} s")
            lines.append(f"  * Adiabatic k factor: {check['k_factor']}")
            lines.append(f"  * Let-through energy: {check['fault_let_through_energy_a2s']} A2s")
            lines.append(f"  * Model withstand: {check['cable_withstand_capacity_a2s']} A2s")
            lines.append(f"  * Utilization: {check['utilization_pct']}%")
    lines.append(separator)
    return _ascii("\n".join(lines))
