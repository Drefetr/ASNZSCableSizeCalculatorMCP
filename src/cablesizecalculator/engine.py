"""AC cable calculations over separately supplied engineering data.

Results evaluate stated constraints and assumptions; they do not certify an
installation or establish conformity with an electrical standard.
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from cablesizecalculator import tables, validation as validate


class CandidateSummary(TypedDict):
    size_mm2: float
    earth_size_mm2: float | None
    paired_earth_size_mm2: float | None
    earth_pairing_ok: bool
    earth_pairing_error: str | None
    base_capacity_a: float
    derated_capacity_a: float
    derated_capacity_a_raw: float
    current_ok: bool
    voltage_drop_v: float
    voltage_drop_pct: float
    voltage_drop_pct_raw: float
    voltage_drop_ok: bool
    short_circuit_ok: bool | None
    earth_short_circuit_ok: bool | None
    pass_earth_short_circuit: bool | None
    pass_loop_impedance: bool | None
    loop_impedance_ok: bool | None
    failed_constraints: list[str]
    compliant: bool


class SizingResult(TypedDict):
    status: Literal["success", "failed"]
    message: str | None
    recommended_active_size_mm2: float | None
    recommended_earth_size_mm2: float | None
    earth_conductor_material: str
    limiting_factor: str | None
    design_current_ib_a: float
    required_rating_a: float
    cable_continuous_capacity_iz_a: float | None
    capacity_margin_pct: float | None
    voltage_drop_v: float | None
    voltage_drop_pct: float | None
    voltage_drop_pct_raw: float | None
    voltage_drop_limit_pct: float
    derating: dict[str, Any]
    loop_impedance_check: dict[str, Any] | None
    short_circuit_check: dict[str, Any] | None
    earth_short_circuit_check: dict[str, Any] | None
    check_states: dict[str, str]
    inputs: dict[str, Any]
    assumptions: list[str]
    all_candidates: list[CandidateSummary]
    data_provenance: dict[str, Any]
    rating_basis: dict[str, Any]


@dataclass(frozen=True)
class _SizingConditions:
    """Validated values shared by every candidate in one sizing calculation."""

    design_current_a: float
    required_rating_a: float
    derating_factor: float
    length_m: float
    voltage: float
    phase: str
    power_factor: float
    material: str
    earth_material: str
    insulation: str
    voltage_drop_limit_pct: float
    mcb_rating_a: float | None
    mcb_curve: str
    supply_loop_impedance_ohm: float
    check_fault: bool
    fault_current_ka: float
    fault_time_s: float
    earth_fault_current_ka: float
    earth_fault_time_s: float


_CandidateChecks = tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]


def _provenance() -> dict[str, Any]:
    return deepcopy({key: value for key, value in tables.DATA_PROVENANCE.items() if key in (
        "configured", "dataset_id", "source", "licence", "standard_editions",
        "validation_status", "assumptions",
    )})


def _lookup(table: dict, key: Any, label: str) -> Any:
    if key not in table:
        raise ValueError(f"{label} '{key}' is unavailable in the configured dataset.")
    return table[key]


def _ceiling(table: dict, value: float, label: str) -> tuple[float, float]:
    """Bounded next-row lookup for the dataset's decreasing derating factors."""
    keys = sorted(table)
    if not keys or value < keys[0] or value > keys[-1]:
        bounds = f"{keys[0]} to {keys[-1]}" if keys else "no configured rows"
        raise ValueError(f"{label} must be within the configured range ({bounds}).")
    key = next(key for key in keys if key >= value)
    return key, table[key]


def _resistance(material: str, insulation: str, size: float) -> float:
    return _lookup(_lookup(_lookup(tables.RESISTANCE_TABLE, material, "Material"),
                           insulation, "Insulation"), size, "Conductor size (mm2)")


def _reactance(size: float) -> float:
    # Missing data is not replaced with an arbitrary fallback.
    return _lookup(tables.REACTANCE_TABLE, size, "Reactance for conductor size (mm2)")


def _current_rating_basis(
    material: str, insulation: str, method: str, phase: str,
    construction: str, exposure: str | None,
) -> tuple[dict[float, float], dict[str, Any], str]:
    """Select only ratings whose declared cable and installation conditions match."""
    if exposure is None:
        if method == "in_thermal_insulation":
            if construction != "generic":
                raise ValueError("insulation_exposure is required for flat_2c_earth in thermal insulation; use partially_surrounded or completely_surrounded.")
            exposure = "unspecified"
        else:
            exposure = "none"
    else:
        exposure = validate.choice(exposure, "insulation_exposure", validate.INSULATION_EXPOSURES)
        if method == "in_thermal_insulation" and exposure == "none":
            raise ValueError("in_thermal_insulation requires partially_surrounded or completely_surrounded exposure.")
        if method != "in_thermal_insulation" and exposure != "none":
            raise ValueError("partially_surrounded and completely_surrounded require installation_method='in_thermal_insulation'.")

    if construction == "generic":
        if exposure in ("partially_surrounded", "completely_surrounded"):
            raise ValueError("The generic current-rating table does not distinguish insulation exposure; select a supported cable_construction profile.")
        ratings = _lookup(_lookup(_lookup(tables.CURRENT_RATINGS, material, "Material"),
                                  insulation, "Insulation"), method, "Installation method")
        assumptions = tables.DATA_PROVENANCE["assumptions"]
        reference = "reference_ground_temp_c" if method in ("underground_duct", "buried_direct") else "reference_air_temp_c"
        return ratings, {
            "profile_id": None, "source": tables.DATA_PROVENANCE["source"],
            "cable_construction": "generic", "loaded_conductors": None,
            "insulation_exposure": exposure, "reference_temperature_c": assumptions[reference],
            "supported_sizes_mm2": sorted(ratings),
        }, exposure

    if phase != "1phase":
        raise ValueError("flat_2c_earth requires phase='1phase' with two loaded conductors.")
    for profile_id, profile in tables.CURRENT_RATING_PROFILES.items():
        if (
            profile["cable_construction"] == construction
            and profile["conductor_material"] == material
            and profile["insulation"] == insulation
            and profile["loaded_conductors"] == 2
            and profile["installation_method"] == method
            and profile["insulation_exposure"] == exposure
        ):
            return profile["ratings"], {
                "profile_id": profile_id,
                **{key: profile[key] for key in (
                    "source", "cable_construction", "loaded_conductors", "insulation_exposure",
                    "reference_temperature_c",
                )},
                "supported_sizes_mm2": sorted(profile["ratings"]),
            }, exposure
    raise ValueError(
        "No current-rating profile is configured for "
        f"{construction}, {material}, {insulation}, two loaded conductors, {method}, {exposure}."
    )


def calculate_load_current(
    load: float, unit: str = "A", voltage: float = 400.0,
    phase: str = "3phase", power_factor: float = 0.85,
) -> float:
    """Calculate AC design current; kW/hp mean electrical input power."""
    load = validate.finite_number(load, "load", positive=True)
    voltage = validate.finite_number(voltage, "voltage", positive=True)
    pf = validate.finite_number(power_factor, "power_factor", positive=True, maximum=1)
    phase = validate.ac_phase(phase)
    unit = validate.choice(unit, "unit", ("a", "kw", "kva", "hp"))
    if unit == "a":
        return load
    power = load * 0.7457 if unit == "hp" else load
    denominator = voltage * (math.sqrt(3) if phase == "3phase" else 1) * (1 if unit == "kva" else pf)
    if denominator == 0 or not math.isfinite(denominator):
        raise ValueError("Load current inputs exceed the supported numerical range.")
    return validate.finite_number(power * 1000 / denominator, "Calculated load current", positive=True)


def get_derating_factor(
    ambient_temp_c: float | None = None, installation_method: str = "in_conduit_in_air",
    num_circuits: int = 1, depth_m: float = 0.5, insulation: str = "V90",
) -> dict[str, Any]:
    """Multiply configured factors using conservative ceilings without extrapolation."""
    tables.require_dataset()
    insulation = validate.insulation(insulation)
    method = validate.choice(installation_method, "installation method", validate.INSTALLATION_METHODS)
    circuits = validate.circuit_count(num_circuits)
    depth = validate.finite_number(depth_m, "depth_m", positive=True)
    underground = method in ("underground_duct", "buried_direct")
    reference_key = "reference_ground_temp_c" if underground else "reference_air_temp_c"
    reference = _lookup(tables.DATA_PROVENANCE["assumptions"], reference_key, "Reference temperature")
    temperature = reference if ambient_temp_c is None else validate.finite_number(ambient_temp_c, "ambient_temp_c")
    temperature_table = _lookup(tables.TEMP_DERATING_GROUND if underground else tables.TEMP_DERATING_AIR,
                               insulation, "Temperature factors")
    used_temperature, ca = _ceiling(temperature_table, temperature, "ambient_temp_c")
    used_circuits, cg = _ceiling(tables.CIRCUITS_GROUPING_DERATING, circuits, "num_circuits")
    used_depth, cd = _ceiling(tables.DEPTH_DERATING_GROUND, depth, "depth_m") if underground else (None, 1.0)
    total = validate.finite_result(ca * cg * cd, "Derating factor")
    result = {
        "ca_temperature": ca, "cg_grouping": cg, "cd_depth": cd,
        "total_derating_factor": round(total, 4), "total_derating_factor_raw": total,
        "temperature_table": "Configured ground factors" if underground else "Configured air factors",
        "lookup_method": "conservative_ceiling", "ambient_temp_c": temperature,
        "tabulated_temperature_c": used_temperature, "tabulated_num_circuits": used_circuits,
        "tabulated_depth_m": used_depth,
        "notes": f"Ca={ca} at {used_temperature} C; Cg={cg} at {used_circuits} circuits; Cd={cd}.",
        "data_provenance": _provenance(),
    }
    if underground:
        result["underground_warning"] = "ambient_temp_c must represent soil temperature at burial depth."
    return result


def calculate_voltage_drop(
    size_mm2: float, load_amps: float, length_m: float, voltage: float = 400.0,
    phase: str = "3phase", conductor_material: str = "copper", insulation: str = "V90",
    power_factor: float = 0.85, worst_case_pf: bool = False,
) -> dict[str, Any]:
    """AC voltage drop at a lagging load PF, using the configured cable model."""
    tables.require_dataset()
    size = validate.finite_number(size_mm2, "size_mm2", positive=True)
    current = validate.finite_number(load_amps, "load_amps", positive=True)
    length = validate.finite_number(length_m, "length_m", positive=True)
    voltage = validate.finite_number(voltage, "voltage", positive=True)
    pf = validate.finite_number(power_factor, "power_factor", positive=True, maximum=1)
    phase = validate.ac_phase(phase)
    material, insulation = validate.material(conductor_material), validate.insulation(insulation)
    worst = validate.boolean(worst_case_pf, "worst_case_pf")
    rc, xc = _resistance(material, insulation, size), _reactance(size)
    zc = math.hypot(rc, xc) if worst else rc * pf + xc * math.sqrt(1 - pf * pf)
    phase_factor = math.sqrt(3) if phase == "3phase" else 2
    drop_per_m = validate.finite_number(phase_factor * current * zc / 1000, "Voltage drop per metre", positive=True)
    volts = validate.finite_result(drop_per_m * length, "Voltage drop")
    percent = validate.finite_result(volts / voltage * 100, "Voltage drop percentage")
    max_3 = validate.finite_result(voltage * 0.03 / drop_per_m, "Maximum voltage drop length")
    max_5 = validate.finite_result(voltage * 0.05 / drop_per_m, "Maximum voltage drop length")
    return {
        "size_mm2": size, "conductor_material": material, "insulation": insulation,
        "rc_ohm_per_km": rc, "xc_ohm_per_km": xc, "effective_zc_ohm_per_km": round(zc, 4),
        "voltage_drop_v": round(volts, 2), "voltage_drop_v_raw": volts,
        "voltage_drop_pct": round(percent, 2), "voltage_drop_pct_raw": percent,
        "compliant_3pct": percent <= 3, "compliant_5pct": percent <= 5,
        "max_length_for_3pct_m": round(max_3, 1), "max_length_for_5pct_m": round(max_5, 1),
        "data_provenance": _provenance(),
    }


def check_loop_impedance(
    active_size_mm2: float, earth_size_mm2: float, protective_device_rating_amps: float,
    mcb_curve: str = "C", nominal_phase_voltage: float = 230.0,
    conductor_material: str = "copper", insulation: str = "V90",
    earth_conductor_material: str = "copper", length_m: float | None = None,
    supply_loop_impedance_ohm: float = 0.0,
) -> dict[str, Any]:
    """Estimate instantaneous disconnection with resistance and reactance.

    Cable loop impedance is hypot(R_active+R_earth, X_active+X_earth).
    Adding the supplied source magnitude bounds unknown source/cable phase
    angles conservatively. Zero source impedance is an ideal-source assumption.
    Verify the configured device multiplier and 0.8 voltage factor for the actual
    device, supply and installation. A length must be provided for a pass/fail.
    """
    tables.require_dataset()
    active = validate.finite_number(active_size_mm2, "active_size_mm2", positive=True)
    earth = validate.finite_number(earth_size_mm2, "earth_size_mm2", positive=True)
    rating = validate.finite_number(protective_device_rating_amps, "protective_device_rating_amps", positive=True)
    voltage = validate.finite_number(nominal_phase_voltage, "nominal_phase_voltage", positive=True)
    source = validate.finite_number(supply_loop_impedance_ohm, "supply_loop_impedance_ohm", minimum=0)
    material, earth_material = validate.material(conductor_material), validate.material(earth_conductor_material)
    insulation = validate.insulation(insulation)
    curve = validate.choice(mcb_curve, "MCB curve", ("B", "C", "D"), upper=True)
    length = None if length_m is None else validate.finite_number(length_m, "length_m", positive=True)
    trip = validate.finite_number(_lookup(tables.MCB_TRIP_MULTIPLIERS, curve, "Trip multiplier") * rating,
                                  "Trip current", positive=True)
    ra, re = _resistance(material, insulation, active), _resistance(earth_material, insulation, earth)
    xa, xe = _reactance(active), _reactance(earth)
    cable_z = validate.finite_number(math.hypot(ra + re, xa + xe), "Cable loop impedance", positive=True)
    allowed_z = validate.finite_number(0.8 * voltage / trip, "Allowed loop impedance", positive=True)
    max_length = validate.finite_result(max(0.0, allowed_z - source) * 1000 / cable_z, "Maximum disconnection length")
    actual_z = None if length is None else validate.finite_result(source + cable_z * length / 1000, "Route loop impedance")
    passed = None if actual_z is None else actual_z <= allowed_z
    return {
        "mcb_curve": curve, "rating_in_amps": rating, "trip_current_ia_amps": trip,
        "conductor_material": material, "earth_conductor_material": earth_material,
        "active_rc_ohm_per_km": ra, "earth_rc_ohm_per_km": re,
        "active_xc_ohm_per_km": xa, "earth_xc_ohm_per_km": xe,
        "total_impedance_ohm_per_km": round(cable_z, 4),
        "supply_loop_impedance_ohm": source,
        "source_impedance_assumption": "ideal_source" if source == 0 else "supplied_magnitude_upper_bound",
        "route_loop_impedance_ohm": actual_z, "maximum_loop_impedance_ohm": allowed_z,
        "max_length_for_disconnection_m": round(max_length, 1),
        "max_length_for_disconnection_m_raw": max_length, "length_m": length, "compliant": passed,
        "status": "calculated" if passed is None else "passed" if passed else "failed",
        "impedance_basis": "resistance_plus_reactance",
        "data_provenance": _provenance(),
    }


def check_short_circuit_capacity(
    size_mm2: float, fault_current_ka: float, fault_time_s: float = 0.1,
    conductor_material: str = "copper", insulation: str = "V90",
) -> dict[str, Any]:
    """Adiabatic I squared t <= k squared S squared for clearing times up to 5 s."""
    tables.require_dataset()
    size = validate.finite_number(size_mm2, "size_mm2", positive=True)
    current = validate.finite_number(fault_current_ka, "fault_current_ka", positive=True)
    time = validate.finite_number(fault_time_s, "fault_time_s", positive=True, maximum=5)
    material, insulation = validate.material(conductor_material), validate.insulation(insulation)
    k = _lookup(tables.ADIABATIC_K_FACTORS, (material, insulation), "Adiabatic k factor")
    current_a = validate.finite_number(current * 1000, "Fault current", positive=True)
    ks = validate.finite_number(k * size, "Thermal capacity", positive=True)
    energy = validate.finite_result(current_a * current_a * time, "Fault energy")
    capacity = validate.finite_number(ks * ks, "Thermal capacity", positive=True)
    max_current = validate.finite_result(ks / math.sqrt(time) / 1000, "Maximum withstand current")
    utilization = validate.finite_result(energy / capacity * 100, "Thermal utilization")
    return {
        "size_mm2": size, "conductor_material": material, "insulation": insulation,
        "k_factor": k, "fault_current_ka": current, "fault_time_s": time,
        "fault_let_through_energy_a2s": round(energy, 1), "cable_withstand_capacity_a2s": round(capacity, 1),
        "max_withstand_current_ka": round(max_current, 2), "compliant": energy <= capacity,
        "utilization_pct": round(utilization, 1),
        "data_provenance": _provenance(),
    }


class _EarthPairingUnavailable(ValueError):
    """A candidate has no resistance-equivalent pairing in a valid dataset."""


def _earth_size(active_size: float, active_material: str, earth_material: str, insulation: str) -> float:
    """Use configured copper pairings, with explicit resistance equivalence for Al.

    The equivalence assumption must be validated against the supplied dataset.
    No fixed conductivity ratio or interpolation is invented.
    """
    copper = _lookup(_lookup(tables.RESISTANCE_TABLE, "copper", "Material"), insulation, "Insulation")
    equivalent: float | None
    if active_material == "copper":
        equivalent = active_size
    else:
        active_r = _resistance(active_material, insulation, active_size)
        equivalent = next((size for size in sorted(tables.TABLE_5_1_EARTH)
                           if copper.get(size, math.inf) <= active_r), None)
        if equivalent is None:
            raise _EarthPairingUnavailable("No copper-equivalent active size is configured for earth pairing.")
    copper_earth = _lookup(tables.TABLE_5_1_EARTH, equivalent, "Earth pairing for active size")
    if earth_material == "copper":
        _resistance(earth_material, insulation, copper_earth)
        return copper_earth
    required_r = _resistance("copper", insulation, copper_earth)
    earth_table = _lookup(_lookup(tables.RESISTANCE_TABLE, earth_material, "Earth material"), insulation, "Earth insulation")
    earth_size = next((size for size in sorted(earth_table) if earth_table[size] <= required_r), None)
    if earth_size is None:
        raise _EarthPairingUnavailable("No aluminium earth size meets the configured copper-earth resistance equivalence.")
    return earth_size


def _evaluate_candidate(
    size: float, base_capacity: float, conditions: _SizingConditions,
) -> tuple[CandidateSummary, _CandidateChecks]:
    """Evaluate one active size, including its pairing and independent earth upsize."""
    capacity = validate.finite_result(base_capacity * conditions.derating_factor, "Derated current capacity")
    current_ok = capacity >= conditions.required_rating_a
    vd = calculate_voltage_drop(
        size, conditions.design_current_a, conditions.length_m, conditions.voltage,
        conditions.phase, conditions.material, conditions.insulation, conditions.power_factor,
    )
    vd_ok = vd["voltage_drop_pct_raw"] <= conditions.voltage_drop_limit_pct
    pairing_error = None
    earth_size: float | None
    paired_earth: float | None
    try:
        earth_size = paired_earth = _earth_size(
            size, conditions.material, conditions.earth_material, conditions.insulation,
        )
    except _EarthPairingUnavailable as error:
        earth_size = paired_earth = None
        pairing_error = str(error)
    pairing_ok = paired_earth is not None
    active_sc: dict[str, Any] | None = None
    earth_sc: dict[str, Any] | None = None
    if conditions.check_fault:
        active_sc = check_short_circuit_capacity(
            size, conditions.fault_current_ka, conditions.fault_time_s,
            conditions.material, conditions.insulation,
        )
    if conditions.check_fault and paired_earth is not None:
        earth_sizes = _lookup(
            _lookup(tables.RESISTANCE_TABLE, conditions.earth_material, "Earth material"),
            conditions.insulation, "Earth insulation",
        )
        for earth_candidate in sorted(s for s in earth_sizes if s >= paired_earth):
            earth_size = earth_candidate
            earth_sc = check_short_circuit_capacity(
                earth_size, conditions.earth_fault_current_ka, conditions.earth_fault_time_s,
                conditions.earth_material, conditions.insulation,
            )
            if earth_sc["compliant"]:
                break
    sc_ok = None if active_sc is None else active_sc["compliant"]
    earth_sc_ok = None if earth_sc is None else earth_sc["compliant"]
    loop: dict[str, Any] | None = None
    if conditions.mcb_rating_a is not None and earth_size is not None:
        phase_voltage = conditions.voltage / math.sqrt(3) if conditions.phase == "3phase" else conditions.voltage
        loop = check_loop_impedance(
            size, earth_size, conditions.mcb_rating_a, conditions.mcb_curve, phase_voltage,
            conditions.material, conditions.insulation, conditions.earth_material,
            conditions.length_m, conditions.supply_loop_impedance_ohm,
        )
    loop_ok = None if loop is None else loop["compliant"]
    failed = [name for name, passed in (
        ("current_capacity", current_ok), ("voltage_drop", vd_ok), ("earth_pairing", pairing_ok),
        ("short_circuit", sc_ok), ("earth_short_circuit", earth_sc_ok), ("loop_impedance", loop_ok),
    ) if passed is False]
    summary: CandidateSummary = {
        "size_mm2": size, "earth_size_mm2": earth_size, "paired_earth_size_mm2": paired_earth,
        "earth_pairing_ok": pairing_ok, "earth_pairing_error": pairing_error,
        "base_capacity_a": base_capacity, "derated_capacity_a": round(capacity, 1), "derated_capacity_a_raw": capacity,
        "current_ok": current_ok, "voltage_drop_v": vd["voltage_drop_v"], "voltage_drop_pct": vd["voltage_drop_pct"],
        "voltage_drop_pct_raw": vd["voltage_drop_pct_raw"], "voltage_drop_ok": vd_ok,
        "short_circuit_ok": sc_ok, "earth_short_circuit_ok": earth_sc_ok,
        "pass_earth_short_circuit": earth_sc_ok, "pass_loop_impedance": loop_ok,
        "loop_impedance_ok": loop_ok, "failed_constraints": failed, "compliant": not failed,
    }
    return summary, (loop, active_sc, earth_sc)


def _limiting_constraint(selected: CandidateSummary, evaluations: list[CandidateSummary]) -> str:
    """Identify the final active-size constraint, or a selected earth thermal upsize."""
    chosen = selected["size_mm2"]
    first_current = next(candidate for candidate in evaluations if candidate["current_ok"])
    drivers = set()
    if chosen > first_current["size_mm2"]:
        # Earlier failures can cease to bind before the final active size. The
        # preceding viable-current candidate identifies what forced that upsize.
        preceding = max((candidate for candidate in evaluations
                         if candidate["current_ok"] and candidate["size_mm2"] < chosen),
                        key=lambda candidate: candidate["size_mm2"])
        drivers.update(preceding["failed_constraints"])
    earth_size, paired_earth = selected["earth_size_mm2"], selected["paired_earth_size_mm2"]
    if not drivers and earth_size is not None and paired_earth is not None and earth_size > paired_earth:
        drivers.add("short_circuit")
    if "earth_short_circuit" in drivers:
        drivers.add("short_circuit")
    return next((name for name in ("loop_impedance", "short_circuit", "voltage_drop", "earth_pairing")
                 if name in drivers), "current_capacity")


def size_cable(
    load: float, length_m: float, unit: str = "A", voltage: float = 400.0,
    phase: str = "3phase", power_factor: float = 0.85, conductor_material: str = "copper",
    insulation: str = "V90", installation_method: str = "in_conduit_in_air",
    ambient_temp_c: float | None = None, num_circuits: int = 1, depth_m: float = 0.5,
    max_volt_drop_pct: float = 3.0, mcb_rating_amps: float | None = None, mcb_curve: str = "C",
    check_fault: bool = False, fault_current_ka: float = 6.0, fault_time_s: float = 0.1,
    earth_conductor_material: str = "copper", earth_fault_current_ka: float | None = None,
    earth_fault_time_s: float | None = None, supply_loop_impedance_ohm: float = 0.0,
    cable_construction: str = "generic", insulation_exposure: str | None = None,
) -> SizingResult:
    """Select the smallest active size passing all requested calculations.

    Each candidate's earth conductor begins at the configured pairing and is
    independently enlarged if its requested thermal check requires it. Loop
    limits are enforced whenever an MCB rating is supplied. Success records
    calculation results; it does not approve the installation.
    """
    tables.require_dataset()
    material, earth_material = validate.material(conductor_material), validate.material(earth_conductor_material)
    insulation, phase = validate.insulation(insulation), validate.ac_phase(phase)
    method = validate.choice(installation_method, "installation method", validate.INSTALLATION_METHODS)
    construction = validate.choice(cable_construction, "cable_construction", validate.CABLE_CONSTRUCTIONS)
    unit = validate.choice(unit, "unit", ("a", "kw", "kva", "hp"))
    load = validate.finite_number(load, "load", positive=True)
    voltage = validate.finite_number(voltage, "voltage", positive=True)
    length = validate.finite_number(length_m, "length_m", positive=True)
    pf = validate.finite_number(power_factor, "power_factor", positive=True, maximum=1)
    limit = validate.finite_number(max_volt_drop_pct, "max_volt_drop_pct", positive=True, maximum=100)
    curve = validate.choice(mcb_curve, "MCB curve", ("B", "C", "D"), upper=True)
    source = validate.finite_number(supply_loop_impedance_ohm, "supply_loop_impedance_ohm", minimum=0)
    fault = validate.boolean(check_fault, "check_fault")
    if not fault and (earth_fault_current_ka is not None or earth_fault_time_s is not None):
        raise ValueError("Earth fault overrides require check_fault=True so both thermal checks are evaluated.")
    fault_current = validate.finite_number(fault_current_ka, "fault_current_ka", positive=True)
    fault_time = validate.finite_number(fault_time_s, "fault_time_s", positive=True, maximum=5)
    earth_current = fault_current if earth_fault_current_ka is None else validate.finite_number(earth_fault_current_ka, "earth_fault_current_ka", positive=True)
    earth_time = fault_time if earth_fault_time_s is None else validate.finite_number(earth_fault_time_s, "earth_fault_time_s", positive=True, maximum=5)
    ib = calculate_load_current(load, unit, voltage, phase, pf)
    rating = None if mcb_rating_amps is None else validate.finite_number(mcb_rating_amps, "mcb_rating_amps", positive=True)
    if rating is None and source != 0:
        raise ValueError("supply_loop_impedance_ohm requires mcb_rating_amps to evaluate disconnection.")
    if rating is not None and rating < ib:
        raise ValueError("mcb_rating_amps must be at least the design load current.")
    ratings, rating_basis, exposure = _current_rating_basis(
        material, insulation, method, phase, construction, insulation_exposure,
    )
    derating = get_derating_factor(ambient_temp_c, method, num_circuits, depth_m, insulation)
    sizes = sorted(ratings)
    if not sizes:
        raise ValueError("No conductor sizes are configured.")
    required = ib if rating is None else rating
    conditions = _SizingConditions(
        design_current_a=ib, required_rating_a=required,
        derating_factor=derating["total_derating_factor_raw"], length_m=length,
        voltage=voltage, phase=phase, power_factor=pf, material=material,
        earth_material=earth_material, insulation=insulation, voltage_drop_limit_pct=limit,
        mcb_rating_a=rating, mcb_curve=curve, supply_loop_impedance_ohm=source,
        check_fault=fault, fault_current_ka=fault_current, fault_time_s=fault_time,
        earth_fault_current_ka=earth_current, earth_fault_time_s=earth_time,
    )
    evaluations: list[CandidateSummary] = []
    details: dict[float, _CandidateChecks] = {}
    selected: CandidateSummary | None = None
    for size in sizes:
        base = _lookup(ratings, size, "Current rating for conductor size")
        evaluation, checks = _evaluate_candidate(size, base, conditions)
        evaluations.append(evaluation)
        details[size] = checks
        if selected is None and evaluation["compliant"]:
            selected = evaluation
    states = {
        "current_capacity": "no_selection", "voltage_drop": "no_selection",
        "loop_impedance": "not_requested" if rating is None else "no_selection",
        "short_circuit": "no_selection" if fault else "not_requested",
        "earth_short_circuit": "no_selection" if fault else "not_requested",
    }
    inputs = {
        "load": load, "unit": unit, "length_m": length, "voltage": voltage, "phase": phase, "power_factor": pf,
        "conductor_material": material, "earth_conductor_material": earth_material, "insulation": insulation,
        "installation_method": method, "ambient_temp_c": derating["ambient_temp_c"],
        "cable_construction": construction, "loaded_conductors": 2 if phase == "1phase" else 3,
        "insulation_exposure": exposure,
        "num_circuits": validate.circuit_count(num_circuits), "depth_m": float(depth_m),
        "max_volt_drop_pct": limit, "mcb_rating_amps": rating, "mcb_curve": curve, "check_fault": fault,
        "fault_current_ka": fault_current, "fault_time_s": fault_time, "earth_fault_current_ka": earth_current,
        "earth_fault_time_s": earth_time, "supply_loop_impedance_ohm": source,
    }
    assumptions = [
        "Configured AC ratings and impedance must match the actual cable construction and installation.",
        "Voltage drop assumes lagging load PF and configured operating-temperature resistance.",
        "Earth pairing is supplied by the dataset; aluminium equivalence uses a same/lower-resistance copper equivalent.",
        "Active and earth conductors use the same insulation temperature model.",
        "Requested calculations do not certify standards compliance.",
    ]
    if rating is not None:
        assumptions.append(f"Source loop magnitude is {source} ohm; source and cable magnitudes are added. Trip current uses the configured curve multiplier and 0.8 voltage factor.")
    if fault:
        assumptions.append("Earth fault current/time default to active fault conditions unless separately supplied; verify with the actual protective device.")
    if rating_basis["profile_id"] is not None:
        assumptions.append("The selected profile specifies current capacity; impedance, earth pairings and thermal withstand use the dataset's other configured tables.")
    else:
        assumptions.append("Legacy generic current ratings do not distinguish cable construction, loaded conductors or detailed insulation exposure.")
    result: SizingResult = {
        "status": "failed" if selected is None else "success",
        "message": (
            "No configured cable size within the selected current-rating profile coverage satisfied all requested calculations."
            if rating_basis["profile_id"] is not None else
            "No configured cable size satisfied all requested calculations."
        ) if selected is None else None,
        "recommended_active_size_mm2": None, "recommended_earth_size_mm2": None,
        "earth_conductor_material": earth_material, "limiting_factor": None,
        "design_current_ib_a": round(ib, 2), "required_rating_a": round(required, 2),
        "cable_continuous_capacity_iz_a": None, "capacity_margin_pct": None,
        "voltage_drop_v": None, "voltage_drop_pct": None, "voltage_drop_pct_raw": None,
        "voltage_drop_limit_pct": limit, "derating": derating, "loop_impedance_check": None,
        "short_circuit_check": None, "earth_short_circuit_check": None,
        "check_states": states, "inputs": inputs, "assumptions": assumptions, "all_candidates": evaluations,
        "data_provenance": _provenance(), "rating_basis": deepcopy(rating_basis),
    }
    if selected is None:
        return result
    chosen = selected["size_mm2"]
    for name in states:
        if states[name] != "not_requested":
            states[name] = "passed"
    loop, active_sc, earth_sc = details[chosen]
    margin = validate.finite_result(
        (selected["derated_capacity_a_raw"] / required - 1) * 100, "Capacity margin",
    )
    result.update({
        "recommended_active_size_mm2": chosen, "recommended_earth_size_mm2": selected["earth_size_mm2"],
        "limiting_factor": _limiting_constraint(selected, evaluations),
        "cable_continuous_capacity_iz_a": selected["derated_capacity_a"],
        "capacity_margin_pct": round(margin, 1),
        "voltage_drop_v": selected["voltage_drop_v"], "voltage_drop_pct": selected["voltage_drop_pct"],
        "voltage_drop_pct_raw": selected["voltage_drop_pct_raw"], "loop_impedance_check": loop,
        "short_circuit_check": active_sc, "earth_short_circuit_check": earth_sc,
    })
    return result
