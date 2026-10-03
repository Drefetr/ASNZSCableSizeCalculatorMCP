"""MCP tools for an experimental electrical cable calculation model."""

from __future__ import annotations

from typing import Annotated, Any, Callable, Literal

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, InputRequiredResult, Tool as MCPTool
from pydantic import Field

from cablesizecalculator.engine import (
    calculate_voltage_drop as engine_calc_vd,
    check_loop_impedance as engine_check_loop,
    check_short_circuit_capacity as engine_check_sc,
    get_derating_factor,
    size_cable as engine_size_cable,
)
from cablesizecalculator.tables import CURRENT_RATING_PROFILES, DATA_PROVENANCE

# Schema constraints protect MCP callers; the engine independently validates direct calls.
PositiveNumber = Annotated[float, Field(gt=0, allow_inf_nan=False, strict=True)]
NonNegativeNumber = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
FiniteNumber = Annotated[float, Field(allow_inf_nan=False, strict=True)]
PowerFactor = Annotated[float, Field(gt=0, le=1, allow_inf_nan=False, strict=True)]
DropLimit = Annotated[float, Field(gt=0, le=100, allow_inf_nan=False, strict=True)]
ClearingTime = Annotated[float, Field(gt=0, le=5, allow_inf_nan=False, strict=True)]
CircuitCount = Annotated[int, Field(ge=1, strict=True)]
StrictBoolean = Annotated[bool, Field(strict=True)]
Phase = Literal["3phase", "1phase"]
Material = Literal["copper", "aluminium"]
Insulation = Literal["V90", "X90"]
LoadUnit = Literal["A", "kW", "kVA", "hp"]
Curve = Literal["B", "C", "D"]
CableConstruction = Literal["generic", "flat_2c_earth"]
InsulationExposure = Literal["none", "partially_surrounded", "completely_surrounded"]
Installation = Literal[
    "in_conduit_in_air", "unenclosed_in_air", "in_thermal_insulation",
    "underground_duct", "buried_direct",
]


class StrictMCPServer(MCPServer):
    """Reject misspelled arguments before the SDK can discard unknown fields."""

    async def list_tools(self) -> list[MCPTool]:
        tools = await super().list_tools()
        for tool in tools:
            tool.input_schema = {**tool.input_schema, "additionalProperties": False}
        return tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Context | None = None,
    ) -> CallToolResult | InputRequiredResult:
        for tool in await self.list_tools():
            if tool.name == name:
                unknown = arguments.keys() - tool.input_schema.get("properties", {}).keys()
                if unknown:
                    fields = ", ".join(repr(field) for field in sorted(unknown))
                    raise ToolError(
                        f"Unknown argument(s) for {name}: {fields}. Use the names in the tool input schema."
                    )
                break
        return await super().call_tool(name, arguments, context)


mcp = StrictMCPServer(
    "CableSizeCalculator",
    instructions=(
        "Experimental cable calculation model. Results show only the checks requested "
        "against a configured dataset, whose provenance and permitted use must be verified. "
        "They do not certify standards compliance or suitability for installation. "
        "Qualified design review and protective-device verification remain necessary."
    ),
)


def _call_engine(calculator: Callable[..., Any], **arguments: Any) -> Any:
    """Surface anticipated engine validation failures as actionable MCP tool errors."""
    try:
        return calculator(**arguments)
    except ValueError as error:
        raise ToolError(str(error)) from error


@mcp.tool()
def size_cable(
    load: PositiveNumber,
    length_m: PositiveNumber,
    unit: LoadUnit = "A",
    voltage: PositiveNumber = 400.0,
    phase: Phase = "3phase",
    power_factor: PowerFactor = 0.85,
    conductor_material: Material = "copper",
    insulation: Insulation = "V90",
    installation_method: Installation = "in_conduit_in_air",
    ambient_temp_c: FiniteNumber | None = None,
    num_circuits: CircuitCount = 1,
    depth_m: PositiveNumber = 0.5,
    max_volt_drop_pct: DropLimit = 3.0,
    mcb_rating_amps: PositiveNumber | None = None,
    mcb_curve: Curve = "C",
    check_fault: StrictBoolean = False,
    fault_current_ka: PositiveNumber = 6.0,
    fault_time_s: ClearingTime = 0.1,
    earth_conductor_material: Material = "copper",
    earth_fault_current_ka: PositiveNumber | None = None,
    earth_fault_time_s: ClearingTime | None = None,
    supply_loop_impedance_ohm: NonNegativeNumber = 0.0,
    cable_construction: CableConstruction = "generic",
    insulation_exposure: InsulationExposure | None = None,
) -> dict[str, Any]:
    """Select the smallest candidate passing all requested experimental model checks.

    load/unit describe the design load; length_m is the positive one-way route length.
    voltage is line-to-line for 3phase and phase-to-neutral for 1phase. DC is unsupported.
    ambient_temp_c is air temperature above ground and soil temperature underground;
    None uses the configured dataset's reference temperature. num_circuits and depth_m
    must fall within the configured data. The model uses V90 at 75 C or X90 at 90 C.

    mcb_rating_amps enables the approximate AC loop check; supply_loop_impedance_ohm
    is the non-negative external source-loop magnitude (default zero is an assumption).
    Supplying a nonzero source-loop magnitude requires mcb_rating_amps.
    Check the actual protective device and measured supply impedance independently.
    check_fault enables active AND earth adiabatic checking using fault_current_ka and
    fault_time_s. The earth conditions default to those active conditions; override them
    with earth_fault_current_ka and/or earth_fault_time_s for the actual earth fault.
    Earth fault overrides require check_fault=True.
    Omitted fault checks are reported as not requested. The earth material is independent
    of the active material. An experimental selection is not an installation approval.
    cable_construction="flat_2c_earth" selects a dedicated two-loaded-conductor profile
    and requires phase="1phase". For in_thermal_insulation, explicitly choose
    partially_surrounded or completely_surrounded; unenclosed_in_air uses none.
    Unsupported construction/condition combinations are rejected without generic fallback.
    The generic default preserves the existing dataset column and does not identify a
    physical cable construction or insulation exposure.
    """
    return _call_engine(
        engine_size_cable,
        load=load, length_m=length_m, unit=unit, voltage=voltage, phase=phase,
        power_factor=power_factor, conductor_material=conductor_material,
        insulation=insulation, installation_method=installation_method,
        ambient_temp_c=ambient_temp_c, num_circuits=num_circuits, depth_m=depth_m,
        max_volt_drop_pct=max_volt_drop_pct, mcb_rating_amps=mcb_rating_amps,
        mcb_curve=mcb_curve, check_fault=check_fault, fault_current_ka=fault_current_ka,
        fault_time_s=fault_time_s, earth_conductor_material=earth_conductor_material,
        earth_fault_current_ka=earth_fault_current_ka, earth_fault_time_s=earth_fault_time_s,
        supply_loop_impedance_ohm=supply_loop_impedance_ohm,
        cable_construction=cable_construction, insulation_exposure=insulation_exposure,
    )


@mcp.tool()
def calculate_voltage_drop(
    size_mm2: PositiveNumber,
    load_amps: PositiveNumber,
    length_m: PositiveNumber,
    voltage: PositiveNumber = 400.0,
    phase: Phase = "3phase",
    conductor_material: Material = "copper",
    insulation: Insulation = "V90",
    power_factor: PowerFactor = 0.85,
    worst_case_pf: StrictBoolean = False,
) -> dict[str, Any]:
    """Calculate experimental AC voltage drop using the configured resistance/reactance data.

    length_m is the one-way route length, size_mm2 must exist in the configured dataset,
    and voltage is line-to-line for 3phase or phase-to-neutral for 1phase. worst_case_pf
    uses the impedance magnitude instead of the specified power factor. Threshold flags
    apply only to this voltage-drop model and do not establish standards compliance.
    """
    return _call_engine(
        engine_calc_vd,
        size_mm2=size_mm2, load_amps=load_amps, length_m=length_m, voltage=voltage,
        phase=phase, conductor_material=conductor_material, insulation=insulation,
        power_factor=power_factor, worst_case_pf=worst_case_pf,
    )


@mcp.tool()
def calculate_derating(
    ambient_temp_c: FiniteNumber | None = None,
    installation_method: Installation = "in_conduit_in_air",
    num_circuits: CircuitCount = 1,
    depth_m: PositiveNumber = 0.5,
    insulation: Insulation = "V90",
) -> dict[str, Any]:
    """Calculate model temperature, grouping and depth factors from the configured data.

    ambient_temp_c denotes soil temperature for underground installations and air
    temperature otherwise. None uses the dataset's reference value. Unsupported
    ranges are rejected; factors between rows use conservative lookup. Installation
    labels do not describe every cable construction, grouping layout or soil condition.
    """
    return _call_engine(
        get_derating_factor,
        ambient_temp_c=ambient_temp_c, installation_method=installation_method,
        num_circuits=num_circuits, depth_m=depth_m, insulation=insulation,
    )


@mcp.tool()
def check_loop_impedance(
    active_size_mm2: PositiveNumber,
    earth_size_mm2: PositiveNumber,
    protective_device_rating_amps: PositiveNumber,
    mcb_curve: Curve = "C",
    nominal_phase_voltage: PositiveNumber = 230.0,
    conductor_material: Material = "copper",
    insulation: Insulation = "V90",
    earth_conductor_material: Material = "copper",
    supply_loop_impedance_ohm: NonNegativeNumber = 0.0,
    length_m: PositiveNumber | None = None,
) -> dict[str, Any]:
    """Estimate an AC earth-fault loop length limit using resistance plus reactance.

    nominal_phase_voltage is phase-to-earth voltage. The external source-loop magnitude
    defaults to zero and must be supplied for a realistic check. length_m enables the
    route-length pass/fail comparison; without it no route check is performed. Material
    is specified separately for active and earth conductors. Generic B/C/D trip
    multipliers do not verify a specific device's clearing time or guarantee disconnection.
    """
    return _call_engine(
        engine_check_loop,
        active_size_mm2=active_size_mm2, earth_size_mm2=earth_size_mm2,
        protective_device_rating_amps=protective_device_rating_amps,
        mcb_curve=mcb_curve, nominal_phase_voltage=nominal_phase_voltage,
        conductor_material=conductor_material, insulation=insulation,
        earth_conductor_material=earth_conductor_material,
        supply_loop_impedance_ohm=supply_loop_impedance_ohm, length_m=length_m,
    )


@mcp.tool()
def check_short_circuit(
    size_mm2: PositiveNumber,
    fault_current_ka: PositiveNumber,
    fault_time_s: ClearingTime = 0.1,
    conductor_material: Material = "copper",
    insulation: Insulation = "V90",
) -> dict[str, Any]:
    """Check experimental adiabatic thermal withstand: I^2*t <= k^2*S^2.

    Supply the applicable fault current in kA and actual protective-device clearing
    time in seconds (0 < time <= 5). The default 0.1 s is an assumption, not a device
    rating. k comes from configured data and assumes the model's initial/final conductor
    temperatures. This check alone does not approve an active or earth conductor.
    """
    return _call_engine(
        engine_check_sc,
        size_mm2=size_mm2, fault_current_ka=fault_current_ka, fault_time_s=fault_time_s,
        conductor_material=conductor_material, insulation=insulation,
    )


@mcp.tool()
def get_standards_info() -> dict[str, Any]:
    """Return model scope and provenance status, without reproducing reference tables."""
    # Return an explicit metadata allowlist so local paths or unrelated private fields
    # supplied in a private dataset do not escape through this information tool.
    provenance = {
        key: DATA_PROVENANCE[key]
        for key in ("configured", "dataset_id", "source", "licence", "standard_editions", "assumptions", "validation_status")
        if key in DATA_PROVENANCE
    }
    return {
        "standard": "Historical references: AS/NZS 3008.1.1 and AS/NZS 3000; no edition certified",
        "model_status": "experimental; no standards-compliance certification",
        "data_provenance": provenance,
        "supported_materials": ["copper", "aluminium"],
        "supported_phases": ["1phase", "3phase"],
        "supported_insulations": {"V90": "Model operating temperature 75 C", "X90": "Model operating temperature 90 C"},
        "supported_installation_methods": [
            "in_conduit_in_air", "unenclosed_in_air", "in_thermal_insulation",
            "underground_duct", "buried_direct",
        ],
        "current_rating_profiles": [
            {
                "profile_id": profile_id,
                "supported_sizes_mm2": sorted(profile["ratings"]),
                **{
                    key: profile[key]
                    for key in (
                        "source", "cable_construction", "conductor_material", "insulation",
                        "loaded_conductors", "installation_method", "insulation_exposure",
                        "reference_temperature_c",
                    )
                    if key in profile
                },
            }
            for profile_id, profile in CURRENT_RATING_PROFILES.items()
        ],
        "assumptions": [
            "AC calculations only; DC data has not been independently validated.",
            "Cable construction, loaded conductors, grouping geometry and soil resistivity require dataset-specific validation.",
            "Earth material is independent of active material; earth-fault conditions default to active-fault conditions when check_fault is enabled and require verification.",
            "Zero external source-loop impedance and default fault clearing time are assumptions, not measurements or device guarantees.",
            "Optional checks not requested are not evidence of a passed design check.",
        ],
    }


@mcp.tool()
def generate_calculation_report(
    load: PositiveNumber,
    length_m: PositiveNumber,
    unit: LoadUnit = "A",
    voltage: PositiveNumber = 400.0,
    phase: Phase = "3phase",
    power_factor: PowerFactor = 0.85,
    conductor_material: Material = "copper",
    insulation: Insulation = "V90",
    installation_method: Installation = "in_conduit_in_air",
    ambient_temp_c: FiniteNumber | None = None,
    num_circuits: CircuitCount = 1,
    depth_m: PositiveNumber = 0.5,
    max_volt_drop_pct: DropLimit = 3.0,
    mcb_rating_amps: PositiveNumber | None = None,
    mcb_curve: Curve = "C",
    check_fault: StrictBoolean = False,
    fault_current_ka: PositiveNumber = 6.0,
    fault_time_s: ClearingTime = 0.1,
    earth_conductor_material: Material = "copper",
    earth_fault_current_ka: PositiveNumber | None = None,
    earth_fault_time_s: ClearingTime | None = None,
    supply_loop_impedance_ohm: NonNegativeNumber = 0.0,
    cable_construction: CableConstruction = "generic",
    insulation_exposure: InsulationExposure | None = None,
) -> str:
    """Generate an ASCII report of inputs, assumptions and requested experimental checks.

    Parameters and assumptions match size_cable. The report identifies optional checks
    as passed, failed or not requested; it does not certify standards compliance.
    """
    from cablesizecalculator.report import format_calculation_report

    result = size_cable(
        load=load, length_m=length_m, unit=unit, voltage=voltage, phase=phase,
        power_factor=power_factor, conductor_material=conductor_material,
        insulation=insulation, installation_method=installation_method,
        ambient_temp_c=ambient_temp_c, num_circuits=num_circuits, depth_m=depth_m,
        max_volt_drop_pct=max_volt_drop_pct, mcb_rating_amps=mcb_rating_amps,
        mcb_curve=mcb_curve, check_fault=check_fault, fault_current_ka=fault_current_ka,
        fault_time_s=fault_time_s, earth_conductor_material=earth_conductor_material,
        earth_fault_current_ka=earth_fault_current_ka, earth_fault_time_s=earth_fault_time_s,
        supply_loop_impedance_ohm=supply_loop_impedance_ohm,
        cable_construction=cable_construction, insulation_exposure=insulation_exposure,
    )
    return format_calculation_report(result)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
