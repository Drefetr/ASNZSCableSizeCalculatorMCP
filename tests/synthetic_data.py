"""Independent, deliberately artificial data for software tests, never cable design.

Resistance = arbitrary coefficient / area; conduit capacity = 4 * area + 1.
All other values below are invented test inputs, not copied standards tables.
"""


def make_dataset() -> dict:
    copper = [1.5, 2.5, 4, 6, 10, 16, 25, 35, 50, 70, 95, 120, 150, 185, 240, 300, 400, 500]
    aluminium = [size for size in copper if size >= 10]
    sizes = {"copper": copper, "aluminium": aluminium}
    methods = {
        "in_conduit_in_air": 1.0, "unenclosed_in_air": 1.2,
        "in_thermal_insulation": 0.5, "underground_duct": 1.1, "buried_direct": 1.3,
    }
    return {
        "schema_version": 1,
        "metadata": {
            "dataset_id": "synthetic-software-tests-v1",
            "source": "Independent formulas in tests/synthetic_data.py; not standards extracts.",
            "licence": "MIT; artificial software test fixture only.",
            "standard_editions": ["None: synthetic software test data"],
            "validation_status": "synthetic_test_only",
            "assumptions": {
                "frequency_hz": 50, "reference_air_temp_c": 40, "reference_ground_temp_c": 25,
                "cable_construction": "Artificial test model, no real cable construction",
                "loaded_conductors": "Artificial AC test circuits",
                "grouping_layout": "Artificial monotonic test factors",
                "soil_thermal_resistivity": "Not a physical soil model",
            },
        },
        "tables": {
            "CONDUCTOR_SIZES_COPPER": copper, "CONDUCTOR_SIZES_ALUMINIUM": aluminium,
            "TABLE_5_1_EARTH": {
                str(size): max(s for s in copper if s <= max(1.5, size / 2)) for size in copper
            },
            "RESISTANCE_TABLE": {
                mat: {
                    ins: {str(s): coefficient / s for s in sizes[mat]}
                    for ins, coefficient in (
                        ("V90", 24 if mat == "copper" else 40),
                        ("X90", 26 if mat == "copper" else 44),
                    )
                } for mat in sizes
            },
            "REACTANCE_TABLE": {str(s): 0.1 for s in copper},
            "CURRENT_RATINGS": {
                mat: {
                    ins: {
                        method: {
                            str(s): (4 * s + 1) * factor * (1 if mat == "copper" else 0.8)
                            * (1 if ins == "V90" else 1.1) for s in sizes[mat]
                        } for method, factor in methods.items()
                    } for ins in ("V90", "X90")
                } for mat in sizes
            },
            "TEMP_DERATING_AIR": {
                ins: {"20": 1.2, "25": 1.15, "30": 1.1, "35": 1.05, "40": 1,
                      "45": 0.9, "50": 0.8, "55": 0.7, "60": 0.6, "65": 0.5, "70": 0.4}
                for ins in ("V90", "X90")
            },
            "TEMP_DERATING_GROUND": {
                ins: {"10": 1.15, "15": 1.1, "20": 1.05, "25": 1,
                      "30": 0.9, "35": 0.8, "40": 0.7, "45": 0.6, "50": 0.5, "55": 0.4, "60": 0.3}
                for ins in ("V90", "X90")
            },
            "CIRCUITS_GROUPING_DERATING": {
                "1": 1, "2": 0.8, "3": 0.7, "4": 0.6, "5": 0.5, "10": 0.4, "12": 0.3, "20": 0.2,
            },
            "DEPTH_DERATING_GROUND": {"0.5": 1, "1": 0.9, "2": 0.8, "3": 0.7},
            "ADIABATIC_K_FACTORS": {"copper": {"V90": 100, "X90": 120}, "aluminium": {"V90": 60, "X90": 80}},
            "MCB_TRIP_MULTIPLIERS": {"B": 5, "C": 10, "D": 20},
        },
    }
