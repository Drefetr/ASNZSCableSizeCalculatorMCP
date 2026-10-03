# CableSizeCalculatorMCP

An experimental, offline [Model Context Protocol](https://modelcontextprotocol.io/) server for AC cable-sizing calculations. It evaluates current capacity, voltage drop, selected protection constraints, and conductor thermal withstand using bundled or operator-supplied reference data.

Calculations are experimental and do not certify an installation or establish standards compliance. The bundled reference data is public domain; its numerical accuracy has not been independently validated.

## Scope

The model supports single-phase and three-phase AC calculations, copper or aluminium active conductors, and separately specified earth conductor material. DC is unsupported because the model uses AC resistance and reactance data. Installation categories and insulation labels describe the model; their suitability depends on the supplied dataset's construction, conductor loading, reference conditions, and other assumptions.

Sizing evaluates every candidate against current capacity and unrounded voltage drop. Optional breaker and fault inputs enable further checks. Results and reports identify which checks were performed or omitted. A successful selection means that the requested calculations passed under the supplied assumptions; it is not a declaration of standards compliance. Network fault impedance, breaker characteristics, actual clearing times, and reference-data applicability must be established for the installation.

## Setup

Python 3.12 or later and [uv](https://docs.astral.sh/uv/) are required for the development setup:

```bash
uv sync
```

The server loads the bundled dataset by default. To use a different dataset, set `CABLESIZE_DATA_FILE` to a local JSON file before starting the process:

```powershell
$env:CABLESIZE_DATA_FILE = (Resolve-Path "./private-data/cable-data.local.json").Path
uv run cablesizecalculator
```

```bash
export CABLESIZE_DATA_FILE=/absolute/path/to/cable-data.local.json
uv run cablesizecalculator
```

Use [reference_data.json](src/cablesizecalculator/reference_data.json) as the format for a custom dataset. The loader checks metadata and numerical table completeness. Restart the server after changing the dataset. An invalid override fails with a configuration error.

Tests generate a synthetic dataset to exercise the code. Those values are deliberately not electrical design references and must not be used for installation design.

### MCP client configuration

Replace the paths in this example with your checkout and private data locations. Omit the `env` entry to use the bundled data:

```json
{
  "mcpServers": {
    "cablesizecalculator": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/CableSizeCalculatorMCP",
        "run",
        "cablesizecalculator"
      ],
      "env": {
        "CABLESIZE_DATA_FILE": "/absolute/path/to/cable-data.local.json"
      }
    }
  }
}
```

## Tools

- `size_cable`: Select a candidate against the requested model constraints.
- `calculate_voltage_drop`: Calculate AC voltage drop for a given conductor and route.
- `calculate_derating`: Apply conservative temperature, grouping, and burial-depth lookups within supported ranges.
- `check_loop_impedance`: Evaluate the model's earth-fault loop constraint with resistance, reactance, and specified source impedance.
- `check_short_circuit`: Evaluate adiabatic conductor thermal withstand for specified fault conditions.
- `get_standards_info`: Describe model scope and dataset availability and provenance.
- `generate_calculation_report`: Produce an ASCII report of inputs, assumptions, candidate checks, and omitted checks.

## Development

```bash
uv run pytest
uv build
```

## License

The code is licensed under [MIT](LICENSE). The bundled reference data is public domain.
