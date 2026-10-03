"""Exercise actual schema publication and validation over the stdio transport."""

import asyncio
import json
import os
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from synthetic_data import make_dataset


def _json_result(response: Any) -> dict[str, Any]:
    assert not response.is_error, f"Unexpected MCP tool error: {response.content}"
    if isinstance(response.structured_content, dict):
        return response.structured_content
    assert response.content and hasattr(response.content[0], "text")
    return json.loads(response.content[0].text)


async def run_protocol_checks() -> dict[str, Any]:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "cablesizecalculator.server"],
        env={"CABLESIZE_DATA_FILE": os.environ["CABLESIZE_DATA_FILE"]},
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            initialization = await session.initialize()
            assert initialization.server_info.name == "CableSizeCalculator"
            discovered = await session.list_tools()
            tools = {tool.name: tool for tool in discovered.tools}
            assert set(tools) == {
                "size_cable", "calculate_voltage_drop", "calculate_derating",
                "check_loop_impedance", "check_short_circuit", "get_standards_info",
                "generate_calculation_report",
            }
            assert all(tool.input_schema["additionalProperties"] is False for tool in tools.values())

            schema = tools["size_cable"].input_schema["properties"]
            assert schema["length_m"]["exclusiveMinimum"] == 0
            assert schema["power_factor"]["maximum"] == 1
            assert schema["num_circuits"]["minimum"] == 1
            assert schema["fault_time_s"]["maximum"] == 5
            assert set(schema["phase"]["enum"]) == {"1phase", "3phase"}
            assert set(schema["conductor_material"]["enum"]) == {"copper", "aluminium"}
            assert "earth_fault_current_ka" in schema
            assert "supply_loop_impedance_ohm" in schema

            invalid_calls = [
                ("size_cable", {"load": 10, "length_m": -1}),
                ("size_cable", {"load": 10, "length_m": 5, "phase": "unknown"}),
                ("size_cable", {"load": 10, "length_m": 5, "phase": "dc"}),
                ("size_cable", {"load": 10, "length_m": 5, "num_circuits": 1.5}),
                ("size_cable", {"load": True, "length_m": 5}),
                ("size_cable", {"load": "Infinity", "length_m": 5}),
                ("size_cable", {"load": 10, "length_m": 5, "check_fault": "true"}),
                ("size_cable", {"load": 10, "length_m": 5, "supply_loop_impedance_ohm": -1}),
                ("size_cable", {"load": 10, "length_m": 5, "supply_loop_impedance_ohm": 0.1}),
                ("size_cable", {"load": 10, "length_m": 5, "earth_fault_current_ka": 1}),
                ("size_cable", {"load": 10, "length_m": 5, "mcb_rating_amps": 1}),
                ("size_cable", {"load": 10, "length_m": 5, "ambient_temp_c": 100}),
                ("calculate_voltage_drop", {"size_mm2": 4, "load_amps": 10, "length_m": 5, "power_factor": -0.5}),
                ("calculate_voltage_drop", {"size_mm2": 4, "load_amps": 10, "length_m": 5, "voltage": 0}),
                ("check_short_circuit", {"size_mm2": 4, "fault_current_ka": 1, "fault_time_s": 6}),
            ]
            for name, arguments in invalid_calls:
                response = await session.call_tool(name, arguments)
                assert response.is_error, f"{name} accepted invalid arguments: {arguments}"
                assert response.content, "Tool errors must explain the failure"
                message = " ".join(block.text for block in response.content if hasattr(block, "text"))
                assert message != f"Error executing tool {name}", "Expected failures need an actionable reason"

            # Valid required arguments plus a typo must never silently omit a check.
            unknown_argument_calls = [
                ("size_cable", {"load": 10, "length_m": 5}, "check_faults"),
                ("size_cable", {"load": 10, "length_m": 5}, "mcb_rating_amp"),
                ("generate_calculation_report", {"load": 10, "length_m": 5}, "check_faults"),
                ("get_standards_info", {}, "dataset"),
                ("calculate_voltage_drop", {"size_mm2": 4, "load_amps": 10, "length_m": 5}, "worst_case"),
                ("calculate_derating", {}, "num_circuit"),
                ("check_loop_impedance", {"active_size_mm2": 4, "earth_size_mm2": 4,
                                          "protective_device_rating_amps": 10}, "length"),
                ("check_short_circuit", {"size_mm2": 4, "fault_current_ka": 1}, "fault_time"),
            ]
            for name, arguments, unknown_field in unknown_argument_calls:
                response = await session.call_tool(name, {**arguments, unknown_field: True})
                assert response.is_error, f"{name} discarded the unknown argument {unknown_field}"
                message = " ".join(block.text for block in response.content if hasattr(block, "text"))
                assert "Unknown argument" in message
                assert unknown_field in message

            info = _json_result(await session.call_tool("get_standards_info", {}))
            assert info["model_status"].startswith("experimental")
            assert "standard_earth_sizes_table_5_1" not in info
            assert "copper_sizes_mm2" not in info
            assert info["data_provenance"]["configured"] is True
            assert info["data_provenance"]["dataset_id"] == make_dataset()["metadata"]["dataset_id"]
            arguments = {"load": 10.0, "length_m": 5.0, "mcb_rating_amps": 10.0}
            selection_response = await session.call_tool("size_cable", arguments)
            result = _json_result(selection_response)
            assert result["status"] == "success", result
            assert result["recommended_active_size_mm2"] > 0
            assert result["check_states"]["loop_impedance"] == "passed"
            assert result["check_states"]["short_circuit"] == "not_requested"
            assert result["check_states"]["earth_short_circuit"] == "not_requested"
            assert result["loop_impedance_check"]["compliant"] is True

            checked = _json_result(await session.call_tool("size_cable", {
                **arguments, "check_fault": True, "fault_current_ka": 0.01, "fault_time_s": 0.01,
            }))
            assert checked["status"] == "success"
            assert checked["check_states"]["short_circuit"] == "passed"
            assert checked["check_states"]["earth_short_circuit"] == "passed"

            report_response = await session.call_tool("generate_calculation_report", arguments)
            assert not report_response.is_error, report_response.content
            report = report_response.content[0].text
            assert "EXPERIMENTAL SELECTION SUMMARY" in report
            assert "COMPLIANT SELECTION SUMMARY" not in report
            assert "Earth-fault loop length: PASS" in report
            assert "Earth conductor thermal withstand: NOT CHECKED" in report
            assert report.isascii()

            # A valid request with no feasible candidate is a structured result,
            # distinct from a rejected input or missing dataset tool error.
            failed = _json_result(await session.call_tool("size_cable", {"load": 1e12, "length_m": 5}))
            assert failed["status"] == "failed"
            assert failed["recommended_active_size_mm2"] is None
            assert failed["message"]
            assert failed["all_candidates"]
            return {"configured": True, "tool_count": len(tools), "invalid_inputs_rejected": len(invalid_calls)}


def test_stdio_mcp_schema_and_invalid_inputs():
    summary = asyncio.run(run_protocol_checks())
    assert summary["configured"] is True
    assert summary["tool_count"] == 7
    assert summary["invalid_inputs_rejected"] >= 8


def test_stdio_missing_dataset_override_fails_without_using_bundled_data(tmp_path):
    async def check_missing_override():
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "cablesizecalculator.server"],
            env={"CABLESIZE_DATA_FILE": str(tmp_path / "missing.json")},
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                info = _json_result(await session.call_tool("get_standards_info", {}))
                assert info["data_provenance"]["configured"] is False
                response = await session.call_tool("size_cable", {"load": 10, "length_m": 5})
                assert response.is_error
                message = " ".join(block.text for block in response.content if hasattr(block, "text"))
                assert "CABLESIZE_DATA_FILE" in message
                assert str(tmp_path) not in message

    asyncio.run(check_missing_override())
