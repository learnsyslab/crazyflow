"""Dependency-free IEEE-754 Float32 gate for independent controller defaults."""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

CONTROLLER_PARAMETER_NAMES = (
    "mass",
    "mass_thrust",
    "kp_xy",
    "kd_xy",
    "ki_xy",
    "i_range_xy",
    "kp_z",
    "kd_z",
    "ki_z",
    "i_range_z",
    "kR_xy",
    "kw_xy",
    "ki_m_xy",
    "i_range_m_xy",
    "kR_z",
    "kw_z",
    "ki_m_z",
    "i_range_m_z",
    "kd_omega_rp",
)
CONTROLLER_INVARIANT_NAMES = (
    "gravity_m_s2",
    "controller_frequency_hz",
    "torque_limit_pwm",
    "yaw_kd_omega",
)


def _source_value(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, (int, float)):
        return value
    return str(value)


def normalize_float32(value: Any) -> dict[str, Any]:
    """Return the deterministic IEEE-754 binary32 value and bit pattern."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"expected a finite numeric value, got {value!r}")
    try:
        packed = struct.pack("<f", numeric)
    except OverflowError as error:
        raise ValueError(f"value is outside the finite Float32 range: {value!r}") from error
    decimal = struct.unpack("<f", packed)[0]
    bits = struct.unpack("<I", packed)[0]
    return {"decimal": decimal, "hex": f"0x{bits:08x}", "bits": bits}


def _comparison(
    semantic_name: str,
    category: str,
    crazyflow_values: Mapping[str, Any],
    firmware_values: Mapping[str, Any],
    source_record: Mapping[str, str] | None,
) -> dict[str, Any]:
    crazyflow_present = semantic_name in crazyflow_values
    firmware_present = semantic_name in firmware_values
    crazyflow_source = crazyflow_values.get(semantic_name)
    firmware_source = firmware_values.get(semantic_name)
    source_record = {} if source_record is None else source_record
    result = {
        "semantic_name": semantic_name,
        "category": category,
        "crazyflow_source": source_record.get("crazyflow_source"),
        "crazyflow_symbol": source_record.get("crazyflow_symbol"),
        "firmware_source": source_record.get("firmware_source"),
        "firmware_symbol": source_record.get("firmware_symbol"),
        "crazyflow_source_value": (_source_value(crazyflow_source) if crazyflow_present else None),
        "firmware_source_value": (_source_value(firmware_source) if firmware_present else None),
        "crazyflow_source_type": (
            type(crazyflow_source).__name__ if crazyflow_present else "missing"
        ),
        "firmware_source_type": (type(firmware_source).__name__ if firmware_present else "missing"),
        "crazyflow_float32_decimal": None,
        "firmware_float32_decimal": None,
        "crazyflow_float32_hex": None,
        "firmware_float32_hex": None,
        "status": "MISMATCH",
    }
    if "note" in source_record:
        result["source_note"] = source_record["note"]

    errors: list[str] = []
    if not crazyflow_present:
        errors.append("Crazyflow value is missing")
    if not firmware_present:
        errors.append("firmware value is missing")
    normalized_crazyflow = None
    normalized_firmware = None
    if crazyflow_present:
        try:
            normalized_crazyflow = normalize_float32(crazyflow_source)
        except (TypeError, ValueError) as error:
            errors.append(f"Crazyflow value cannot be normalized: {error}")
    if firmware_present:
        try:
            normalized_firmware = normalize_float32(firmware_source)
        except (TypeError, ValueError) as error:
            errors.append(f"firmware value cannot be normalized: {error}")
    if normalized_crazyflow is not None:
        result["crazyflow_float32_decimal"] = normalized_crazyflow["decimal"]
        result["crazyflow_float32_hex"] = normalized_crazyflow["hex"]
    if normalized_firmware is not None:
        result["firmware_float32_decimal"] = normalized_firmware["decimal"]
        result["firmware_float32_hex"] = normalized_firmware["hex"]
    if (
        not errors
        and normalized_crazyflow is not None
        and normalized_firmware is not None
        and normalized_crazyflow["bits"] == normalized_firmware["bits"]
    ):
        result["status"] = "PASS"
    elif not errors:
        errors.append("normalized Float32 bit patterns differ")
    if errors:
        result["errors"] = errors
    return result


def _category_result(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    exact_matches = sum(item["status"] == "PASS" for item in comparisons)
    mismatch_count = len(comparisons) - exact_matches
    return {
        "count": len(comparisons),
        "exact_float32_matches": exact_matches,
        "mismatches": mismatch_count,
        "status": "PASS" if mismatch_count == 0 else "FAIL",
        "comparisons": comparisons,
    }


def evaluate_parameter_gate(
    crazyflow_parameters: Mapping[str, Any],
    firmware_parameters: Mapping[str, Any],
    crazyflow_invariants: Mapping[str, Any],
    firmware_invariants: Mapping[str, Any],
    *,
    source_records: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Collect every missing, invalid, or bit-different value before work."""
    records = {} if source_records is None else source_records
    parameter_comparisons = [
        _comparison(
            name,
            "controller_parameter",
            crazyflow_parameters,
            firmware_parameters,
            records.get(name),
        )
        for name in CONTROLLER_PARAMETER_NAMES
    ]
    invariant_comparisons = [
        _comparison(
            name,
            "controller_invariant",
            crazyflow_invariants,
            firmware_invariants,
            records.get(name),
        )
        for name in CONTROLLER_INVARIANT_NAMES
    ]
    parameters = _category_result(parameter_comparisons)
    invariants = _category_result(invariant_comparisons)
    checked_values = parameters["count"] + invariants["count"]
    exact_matches = parameters["exact_float32_matches"] + invariants["exact_float32_matches"]
    mismatch_count = parameters["mismatches"] + invariants["mismatches"]
    return {
        "schema_version": 1,
        "normalization": "IEEE-754 binary32 bit-pattern equality",
        "controller_parameters": parameters,
        "controller_invariants": invariants,
        "totals": {
            "checked_values": checked_values,
            "exact_float32_matches": exact_matches,
            "mismatches": mismatch_count,
        },
        "status": "PASS" if mismatch_count == 0 else "FAIL",
    }


def mismatches(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return every mismatch in stable parameter-then-invariant order."""
    return [
        comparison
        for category in ("controller_parameters", "controller_invariants")
        for comparison in result[category]["comparisons"]
        if comparison["status"] != "PASS"
    ]


def format_parameter_gate(result: Mapping[str, Any]) -> str:
    """Render a compact deterministic diagnostic while retaining all differences."""
    lines = ["INDEPENDENT FLOAT32 PARAMETER GATE"]
    for category_name, label in (
        ("controller_parameters", "controller_parameters"),
        ("controller_invariants", "controller_invariants"),
    ):
        category = result[category_name]
        lines.append(
            f"{label}: {category['exact_float32_matches']}/{category['count']} {category['status']}"
        )
        for comparison in category["comparisons"]:
            if comparison["status"] == "PASS":
                continue
            lines.append(
                f"  {comparison['semantic_name']}: "
                f"{comparison['crazyflow_source']}::"
                f"{comparison['crazyflow_symbol']}="
                f"{comparison['crazyflow_float32_decimal']!r}/"
                f"{comparison['crazyflow_float32_hex']} vs "
                f"{comparison['firmware_source']}::"
                f"{comparison['firmware_symbol']}="
                f"{comparison['firmware_float32_decimal']!r}/"
                f"{comparison['firmware_float32_hex']}"
            )
    return "\n".join(lines)
