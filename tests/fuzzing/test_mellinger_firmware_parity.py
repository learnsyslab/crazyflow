"""Compare the Crazyflow Mellinger controller against the pinned firmware."""

from __future__ import annotations

import json
import warnings
from typing import Any

import mellinger_fuzz_oracle

from .mellinger.parameter_gate import mismatches
from .mellinger.parity import run_parity_smoke


def _compact_result(parity: dict[str, Any]) -> dict[str, Any]:
    def compact_worst_case(level: dict[str, Any]) -> dict[str, Any]:
        worst = level["worst_case"]
        return {
            "sample_index": worst["sample_index"],
            "motor": worst["motor"],
            "native_m1_m4": worst["native_m1_m4"],
            "jax_m1_m4": worst["jax_m1_m4"],
            "abs_errors_m1_m4": worst["abs_errors_m1_m4"],
        }

    independent = parity["independent_samples"]
    stateful = parity["stateful"]
    level_a = independent["interface_matched_mixer"]
    level_b = independent["production_end_to_end_mixer"]
    stateful_a = stateful["interface_matched_mixer"]
    stateful_b = stateful["production_end_to_end_mixer"]
    return {
        "status": parity["status"],
        "parameter_mode": parity["parameter_mode"],
        "comparison_kd_z": parity["comparison_kd_z"],
        "parameters_transferred": parity["parameters_transferred"],
        "stock_parameter_differences": {
            "status": parity["stock_parameter_gate"]["status"],
            "mismatches": mismatches(parity["stock_parameter_gate"]),
        },
        "parameter_gate": {
            "status": parity["parameter_gate"]["status"],
            "controller_parameters": (
                f"{parity['parameter_gate']['controller_parameters']['exact_float32_matches']}/"
                f"{parity['parameter_gate']['controller_parameters']['count']}"
            ),
            "controller_invariants": (
                f"{parity['parameter_gate']['controller_invariants']['exact_float32_matches']}/"
                f"{parity['parameter_gate']['controller_invariants']['count']}"
            ),
        },
        "mixer_configuration_gate": {
            "status": parity["mixer_configuration_values"]["status"],
            "values": (
                f"{parity['mixer_configuration_values']['values_passed']}/"
                f"{parity['mixer_configuration_values']['values_total']}"
            ),
            "full_prework_checks": (
                f"{parity['raw_mixer_configuration_gate']['checks_passed']}/"
                f"{parity['raw_mixer_configuration_gate']['checks_total']}"
            ),
        },
        "independent_samples": {
            "seed": independent["seed"],
            "samples": independent["samples"],
            "jax_batch_size": independent["jax_batch_size"],
            "native_controller_runs": independent["native_controller_runs"],
            "jax_controller_runs": independent["jax_controller_runs"],
            "pre_mixer": independent["pre_mixer"],
            "raw_level_a": {
                "analytical_bound": level_a["analytical_bound"],
                "acceptance_tolerance_pwm": level_a["acceptance_tolerance_pwm"],
                "overall_max_abs_error_pwm": level_a["overall_max_abs_error_pwm"],
                "per_motor": level_a["per_motor"],
                "worst_case": compact_worst_case(level_a),
                "error_classes": level_a["error_classes"],
            },
            "raw_level_b": {
                "analytical_bound": level_b["analytical_bound"],
                "acceptance_tolerance_pwm": level_b["acceptance_tolerance_pwm"],
                "overall_max_abs_error_pwm": level_b["overall_max_abs_error_pwm"],
                "per_motor": level_b["per_motor"],
                "worst_case": compact_worst_case(level_b),
                "error_classes": level_b["error_classes"],
            },
        },
        "stateful": {
            "status": stateful["status"],
            "steps": stateful["steps"],
            "comparable_output_steps": stateful["comparable_output_steps"],
            "comparable_integrator_steps": stateful["comparable_integrator_steps"],
            "reset_observation": stateful["reset_observation"],
            "post_reset": stateful["post_reset"],
            "max_thrust_error_pwm": stateful["max_thrust_error_pwm"],
            "max_torque_error_lsb": stateful["max_torque_error_lsb"],
            "max_internal_state_error": stateful["max_internal_state_error"],
            "reset_replay_equal": stateful["reset_replay_equal"],
            "raw_mixer_replay_equal": stateful["raw_mixer_replay_equal"],
            "raw_level_a_max_abs_error_pwm": stateful_a["overall_max_abs_error_pwm"],
            "raw_level_b_max_abs_error_pwm": stateful_b["overall_max_abs_error_pwm"],
        },
        "old_four_pwm_regression": parity["permanent_four_pwm_counterexample"],
        "runtime": parity["runtime"],
    }


def test_parameter_mismatch_fails_before_rng_and_controllers() -> None:
    from .mellinger.parameter_gate import evaluate_parameter_gate
    from .mellinger.parity import build_parameter_gate_failure

    parameter_names = (
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
    crazyflow_values = {name: 1.0 for name in parameter_names}
    firmware_values = dict(crazyflow_values)
    crazyflow_values["mass"] = 2.0
    crazyflow_values["kd_z"] = 0.5
    invariants = {
        "gravity_m_s2": 9.81,
        "controller_frequency_hz": 500,
        "torque_limit_pwm": 32000,
        "yaw_kd_omega": 0,
    }
    source_records = {
        name: {
            "crazyflow_source": "crazyflow/control/mellinger/params.toml",
            "crazyflow_symbol": f"cf2x_L250.{name}",
            "firmware_source": ("src/modules/src/controller/controller_mellinger.c"),
            "firmware_symbol": f"controllerMellinger.{name}",
        }
        for name in (*parameter_names, *invariants)
    }
    gate = evaluate_parameter_gate(
        crazyflow_values, firmware_values, invariants, invariants, source_records=source_records
    )
    summary = build_parameter_gate_failure(gate)
    differences = mismatches(gate)
    assert gate["status"] == "FAIL"
    assert {item["semantic_name"] for item in differences} == {"mass", "kd_z"}
    assert summary["status"] == "FAIL"
    assert summary["samples_generated"] == 0
    assert summary["controller_runs"] == 0
    assert summary["parameters_transferred"] is False
    for difference in differences:
        assert difference["crazyflow_source"]
        assert difference["crazyflow_symbol"]
        assert difference["firmware_source"]
        assert difference["firmware_symbol"]
        assert difference["crazyflow_float32_decimal"] is not None
        assert difference["firmware_float32_decimal"] is not None
        assert difference["crazyflow_float32_hex"].startswith("0x")
        assert difference["firmware_float32_hex"].startswith("0x")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "failure_stage": summary["failure_stage"],
                "samples_generated": summary["samples_generated"],
                "controller_runs": summary["controller_runs"],
                "parameters_transferred": summary["parameters_transferred"],
                "mismatches": differences,
            },
            sort_keys=True,
        )
    )


def test_mellinger_firmware_parity():
    result = _compact_result(run_parity_smoke(mellinger_fuzz_oracle))
    if result["status"] == "PASS_CHECKED_SCOPE_WITH_DIFFERENCES":
        stock = [
            item["semantic_name"] for item in result["stock_parameter_differences"]["mismatches"]
        ]
        warnings.warn(
            f"Mellinger stock parameter differences: {stock}; "
            f"stateful result: {result['stateful']['status']}. "
            "The passing checks are limited to the matched configuration and comparable steps.",
            stacklevel=2,
        )
    print("MELLINGER_FUZZ_RESULT=" + json.dumps(result, sort_keys=True))
    print(
        "PASS: on-demand pinned firmware fetch, transient build, "
        "matched-configuration controller comparisons, raw levels A/B, "
        "and cleanup. Stock defaults and the reset transition are reported separately."
    )
