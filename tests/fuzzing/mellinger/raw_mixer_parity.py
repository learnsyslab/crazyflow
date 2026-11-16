"""Pre-run proof gate and aggregation for two raw Legacy-X mixer claims."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

MOTOR_ORDER = ("M1", "M2", "M3", "M4")
LEGACY_X_MIXING_MATRIX = ((-1.0, -1.0, 1.0, 1.0), (-1.0, 1.0, 1.0, -1.0), (-1.0, 1.0, -1.0, 1.0))
RAW_MIXER_BOUNDARY = (
    "Legacy-CF2-X signed M1-M4 immediately after linear mixing; before "
    "Crazyflow clipping, battery compensation, powerDistributionCap(), "
    "arming, idle thrust, ESCs, and hardware semantics"
)
EXCLUDED_CLAIMS = (
    "controller-output equality in the interface-matched mixer claim",
    "Crazyflow production clipping and pwm_min/pwm_max",
    "battery compensation",
    "powerDistributionCap() and common high-side desaturation",
    "arming and idle thrust",
    "ESC, motor-driver, and hardware behavior",
    "STM32 or firmware-in-the-loop identity",
    "final physical motor PWM",
)

INTERFACE_MATCHED_TOLERANCE_PWM = 3
PRODUCTION_END_TO_END_TOLERANCE_PWM = 8
CONTROLLER_THRUST_ABS_BOUND_PWM = 1_500_000.0
TORQUE_ABS_BOUND_PWM = 32_000.0
RAW_INTERMEDIATE_ABS_BOUND_PWM = CONTROLLER_THRUST_ABS_BOUND_PWM + 2 * TORQUE_ABS_BOUND_PWM
FLOAT32_MAGNITUDE_CEILING_PWM = float(2**21)

# At |x| < 2**21, a correctly rounded Float32 addition has at most 0.0625
# absolute error. In any binary reduction of the JAX dot's three signed terms,
# both the pairwise and final sums stay below 2**16 (error <= 0.001953125
# each). Multiplication by +/-1 is exact.
NATIVE_MIXER_FLOAT32_ROUNDING_PWM = 3 * 0.0625
JAX_DOT_FLOAT32_ROUNDING_PWM = 2 * 0.001953125
JAX_FINAL_ADD_FLOAT32_ROUNDING_PWM = 0.0625
JAX_MIXER_FLOAT32_ROUNDING_PWM = JAX_DOT_FLOAT32_ROUNDING_PWM + JAX_FINAL_ADD_FLOAT32_ROUNDING_PWM
COMBINED_FLOAT32_ROUNDING_PWM = NATIVE_MIXER_FLOAT32_ROUNDING_PWM + JAX_MIXER_FLOAT32_ROUNDING_PWM

EXPECTED_JAX_PRIMITIVES = (
    "slice",
    "div",
    "slice",
    "concatenate",
    "broadcast_in_dim",
    "dot_general",
    "add",
)
FORBIDDEN_PRODUCTION_QUANTIZATION_TOKENS = (
    "round(",
    "floor(",
    "trunc(",
    "astype(",
    "int16",
    "int32",
    "int64",
    "stop_gradient",
)


def _f32_tuple(values: Sequence[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values, dtype=np.float32))


EXPECTED_ANALYSIS_CONTRACT = {
    "fuzzer_position_state_abs_max_m": 2.0,
    "fuzzer_position_setpoint_abs_max_m": 2.0,
    "fuzzer_velocity_state_abs_max_m_s": 1.0,
    "fuzzer_velocity_setpoint_abs_max_m_s": 1.0,
    "fuzzer_acceleration_setpoint_abs_max_m_s2": 5.0,
    "accepted_quaternion_norm_max": 1.001,
    "mass_kg": float(np.float32(0.029)),
    "mass_thrust_pwm_per_n": float(np.float32(132000.0)),
    "gravity_vec_m_s2": _f32_tuple((0.0, 0.0, -9.81)),
    "kp": _f32_tuple((0.4, 0.4, 1.25)),
    "kd": _f32_tuple((0.2, 0.2, 0.4)),
    "ki": _f32_tuple((0.05, 0.05, 0.05)),
    "position_integral_abs_max": _f32_tuple((2.0, 2.0, 0.4)),
    "torque_clip_abs_max_pwm": _f32_tuple((32000.0, 32000.0, 32000.0)),
    "pre_mixer_thrust_atol_pwm": 1.0,
    "pre_mixer_torque_tolerance_lsb": 1,
    "controller_thrust_abs_bound_pwm": CONTROLLER_THRUST_ABS_BOUND_PWM,
    "raw_intermediate_abs_bound_pwm": RAW_INTERMEDIATE_ABS_BOUND_PWM,
    "float32_magnitude_ceiling_pwm": FLOAT32_MAGNITUDE_CEILING_PWM,
    "jax_x64_enabled": False,
}


def derive_controller_thrust_range(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Derive a code-contract thrust envelope without generated samples."""
    position_error = float(contract["fuzzer_position_state_abs_max_m"]) + float(
        contract["fuzzer_position_setpoint_abs_max_m"]
    )
    velocity_error = float(contract["fuzzer_velocity_state_abs_max_m_s"]) + float(
        contract["fuzzer_velocity_setpoint_abs_max_m_s"]
    )
    acceleration = float(contract["fuzzer_acceleration_setpoint_abs_max_m_s2"])
    mass = abs(float(contract["mass_kg"]))
    gravity = tuple(abs(float(value)) for value in contract["gravity_vec_m_s2"])
    kp = tuple(abs(float(value)) for value in contract["kp"])
    kd = tuple(abs(float(value)) for value in contract["kd"])
    ki = tuple(abs(float(value)) for value in contract["ki"])
    integral = tuple(abs(float(value)) for value in contract["position_integral_abs_max"])
    target_axis_bounds = tuple(
        mass * (acceleration + gravity[axis])
        + kp[axis] * position_error
        + kd[axis] * velocity_error
        + ki[axis] * integral[axis]
        for axis in range(3)
    )

    # For q norm <= n, the first two z-axis components are <= n**2 by
    # 2|ab| <= a**2+b**2. The third is 1-2(x**2+y**2).
    quaternion_norm = float(contract["accepted_quaternion_norm_max"])
    norm_squared = quaternion_norm**2
    rotation_component_bound = max(norm_squared, abs(1.0 - 2.0 * norm_squared), 1.0)
    current_thrust_bound_n = rotation_component_bound * sum(target_axis_bounds)
    exact_real_pwm_bound = current_thrust_bound_n * abs(float(contract["mass_thrust_pwm_per_n"]))

    # 0.1% is a deliberately loose forward-error allowance. At u=2**-24 it
    # covers over 16,000 unit roundoffs, whereas the pinned target-thrust,
    # quaternion-column, dot, and final-scale expressions use far fewer than
    # 100 Float32 arithmetic operations.
    float32_forward_error_factor = 1.001
    bound_with_float32_allowance = exact_real_pwm_bound * float32_forward_error_factor
    return {
        "position_error_abs_max_m": position_error,
        "velocity_error_abs_max_m_s": velocity_error,
        "target_thrust_axis_abs_bounds_n": target_axis_bounds,
        "rotation_z_axis_component_abs_bound": rotation_component_bound,
        "exact_real_thrust_abs_bound_pwm": exact_real_pwm_bound,
        "float32_forward_error_factor": float32_forward_error_factor,
        "bound_with_float32_allowance_pwm": bound_with_float32_allowance,
        "selected_contract_bound_pwm": float(contract["controller_thrust_abs_bound_pwm"]),
        "status": (
            "PASS"
            if bound_with_float32_allowance <= float(contract["controller_thrust_abs_bound_pwm"])
            else "FAIL"
        ),
        "observations_used": False,
    }


CONTROLLER_THRUST_RANGE_DERIVATION = derive_controller_thrust_range(EXPECTED_ANALYSIS_CONTRACT)

COMMON_BOUND_ASSUMPTIONS = (
    "The deterministic fuzzer uses the checked position, velocity, acceleration, "
    "quaternion, parameter, and integrator ranges recorded by the pre-run gate.",
    "The existing pre-mixer gate accepts at most 1 PWM thrust error and at most "
    "1 LSB after JAX torque truncation toward zero.",
    "Native roll, pitch, and yaw are valid int16_t values clipped to +/-32000; "
    "the productive JAX torque values are Float32 and clipped to the same range.",
    "The target-thrust formula, quaternion acceptance gate, fuzzer ranges, and "
    "exact parameters derive |thrust| < 1,424,710 PWM including a conservative "
    "Float32 allowance; the selected 1,500,000-PWM contract is therefore safe. "
    "All raw intermediates are <=1,564,000 PWM, strictly below 2**21.",
    "The pinned native object uses IEEE-754 Float32 with FP contraction disabled. "
    "The gated JAXPR uses Float32 divide, dot_general, and add with x64 disabled.",
)

INTERFACE_MATCHED_BOUND_DERIVATION = {
    "level": "A",
    "kind": "analytical pre-run bound",
    "proof_status": "PASS",
    "observations_used": False,
    "symbolic_inequality": (
        "|M_native-M_jax_matched| < "
        "|trunc0(R/2)-R/2| + |trunc0(P/2)-P/2| "
        "+ 1(native int32 cast) + 0.25390625(Float32) < 2.25390625"
    ),
    "assumptions": COMMON_BOUND_ASSUMPTIONS
    + (
        "Both mixers receive the identical firmware-compatible tuple: the same "
        "Float32 thrust and the same int16 roll/pitch/yaw values after only the "
        "documented Crazyflow pitch/yaw sign adapter.",
    ),
    "linear_input_bound_pwm": {
        "thrust": 0.0,
        "roll_half_truncation": "<=0.5",
        "pitch_half_truncation": "<=0.5",
        "yaw": 0.0,
        "sum": "<=1.0",
    },
    "controller_thrust_range_derivation": CONTROLLER_THRUST_RANGE_DERIVATION,
    "open_real_bound_before_float32_rounding_pwm": "<2.0",
    "native_final_int32_truncation_pwm": "<1.0",
    "float32_rounding_addendum_pwm": COMBINED_FLOAT32_ROUNDING_PWM,
    "strict_total_bound_pwm": f"<{2.0 + COMBINED_FLOAT32_ROUNDING_PWM}",
    "integer_acceptance_tolerance_pwm": INTERFACE_MATCHED_TOLERANCE_PWM,
    "independent_of_observations": (
        "Every term follows from the pinned formulas, fixed interface types, "
        "pre-mixer contracts, and pre-run fuzzer range gate; no sampled maximum "
        "is used."
    ),
}

PRODUCTION_END_TO_END_BOUND_DERIVATION = {
    "level": "B",
    "kind": "analytical pre-run bound",
    "proof_status": "PASS",
    "observations_used": False,
    "symbolic_inequality": (
        "|M_native-M_jax_production| < "
        "1(thrust) + (1+1)/2+0.5(roll) + (1+1)/2+0.5(pitch) "
        "+ (1+1)(yaw) + 1(native int32 cast) "
        "+ 0.25390625(Float32) < 7.25390625"
    ),
    "assumptions": COMMON_BOUND_ASSUMPTIONS
    + (
        "The native mixer consumes the real quantized firmware-controller tuple. "
        "The JAX mixer consumes the real unquantized productive controller tuple "
        "from the same state, setpoint, and time sample.",
        "For every JAX torque q, q=trunc0(q)+f with |f|<1. The pre-mixer gate "
        "allows the native int16 torque to differ from trunc0(q) by at most 1 LSB.",
    ),
    "linear_input_bound_pwm": {
        "thrust": "<=1.0",
        "roll": "<1.5",
        "pitch": "<1.5",
        "yaw": "<2.0",
        "sum": "<6.0",
    },
    "controller_thrust_range_derivation": CONTROLLER_THRUST_RANGE_DERIVATION,
    "open_real_bound_before_float32_rounding_pwm": "<7.0",
    "native_final_int32_truncation_pwm": "<1.0",
    "float32_rounding_addendum_pwm": COMBINED_FLOAT32_ROUNDING_PWM,
    "strict_total_bound_pwm": f"<{7.0 + COMBINED_FLOAT32_ROUNDING_PWM}",
    "integer_acceptance_tolerance_pwm": PRODUCTION_END_TO_END_TOLERANCE_PWM,
    "includes_productive_interface_quantization_difference": True,
    "independent_of_observations": (
        "The bound is fixed before RNG from coefficient magnitudes, truncation "
        "contracts, Float32 ranges, and operation counts. Empirical maxima do "
        "not select or enlarge it."
    ),
    "permanent_counterexample_to_old_four_pwm_claim": {
        "native_controller_output_firmware_signs": [-2.0537006713209482, -9, 5, 8],
        "jax_controller_output_crazyflow_signs": [
            -1.086831395793519,
            -9.957264058160904,
            -5.72288559208756,
            -8.991270727389749,
        ],
        "jax_torque_after_truncation_common_signs": [-9, -5, -8],
        "native_raw_m1_m4": [11, -8, 0, -12],
        "jax_raw_m1_m4": [
            15.744513511657715,
            -7.960913181304932,
            0.06436502933502197,
            -12.195291519165039,
        ],
        "abs_errors_m1_m4": [
            4.744513511657715,
            0.03908681869506836,
            0.06436502933502197,
            0.19529151916503906,
        ],
        "conclusion": (
            "M1 disproves the old universal 4-PWM claim and remains inside the "
            "analytical 8-PWM production end-to-end acceptance tolerance."
        ),
    },
}


_BASIS_CASES = (
    {
        "name": "collective",
        "common_input": (10.0, 0, 0, 0),
        "native_input": (10.0, 0, 0, 0),
        "expected": (10.0, 10.0, 10.0, 10.0),
    },
    {
        "name": "roll",
        "common_input": (0.0, 2, 0, 0),
        "native_input": (0.0, 2, 0, 0),
        "expected": (-1.0, -1.0, 1.0, 1.0),
    },
    {
        "name": "pitch_sign_adapter",
        "common_input": (0.0, 0, 2, 0),
        "native_input": (0.0, 0, -2, 0),
        "expected": (-1.0, 1.0, 1.0, -1.0),
    },
    {
        "name": "yaw_sign_adapter",
        "common_input": (0.0, 0, 0, 1),
        "native_input": (0.0, 0, 0, -1),
        "expected": (-1.0, 1.0, -1.0, 1.0),
    },
)


def _as_float_tuple(values: Sequence[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values).reshape(-1))


def _check(
    checks: list[dict[str, Any]], name: str, condition: bool, *, expected: Any, actual: Any
) -> None:
    checks.append(
        {
            "name": name,
            "status": "PASS" if condition else "FAIL",
            "expected": expected,
            "actual": actual,
        }
    )


def analytical_bounds_are_valid() -> bool:
    """Return whether both observation-independent proofs meet their hard caps."""
    return (
        not INTERFACE_MATCHED_BOUND_DERIVATION["observations_used"]
        and not PRODUCTION_END_TO_END_BOUND_DERIVATION["observations_used"]
        and INTERFACE_MATCHED_BOUND_DERIVATION["proof_status"] == "PASS"
        and PRODUCTION_END_TO_END_BOUND_DERIVATION["proof_status"] == "PASS"
        and CONTROLLER_THRUST_RANGE_DERIVATION["status"] == "PASS"
        and INTERFACE_MATCHED_TOLERANCE_PWM <= 3
        and PRODUCTION_END_TO_END_TOLERANCE_PWM <= 8
        and 2.0 + COMBINED_FLOAT32_ROUNDING_PWM < INTERFACE_MATCHED_TOLERANCE_PWM
        and 7.0 + COMBINED_FLOAT32_ROUNDING_PWM < PRODUCTION_END_TO_END_TOLERANCE_PWM
        and RAW_INTERMEDIATE_ABS_BOUND_PWM < FLOAT32_MAGNITUDE_CEILING_PWM
    )


def evaluate_raw_mixer_configuration(
    jax_mixer: Callable[[float, float, float, float], Sequence[Any]],
    native_mixer: Callable[[float, int, int, int], Sequence[Any]],
    mixing_matrix: Sequence[Sequence[Any]],
    bridge_metadata: Mapping[str, Any],
    *,
    analysis_contract: Mapping[str, Any],
    production_helper_source: str,
    jax_primitives: Sequence[str],
) -> dict[str, Any]:
    """Verify both proofs and the exact raw boundary before RNG/controllers."""
    checks: list[dict[str, Any]] = []
    actual_matrix = tuple(tuple(float(value) for value in row) for row in np.asarray(mixing_matrix))
    _check(
        checks,
        "crazyflow_legacy_x_matrix",
        actual_matrix == LEGACY_X_MIXING_MATRIX,
        expected=LEGACY_X_MIXING_MATRIX,
        actual=actual_matrix,
    )
    for field, expected in (
        ("mixer_motor_order", "M1,M2,M3,M4"),
        ("mixer_raw", True),
        ("mixer_uncapped", True),
        ("mixer_pre_battery", True),
        ("mixer_final_motor_pwm", False),
    ):
        _check(
            checks,
            f"native_{field}",
            bridge_metadata.get(field) == expected,
            expected=expected,
            actual=bridge_metadata.get(field),
        )

    safe_range = bridge_metadata.get("mixer_thrust_pwm_float32_safe_range")
    safe_range_ok = (
        isinstance(safe_range, (tuple, list))
        and len(safe_range) == 2
        and float(safe_range[0]) <= -CONTROLLER_THRUST_ABS_BOUND_PWM
        and float(safe_range[1]) >= CONTROLLER_THRUST_ABS_BOUND_PWM
    )
    _check(
        checks,
        "native_safe_range_covers_fuzzer_bound",
        safe_range_ok,
        expected=(f"a range covering +/-{CONTROLLER_THRUST_ABS_BOUND_PWM:g} PWM"),
        actual=safe_range,
    )

    for name, expected in EXPECTED_ANALYSIS_CONTRACT.items():
        actual = analysis_contract.get(name)
        _check(
            checks, f"bound_assumption_{name}", actual == expected, expected=expected, actual=actual
        )
    live_thrust_derivation = derive_controller_thrust_range(analysis_contract)
    _check(
        checks,
        "derived_fuzzer_thrust_range",
        live_thrust_derivation["status"] == "PASS",
        expected=(
            "code-contract derivation including Float32 allowance <= "
            f"{CONTROLLER_THRUST_ABS_BOUND_PWM:g} PWM"
        ),
        actual=live_thrust_derivation,
    )
    derived_raw_bound = float(analysis_contract["controller_thrust_abs_bound_pwm"]) + 2.0 * max(
        abs(float(value)) for value in analysis_contract["torque_clip_abs_max_pwm"]
    )
    _check(
        checks,
        "derived_raw_intermediate_range",
        derived_raw_bound == float(analysis_contract["raw_intermediate_abs_bound_pwm"])
        and derived_raw_bound < float(analysis_contract["float32_magnitude_ceiling_pwm"]),
        expected=(
            f"{RAW_INTERMEDIATE_ABS_BOUND_PWM:g} PWM < {FLOAT32_MAGNITUDE_CEILING_PWM:g} PWM"
        ),
        actual=derived_raw_bound,
    )
    _check(
        checks,
        "interface_matched_bound_at_most_3_pwm",
        analytical_bounds_are_valid() and INTERFACE_MATCHED_TOLERANCE_PWM <= 3,
        expected="proved observation-independent integer tolerance <= 3",
        actual=INTERFACE_MATCHED_BOUND_DERIVATION,
    )
    _check(
        checks,
        "production_end_to_end_bound_at_most_8_pwm",
        analytical_bounds_are_valid() and PRODUCTION_END_TO_END_TOLERANCE_PWM <= 8,
        expected="proved observation-independent integer tolerance <= 8",
        actual=PRODUCTION_END_TO_END_BOUND_DERIVATION,
    )

    primitive_tuple = tuple(jax_primitives)
    _check(
        checks,
        "jax_float32_operation_order",
        primitive_tuple == EXPECTED_JAX_PRIMITIVES,
        expected=EXPECTED_JAX_PRIMITIVES,
        actual=primitive_tuple,
    )
    forbidden_found = tuple(
        token
        for token in FORBIDDEN_PRODUCTION_QUANTIZATION_TOKENS
        if token in production_helper_source
    )
    _check(
        checks,
        "production_jax_helper_has_no_quantization",
        not forbidden_found,
        expected="no rounding, truncation, integer cast, or gradient trick",
        actual=forbidden_found,
    )

    basis_results = []
    for case in _BASIS_CASES:
        jax_values = _as_float_tuple(jax_mixer(*case["common_input"]))
        native_values = _as_float_tuple(native_mixer(*case["native_input"]))
        expected = tuple(case["expected"])
        basis_results.append(
            {
                "name": case["name"],
                "common_input": case["common_input"],
                "native_input": case["native_input"],
                "expected_m1_m4": expected,
                "jax_m1_m4": jax_values,
                "native_m1_m4": native_values,
            }
        )
        _check(
            checks,
            f"basis_{case['name']}_jax",
            jax_values == expected,
            expected=expected,
            actual=jax_values,
        )
        _check(
            checks,
            f"basis_{case['name']}_native",
            native_values == expected,
            expected=expected,
            actual=native_values,
        )

    fractional_values = _as_float_tuple(jax_mixer(0.25, 1.0, 0.0, 0.0))
    fractional_expected = (-0.25, -0.25, 0.75, 0.75)
    _check(
        checks,
        "production_jax_fractional_inputs_preserved",
        fractional_values == fractional_expected,
        expected=fractional_expected,
        actual=fractional_values,
    )

    negative_jax = _as_float_tuple(jax_mixer(0.0, 100.0, 0.0, 0.0))
    negative_native = _as_float_tuple(native_mixer(0.0, 100, 0, 0))
    high_jax = _as_float_tuple(jax_mixer(70000.0, 0.0, 0.0, 0.0))
    high_native = _as_float_tuple(native_mixer(70000.0, 0, 0, 0))
    _check(
        checks,
        "jax_boundary_pre_clipping",
        min(negative_jax) < 0.0 and max(high_jax) > 65535.0,
        expected="negative and above-65535 outputs remain visible",
        actual={"negative": negative_jax, "high": high_jax},
    )
    _check(
        checks,
        "native_boundary_pre_cap",
        min(negative_native) < 0.0 and max(high_native) > 65535.0,
        expected="negative and above-65535 outputs remain visible",
        actual={"negative": negative_native, "high": high_native},
    )

    failed = [item for item in checks if item["status"] == "FAIL"]
    return {
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "motor_order": MOTOR_ORDER,
        "basis_vectors": basis_results,
        "boundary": RAW_MIXER_BOUNDARY,
        "excluded_claims": EXCLUDED_CLAIMS,
    }


def mixer_configuration_summary(configuration_gate: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable 14-value matrix/metadata/basis sub-gate.

    The full pre-work gate also proves ranges, Float32 operation order, and
    both analytical bounds. This sub-gate preserves the original 14-value
    mixer-configuration contract: one matrix, five native boundary metadata
    values, and eight real-function basis results.
    """
    native_configuration_names = {
        "native_mixer_motor_order",
        "native_mixer_raw",
        "native_mixer_uncapped",
        "native_mixer_pre_battery",
        "native_mixer_final_motor_pwm",
    }
    selected = [
        check
        for check in configuration_gate["checks"]
        if check["name"] == "crazyflow_legacy_x_matrix"
        or check["name"] in native_configuration_names
        or check["name"].startswith("basis_")
    ]
    if len(selected) != 14:
        raise ValueError(f"expected exactly 14 mixer configuration values, got {len(selected)}")
    passed = sum(check["status"] == "PASS" for check in selected)
    return {
        "status": "PASS" if passed == len(selected) else "FAIL",
        "values_passed": passed,
        "values_total": len(selected),
        "values": selected,
    }


def aggregate_mixer_errors(
    native_raw: Sequence[Sequence[Any]], jax_raw: Sequence[Sequence[Any]], *, tolerance_pwm: int
) -> dict[str, Any]:
    """Return compact per-motor, exclusive-bucket, and worst-case statistics."""
    native = np.asarray(native_raw)
    jax_values = np.asarray(jax_raw)
    if native.shape != jax_values.shape or native.ndim != 2 or native.shape[1] != 4:
        raise ValueError(
            "native_raw and jax_raw must have the same shape (samples, 4), got "
            f"{native.shape} and {jax_values.shape}"
        )
    errors = np.abs(
        native.astype(np.float64, copy=False) - jax_values.astype(np.float64, copy=False)
    )
    finite = bool(np.all(np.isfinite(errors)))
    if errors.size:
        flat_index = int(np.argmax(errors))
        sample_index, motor_index = np.unravel_index(flat_index, errors.shape)
        sample_index = int(sample_index)
        motor_index = int(motor_index)
        max_abs_error = float(errors[sample_index, motor_index])
    else:
        sample_index = 0
        motor_index = 0
        max_abs_error = 0.0
    outside = int(np.count_nonzero(errors > tolerance_pwm))
    per_motor = {
        motor: {"max_abs_error_pwm": (float(np.max(errors[:, index])) if len(errors) else 0.0)}
        for index, motor in enumerate(MOTOR_ORDER)
    }
    worst = {
        "sample_index": sample_index,
        "motor": MOTOR_ORDER[motor_index],
        "motor_index": motor_index,
        "native_m1_m4": (native[sample_index].astype(np.int64).tolist() if len(native) else []),
        "jax_m1_m4": (
            jax_values[sample_index].astype(np.float64).tolist() if len(jax_values) else []
        ),
        "abs_errors_m1_m4": errors[sample_index].tolist() if len(errors) else [],
    }

    classes = {"exact": int(np.count_nonzero(errors == 0.0))}
    lower = 0.0
    for upper in range(1, tolerance_pwm + 1):
        classes[f"gt_{int(lower)}_le_{upper}_pwm"] = int(
            np.count_nonzero((errors > lower) & (errors <= upper))
        )
        lower = float(upper)
    classes["outside_tolerance"] = outside
    return {
        "overall_max_abs_error_pwm": max_abs_error,
        "per_motor": per_motor,
        "worst_case": worst,
        "exact_matches": classes["exact"],
        "error_classes": classes,
        "outside_tolerance": outside,
        "finite": finite,
        "status": "PASS" if finite and outside == 0 else "FAIL",
    }


def _section(
    *,
    statistics: Mapping[str, Any],
    configuration_gate: Mapping[str, Any],
    claim: str,
    derivation: Mapping[str, Any],
    tolerance_pwm: int,
    includes_productive_interface_quantization_difference: bool,
) -> dict[str, Any]:
    return {
        "boundary": RAW_MIXER_BOUNDARY,
        "claim": claim,
        "excluded_claims": EXCLUDED_CLAIMS,
        "motor_order": MOTOR_ORDER,
        "configuration_gate": dict(configuration_gate),
        "analytical_bound": derivation["strict_total_bound_pwm"],
        "acceptance_tolerance_pwm": tolerance_pwm,
        "bound_derivation": dict(derivation),
        "includes_productive_interface_quantization_difference": (
            includes_productive_interface_quantization_difference
        ),
        **dict(statistics),
        "controller_runs_reused": True,
        "additional_controller_runs": 0,
        "pass": statistics["status"] == "PASS",
    }


def build_interface_matched_section(
    statistics: Mapping[str, Any], configuration_gate: Mapping[str, Any]
) -> dict[str, Any]:
    return _section(
        statistics=statistics,
        configuration_gate=configuration_gate,
        claim=(
            "Isolated interface-matched mixer parity: both mixers consume the "
            "same Float32 thrust and the same valid int16 roll/pitch/yaw tuple, "
            "with only the documented Crazyflow sign adapter."
        ),
        derivation=INTERFACE_MATCHED_BOUND_DERIVATION,
        tolerance_pwm=INTERFACE_MATCHED_TOLERANCE_PWM,
        includes_productive_interface_quantization_difference=False,
    )


def build_production_end_to_end_section(
    statistics: Mapping[str, Any], configuration_gate: Mapping[str, Any]
) -> dict[str, Any]:
    return _section(
        statistics=statistics,
        configuration_gate=configuration_gate,
        claim=(
            "Authentic production end-to-end raw mixer parity for identical "
            "state/setpoint/time samples within the analytical bound. Native "
            "quantized controller outputs and productive unquantized JAX "
            "controller outputs are intentionally retained."
        ),
        derivation=PRODUCTION_END_TO_END_BOUND_DERIVATION,
        tolerance_pwm=PRODUCTION_END_TO_END_TOLERANCE_PWM,
        includes_productive_interface_quantization_difference=True,
    )
