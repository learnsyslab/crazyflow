"""Opt-in Mellinger controller and two-level raw-mixer comparison."""

from __future__ import annotations

import importlib.util
import inspect
import platform
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.spatial.transform import Rotation

from .parameter_gate import evaluate_parameter_gate, format_parameter_gate
from .raw_mixer_parity import (
    EXPECTED_ANALYSIS_CONTRACT,
    INTERFACE_MATCHED_TOLERANCE_PWM,
    PINNED_CONTROLLER_SHA256,
    PINNED_FIRMWARE_COMMIT,
    PINNED_MIXER_SHA256,
    PRODUCTION_END_TO_END_BOUND_DERIVATION,
    PRODUCTION_END_TO_END_TOLERANCE_PWM,
    aggregate_mixer_errors,
    analytical_bounds_are_valid,
    build_interface_matched_section,
    build_production_end_to_end_section,
    evaluate_raw_mixer_configuration,
    mixer_configuration_summary,
)

SEED = 20260714
SAMPLES = 100
JAX_BATCH_SIZE = 100
STATEFUL_STEPS = 20
CONTROLLER_FREQUENCY_HZ = 500.0
DRONE = "cf2x_L250"
COMPARISON_KD_Z = 0.4  # Pinned firmware value, applied to the test call only.
COMPARISON_MODE = "matched_test_configuration"
THRUST_ATOL_PWM = 1.0
TORQUE_TOLERANCE_LSB = 1
ROTATION_ROUNDTRIP_ATOL_RAD = 1.0e-5
FUZZER_POSITION_ABS_MAX_M = 2.0
FUZZER_VELOCITY_ABS_MAX_M_S = 1.0
FUZZER_ACCELERATION_ABS_MAX_M_S2 = 5.0
ACCEPTED_QUATERNION_NORM_MAX = 1.001


class ParityGateFailure(AssertionError):
    """A complete pre-work gate failure with zero generated/controller work."""

    def __init__(self, summary: dict[str, Any]):
        """Retain the complete structured failure beside the compact message."""
        self.summary = summary
        super().__init__("FAIL: " + format_parameter_gate(summary["parameter_gate"]))


@dataclass(frozen=True)
class ControllerModules:
    """Source-local Crazyflow functions plus the transient native extension."""

    bridge: Any
    load_params: Any
    state2attitude: Any
    state2attitude_debug: Any
    attitude2force_torque: Any
    attitude2force_torque_debug: Any
    raw_mixer: Any


def import_controller_modules(crazyflow_root: Path, bridge_build_dir: Path) -> ControllerModules:
    """Import only source-local controller modules and the transient extension."""
    crazyflow_root = crazyflow_root.resolve(strict=True)
    bridge_build_dir = bridge_build_dir.resolve(strict=True)
    from crazyflow.control.core import load_fn_params
    from crazyflow.control.mellinger import control

    source = Path(control.__file__).resolve()
    expected_source = (crazyflow_root / "crazyflow/control/mellinger/control.py").resolve()
    if source != expected_source:
        raise AssertionError(f"imported Crazyflow controller outside checkout: {source}")
    (extension,) = bridge_build_dir.glob("mellinger_fuzz_oracle*.so")
    if extension.resolve().parent != bridge_build_dir:
        raise AssertionError(f"native oracle resolves outside transient build: {extension}")
    spec = importlib.util.spec_from_file_location("mellinger_fuzz_oracle", extension)
    assert spec is not None and spec.loader is not None
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)

    def state_with_raw_thrust(*, cmd: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        # The C oracle uses yaw plus rates, Crazyflow takes quaternion plus rates.
        yaw_quaternion = Rotation.from_euler("z", np.asarray(cmd[..., 9:10])).as_quat()
        state_command = jnp.concatenate(
            (cmd[..., :9], jnp.asarray(yaw_quaternion, dtype=cmd.dtype), cmd[..., 10:]), axis=-1
        )
        convert = control.pwm2force
        captured = []

        def observe(pwm: Any, *args: Any, **kwargs: Any) -> Any:
            captured.append(pwm)
            return convert(pwm, *args, **kwargs)

        with patch.object(control, "pwm2force", observe):
            command, integral = control.state2attitude(cmd=state_command, **kwargs)
        assert len(captured) == 1, (
            "state2attitude no longer calls the raw-thrust observation seam once"
        )
        (raw_thrust,) = captured
        return command, integral, raw_thrust

    def attitude_with_raw_torque(**kwargs: Any) -> tuple[Any, Any, Any, Any]:
        mixer = control.force_torque_pwms2pwms
        captured = []

        def observe(force: Any, torque: Any, mixing_matrix: Any) -> Any:
            captured.append(torque)
            return mixer(force, torque, mixing_matrix)

        with patch.object(control, "force_torque_pwms2pwms", observe):
            force, torque, integral = control.attitude2force_torque(**kwargs)
        assert len(captured) == 1, "attitude2force_torque no longer calls the raw-mixer seam once"
        (raw_torque,) = captured
        return force, torque, integral, raw_torque

    return ControllerModules(
        bridge=bridge,
        load_params=load_fn_params,
        state2attitude=control.state2attitude,
        state2attitude_debug=state_with_raw_thrust,
        attitude2force_torque=control.attitude2force_torque,
        attitude2force_torque_debug=attitude_with_raw_torque,
        raw_mixer=control.force_torque_pwms2pwms,
    )


def _source_records() -> dict[str, dict[str, str]]:
    controller_source = "src/modules/src/controller/controller_mellinger.c"
    records: dict[str, dict[str, str]] = {}
    parameter_symbols = {
        "mass": ("cf2x_L250.core.mass", "CF_MASS"),
        "mass_thrust": ("cf2x_L250.core.mass_thrust", "controllerMellinger.massThrust"),
        "kp_xy": ("cf2x_L250.state2attitude.kp[0]", "controllerMellinger.kp_xy"),
        "kd_xy": ("cf2x_L250.state2attitude.kd[0]", "controllerMellinger.kd_xy"),
        "ki_xy": ("cf2x_L250.state2attitude.ki[0]", "controllerMellinger.ki_xy"),
        "i_range_xy": ("cf2x_L250.state2attitude.int_err_max[0]", "controllerMellinger.i_range_xy"),
        "kp_z": ("cf2x_L250.state2attitude.kp[2]", "controllerMellinger.kp_z"),
        "kd_z": ("cf2x_L250.state2attitude.kd[2]", "controllerMellinger.kd_z"),
        "ki_z": ("cf2x_L250.state2attitude.ki[2]", "controllerMellinger.ki_z"),
        "i_range_z": ("cf2x_L250.state2attitude.int_err_max[2]", "controllerMellinger.i_range_z"),
        "kR_xy": ("cf2x_L250.attitude2force_torque.kR[0]", "controllerMellinger.kR_xy"),
        "kw_xy": ("cf2x_L250.attitude2force_torque.kw[0]", "controllerMellinger.kw_xy"),
        "ki_m_xy": ("cf2x_L250.attitude2force_torque.ki_m[0]", "controllerMellinger.ki_m_xy"),
        "i_range_m_xy": (
            "cf2x_L250.attitude2force_torque.int_err_max[0]",
            "controllerMellinger.i_range_m_xy",
        ),
        "kR_z": ("cf2x_L250.attitude2force_torque.kR[2]", "controllerMellinger.kR_z"),
        "kw_z": ("cf2x_L250.attitude2force_torque.kw[2]", "controllerMellinger.kw_z"),
        "ki_m_z": ("cf2x_L250.attitude2force_torque.ki_m[2]", "controllerMellinger.ki_m_z"),
        "i_range_m_z": (
            "cf2x_L250.attitude2force_torque.int_err_max[2]",
            "controllerMellinger.i_range_m_z",
        ),
        "kd_omega_rp": (
            "cf2x_L250.attitude2force_torque.kd_omega[0]",
            "controllerMellinger.kd_omega_rp",
        ),
    }
    for name, (crazyflow_symbol, firmware_symbol) in parameter_symbols.items():
        records[name] = {
            "crazyflow_source": "crazyflow/control/mellinger/params.toml",
            "crazyflow_symbol": crazyflow_symbol,
            "firmware_source": (
                "src/platform/interface/platform_defaults_cf2.h"
                if name == "mass"
                else controller_source
            ),
            "firmware_symbol": firmware_symbol,
        }
    records.update(
        {
            "gravity_m_s2": {
                "crazyflow_source": "crazyflow/control/mellinger/params.toml",
                "crazyflow_symbol": "cf2x_L250.core.gravity_vec[2]",
                "firmware_source": "src/utils/interface/physicalConstants.h",
                "firmware_symbol": "GRAVITY_MAGNITUDE",
            },
            "controller_frequency_hz": {
                "crazyflow_source": "tests/fuzzing/mellinger/parity.py",
                "crazyflow_symbol": "CONTROLLER_FREQUENCY_HZ",
                "firmware_source": "src/modules/interface/stabilizer_types.h",
                "firmware_symbol": "ATTITUDE_RATE",
            },
            "torque_limit_pwm": {
                "crazyflow_source": "crazyflow/control/mellinger/params.toml",
                "crazyflow_symbol": "cf2x_L250.core.torque_pwm_max",
                "firmware_source": controller_source,
                "firmware_symbol": "clamp(...,-32000,32000), pinned source literal",
            },
            "yaw_kd_omega": {
                "crazyflow_source": "crazyflow/control/mellinger/params.toml",
                "crazyflow_symbol": "cf2x_L250.attitude2force_torque.kd_omega[2]",
                "firmware_source": controller_source,
                "firmware_symbol": "no yaw kd_omega term, pinned source inspection",
            },
        }
    )
    return records


def _controller_parameters(state: dict[str, Any], attitude: dict[str, Any]) -> dict[str, Any]:
    return {
        "mass": state["mass"],
        "mass_thrust": state["mass_thrust"],
        "kp_xy": state["kp"][0],
        "kd_xy": state["kd"][0],
        "ki_xy": state["ki"][0],
        "i_range_xy": state["int_err_max"][0],
        "kp_z": state["kp"][2],
        "kd_z": state["kd"][2],
        "ki_z": state["ki"][2],
        "i_range_z": state["int_err_max"][2],
        "kR_xy": attitude["kR"][0],
        "kw_xy": attitude["kw"][0],
        "ki_m_xy": attitude["ki_m"][0],
        "i_range_m_xy": attitude["int_err_max"][0],
        "kR_z": attitude["kR"][2],
        "kw_z": attitude["kw"][2],
        "ki_m_z": attitude["ki_m"][2],
        "i_range_m_z": attitude["int_err_max"][2],
        "kd_omega_rp": attitude["kd_omega"][0],
    }


def _configuration_gate(
    modules: ControllerModules,
    firmware_defaults: dict[str, Any],
    state: dict[str, Any],
    attitude: dict[str, Any],
    *,
    comparison: bool = False,
) -> dict[str, Any]:
    """Compare stock inputs or the exact dictionaries passed to the controller."""
    torque_limits = np.asarray(attitude["torque_pwm_max"])
    uniform_torque_limit = (
        torque_limits[0]
        if all(value == torque_limits[0] for value in torque_limits)
        else tuple(torque_limits)
    )
    crazyflow_invariants = {
        "gravity_m_s2": -state["gravity_vec"][2],
        "controller_frequency_hz": CONTROLLER_FREQUENCY_HZ,
        "torque_limit_pwm": uniform_torque_limit,
        "yaw_kd_omega": attitude["kd_omega"][2],
    }
    firmware_invariants = {
        "gravity_m_s2": modules.bridge.gravity_m_s2,
        "controller_frequency_hz": getattr(modules.bridge, "attitude_rate_hz", None),
        "torque_limit_pwm": 32000,
        "yaw_kd_omega": 0.0,
    }
    records = _source_records()
    if comparison:
        records["kd_z"].update(
            crazyflow_source="tests/fuzzing/mellinger/parity.py",
            crazyflow_symbol="COMPARISON_KD_Z in the actual test input",
        )
    return evaluate_parameter_gate(
        _controller_parameters(state, attitude),
        firmware_defaults,
        crazyflow_invariants,
        firmware_invariants,
        source_records=records,
    )


def build_parameter_gate_failure(
    parameter_gate: dict[str, Any], *, parameter_mode: str = "independent_defaults"
) -> dict[str, Any]:
    """Return the mandatory machine-readable zero-work failure record."""
    return {
        "status": "FAIL",
        "failure_stage": "parameter_gate",
        "parameter_mode": parameter_mode,
        "parameters_transferred": False,
        "parameter_transfer_direction": "none",
        "seed": SEED,
        "requested_samples": SAMPLES,
        "samples_generated": 0,
        "controller_runs": 0,
        "native_controller_runs": 0,
        "jax_controller_runs": 0,
        "parameter_gate": parameter_gate,
    }


def _raw_analysis_contract(
    state_params: dict[str, Any], attitude_params: dict[str, Any]
) -> dict[str, Any]:
    return {
        "fuzzer_position_state_abs_max_m": FUZZER_POSITION_ABS_MAX_M,
        "fuzzer_position_setpoint_abs_max_m": FUZZER_POSITION_ABS_MAX_M,
        "fuzzer_velocity_state_abs_max_m_s": FUZZER_VELOCITY_ABS_MAX_M_S,
        "fuzzer_velocity_setpoint_abs_max_m_s": FUZZER_VELOCITY_ABS_MAX_M_S,
        "fuzzer_acceleration_setpoint_abs_max_m_s2": (FUZZER_ACCELERATION_ABS_MAX_M_S2),
        "accepted_quaternion_norm_max": ACCEPTED_QUATERNION_NORM_MAX,
        "mass_kg": float(np.float32(state_params["mass"])),
        "mass_thrust_pwm_per_n": float(np.float32(state_params["mass_thrust"])),
        "gravity_vec_m_s2": tuple(
            float(value) for value in np.asarray(state_params["gravity_vec"], dtype=np.float32)
        ),
        "kp": tuple(float(value) for value in np.asarray(state_params["kp"], dtype=np.float32)),
        "kd": tuple(float(value) for value in np.asarray(state_params["kd"], dtype=np.float32)),
        "ki": tuple(float(value) for value in np.asarray(state_params["ki"], dtype=np.float32)),
        "position_integral_abs_max": tuple(
            float(value) for value in np.asarray(state_params["int_err_max"], dtype=np.float32)
        ),
        "torque_clip_abs_max_pwm": tuple(
            float(value)
            for value in np.asarray(attitude_params["torque_pwm_max"], dtype=np.float32)
        ),
        "pre_mixer_thrust_atol_pwm": THRUST_ATOL_PWM,
        "pre_mixer_torque_tolerance_lsb": TORQUE_TOLERANCE_LSB,
        "controller_thrust_abs_bound_pwm": EXPECTED_ANALYSIS_CONTRACT[
            "controller_thrust_abs_bound_pwm"
        ],
        "raw_intermediate_abs_bound_pwm": EXPECTED_ANALYSIS_CONTRACT[
            "raw_intermediate_abs_bound_pwm"
        ],
        "float32_magnitude_ceiling_pwm": EXPECTED_ANALYSIS_CONTRACT[
            "float32_magnitude_ceiling_pwm"
        ],
        "jax_x64_enabled": bool(jax.config.jax_enable_x64),
    }


def _bridge_metadata(bridge: Any) -> dict[str, Any]:
    return {
        name: getattr(bridge, name, None)
        for name in (
            "firmware_commit",
            "controller_sha256",
            "mixer_sha256",
            "mixer_motor_order",
            "mixer_raw",
            "mixer_uncapped",
            "mixer_pre_battery",
            "mixer_final_motor_pwm",
            "mixer_thrust_pwm_float32_safe_range",
        )
    }


def evaluate_raw_gate(
    modules: ControllerModules, state_params: dict[str, Any], attitude_params: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prove the 17-value mixer configuration and both bounds before RNG."""
    mixing_matrix = attitude_params["mixing_matrix"]

    def jax_mixer(thrust: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
        return np.asarray(
            modules.raw_mixer(
                jnp.asarray(thrust, dtype=jnp.float32),
                jnp.asarray([roll, pitch, yaw], dtype=jnp.float32),
                mixing_matrix,
            )
        )

    jaxpr = jax.make_jaxpr(modules.raw_mixer)(
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([2.0, 3.0, 4.0], dtype=jnp.float32),
        jnp.asarray(mixing_matrix, dtype=jnp.float32),
    )
    gate = evaluate_raw_mixer_configuration(
        jax_mixer,
        modules.bridge.mix_legacy_raw,
        np.asarray(mixing_matrix),
        _bridge_metadata(modules.bridge),
        analysis_contract=_raw_analysis_contract(state_params, attitude_params),
        production_helper_source=inspect.getsource(modules.raw_mixer),
        jax_primitives=tuple(equation.primitive.name for equation in jaxpr.jaxpr.eqns),
    )
    summary = mixer_configuration_summary(gate)
    return gate, summary


def _permanent_four_pwm_counterexample(
    modules: ControllerModules, attitude_params: dict[str, Any]
) -> dict[str, Any]:
    record = PRODUCTION_END_TO_END_BOUND_DERIVATION[
        "permanent_counterexample_to_old_four_pwm_claim"
    ]
    native_input = record["native_controller_output_firmware_signs"]
    jax_input = record["jax_controller_output_crazyflow_signs"]
    native = np.asarray(
        modules.bridge.mix_legacy_raw(
            float(native_input[0]), int(native_input[1]), int(native_input[2]), int(native_input[3])
        ),
        dtype=np.int32,
    )
    jax_raw = np.asarray(
        modules.raw_mixer(
            jnp.asarray(jax_input[0], dtype=jnp.float32),
            jnp.asarray(jax_input[1:], dtype=jnp.float32),
            attitude_params["mixing_matrix"],
        ),
        dtype=np.float32,
    )
    errors = np.abs(native.astype(np.float64) - jax_raw.astype(np.float64))
    maximum = float(np.max(errors))
    if native.tolist() != record["native_raw_m1_m4"]:
        raise AssertionError("permanent counterexample native result changed")
    if not (maximum > 4.0 and maximum <= PRODUCTION_END_TO_END_TOLERANCE_PWM):
        raise AssertionError("permanent counterexample no longer disproves 4 PWM within 8 PWM")
    return {
        "status": "PASS",
        "old_universal_four_pwm_claim_disproved": True,
        "overall_max_abs_error_pwm": maximum,
        "native_m1_m4": native.tolist(),
        "jax_m1_m4": jax_raw.astype(np.float64).tolist(),
        "abs_errors_m1_m4": errors.tolist(),
    }


def _generate_random_samples(
    state_params: dict[str, Any], attitude_params: dict[str, Any]
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(SEED)
    position = rng.uniform(
        -FUZZER_POSITION_ABS_MAX_M, FUZZER_POSITION_ABS_MAX_M, size=(SAMPLES, 3)
    ).astype(np.float32)
    velocity = rng.uniform(
        -FUZZER_VELOCITY_ABS_MAX_M_S, FUZZER_VELOCITY_ABS_MAX_M_S, size=(SAMPLES, 3)
    ).astype(np.float32)
    quaternion = rng.normal(0.0, 1.0, size=(SAMPLES, 4)).astype(np.float32)
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    angular_velocity = rng.uniform(-20.0, 20.0, size=(SAMPLES, 3)).astype(np.float32)
    previous_angular_velocity = rng.uniform(-20.0, 20.0, size=(SAMPLES, 3)).astype(np.float32)
    setpoint_rates = np.zeros((SAMPLES, 3), dtype=np.float32)
    previous_setpoint_omega = np.zeros((SAMPLES, 2), dtype=np.float32)
    setpoint_position = rng.uniform(
        -FUZZER_POSITION_ABS_MAX_M, FUZZER_POSITION_ABS_MAX_M, size=(SAMPLES, 3)
    ).astype(np.float32)
    setpoint_velocity = rng.uniform(
        -FUZZER_VELOCITY_ABS_MAX_M_S, FUZZER_VELOCITY_ABS_MAX_M_S, size=(SAMPLES, 3)
    ).astype(np.float32)
    setpoint_acceleration = rng.uniform(
        -FUZZER_ACCELERATION_ABS_MAX_M_S2, FUZZER_ACCELERATION_ABS_MAX_M_S2, size=(SAMPLES, 3)
    ).astype(np.float32)
    setpoint_yaw = rng.uniform(-np.pi, np.pi, size=(SAMPLES, 1)).astype(np.float32)
    setpoint = np.concatenate(
        (setpoint_position, setpoint_velocity, setpoint_acceleration, setpoint_yaw, setpoint_rates),
        axis=1,
    )
    position_limits = np.asarray(state_params["int_err_max"], dtype=np.float32)
    attitude_limits = np.asarray(attitude_params["int_err_max"], dtype=np.float32)
    position_integral = rng.uniform(-position_limits, position_limits, size=(SAMPLES, 3)).astype(
        np.float32
    )
    attitude_integral = rng.uniform(-attitude_limits, attitude_limits, size=(SAMPLES, 3)).astype(
        np.float32
    )
    return (
        position,
        velocity,
        quaternion,
        angular_velocity,
        setpoint,
        position_integral,
        attitude_integral,
        previous_angular_velocity,
        previous_setpoint_omega,
    )


def _set_firmware_memory(
    controller: Any,
    position_integral: np.ndarray,
    attitude_integral: np.ndarray,
    previous_angular_velocity: np.ndarray,
    previous_setpoint_omega: np.ndarray,
) -> None:
    controller.set_internal_state(
        float(position_integral[0]),
        float(position_integral[1]),
        float(position_integral[2]),
        float(attitude_integral[0]),
        float(-attitude_integral[1]),
        float(attitude_integral[2]),
        float(previous_angular_velocity[0]),
        float(-previous_angular_velocity[1]),
        float(previous_setpoint_omega[0]),
        float(previous_setpoint_omega[1]),
    )


def _quantize_torque(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), -32000.0, 32000.0)
    return np.trunc(clipped).astype(np.int16).astype(np.int32)


def _analyse_rotation_roundtrip(
    position: np.ndarray,
    velocity: np.ndarray,
    setpoint: np.ndarray,
    position_integral: np.ndarray,
    command_rpyt: np.ndarray,
    state_params: dict[str, Any],
) -> dict[str, Any]:
    dt = np.float32(1.0 / CONTROLLER_FREQUENCY_HZ)
    position_error = setpoint[:, 0:3] - position
    velocity_error = setpoint[:, 3:6] - velocity
    integral_limit = np.asarray(state_params["int_err_max"], dtype=np.float32)
    current_integral = np.clip(
        position_integral + position_error * dt, -integral_limit, integral_limit
    )
    target_thrust = (
        np.float32(state_params["mass"])
        * (setpoint[:, 6:9] - np.asarray(state_params["gravity_vec"], dtype=np.float32))
        + np.asarray(state_params["kp"], dtype=np.float32) * position_error
        + np.asarray(state_params["kd"], dtype=np.float32) * velocity_error
        + np.asarray(state_params["ki"], dtype=np.float32) * current_integral
    )
    target_norm = np.linalg.norm(target_thrust, axis=1)
    valid_target = target_norm > 1.0e-7
    z_des = np.zeros_like(target_thrust)
    z_des[valid_target] = target_thrust[valid_target] / target_norm[valid_target, None]
    yaw = setpoint[:, 9]
    x_c = np.column_stack((np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)))
    y_unnormalized = np.cross(z_des, x_c)
    cross_norm = np.linalg.norm(y_unnormalized, axis=1)
    valid = valid_target & (cross_norm > 1.0e-7)
    if not np.any(valid):
        return {
            "invalid_geometry_samples": int(len(position)),
            "max_rotation_roundtrip_error_rad": None,
        }
    y_des = y_unnormalized[valid] / cross_norm[valid, None]
    x_des = np.cross(y_des, z_des[valid])
    desired_matrix = np.stack((x_des, y_des, z_des[valid]), axis=-1)
    roundtrip_matrix = Rotation.from_euler("xyz", command_rpyt[valid, 0:3]).as_matrix()
    relative = np.swapaxes(desired_matrix, -1, -2) @ roundtrip_matrix
    angle_error = Rotation.from_matrix(relative).magnitude()
    return {
        "invalid_geometry_samples": int(np.count_nonzero(~valid)),
        "max_rotation_roundtrip_error_rad": float(np.max(angle_error)),
        "mean_rotation_roundtrip_error_rad": float(np.mean(angle_error)),
    }


def _attach_worst_sample(
    statistics: dict[str, Any],
    position: np.ndarray,
    velocity: np.ndarray,
    quaternion: np.ndarray,
    angular_velocity: np.ndarray,
    setpoint: np.ndarray,
    native_controller_output: np.ndarray,
    jax_controller_output: np.ndarray,
) -> None:
    index = statistics["worst_case"]["sample_index"]
    statistics["worst_case"]["sample"] = {
        "state": np.concatenate(
            (position[index], velocity[index], quaternion[index], angular_velocity[index])
        )
        .astype(np.float32)
        .tolist(),
        "setpoint": setpoint[index].astype(np.float32).tolist(),
        "native_controller_output_firmware_signs": (
            native_controller_output[index].astype(np.float32).tolist()
        ),
        "jax_controller_output_crazyflow_signs": (
            jax_controller_output[index].astype(np.float32).tolist()
        ),
    }


def _run_independent_samples(
    modules: ControllerModules,
    state_params: dict[str, Any],
    attitude_params: dict[str, Any],
    raw_gate: dict[str, Any],
) -> dict[str, Any]:
    (
        position,
        velocity,
        quaternion,
        angular_velocity,
        setpoint,
        position_integral,
        attitude_integral,
        previous_angular_velocity,
        previous_setpoint_omega,
    ) = _generate_random_samples(state_params, attitude_params)

    command_rpyt, _, jax_thrust = modules.state2attitude_debug(
        pos=jnp.asarray(position),
        quat=jnp.asarray(quaternion),
        vel=jnp.asarray(velocity),
        cmd=jnp.asarray(setpoint),
        pos_err_i=jnp.asarray(position_integral),
        ctrl_freq=CONTROLLER_FREQUENCY_HZ,
        **state_params,
    )
    _, _, _, jax_torque = modules.attitude2force_torque_debug(
        quat=jnp.asarray(quaternion),
        ang_vel=jnp.asarray(angular_velocity),
        cmd=command_rpyt,
        prev_ang_vel=jnp.asarray(previous_angular_velocity),
        r_int_error=jnp.asarray(attitude_integral),
        ctrl_freq=CONTROLLER_FREQUENCY_HZ,
        **attitude_params,
    )
    jax_thrust = np.asarray(jax_thrust, dtype=np.float32)
    jax_torque = np.asarray(jax_torque, dtype=np.float32)
    command_rpyt = np.asarray(command_rpyt, dtype=np.float32)
    production_jax_raw = np.asarray(
        modules.raw_mixer(
            jnp.asarray(jax_thrust), jnp.asarray(jax_torque), attitude_params["mixing_matrix"]
        ),
        dtype=np.float32,
    )

    native_controller = modules.bridge.MellingerController()
    native_output = np.empty((SAMPLES, 4), dtype=np.float32)
    native_raw = np.empty((SAMPLES, 4), dtype=np.int32)
    for index in range(SAMPLES):
        _set_firmware_memory(
            native_controller,
            position_integral[index],
            attitude_integral[index],
            previous_angular_velocity[index],
            previous_setpoint_omega[index],
        )
        state = np.concatenate(
            (position[index], velocity[index], quaternion[index], angular_velocity[index])
        )
        native_output[index] = native_controller.compute_control(
            state.tolist(), setpoint[index].tolist()
        )
        native_raw[index] = modules.bridge.mix_legacy_raw(
            float(native_output[index, 0]),
            int(native_output[index, 1]),
            int(native_output[index, 2]),
            int(native_output[index, 3]),
        )

    native_torque_common = np.column_stack(
        (native_output[:, 1], -native_output[:, 2], -native_output[:, 3])
    )
    jax_quantized = _quantize_torque(jax_torque)
    native_torque_int = native_torque_common.astype(np.int32)
    thrust_error = np.abs(native_output[:, 0] - jax_thrust)
    torque_error = np.abs(native_torque_int - jax_quantized)
    gate_mismatches = int(np.count_nonzero((native_output[:, 0] > 0) != (jax_thrust > 0)))
    rotation = _analyse_rotation_roundtrip(
        position, velocity, setpoint, position_integral, command_rpyt, state_params
    )
    if not np.all(np.isfinite(thrust_error)):
        raise AssertionError("non-finite pre-mixer error")
    if float(np.max(thrust_error)) > THRUST_ATOL_PWM:
        raise AssertionError("pre-mixer thrust tolerance exceeded")
    if int(np.max(torque_error)) > TORQUE_TOLERANCE_LSB:
        raise AssertionError("pre-mixer torque tolerance exceeded")
    if gate_mismatches:
        raise AssertionError("pre-mixer thrust gate mismatch")
    if rotation["invalid_geometry_samples"]:
        raise AssertionError("invalid desired-rotation geometry")
    if (
        rotation["max_rotation_roundtrip_error_rad"] is None
        or rotation["max_rotation_roundtrip_error_rad"] > ROTATION_ROUNDTRIP_ATOL_RAD
    ):
        raise AssertionError("rotation round-trip tolerance exceeded")

    interface_jax_raw = np.asarray(
        modules.raw_mixer(
            jnp.asarray(native_output[:, 0], dtype=jnp.float32),
            jnp.asarray(native_torque_common, dtype=jnp.float32),
            attitude_params["mixing_matrix"],
        ),
        dtype=np.float32,
    )
    interface_statistics = aggregate_mixer_errors(
        native_raw, interface_jax_raw, tolerance_pwm=INTERFACE_MATCHED_TOLERANCE_PWM
    )
    production_statistics = aggregate_mixer_errors(
        native_raw, production_jax_raw, tolerance_pwm=PRODUCTION_END_TO_END_TOLERANCE_PWM
    )
    jax_controller_output = np.column_stack((jax_thrust, jax_torque))
    _attach_worst_sample(
        interface_statistics,
        position,
        velocity,
        quaternion,
        angular_velocity,
        setpoint,
        native_output,
        jax_controller_output,
    )
    _attach_worst_sample(
        production_statistics,
        position,
        velocity,
        quaternion,
        angular_velocity,
        setpoint,
        native_output,
        jax_controller_output,
    )
    interface = build_interface_matched_section(interface_statistics, raw_gate)
    production = build_production_end_to_end_section(production_statistics, raw_gate)
    if not interface["pass"] or not production["pass"]:
        raise AssertionError(
            {
                "status": "FAIL",
                "failure_stage": "raw_mixer_comparison",
                "interface_matched_mixer": interface,
                "production_end_to_end_mixer": production,
            }
        )
    return {
        "status": "PASS",
        "seed": SEED,
        "samples": SAMPLES,
        "samples_generated": SAMPLES,
        "jax_batch_size": JAX_BATCH_SIZE,
        "parameter_mode": COMPARISON_MODE,
        "parameters_transferred": False,
        "native_controller_runs": SAMPLES,
        "jax_controller_runs": SAMPLES,
        "native_controller_runs_per_sample": 1,
        "jax_controller_runs_per_sample": 1,
        "pre_mixer": {
            "max_thrust_error_pwm": float(np.max(thrust_error)),
            "max_torque_error_lsb": int(np.max(torque_error)),
            "positive_thrust_samples": int(np.count_nonzero(native_output[:, 0] > 0)),
            "thrust_gate_mismatches": gate_mismatches,
            "rotation_roundtrip": rotation,
        },
        "interface_matched_mixer": interface,
        "production_end_to_end_mixer": production,
    }


def _sequence_sample(index: int, gate_index: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.float32(index / CONTROLLER_FREQUENCY_HZ)
    position = np.array(
        [0.08 * np.sin(1.3 * t), -0.06 * np.cos(0.9 * t), 0.03 * np.sin(0.7 * t)], dtype=np.float32
    )
    velocity = np.array(
        [0.104 * np.cos(1.3 * t), 0.054 * np.sin(0.9 * t), 0.021 * np.cos(0.7 * t)],
        dtype=np.float32,
    )
    euler = np.array(
        [0.08 * np.sin(0.8 * t), -0.06 * np.cos(0.6 * t), 0.1 * np.sin(0.4 * t)], dtype=np.float32
    )
    quaternion = Rotation.from_euler("xyz", euler).as_quat().astype(np.float32)
    angular_velocity = np.array(
        [0.064 * np.cos(0.8 * t), 0.036 * np.sin(0.6 * t), 0.04 * np.cos(0.4 * t)], dtype=np.float32
    )
    state = np.concatenate((position, velocity, quaternion, angular_velocity))
    setpoint = np.zeros(13, dtype=np.float32)
    setpoint[:3] = [0.15, -0.1, 0.25]
    setpoint[3:6] = [0.02, -0.01, 0.0]
    setpoint[6:9] = [0.1 * np.sin(t), -0.08 * np.cos(t), 0.05 * np.sin(0.5 * t)]
    setpoint[9] = 0.2 * np.sin(0.3 * t)
    if index == gate_index:
        setpoint[8] = -25.0
    return state, setpoint


def _run_stateful(
    modules: ControllerModules,
    state_params: dict[str, Any],
    attitude_params: dict[str, Any],
    raw_gate: dict[str, Any],
) -> dict[str, Any]:
    native = modules.bridge.MellingerController()
    native.reset()
    position_integral = jnp.zeros(3, dtype=jnp.float32)
    attitude_integral = jnp.zeros(3, dtype=jnp.float32)
    previous_angular_velocity = jnp.zeros(3, dtype=jnp.float32)
    gate_index = STATEFUL_STEPS // 2
    thrust_errors: list[float] = []
    torque_errors: list[int] = []
    internal_state_errors: list[float] = []
    native_raw_values: list[tuple[int, int, int, int]] = []
    interface_jax_raw_values: list[np.ndarray] = []
    production_jax_raw_values: list[np.ndarray] = []
    native_outputs: list[tuple[float, int, int, int]] = []
    jax_outputs: list[tuple[np.ndarray, np.ndarray]] = []
    first_native_output = None
    reset_observation = {}

    for index in range(STATEFUL_STEPS):
        state, setpoint = _sequence_sample(index, gate_index)
        command, position_integral, thrust_pwm = modules.state2attitude_debug(
            pos=jnp.asarray(state[:3]),
            quat=jnp.asarray(state[6:10]),
            vel=jnp.asarray(state[3:6]),
            cmd=jnp.asarray(setpoint),
            pos_err_i=position_integral,
            ctrl_freq=CONTROLLER_FREQUENCY_HZ,
            **state_params,
        )
        _, _, attitude_integral, torque_pwm = modules.attitude2force_torque_debug(
            quat=jnp.asarray(state[6:10]),
            ang_vel=jnp.asarray(state[10:13]),
            cmd=command,
            prev_ang_vel=previous_angular_velocity,
            r_int_error=attitude_integral,
            ctrl_freq=CONTROLLER_FREQUENCY_HZ,
            **attitude_params,
        )
        previous_angular_velocity = jnp.asarray(state[10:13])
        native_output = native.compute_control(state.tolist(), setpoint.tolist())
        if first_native_output is None:
            first_native_output = native_output
        native_common = np.asarray(
            [native_output[1], -native_output[2], -native_output[3]], dtype=np.int32
        )
        jax_quantized = _quantize_torque(np.asarray(torque_pwm))
        thrust_errors.append(abs(float(native_output[0]) - float(thrust_pwm)))
        torque_errors.append(int(np.max(np.abs(native_common - jax_quantized))))
        native_outputs.append(native_output)
        jax_outputs.append(
            (np.asarray(thrust_pwm, dtype=np.float32), np.asarray(torque_pwm, dtype=np.float32))
        )
        native_raw_values.append(
            modules.bridge.mix_legacy_raw(
                float(native_output[0]),
                int(native_output[1]),
                int(native_output[2]),
                int(native_output[3]),
            )
        )
        interface_jax_raw_values.append(
            np.asarray(
                modules.raw_mixer(
                    jnp.asarray(native_output[0], dtype=jnp.float32),
                    jnp.asarray(native_common, dtype=jnp.float32),
                    attitude_params["mixing_matrix"],
                ),
                dtype=np.float32,
            )
        )
        production_jax_raw_values.append(
            np.asarray(
                modules.raw_mixer(thrust_pwm, torque_pwm, attitude_params["mixing_matrix"]),
                dtype=np.float32,
            )
        )

        internal = dict(native.get_internal_state())
        expected_internal = np.asarray(
            [
                *np.asarray(position_integral),
                float(attitude_integral[0]),
                -float(attitude_integral[1]),
                float(attitude_integral[2]),
            ],
            dtype=np.float32,
        )
        actual_internal = np.asarray(
            [
                internal["i_error_x"],
                internal["i_error_y"],
                internal["i_error_z"],
                internal["i_error_m_x"],
                internal["i_error_m_y"],
                internal["i_error_m_z"],
            ],
            dtype=np.float32,
        )
        internal_state_errors.append(float(np.max(np.abs(actual_internal - expected_internal))))
        if index == gate_index:
            if native_output[0] > 0 or any(native_output[1:]):
                raise AssertionError("stateful non-positive-thrust gate did not zero torque")
            if np.any(actual_internal != 0):
                raise AssertionError("native non-positive-thrust gate did not reset state")
            reset_observation = {
                "step": index,
                "firmware_integrators": actual_internal.tolist(),
                "crazyflow_integrators": expected_internal.tolist(),
                "same_integrator_reset": bool(np.array_equal(actual_internal, expected_internal)),
                "firmware_previous_gyro_xy_rad_s": [
                    float(internal["prev_omega_roll"]),
                    -float(internal["prev_omega_pitch"]),
                ],
                "crazyflow_previous_gyro_xy_rad_s": (
                    np.asarray(previous_angular_velocity)[:2].tolist()
                ),
            }

    native.reset()
    replay_state, replay_setpoint = _sequence_sample(0, gate_index)
    reset_replay_output = native.compute_control(replay_state.tolist(), replay_setpoint.tolist())
    reset_replay_equal = reset_replay_output == first_native_output

    same_transition = reset_observation["same_integrator_reset"]
    # At the gate both controllers gate their torque. Subsequent commands
    # start from different integrator states when only the firmware resets.
    compared_steps = STATEFUL_STEPS if same_transition else gate_index + 1
    compared_integrals = STATEFUL_STEPS if same_transition else gate_index
    interface_statistics = aggregate_mixer_errors(
        native_raw_values, interface_jax_raw_values, tolerance_pwm=INTERFACE_MATCHED_TOLERANCE_PWM
    )
    production_statistics = aggregate_mixer_errors(
        native_raw_values[:compared_steps],
        production_jax_raw_values[:compared_steps],
        tolerance_pwm=PRODUCTION_END_TO_END_TOLERANCE_PWM,
    )

    def attach_stateful_sample(statistics: dict[str, Any]) -> None:
        index = statistics["worst_case"]["sample_index"]
        state, setpoint = _sequence_sample(index, gate_index)
        statistics["worst_case"]["sample"] = {
            "state": state.tolist(),
            "setpoint": setpoint.tolist(),
            "native_controller_output_firmware_signs": list(native_outputs[index]),
            "jax_controller_output_crazyflow_signs": [
                float(jax_outputs[index][0]),
                *jax_outputs[index][1].tolist(),
            ],
        }

    attach_stateful_sample(interface_statistics)
    attach_stateful_sample(production_statistics)
    native_replay = np.asarray(
        [modules.bridge.mix_legacy_raw(*values) for values in native_outputs], dtype=np.int32
    )
    interface_replay = np.asarray(
        [
            modules.raw_mixer(
                jnp.asarray(values[0], dtype=jnp.float32),
                jnp.asarray([values[1], -values[2], -values[3]], dtype=jnp.float32),
                attitude_params["mixing_matrix"],
            )
            for values in native_outputs
        ],
        dtype=np.float32,
    )
    production_replay = np.asarray(
        [
            modules.raw_mixer(
                jnp.asarray(thrust, dtype=jnp.float32),
                jnp.asarray(torque, dtype=jnp.float32),
                attitude_params["mixing_matrix"],
            )
            for thrust, torque in jax_outputs
        ],
        dtype=np.float32,
    )
    raw_replay_equal = {
        "native_shared": bool(np.array_equal(native_replay, np.asarray(native_raw_values))),
        "interface_matched_mixer": bool(
            np.array_equal(interface_replay, np.asarray(interface_jax_raw_values))
        ),
        "production_end_to_end_mixer": bool(
            np.array_equal(production_replay, np.asarray(production_jax_raw_values))
        ),
    }
    interface = build_interface_matched_section(interface_statistics, raw_gate)
    production = build_production_end_to_end_section(production_statistics, raw_gate)
    summary = {
        "status": "PASS" if same_transition else "RESET_DIVERGENCE",
        "steps": STATEFUL_STEPS,
        "gate_step": gate_index,
        "comparable_output_steps": compared_steps,
        "comparable_integrator_steps": compared_integrals,
        "reset_observation": reset_observation,
        "post_reset": {
            "max_thrust_error_pwm": max(thrust_errors[compared_steps:], default=None),
            "max_torque_error_lsb": max(torque_errors[compared_steps:], default=None),
        },
        "parameter_mode": COMPARISON_MODE,
        "parameters_transferred": False,
        "native_controller_runs_primary": STATEFUL_STEPS,
        "native_controller_replay_runs": 1,
        "native_controller_runs_total": STATEFUL_STEPS + 1,
        "jax_controller_runs_primary": STATEFUL_STEPS,
        "jax_controller_replay_runs": 0,
        "jax_controller_runs_total": STATEFUL_STEPS,
        "native_controller_runs_per_step": 1,
        "jax_controller_runs_per_step": 1,
        "max_thrust_error_pwm": max(thrust_errors[:compared_steps]),
        "max_torque_error_lsb": max(torque_errors[:compared_steps]),
        "max_internal_state_error": max(internal_state_errors[:compared_integrals]),
        "reset_replay_equal": reset_replay_equal,
        "raw_mixer_replay_equal": raw_replay_equal,
        "interface_matched_mixer": interface,
        "production_end_to_end_mixer": production,
    }
    if summary["max_thrust_error_pwm"] > THRUST_ATOL_PWM:
        raise AssertionError(summary)
    if summary["max_torque_error_lsb"] > TORQUE_TOLERANCE_LSB:
        raise AssertionError(summary)
    if summary["max_internal_state_error"] > 2.0e-5:
        raise AssertionError(summary)
    if (
        not reset_replay_equal
        or not all(raw_replay_equal.values())
        or not interface["pass"]
        or not production["pass"]
    ):
        raise AssertionError(summary)
    return summary


def runtime_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "jax": jax.__version__,
    }


def run_parity_smoke(crazyflow_root: Path, bridge_build_dir: Path) -> dict[str, Any]:
    """Measure stock differences and compare outputs at matching gains."""
    machine = platform.machine().lower()
    if not sys.platform.startswith("linux") or machine not in {"x86_64", "amd64"}:
        raise AssertionError(
            f"mellinger-fuzzing is Linux x86-64 only, got {sys.platform}/{machine}"
        )
    if not analytical_bounds_are_valid():
        raise AssertionError("analytical raw bounds are not valid before RNG/controller work")
    modules = import_controller_modules(crazyflow_root, bridge_build_dir)
    bridge = modules.bridge
    if bridge.firmware_commit != PINNED_FIRMWARE_COMMIT:
        raise AssertionError("transient oracle firmware commit mismatch")
    if bridge.controller_sha256 != PINNED_CONTROLLER_SHA256:
        raise AssertionError("transient oracle controller hash mismatch")
    if bridge.mixer_sha256 != PINNED_MIXER_SHA256:
        raise AssertionError("transient oracle mixer hash mismatch")
    if not np.isclose(float(bridge.platform_mass_kg), float(np.float32(0.029)), rtol=0.0, atol=0.0):
        raise AssertionError("transient oracle CF_MASS mismatch")

    state_params = modules.load_params(modules.state2attitude, DRONE, xp=jnp)
    attitude_params = modules.load_params(modules.attitude2force_torque, DRONE, xp=jnp)
    native_defaults = dict(bridge.MellingerController().get_params())
    params_path = Path(inspect.getfile(modules.state2attitude)).resolve().with_name("params.toml")
    with params_path.open("rb") as stream:
        profile = tomllib.load(stream)[DRONE]
    stock_gate = _configuration_gate(
        modules,
        native_defaults,
        profile["core"] | profile["state2attitude"],
        profile["core"] | profile["attitude2force_torque"],
    )
    state_params = {
        **state_params,
        "kd": jnp.asarray(state_params["kd"]).at[2].set(COMPARISON_KD_Z),
    }
    parameter_gate = _configuration_gate(
        modules, native_defaults, state_params, attitude_params, comparison=True
    )
    if parameter_gate["status"] != "PASS":
        raise ParityGateFailure(
            build_parameter_gate_failure(parameter_gate, parameter_mode=COMPARISON_MODE)
            | {"stock_parameter_gate": stock_gate}
        )

    raw_gate, mixer_values = evaluate_raw_gate(modules, state_params, attitude_params)
    if raw_gate["status"] != "PASS" or mixer_values["status"] != "PASS":
        raise AssertionError(
            {
                "status": "FAIL",
                "failure_stage": "raw_mixer_configuration_gate",
                "parameter_mode": COMPARISON_MODE,
                "parameters_transferred": False,
                "samples_generated": 0,
                "controller_runs": 0,
                "parameter_gate": parameter_gate,
                "mixer_configuration_values": mixer_values,
                "raw_mixer_configuration_gate": raw_gate,
            }
        )

    counterexample = _permanent_four_pwm_counterexample(modules, attitude_params)
    independent = _run_independent_samples(modules, state_params, attitude_params, raw_gate)
    stateful = _run_stateful(modules, state_params, attitude_params, raw_gate)
    return {
        "status": (
            "PASS_CHECKED_SCOPE_WITH_DIFFERENCES"
            if stock_gate["status"] != "PASS" or stateful["status"] != "PASS"
            else "PASS_CHECKED_SCOPE"
        ),
        "scope": "on-demand Linux x86-64 Mellinger and raw-mixer smoke",
        "firmware_commit": bridge.firmware_commit,
        "controller_sha256": bridge.controller_sha256,
        "mixer_sha256": bridge.mixer_sha256,
        "cf_mass": float(bridge.platform_mass_kg),
        "firmware_c_compiler": bridge.c_compiler,
        "parameter_mode": COMPARISON_MODE,
        "comparison_kd_z": COMPARISON_KD_Z,
        "parameters_transferred": False,
        "stock_parameter_gate": stock_gate,
        "parameter_gate": parameter_gate,
        "mixer_configuration_values": mixer_values,
        "raw_mixer_configuration_gate": raw_gate,
        "analytical_bounds": {
            "interface_matched_mixer": "<2.25390625 PWM",
            "interface_matched_acceptance_pwm": 3,
            "production_end_to_end_mixer": "<7.25390625 PWM",
            "production_end_to_end_acceptance_pwm": 8,
            "observations_used": False,
        },
        "permanent_four_pwm_counterexample": counterexample,
        "independent_samples": independent,
        "stateful": stateful,
        "runtime": runtime_metadata(),
    }
