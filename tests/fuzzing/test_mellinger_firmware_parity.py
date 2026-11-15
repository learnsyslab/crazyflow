"""On-demand source-pinned native Mellinger parity smoke for Linux x86-64."""

from __future__ import annotations

import json
import os
import platform
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable

import jax
import pytest

from .mellinger.fetch_build import (
    EXPECTED_MANIFEST_RELATIVE_PATHS,
    FirmwareFetchUnavailable,
    FirmwareVerificationError,
    build_verified_oracle,
    fetch_exact_firmware,
    verify_git_checkout,
    verify_manifest,
)
from .mellinger.parameter_gate import mismatches

FUZZING_DIR = Path(__file__).resolve().parent
HARNESS_DIR = FUZZING_DIR / "mellinger"
CRAZYFLOW_ROOT = FUZZING_DIR.parents[1]

pytestmark = pytest.mark.fuzzing


@contextmanager
def _owned_workspace():
    """Yield one exact TemporaryDirectory and let its owner remove only it."""
    with tempfile.TemporaryDirectory(prefix="crazyflow-mellinger-fuzz-") as temporary_name:
        workspace = Path(temporary_name).resolve(strict=True)
        yield workspace


def _fetch_with_network_policy(
    destination: Path,
    *,
    network_required: bool,
    fetcher: Callable[[Path], dict[str, Any]] = fetch_exact_firmware,
) -> dict[str, Any]:
    try:
        return fetcher(destination)
    except FirmwareFetchUnavailable as error:
        if network_required:
            raise AssertionError(
                f"FAIL: required firmware retrieval infrastructure failed: {error}"
            ) from error
        pytest.skip(
            "SKIP: optional local firmware retrieval failed with a positively "
            f"classified network/transport error: {error}"
        )


def _network_required() -> bool:
    raw_value = os.environ.get("MELLINGER_FUZZ_NETWORK_REQUIRED", "0")
    if raw_value not in {"0", "1"}:
        raise AssertionError("MELLINGER_FUZZ_NETWORK_REQUIRED must be exactly 0 or 1")
    return raw_value == "1"


def _generated_checkout_artifacts() -> list[str]:
    """Detect build, firmware, binary, or result residue in fuzzing."""
    forbidden_directory_names = {"build", "crazyflie-firmware", "results"}
    forbidden_file_names = {"CMakeCache.txt", "compile_commands.json"}
    forbidden_suffixes = {".o", ".so", ".pyd", ".dylib"}
    residue = []
    for path in FUZZING_DIR.rglob("*"):
        relative = path.relative_to(FUZZING_DIR).as_posix()
        if (
            (path.is_dir() and path.name in forbidden_directory_names)
            or path.name in forbidden_file_names
            or path.suffix in forbidden_suffixes
        ):
            residue.append(relative)
    return sorted(residue)


def _compact_result(
    fetch: dict[str, Any],
    compiled_inputs: tuple[str, ...],
    compiled_sources: tuple[str, ...],
    parity: dict[str, Any],
    *,
    workspace_removed: bool,
) -> dict[str, Any]:
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
        "firmware_fetch": fetch,
        "firmware_manifest": {
            "status": "PASS",
            "compiled_inputs_verified": len(compiled_inputs),
            "compiled_inputs": list(compiled_inputs),
        },
        "native_build": {"status": "PASS", "compiled_sources": list(compiled_sources)},
        "cleanup": {
            "workspace_removed": workspace_removed,
            "checkout_residue": _generated_checkout_artifacts(),
        },
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


def test_optional_network_failure_maps_to_pytest_skip() -> None:
    def unavailable(_destination: Path):
        raise FirmwareFetchUnavailable("simulated: network is unreachable")

    with pytest.raises(pytest.skip.Exception, match="SKIP: optional local"):
        _fetch_with_network_policy(
            Path("/not-created"), network_required=False, fetcher=unavailable
        )


def test_required_network_failure_is_nonzero_failure() -> None:
    def unavailable(_destination: Path):
        raise FirmwareFetchUnavailable("simulated: network is unreachable")

    with pytest.raises(AssertionError, match="FAIL: required firmware retrieval"):
        _fetch_with_network_policy(Path("/not-created"), network_required=True, fetcher=unavailable)


def test_hash_mismatch_is_failure_not_skip() -> None:
    workspace = None
    try:
        with _owned_workspace() as owned:
            workspace = owned
            firmware = owned / "firmware"
            for relative_path in EXPECTED_MANIFEST_RELATIVE_PATHS:
                target = firmware / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"deliberate hash mismatch\n")
            with pytest.raises(
                FirmwareVerificationError, match="firmware source manifest mismatch"
            ):
                verify_manifest(firmware, HARNESS_DIR / "firmware_sources.sha256")
    finally:
        assert workspace is None or not workspace.exists()


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


def test_owned_workspace_cleanup_after_success_and_error() -> None:
    successful_workspace = None
    with _owned_workspace() as owned:
        successful_workspace = owned
        (owned / "firmware").mkdir()
        (owned / "build").mkdir()
    assert successful_workspace is not None
    assert not successful_workspace.exists()

    failed_workspace = None
    with pytest.raises(RuntimeError, match="intentional lifecycle failure"):
        with _owned_workspace() as owned:
            failed_workspace = owned
            (owned / "firmware").mkdir()
            (owned / "build").mkdir()
            raise RuntimeError("intentional lifecycle failure")
    assert failed_workspace is not None
    assert not failed_workspace.exists()


@pytest.mark.skipif(
    os.environ.get("CRAZYFLOW_RUN_MELLINGER_FUZZING") != "1",
    reason="set CRAZYFLOW_RUN_MELLINGER_FUZZING=1 to fetch and build the native reference",
)
@pytest.mark.skipif(
    platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"},
    reason="native Mellinger harness supports Linux x86-64",
)
def test_mellinger_firmware_parity_online() -> None:
    missing = [tool for tool in ("git", "cmake", "ninja") if shutil.which(tool) is None]
    assert not missing, f"Install the native Mellinger build tools: {', '.join(missing)}"
    assert find_spec("pybind11") is not None, "Install pybind11 in the active Crazyflow environment"
    residue_before = _generated_checkout_artifacts()
    workspace = None
    fetch_result = None
    native_build = None
    parity_result = None
    previous_cache_setting = jax.config.jax_enable_compilation_cache
    try:
        jax.config.update("jax_enable_compilation_cache", False)
        with _owned_workspace() as owned:
            workspace = owned
            firmware = owned / "crazyflie-firmware"
            build = owned / "native-build"
            fetch_result = _fetch_with_network_policy(
                firmware, network_required=_network_required()
            )
            verify_git_checkout(firmware)
            manifest = verify_manifest(firmware, HARNESS_DIR / "firmware_sources.sha256")
            assert manifest["manifest_entries"] == 13
            native_build = build_verified_oracle(HARNESS_DIR, firmware, build)
            from .mellinger.parity import run_parity_smoke

            parity_result = run_parity_smoke(CRAZYFLOW_ROOT, native_build.build_dir)
            verify_manifest(firmware, HARNESS_DIR / "firmware_sources.sha256")
            verify_git_checkout(firmware)
    finally:
        jax.config.update("jax_enable_compilation_cache", previous_cache_setting)
        assert workspace is None or not workspace.exists()

    assert fetch_result is not None
    assert native_build is not None
    assert parity_result is not None
    assert _generated_checkout_artifacts() == residue_before
    result = _compact_result(
        fetch_result,
        native_build.compiled_firmware_inputs,
        native_build.compiled_sources,
        parity_result,
        workspace_removed=not workspace.exists(),
    )
    assert result["cleanup"]["workspace_removed"] is True
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
