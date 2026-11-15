"""Exact firmware fetch, source verification, native build, and dependency audit."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

FIRMWARE_URL = "https://github.com/bitcraze/crazyflie-firmware.git"
PINNED_FIRMWARE_COMMIT = "f45ff8fdce4e73e5ce88ec06eee76309d2fa9a39"
PINNED_CONTROLLER_SHA256 = "8942b2cc12b235915dae7f02759396ab0147360e4ca98564f7b2e3b54092d325"
PINNED_MIXER_SHA256 = "07017878341e15cc2f70406f3a9cc5ea97cb77acf57aac369205154a3614b1d6"
CONTROLLER_SOURCE = Path("src/modules/src/controller/controller_mellinger.c")
MIXER_SOURCE = Path("src/modules/src/power_distribution_quadrotor.c")
EXPECTED_COMPILED_SOURCE_NAMES = {
    "controller_mellinger.c",
    "power_distribution_quadrotor.c",
    "wrapper.cpp",
}
EXPECTED_MANIFEST_RELATIVE_PATHS = (
    CONTROLLER_SOURCE,
    Path("src/modules/interface/controller/controller_mellinger.h"),
    Path("src/modules/interface/controller/position_controller.h"),
    Path("src/modules/interface/stabilizer_types.h"),
    Path("src/modules/interface/math3d.h"),
    Path("src/hal/interface/imu_types.h"),
    Path("src/utils/interface/lighthouse/lighthouse_types.h"),
    Path("src/utils/interface/physicalConstants.h"),
    MIXER_SOURCE,
    Path("src/modules/interface/power_distribution.h"),
    Path("src/utils/interface/num.h"),
    Path("src/platform/interface/platform_defaults.h"),
    Path("src/platform/interface/platform_defaults_cf2.h"),
)

_NETWORK_FAILURE_MARKERS = (
    "could not resolve host",
    "could not resolve proxy",
    "failed to connect",
    "connection timed out",
    "operation timed out",
    "network is unreachable",
    "temporary failure in name resolution",
    "name or service not known",
    "tls",
    "ssl_error",
    "rpc failed",
    "remote end hung up unexpectedly",
    "early eof",
    "the requested url returned error: 500",
    "the requested url returned error: 502",
    "the requested url returned error: 503",
    "the requested url returned error: 504",
)


class FirmwareFetchUnavailable(RuntimeError):
    """A positively classified transient network/transport retrieval failure."""


class FirmwareFetchFailure(RuntimeError):
    """A non-network fetch failure that must never be converted into a skip."""


class FirmwareVerificationError(AssertionError):
    """The commit, tree, source manifest, or compiled-input boundary differs."""


class NativeBuildError(AssertionError):
    """The transient native configure, build, or compile audit failed."""


@dataclass(frozen=True)
class NativeBuild:
    """Verified paths and compact build evidence inside one owned workspace."""

    firmware_dir: Path
    build_dir: Path
    module_path: Path
    compiled_firmware_inputs: tuple[str, ...]
    compiled_sources: tuple[str, ...]


def _run(
    command: Sequence[str], *, cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [str(value) for value in command],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )
    if check and result.returncode != 0:
        output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{output}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(manifest_path: Path) -> tuple[tuple[str, Path], ...]:
    """Read the immutable 13-entry manifest and reject malformed/path-escape input."""
    if not manifest_path.is_file():
        raise FirmwareVerificationError(f"manifest is missing: {manifest_path}")
    entries: list[tuple[str, Path]] = []
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = raw_line.split(maxsplit=1)
        if (
            len(fields) != 2
            or len(fields[0]) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in fields[0])
        ):
            raise FirmwareVerificationError(f"malformed manifest line {line_number}: {raw_line!r}")
        relative_path = Path(fields[1])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise FirmwareVerificationError(
                f"unsafe manifest path at line {line_number}: {relative_path}"
            )
        entries.append((fields[0].lower(), relative_path))
    actual_paths = tuple(path for _, path in entries)
    if actual_paths != EXPECTED_MANIFEST_RELATIVE_PATHS:
        raise FirmwareVerificationError(
            "unexpected firmware manifest entries: "
            f"expected={EXPECTED_MANIFEST_RELATIVE_PATHS!r}, actual={actual_paths!r}"
        )
    if len(set(actual_paths)) != len(actual_paths):
        raise FirmwareVerificationError("duplicate firmware manifest path")
    return tuple(entries)


def verify_manifest(
    firmware_dir: Path,
    manifest_path: Path,
    *,
    expected_controller_hash: str = PINNED_CONTROLLER_SHA256,
    expected_mixer_hash: str = PINNED_MIXER_SHA256,
) -> dict[str, Any]:
    """Verify every declared compiler input and both explicit source pins."""
    firmware_dir = firmware_dir.resolve(strict=True)
    mismatches: list[dict[str, str]] = []
    entries = read_manifest(manifest_path)
    for expected_hash, relative_path in entries:
        source = firmware_dir / relative_path
        if not source.is_file() or source.is_symlink():
            mismatches.append(
                {
                    "path": relative_path.as_posix(),
                    "expected": expected_hash,
                    "actual": "missing-or-not-regular",
                }
            )
            continue
        resolved = source.resolve()
        if firmware_dir not in resolved.parents:
            mismatches.append(
                {
                    "path": relative_path.as_posix(),
                    "expected": expected_hash,
                    "actual": "path-escape",
                }
            )
            continue
        actual_hash = _sha256(source)
        if actual_hash != expected_hash:
            mismatches.append(
                {"path": relative_path.as_posix(), "expected": expected_hash, "actual": actual_hash}
            )
    if mismatches:
        raise FirmwareVerificationError(
            "firmware source manifest mismatch: " + json.dumps(mismatches, sort_keys=True)
        )

    controller_hash = _sha256(firmware_dir / CONTROLLER_SOURCE)
    mixer_hash = _sha256(firmware_dir / MIXER_SOURCE)
    explicit_mismatches = []
    if controller_hash != expected_controller_hash:
        explicit_mismatches.append(
            {
                "source": CONTROLLER_SOURCE.as_posix(),
                "expected": expected_controller_hash,
                "actual": controller_hash,
            }
        )
    if mixer_hash != expected_mixer_hash:
        explicit_mismatches.append(
            {
                "source": MIXER_SOURCE.as_posix(),
                "expected": expected_mixer_hash,
                "actual": mixer_hash,
            }
        )
    if explicit_mismatches:
        raise FirmwareVerificationError(
            "explicit controller/mixer hash mismatch: "
            + json.dumps(explicit_mismatches, sort_keys=True)
        )
    return {
        "status": "PASS",
        "manifest_entries": len(entries),
        "controller_sha256": controller_hash,
        "mixer_sha256": mixer_hash,
        "verified_paths": [path.as_posix() for _, path in entries],
    }


def verify_git_checkout(
    firmware_dir: Path, *, expected_commit: str = PINNED_FIRMWARE_COMMIT
) -> dict[str, Any]:
    """Require the exact detached clean commit after retrieval and after build."""
    firmware_dir = firmware_dir.resolve(strict=True)
    head = _run(["git", "-C", str(firmware_dir), "rev-parse", "HEAD"]).stdout.strip()
    status = _run(
        ["git", "-C", str(firmware_dir), "status", "--porcelain=v1", "--untracked-files=all"]
    ).stdout
    if head != expected_commit:
        raise FirmwareVerificationError(
            f"firmware commit mismatch: expected {expected_commit}, got {head}"
        )
    if status:
        raise FirmwareVerificationError(f"firmware checkout has unexpected changes: {status!r}")
    return {"status": "PASS", "head": head, "worktree_clean": True}


def _classified_network_failure(output: str) -> bool:
    lowered = output.lower()
    return any(marker in lowered for marker in _NETWORK_FAILURE_MARKERS)


def fetch_exact_firmware(
    destination: Path, *, run: Callable[..., subprocess.CompletedProcess[str]] = _run
) -> dict[str, Any]:
    """Fetch only the immutable commit into a previously nonexistent directory."""
    destination = destination.resolve()
    if destination.exists():
        raise FirmwareFetchFailure(f"firmware destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    commands = (
        ("git", "-c", "init.defaultBranch=main", "init", str(destination)),
        ("git", "-C", str(destination), "remote", "add", "origin", FIRMWARE_URL),
    )
    try:
        for command in commands:
            run(command)
    except Exception as error:
        raise FirmwareFetchFailure(f"local Git setup failed: {error}") from error

    fetch_command = (
        "git",
        "-C",
        str(destination),
        "fetch",
        "--depth=1",
        "--no-tags",
        "origin",
        PINNED_FIRMWARE_COMMIT,
    )
    try:
        fetch_result = run(fetch_command, check=False)
    except Exception as error:
        output = str(error)
        if _classified_network_failure(output):
            raise FirmwareFetchUnavailable(output) from error
        raise FirmwareFetchFailure(f"firmware fetch failed: {output}") from error
    if fetch_result.returncode != 0:
        output = "\n".join(
            part.strip() for part in (fetch_result.stdout, fetch_result.stderr) if part.strip()
        )
        if _classified_network_failure(output):
            raise FirmwareFetchUnavailable(output)
        raise FirmwareFetchFailure(
            "firmware fetch failed without a classifiable network cause: " + output
        )

    checkout_command = ("git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD")
    try:
        run(checkout_command)
    except Exception as error:
        raise FirmwareFetchFailure(f"exact checkout failed: {error}") from error
    verification = verify_git_checkout(destination)
    effective_url = run(["git", "-C", str(destination), "remote", "get-url", "origin"])
    return {
        "status": "PASS",
        "url": FIRMWARE_URL,
        "effective_url": effective_url.stdout.strip(),
        "requested_commit": PINNED_FIRMWARE_COMMIT,
        "resolved_commit": verification["head"],
        "movable_ref_used": False,
    }


def _compile_arguments(entry: dict[str, Any]) -> list[str]:
    if "arguments" in entry:
        return [str(value) for value in entry["arguments"]]
    return shlex.split(str(entry["command"]))


def _entry_source(entry: dict[str, Any]) -> Path:
    source = Path(str(entry["file"]))
    if not source.is_absolute():
        source = Path(str(entry["directory"])) / source
    return source.resolve()


def audit_compile_database(
    harness_dir: Path, firmware_dir: Path, build_dir: Path
) -> dict[str, Any]:
    """Require the exact two C sources plus wrapper, standards, and FP flags."""
    database_path = build_dir / "compile_commands.json"
    if not database_path.is_file():
        raise NativeBuildError(f"compile database missing: {database_path}")
    entries = json.loads(database_path.read_text(encoding="utf-8"))
    expected_sources = {
        (firmware_dir / CONTROLLER_SOURCE).resolve(),
        (firmware_dir / MIXER_SOURCE).resolve(),
        (harness_dir / "src/wrapper.cpp").resolve(),
    }
    actual_sources = {_entry_source(entry) for entry in entries}
    if actual_sources != expected_sources:
        raise NativeBuildError(
            "unexpected compiled source set: "
            f"missing={sorted(map(str, expected_sources - actual_sources))}, "
            f"unexpected={sorted(map(str, actual_sources - expected_sources))}"
        )

    standards = {
        "controller_mellinger.c": "-std=gnu11",
        "power_distribution_quadrotor.c": "-std=gnu11",
        "wrapper.cpp": "-std=c++17",
    }
    for entry in entries:
        source_name = _entry_source(entry).name
        arguments = _compile_arguments(entry)
        required = {standards[source_name], "-fno-fast-math", "-ffp-contract=off"}
        missing = required - set(arguments)
        if missing:
            raise NativeBuildError(f"{source_name} missing compile flags: {sorted(missing)}")
        if {"-ffast-math", "-Ofast"} & set(arguments):
            raise NativeBuildError(f"{source_name} has a forbidden fast-math option")
    return {
        "status": "PASS",
        "compiled_sources": sorted(path.name for path in actual_sources),
        "controller_standard": "gnu11",
        "mixer_standard": "gnu11",
        "wrapper_standard": "c++17",
        "fp_contract": "off",
        "fast_math": False,
    }


def audit_ninja_dependencies(
    firmware_dir: Path, build_dir: Path, manifest_path: Path
) -> tuple[str, ...]:
    """Match every firmware path seen by the compiler to the 13-file manifest."""
    result = _run(["ninja", "-C", str(build_dir), "-t", "deps"], check=False)
    if result.returncode != 0:
        raise NativeBuildError(f"ninja dependency audit failed: {result.stderr.strip()}")
    firmware_dir = firmware_dir.resolve()
    observed: set[Path] = set()
    firmware_prefix = str(firmware_dir) + os.sep
    for raw_line in result.stdout.splitlines():
        candidate = raw_line.strip()
        if not candidate.startswith(firmware_prefix):
            continue
        path = Path(candidate).resolve()
        if path == firmware_dir or firmware_dir not in path.parents:
            continue
        observed.add(path.relative_to(firmware_dir))
    expected = {path for _, path in read_manifest(manifest_path)}
    if observed != expected:
        raise FirmwareVerificationError(
            "unexpected compiled firmware input set: "
            f"missing={sorted(path.as_posix() for path in expected - observed)}, "
            f"unexpected={sorted(path.as_posix() for path in observed - expected)}"
        )
    return tuple(sorted(path.as_posix() for path in observed))


def build_verified_oracle(harness_dir: Path, firmware_dir: Path, build_dir: Path) -> NativeBuild:
    """Configure, audit, build, dependency-audit, and reverify the source tree."""
    harness_dir = harness_dir.resolve(strict=True)
    firmware_dir = firmware_dir.resolve(strict=True)
    build_dir = build_dir.resolve()
    if build_dir.exists():
        raise NativeBuildError(f"build directory already exists: {build_dir}")
    manifest_path = harness_dir / "firmware_sources.sha256"
    verify_git_checkout(firmware_dir)
    verify_manifest(firmware_dir, manifest_path)

    configure_command = (
        "cmake",
        "-S",
        str(harness_dir),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        f"-DCF_FIRMWARE={firmware_dir}",
        f"-DPython_EXECUTABLE={sys.executable}",
        "-DCMAKE_BUILD_TYPE=Release",
    )
    build_command = ("cmake", "--build", str(build_dir), "--parallel")
    try:
        _run(configure_command)
    except Exception as error:
        raise NativeBuildError(f"native CMake configure failed: {error}") from error
    compile_audit = audit_compile_database(harness_dir, firmware_dir, build_dir)
    try:
        _run(build_command)
    except Exception as error:
        raise NativeBuildError(f"native build failed: {error}") from error
    compiled_inputs = audit_ninja_dependencies(firmware_dir, build_dir, manifest_path)
    post_manifest = verify_manifest(firmware_dir, manifest_path)
    post_git = verify_git_checkout(firmware_dir)

    modules = tuple(build_dir.glob("mellinger_fuzz_oracle*.so"))
    if len(modules) != 1:
        raise NativeBuildError(f"expected one built oracle extension, found {list(modules)!r}")
    if compile_audit["compiled_sources"] != sorted(EXPECTED_COMPILED_SOURCE_NAMES):
        raise NativeBuildError("compiled source-name audit is inconsistent")
    if post_manifest["manifest_entries"] != 13 or not post_git["worktree_clean"]:
        raise NativeBuildError("post-build source verification is inconsistent")
    return NativeBuild(
        firmware_dir=firmware_dir,
        build_dir=build_dir,
        module_path=modules[0],
        compiled_firmware_inputs=compiled_inputs,
        compiled_sources=tuple(compile_audit["compiled_sources"]),
    )
