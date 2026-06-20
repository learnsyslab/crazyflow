"""Tests for identification pipeline."""

from __future__ import annotations

import array_api_compat.numpy as np
import pytest

from crazyflow.dynamics.utils.data_utils import derivatives_svf, preprocessing
from crazyflow.dynamics.utils.identification import sys_id_rotation, sys_id_translation


@pytest.mark.integration
def test_sys_id_rotation():
    time = np.linspace(0.0, 1.0, 100)
    pos = np.stack((np.cos(2 * np.pi * time), np.sin(2 * np.pi * time), time), axis=-1)
    quat = np.zeros((100, 4))
    quat[:, 3] = 1.0  # No rotation
    cmd_rpy = np.zeros((100, 3))
    cmd_f = np.ones(100) * 0.03
    data = {
        "time": np.linspace(0.0, 1.0, 100),
        "pos": pos,
        "quat": quat,
        "cmd_rpy": cmd_rpy,
        "cmd_f": cmd_f,
    }

    data = preprocessing(data)
    data["rpy"] = np.roll(cmd_rpy, shift=10, axis=0)
    data = derivatives_svf(data)

    sys_id_rotation(data)


@pytest.mark.integration
@pytest.mark.parametrize("dynamics", ["so_rpy", "so_rpy_rotor", "so_rpy_rotor_drag"])
def test_sys_id_translation(dynamics: str):
    mass = 0.03
    time = np.linspace(0.0, 1.0, 100)
    phi = 2 * np.pi * time
    pos = np.stack((np.cos(phi) * 0.1, np.sin(phi) * 0.1, np.roll(np.cos(phi), 10)), axis=-1)
    quat = np.zeros((100, 4))
    quat[:, 3] = 1.0  # No rotation
    cmd_rpy = np.zeros((100, 3))
    cmd_f = (np.ones(100) + np.cos(phi)) * mass * 9.81
    data = {
        "time": np.linspace(0.0, 1.0, 100),
        "pos": pos,
        "quat": quat,
        "cmd_rpy": cmd_rpy,
        "cmd_f": cmd_f,
    }

    data = preprocessing(data)
    data = derivatives_svf(data)

    sys_id_translation(dynamics=dynamics, mass=mass, data=data)
