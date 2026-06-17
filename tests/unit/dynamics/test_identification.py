"""Tests for identification utils."""

from __future__ import annotations

import array_api_compat.numpy as np
import pytest

from crazyflow.dynamics.utils.data_utils import derivatives_svf, preprocessing


@pytest.mark.unit
def test_preprocessing():
    """Test preprocessing function."""
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

    data_processed = preprocessing(data)
    data_processed["rpy"]


@pytest.mark.unit
def test_derivatives_svf():
    """Test preprocessing function."""
    time = np.linspace(0.0, 1.0, 100)
    pos = np.stack((np.cos(2 * np.pi * time), np.sin(2 * np.pi * time), time), axis=-1)
    quat = np.zeros((100, 4))
    quat[:, 3] = 1.0  # No rotation
    rpy = np.zeros((100, 3))
    cmd_rpy = np.zeros((100, 3))
    cmd_f = np.ones(100) * 0.03
    data = {
        "time": np.linspace(0.0, 1.0, 100),
        "pos": pos,
        "quat": quat,
        "rpy": rpy,
        "cmd_rpy": cmd_rpy,
        "cmd_f": cmd_f,
    }

    data = derivatives_svf(data)
    # Needed for translational sysid
    data["SVF_pos"]
    data["SVF_vel"]
    data["SVF_acc"]
    data["SVF_quat"]
    data["SVF_cmd_f"]
    # Needed for rotational sysid
    data["SVF_rpy"]
    data["SVF_cmd_rpy"]
