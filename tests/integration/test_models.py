import pytest

from crazyflow import available_drones
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim


@pytest.mark.integration
@pytest.mark.parametrize("dynamics", Dynamics)
@pytest.mark.parametrize("drone", available_drones)
def test_attitude_symbolic(dynamics: Dynamics, drone: "str"):
    """Tests if xml files contain syntax errors."""
    Sim(dynamics=dynamics, drone=drone)
