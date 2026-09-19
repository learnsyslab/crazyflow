import pytest
from conftest import drone_dynamics

from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim


@pytest.mark.integration
@pytest.mark.parametrize("dynamics, drone", drone_dynamics())
def test_attitude_symbolic(dynamics: Dynamics, drone: "str"):
    """Tests if xml files contain syntax errors."""
    Sim(dynamics=dynamics, drone=drone)
