import re
from pathlib import Path

import pytest

from crazyflow.drones import Drone

README = Path(__file__).resolve().parents[2] / "README.md"


@pytest.mark.unit
def test_readme_lists_all_drones():
    """Test that the README drone table lists exactly the members of Drone."""
    section = README.read_text().split("## Supported drones")[1].split("\n## ")[0]
    rows = re.findall(r"^\| `(\w+)` \|", section, re.MULTILINE)
    assert sorted(rows) == sorted(Drone)
