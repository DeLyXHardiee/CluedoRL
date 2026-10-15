import subprocess
import sys
from pathlib import Path

PILOT = Path(__file__).resolve().parent.parent / "pilot" / "cluedo_pilot.py"


def test_pilot_still_runs():
    out = subprocess.run([sys.executable, str(PILOT), "20", "4"], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert "players=4" in out.stdout and "win=" in out.stdout
