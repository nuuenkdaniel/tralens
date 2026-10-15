"""Discoverable diagnostic tool tests."""
import tests
from pathlib import Path

# With `discover -s tests`, unittest calls this package "tools". Let its
# diagnostic imports resolve the real checkout tool without duplicating it.
__path__.append(str(Path(__file__).resolve().parents[2] / "tools"))
