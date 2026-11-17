"""Put contrib/model-deploy on sys.path so tests can import bridge/apply."""

import sys
from pathlib import Path

CONTRIB = Path(__file__).resolve().parents[2] / "contrib" / "model-deploy"
if str(CONTRIB) not in sys.path:
    sys.path.insert(0, str(CONTRIB))
