"""Put the service root on sys.path so tests import server/synthesizer plainly."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
