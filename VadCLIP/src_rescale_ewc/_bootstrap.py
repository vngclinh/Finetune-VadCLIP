"""Make ``VadCLIP/src`` importable so this folder reuses ``model.py``, ``clip/`` and
``utils/`` instead of copying them.

Copying would let the two trees drift apart, and the whole point of the experiment is
that stage 2 differs from the baseline only in the loss. ``src`` is appended rather than
prepended so this folder's own modules always win a name clash.

``VadCLIP/src/utils/layers.py`` builds ``DistanceAdj`` on the parameter's device, unlike
``VadCLIP/baseline/src``, which hardcodes ``.to('cuda')`` -- so importing from ``src`` is
also what lets the smoke test run on CPU.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

if not SRC_DIR.is_dir():
    raise RuntimeError(f"Expected the shared VadCLIP sources at {SRC_DIR}, which does not exist.")

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
