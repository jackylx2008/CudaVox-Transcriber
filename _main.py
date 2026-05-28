"""Deprecated compatibility entrypoint.

Use `main.py` for normal CLI runs. This file is kept only for callers that
still reference the previous root entrypoint during the transition period.
"""

from __future__ import annotations

import sys

from FunASRNano.cli import main


if __name__ == "__main__":
    sys.stderr.write(
        "Deprecated: _main.py is a compatibility entrypoint. "
        "Use main.py instead.\n"
    )
    raise SystemExit(main())
