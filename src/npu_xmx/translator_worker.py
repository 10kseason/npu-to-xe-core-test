from __future__ import annotations

import json
import sys
from typing import Any

from .bridge import BridgeState


def _write_response(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> None:
    state = BridgeState()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            op = str(request.get("op", ""))
            payload = request.get("payload", {})
            if op == "shutdown":
                _write_response({"ok": True, "result": {"shutdown": True}})
                return
            result = state.dispatch(op, payload if isinstance(payload, dict) else {})
            _write_response({"ok": True, "result": result})
        except KeyError as exc:
            _write_response({"ok": False, "error": str(exc), "error_type": exc.__class__.__name__})
        except Exception as exc:  # noqa: BLE001 - worker must surface failures to the Java caller.
            _write_response({"ok": False, "error": str(exc), "error_type": exc.__class__.__name__})


if __name__ == "__main__":
    main()
