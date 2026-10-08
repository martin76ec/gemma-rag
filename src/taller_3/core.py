"""Shared configuration, storage and one error boundary."""

import json
import traceback
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"


class LabError(RuntimeError):
    """Actionable experiment failure; never replace failed metrics with zeros."""


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


@contextmanager
def errors(stage):
    try:
        yield
    except Exception as exc:
        OUT.mkdir(exist_ok=True)
        with (OUT / "errors.jsonl").open("a") as stream:
            stream.write(json.dumps({"stage": stage, "error": str(exc),
                                     "traceback": traceback.format_exc()}, ensure_ascii=False) + "\n")
        raise LabError(f"{stage}: {exc}. Consulte outputs/errors.jsonl") from exc
