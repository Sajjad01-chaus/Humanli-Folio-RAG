import json
import logging
import sys
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO, cast

request_id: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = cast(dict[str, object], getattr(record, "fields", {}))
        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
            "request_id": request_id.get(),
            **fields,
        }
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(stream: TextIO | None = None) -> None:
    logger = logging.getLogger("folio")
    if logger.handlers and stream is None:
        return
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False


def event(name: str, *, level: int = logging.INFO, **fields: object) -> None:
    logging.getLogger("folio").log(level, name, extra={"fields": fields})


def failure(name: str, exc: Exception) -> None:
    event(
        name,
        level=logging.ERROR,
        error_type=type(exc).__name__,
        stack=[
            {"file": Path(frame.filename).name, "function": frame.name, "line": frame.lineno}
            for frame in traceback.extract_tb(exc.__traceback__)
        ],
    )
