"""Self-contained HTML explorer: no CDN, external requests or remote telemetry."""

import json
from pathlib import Path


def write_viewer(analysis: dict, path: Path) -> None:
    # Escaping '<' prevents a dataset sentence from ending the JSON script element.
    payload = json.dumps(analysis, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("viewer.html").read_text()
    path.write_text(template.replace("__SYNTIDE_ANALYSIS_JSON__", payload))
