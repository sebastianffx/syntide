"""Serve local analysis and generation-plan downloads."""

import json
import logging
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from syntide.explore.plans import make_plan, verify_analysis


def make_server(directory: Path, port: int = 8765) -> ThreadingHTTPServer:
    directory = directory.resolve()
    analysis = json.loads((directory / "analysis.json").read_text())
    verify_analysis(analysis)

    class Handler(BaseHTTPRequestHandler):
        def authorized(self) -> bool:
            hosts = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            origin = self.headers.get("Origin")
            if self.headers.get("Host") not in hosts or (
                origin and origin not in {f"http://{host}" for host in hosts}
            ):
                self.reply(
                    403,
                    b"Only same-origin loopback requests are allowed",
                    "text/plain",
                )
                return False
            return True

        def reply(self, code: int, payload: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if not self.authorized():
                return
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self.reply(
                    200,
                    (directory / "index.html").read_bytes(),
                    "text/html; charset=utf-8",
                )
            elif path == "/api/analysis":
                self.reply(
                    200, json.dumps(analysis).encode(), "application/json"
                )
            else:
                self.reply(404, b"Not found", "text/plain")

        def do_POST(self) -> None:
            if not self.authorized():
                return
            if self.path != "/api/plan":
                self.reply(404, b"Not found", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000:
                    raise ValueError(
                        "Selection must be between 1 byte and 2 MB"
                    )
                selection = json.loads(self.rfile.read(length))
                if not isinstance(selection, dict):
                    raise ValueError("Selection must be a JSON object")
                result = make_plan(analysis, selection)
                self.reply(
                    200,
                    json.dumps(result, ensure_ascii=False).encode(),
                    "application/json",
                )
            except (ValueError, KeyError, TypeError) as error:
                self.reply(
                    400,
                    json.dumps({"error": str(error)}).encode(),
                    "application/json",
                )

        def log_message(self, message: str, *args: object) -> None:
            logging.getLogger(__name__).debug(message, *args)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(
    directory: Path, port: int = 8765, *, open_browser: bool = True
) -> None:
    server = make_server(directory, port)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"SYNTIDE explorer: {url} (Ctrl-C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
