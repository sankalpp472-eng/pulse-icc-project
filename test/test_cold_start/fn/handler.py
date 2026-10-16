"""
Stand-in for one MCP tool: a tiny HTTP server behind the OpenFaaS of-watchdog.

Each call sleeps ~100 ms (the PRD's 50-150 ms tool time) and reports
whether it was the first request this process has served. If `first_request`
is true, the call was served by a freshly started pod (a cold start).
"""

import json
import os
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROCESS_START = time.time()
WORK_MS = int(os.getenv("WORK_MS", "100"))
served = 0


class Handler(BaseHTTPRequestHandler):
    def _handle(self):
        global served
        if self.path.startswith("/healthz"):
            # Readiness check: answer instantly and don't count it as a request
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        served += 1
        t0 = time.time()
        time.sleep(WORK_MS / 1000)
        body = json.dumps({
            "pod": socket.gethostname(),
            "first_request": served == 1,
            "requests_served": served,
            "process_age_s": round(t0 - PROCESS_START, 3),
            "work_ms": WORK_MS,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _handle

    def log_message(self, *args):  # keep pod logs quiet
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 5000), Handler).serve_forever()
