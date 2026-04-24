from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config.settings import DASHBOARD_HOST, DASHBOARD_PORT
from services.data_bus import DataBus

HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>Smart Traffic Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; background:#111; color:#eee; margin:20px; }
    .row { display:flex; gap:16px; flex-wrap: wrap; }
    .card { background:#1e1e1e; border:1px solid #333; border-radius:10px; padding:16px; min-width:220px; }
    h1 { margin-top:0; }
    .green { color:#4caf50; font-weight:bold; }
    .red { color:#ff5252; font-weight:bold; }
    pre { background:#0a0a0a; padding:10px; border-radius:8px; overflow:auto; }
  </style>
</head>
<body>
  <h1>Smart Traffic System - Local Demo</h1>
  <div class="row">
    <div class="card"><h3>Total Vehicles</h3><div id="vehicle_count">0</div></div>
    <div class="card"><h3>FPS</h3><div id="fps">0.0</div></div>
    <div class="card"><h3>Frame Index</h3><div id="frame_index">0</div></div>
    <div class="card"><h3>Active Green Lane</h3><div id="active_green_lane">-</div></div>
    <div class="card"><h3>Congestion Score</h3><div id="congestion_score">0%</div></div>
  </div>
  <h3>Signal State</h3>
  <pre id="signal_state"></pre>
  <h3>Lane Density</h3>
  <pre id="lane_counts"></pre>
  <h3>Recent Violations</h3>
  <pre id="violations"></pre>
  <script>
    async function refresh() {
      const res = await fetch('/api/state');
      const s = await res.json();
      document.getElementById('vehicle_count').textContent = s.vehicle_count ?? 0;
      document.getElementById('fps').textContent = s.fps ?? 0;
      document.getElementById('frame_index').textContent = s.frame_index ?? 0;
      document.getElementById('active_green_lane').textContent = s.active_green_lane ?? '-';
      document.getElementById('congestion_score').textContent = `${(s.congestion_score ?? 0).toFixed(1)}%`;
      document.getElementById('signal_state').textContent = JSON.stringify(s.signal_state ?? {}, null, 2);
      document.getElementById('lane_counts').textContent = JSON.stringify(s.lane_counts ?? {}, null, 2);
      document.getElementById('violations').textContent = JSON.stringify((s.violations ?? []).slice(-10), null, 2);
    }
    setInterval(refresh, 1000);
    refresh();
  </script>
</body>
</html>
"""


def _json_response(handler: BaseHTTPRequestHandler, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, html: str) -> None:
    body = html.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class DashboardServer:
    def __init__(self, data_bus: DataBus):
        self.data_bus = data_bus
        self._server: ThreadingHTTPServer | None = None

    def _build_handler(self):
        bus = self.data_bus

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    _html_response(self, HTML)
                    return
                if self.path == "/api/state":
                    _json_response(self, bus.snapshot())
                    return
                self.send_response(404)
                self.end_headers()

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                # Silence default HTTP logs for cleaner demo output.
                return

        return Handler

    def run(self) -> None:
        handler = self._build_handler()
        self._server = ThreadingHTTPServer((DASHBOARD_HOST, DASHBOARD_PORT), handler)
        self._server.serve_forever()

    def run_in_thread(self) -> threading.Thread:
        t = threading.Thread(target=self.run, daemon=True)
        t.start()
        return t

