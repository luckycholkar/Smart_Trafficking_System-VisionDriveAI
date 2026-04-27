"""Non-blocking Twilio SMS notifier with cooldown and CSV audit log."""

from __future__ import annotations

import csv
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from visiondrive.settings import NotificationsConfig

try:  # pragma: no cover - optional dependency
    from twilio.rest import Client as _TwilioClient
except Exception:  # pragma: no cover
    _TwilioClient = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


class EmergencyAlert:
    """Queues alerts on a worker thread; SMS dispatch is best-effort."""

    def __init__(self, config: NotificationsConfig) -> None:
        self._config = config
        self._queue: queue.Queue[tuple[dict[str, Any], str]] = queue.Queue()
        self._cooldown_map: dict[str, float] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, name="vd-emergency", daemon=True)
        self._worker.start()
        self._ensure_csv_header()

    def notify_async(self, payload: dict[str, Any]) -> dict[str, Any]:
        incident_key = self._incident_key(payload)
        now = time.time()
        with self._lock:
            last_sent = self._cooldown_map.get(incident_key)
            if last_sent is not None and (now - last_sent) < self._config.cooldown_seconds:
                return {"queued": False, "status": "COOLDOWN_ACTIVE", "incident_key": incident_key}
            self._cooldown_map[incident_key] = now

        self._queue.put((payload, incident_key))
        return {"queued": True, "status": "QUEUED", "incident_key": incident_key}

    def shutdown(self) -> None:
        self._stop_event.set()
        self._worker.join(timeout=1.0)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload, incident_key = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            status = "FAIL"
            detail = "Twilio not configured"
            try:
                sent, detail = self._send_sms(payload)
                status = "SUCCESS" if sent else "FAIL"
            except Exception as exc:  # pragma: no cover
                status, detail = "FAIL", str(exc)
                log.exception("Twilio SMS dispatch failed")

            self._append_log(payload, incident_key, status, detail)
            self._queue.task_done()

    def _send_sms(self, payload: dict[str, Any]) -> tuple[bool, str]:
        cfg = self._config
        creds = (
            cfg.twilio_account_sid,
            cfg.twilio_auth_token,
            cfg.twilio_from_phone,
            cfg.emergency_phone,
        )
        if not all(creds):
            return False, "Missing Twilio env vars"
        if _TwilioClient is None:
            return False, "twilio package not installed"

        client = _TwilioClient(cfg.twilio_account_sid, cfg.twilio_auth_token)
        msg = client.messages.create(
            body=self._build_body(payload),
            from_=cfg.twilio_from_phone,
            to=cfg.emergency_phone,
        )
        return True, f"sid={msg.sid}"

    @staticmethod
    def _build_body(payload: dict[str, Any]) -> str:
        timestamp = payload.get("timestamp", datetime.now().isoformat(timespec="seconds"))
        lat = payload.get("gps_lat")
        lon = payload.get("gps_lon")
        location = payload.get("location", "Unknown location")
        severity = payload.get("severity_level", "UNKNOWN")
        maps_link = (
            f"https://maps.google.com/?q={lat},{lon}"
            if lat is not None and lon is not None
            else "https://maps.google.com/"
        )
        return (
            "CRASH ALERT\n"
            f"Time: {timestamp}\n"
            f"Location: {location}\n"
            f"Severity: {severity}\n"
            f"Map: {maps_link}"
        )

    @staticmethod
    def _incident_key(payload: dict[str, Any]) -> str:
        camera = str(payload.get("camera_id", "CAM"))
        trigger = str(payload.get("trigger_type", "ACCIDENT"))
        ids = payload.get("involved_vehicle_ids", [])
        ids_text = "-".join(str(x) for x in sorted(ids)) if ids else "na"
        return f"{camera}:{trigger}:{ids_text}"

    def _ensure_csv_header(self) -> None:
        path = Path(self._config.log_csv_path)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["time", "location", "severity", "incident_key", "sms_status", "details"]
            )

    def _append_log(
        self, payload: dict[str, Any], incident_key: str, status: str, detail: str
    ) -> None:
        row = [
            payload.get("timestamp", datetime.now().isoformat(timespec="seconds")),
            payload.get("location", "Unknown location"),
            payload.get("severity_level", "UNKNOWN"),
            incident_key,
            status,
            detail,
        ]
        with Path(self._config.log_csv_path).open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
