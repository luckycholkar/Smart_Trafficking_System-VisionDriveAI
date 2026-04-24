from __future__ import annotations

import csv
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from twilio.rest import Client  # type: ignore
except Exception:  # pragma: no cover - allow running without twilio package
    Client = None

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


@dataclass
class EmergencyAlertConfig:
    account_sid: Optional[str] = None
    auth_token: Optional[str] = None
    from_phone: Optional[str] = None
    emergency_phone: Optional[str] = None
    cooldown_seconds: float = 60.0
    log_csv_path: str = "emergency_logs.csv"
    worker_poll_timeout_sec: float = 0.5


class EmergencyAlert:
    """
    Non-blocking Twilio SMS notifier with anti-spam cooldown and CSV logging.
    """

    def __init__(self, config: Optional[EmergencyAlertConfig] = None) -> None:
        self.config = config or EmergencyAlertConfig()
        if load_dotenv:
            load_dotenv()
        self.account_sid = self.config.account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = self.config.auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        self.from_phone = self.config.from_phone or os.getenv("TWILIO_FROM_PHONE")
        self.emergency_phone = self.config.emergency_phone or os.getenv("EMERGENCY_PHONE")

        self._queue: "queue.Queue[Tuple[Dict[str, Any], str]]" = queue.Queue()
        self._cooldown_map: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._ensure_csv_header()

    def notify_async(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Queue an emergency alert if cooldown allows.
        """
        incident_key = self._incident_key(payload)
        now = time.time()
        with self._lock:
            last_sent = self._cooldown_map.get(incident_key)
            if last_sent is not None and (now - last_sent) < self.config.cooldown_seconds:
                return {
                    "queued": False,
                    "status": "COOLDOWN_ACTIVE",
                    "incident_key": incident_key,
                }
            self._cooldown_map[incident_key] = now

        self._queue.put((payload, incident_key))
        return {"queued": True, "status": "QUEUED", "incident_key": incident_key}

    def shutdown(self) -> None:
        self._stop_event.set()
        self._worker.join(timeout=1.0)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload, incident_key = self._queue.get(timeout=self.config.worker_poll_timeout_sec)
            except queue.Empty:
                continue

            status = "FAIL"
            detail = "Twilio not configured"
            try:
                sent, detail = self._send_twilio_sms(payload)
                status = "SUCCESS" if sent else "FAIL"
            except Exception as exc:  # pragma: no cover
                status = "FAIL"
                detail = str(exc)

            self._append_log(payload, incident_key, status, detail)
            self._queue.task_done()

    def _send_twilio_sms(self, payload: Dict[str, Any]) -> Tuple[bool, str]:
        if not all([self.account_sid, self.auth_token, self.from_phone, self.emergency_phone]):
            return False, "Missing Twilio env vars"
        if Client is None:
            return False, "twilio package not installed"

        client = Client(self.account_sid, self.auth_token)
        body = self._build_sms_body(payload)
        msg = client.messages.create(
            body=body,
            from_=self.from_phone,
            to=self.emergency_phone,
        )
        return True, f"sid={msg.sid}"

    def _build_sms_body(self, payload: Dict[str, Any]) -> str:
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

    def _incident_key(self, payload: Dict[str, Any]) -> str:
        camera = str(payload.get("camera_id", "CAM"))
        trigger = str(payload.get("trigger_type", "ACCIDENT"))
        ids = payload.get("involved_vehicle_ids", [])
        ids_text = "-".join(str(x) for x in sorted(ids)) if ids else "na"
        return f"{camera}:{trigger}:{ids_text}"

    def _ensure_csv_header(self) -> None:
        path = Path(self.config.log_csv_path)
        if path.exists():
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "location", "severity", "incident_key", "sms_status", "details"])

    def _append_log(self, payload: Dict[str, Any], incident_key: str, status: str, detail: str) -> None:
        row = [
            payload.get("timestamp", datetime.now().isoformat(timespec="seconds")),
            payload.get("location", "Unknown location"),
            payload.get("severity_level", "UNKNOWN"),
            incident_key,
            status,
            detail,
        ]
        with Path(self.config.log_csv_path).open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

