"""
ANPR placeholder. Swap with EasyOCR or a paid plate-recognition API by
replacing the function passed into HelmetViolationPipeline.
"""

from __future__ import annotations

from typing import Any


def run_anpr(crop_frame: Any) -> dict[str, str]:
    _ = crop_frame
    return {"plate_number": "PENDING", "confidence": "0.00"}
