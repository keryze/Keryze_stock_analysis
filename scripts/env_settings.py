#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv_once() -> None:
    if os.environ.get("_A_STOCK_MONITOR_DOTENV_LOADED") == "1":
        return

    candidates = [
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    env_path = next((p for p in candidates if p.exists()), None)
    if not env_path:
        os.environ["_A_STOCK_MONITOR_DOTENV_LOADED"] = "1"
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ.setdefault(key, value)

    os.environ["_A_STOCK_MONITOR_DOTENV_LOADED"] = "1"


_load_dotenv_once()

TUSHARE_TOKEN = (os.getenv("TUSHARE_TOKEN") or "").strip()
EASTMONEY_COOKIE = (os.getenv("EASTMONEY_COOKIE") or "").strip()

EMAIL_SENDER = (os.getenv("EMAIL_SENDER") or "").strip()
EMAIL_PASSWORD = (os.getenv("EMAIL_PASSWORD") or "").strip()
EMAIL_RECEIVER = (os.getenv("EMAIL_RECEIVER") or "").strip()

FLASK_SECRET_KEY = (os.getenv("FLASK_SECRET_KEY") or "change-this-secret-key").strip()
