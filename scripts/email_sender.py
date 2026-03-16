#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import mimetypes
import smtplib
from email.message import EmailMessage
from pathlib import Path

from env_settings import EMAIL_PASSWORD, EMAIL_RECEIVER, EMAIL_SENDER


QQ_SMTP_HOST = "smtp.qq.com"
QQ_SMTP_PORT_SSL = 465


def send_email_with_attachment(subject: str, body: str, attachment_path: Path, to_addr: str | None = None) -> None:
    sender = (EMAIL_SENDER or "").strip()
    password = (EMAIL_PASSWORD or "").strip()
    receiver = (to_addr or EMAIL_RECEIVER or sender).strip()

    if not sender or not password:
        raise ValueError("EMAIL_SENDER or EMAIL_PASSWORD is not configured")
    if not receiver:
        raise ValueError("EMAIL_RECEIVER is not configured")

    file_path = Path(attachment_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Attachment not found: {file_path}")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver
    msg.set_content(body)

    mime_type, _ = mimetypes.guess_type(file_path.name)
    if mime_type:
        maintype, subtype = mime_type.split("/", 1)
    else:
        maintype, subtype = "application", "octet-stream"

    with open(file_path, "rb") as f:
        msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=file_path.name)

    with smtplib.SMTP_SSL(QQ_SMTP_HOST, QQ_SMTP_PORT_SSL, timeout=30) as server:
        server.login(sender, password)
        server.send_message(msg)
