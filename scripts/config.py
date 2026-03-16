#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deprecated config file.
Sensitive settings were moved to `.env`.
"""

# Keep this file only for non-sensitive defaults or backward reference.
# Runtime secrets must be provided via environment variables.

WATCHED_STOCKS = [
    '600900',
    '601985',
    '600905',
    '600930',
    '603808',
    '300896',
    '688223',
    '603127',
]

ENABLE_WEBSOCKET = True
WEBSOCKET_UPDATE_INTERVAL = 3

WEB_HOST = '0.0.0.0'
WEB_PORT = 5000
PASSWORD = 'stock2024'

# Deprecated placeholders (do not put real secrets here)
TUSHARE_TOKEN = ''
EMAIL_SENDER = ''
EMAIL_PASSWORD = ''
EMAIL_RECEIVER = ''
EASTMONEY_COOKIE = ''
