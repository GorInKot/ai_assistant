"""Настройки портала как есть плюс раздел «ИИ-ассистент».

Рабочие данные демо-портала (SQLite, загрузки) лежат в PORTAL_DATA_DIR, а не в папке портала.
"""

import os
from pathlib import Path

from portalrnsk.settings import *  # noqa: F401,F403
from portalrnsk.settings import DATABASES, INSTALLED_APPS, TEMPLATES

BRIDGE_DIR = Path(__file__).resolve().parent

INSTALLED_APPS = [*INSTALLED_APPS, "portal_bridge"]
ROOT_URLCONF = "portal_bridge.urls"
TEMPLATES[0]["DIRS"] = [BRIDGE_DIR / "templates", *TEMPLATES[0]["DIRS"]]

# Общий с ассистентом секрет (там он PORTAL_SSO_SECRET), не короче 32 символов.
ASSISTANT_SSO_SECRET = os.environ.get("ASSISTANT_SSO_SECRET", "")
# Адрес ассистента; пусто — тот же хост, что у портала, порт ASSISTANT_PORT.
ASSISTANT_URL = os.environ.get("ASSISTANT_URL", "").rstrip("/")
ASSISTANT_PORT = os.environ.get("ASSISTANT_PORT", "8000")

_data_dir = os.environ.get("PORTAL_DATA_DIR")
if _data_dir:
    DATA_DIR = Path(_data_dir).resolve()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for alias, db in DATABASES.items():
        if db["ENGINE"].endswith("sqlite3"):
            db["NAME"] = DATA_DIR / Path(db["NAME"]).name
    MEDIA_ROOT = DATA_DIR / "media"
