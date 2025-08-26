import os, json, gspread
from typing import Optional
from google.oauth2.service_account import Credentials
import pandas as pd

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

def _load_credentials() -> Credentials:
    json_env = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if json_env:
        data = json.loads(json_env)
        return Credentials.from_service_account_info(data, scopes=_SCOPES)
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if path and os.path.exists(path):
        return Credentials.from_service_account_file(path, scopes=_SCOPES)
    raise RuntimeError("Set GOOGLE_SERVICE_ACCOUNT_JSON o GOOGLE_APPLICATION_CREDENTIALS.")

def get_service_account_email() -> Optional[str]:
    try:
        json_env = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        if json_env:
            return json.loads(json_env).get("client_email")
        path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("client_email")
    except Exception:
        pass
    return None

def _client():
    creds = _load_credentials()
    return gspread.authorize(creds)

def read_ws(sheet_id: str, tab: str) -> pd.DataFrame:
    gc = _client()
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(tab)
    rows = ws.get_all_records()
    df = pd.DataFrame(rows)
    # normaliza fecha si existe
    for c in df.columns:
        if c.strip().lower() in ("fecha","date"):
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    return df
