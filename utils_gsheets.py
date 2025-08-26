from functools import lru_cache
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

    # Leemos todas las celdas (incluye filas vacías arriba)
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    # 1) Detectar la fila de encabezados: primera con >=2 celdas no vacías
    header_idx = 0
    for i, row in enumerate(values[:50]):  # mira hasta las primeras 50 filas
        nonempty = [c.strip() for c in row if c.strip() != ""]
        if len(nonempty) >= 2:
            header_idx = i
            break

    headers_raw = values[header_idx]

    # 2) Sanear encabezados: reemplazar vacíos, evitar duplicados
    seen = {}
    headers = []
    for j, h in enumerate(headers_raw):
        base = (h or "").replace("\n", " ").strip()
        if not base:
            base = f"col_{j+1}"
        name = base
        k = 1
        while name in seen:
            k += 1
            name = f"{base}_{k}"
        seen[name] = True
        headers.append(name)

    # 3) Construir DataFrame
    rows = values[header_idx + 1 : ]
    df = pd.DataFrame(rows, columns=headers)

    # 4) Quitar filas totalmente vacías
    if not df.empty:
        mask_empty = df.apply(lambda r: "".join(map(str, r.values)).strip() == "", axis=1)
        df = df.loc[~mask_empty].copy()

    # 5) Normalizar fechas si existe "Fecha" o "date"
    for c in df.columns:
        if c.strip().lower() in ("fecha", "date"):
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date

    return df
