import os, json, gspread
from typing import Optional
from functools import lru_cache
from google.oauth2.service_account import Credentials
import pandas as pd
import numpy as np

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

@lru_cache(maxsize=1)
def _client():
    creds = _load_credentials()
    return gspread.authorize(creds)

@lru_cache(maxsize=4)
def _open_sheet(sheet_id: str):
    return _client().open_by_key(sheet_id)

def _parse_date_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    d = pd.to_datetime(s, errors="coerce")  # intenta ISO primero
    if d.notna().mean() < 0.8:
        d = pd.to_datetime(s, errors="coerce", dayfirst=True)  # dd/mm/yyyy
    return d.dt.date

def read_ws(sheet_id: str, tab: str) -> pd.DataFrame:
    sh = _open_sheet(sheet_id)
    ws = sh.worksheet(tab)
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    # Detecta fila de encabezado (favor tokens típicos)
    header_idx = 0
    best_i, best_score = 0, -1
    tokens = ("svc","fecha","shipments","spr","delivery","tipo","cantidad","base","e1","total","spot","sdd","rentals")
    for i, row in enumerate(values[:100]):
        row_lower = [c.strip().lower() for c in row]
        nonempty = sum(1 for c in row_lower if c)
        alphas   = sum(1 for c in row_lower if any(ch.isalpha() for ch in c))
        tok = sum(1 for c in row_lower if any(t in c for t in tokens))
        score = (tok*2) + (1 if nonempty>=3 else 0) + (1 if alphas>=3 else 0)
        if score > best_score:
            best_score, best_i = score, i
        if tok>=2 and nonempty>=3:
            header_idx = i
            break
    else:
        header_idx = best_i

    headers_raw = values[header_idx]
    # Sanea headers y evita duplicados
    seen, headers = {}, []
    for j, h in enumerate(headers_raw):
        base = (h or "").replace("\n"," ").strip()
        if not base:
            base = f"col_{j+1}"
        name = base
        k = 1
        while name in seen:
            k += 1
            name = f"{base}_{k}"
        seen[name] = True
        headers.append(name)

    rows = values[header_idx+1:]
    df = pd.DataFrame(rows, columns=headers)

    # Quita filas completamente vacías
    if not df.empty:
        mask_empty = df.apply(lambda r: "".join(map(str, r.values)).strip() == "", axis=1)
        df = df.loc[~mask_empty].copy()

    # Normaliza fechas si existe "Fecha"/"date"
    for c in df.columns:
        if c.strip().lower() in ("fecha","date"):
            df[c] = _parse_date_series(df[c])

    return df
