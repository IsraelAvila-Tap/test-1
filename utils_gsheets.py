# -*- coding: utf-8 -*-
import os, json, io
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

_client_cache = None
_svc_email = None

def _client():
    global _client_cache, _svc_email
    if _client_cache is not None:
        return _client_cache
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON","").strip()
    if not raw:
        raise RuntimeError("No GOOGLE_SERVICE_ACCOUNT_JSON en el entorno.")
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        # por si llegó con comillas escapadas
        info = json.loads(bytes(raw, "utf-8").decode("unicode_escape"))
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    _svc_email = info.get("client_email","")
    _client_cache = gspread.authorize(creds)
    return _client_cache

def get_service_account_email():
    global _svc_email
    if _svc_email:
        return _svc_email
    # forzar construcción
    try:
        _client()
    except Exception:
        return None
    return _svc_email

def _fix_headers(headers):
    seen={}
    fixed=[]
    for j,h in enumerate(headers):
        base = (h or "").replace("\n"," ").strip() or f"col_{j+1}"
        name=base; k=1
        while name in seen:
            k+=1; name=f"{base}_{k}"
        seen[name]=1; fixed.append(name)
    return fixed

def read_ws(sheet_id: str, tab: str) -> pd.DataFrame:
    """Lee una pestaña en modo tolerante:
       - toma la primera fila no-vacía como header
       - rellena encabezados vacíos y desduplica
       - convierte fechas con autoguess (sin forzar dayfirst)
    """
    gc = _client()
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(tab)
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    # header = primera fila con ≥2 celdas no vacías
    header_row = 0
    for i, row in enumerate(values):
        non_empty = [c for c in row if str(c).strip()!=""]
        if len(non_empty) >= 2:
            header_row = i
            break

    header = _fix_headers(values[header_row])
    data   = values[header_row+1:]
    df = pd.DataFrame(data, columns=header)

    # fechas: intenta parsear columnas que luzcan como fecha
    for c in df.columns:
        if "fecha" in c.lower():
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
            except Exception:
                pass
    return df

