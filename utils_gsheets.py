# utils_gsheets.py — utilidades para leer Google Sheets

import os, json
import pandas as pd

def _client():
    """Devuelve gspread client a partir de GOOGLE_SERVICE_ACCOUNT_JSON (environ)."""
    import gspread
    from google.oauth2.service_account import Credentials

    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if not raw:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON vacío. Define tu Service Account en Streamlit Secrets.")
    try:
        info = json.loads(raw)
    except Exception:
        # si vino TOML -> cadena -> dict
        info = json.loads(str(raw))

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)

def get_service_account_email() -> str:
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if not raw:
        return ""
    try:
        info = json.loads(raw)
    except Exception:
        try:
            info = json.loads(str(raw))
        except Exception:
            return ""
    return info.get("client_email","")

def read_ws(sheet_id: str, worksheet_name: str) -> pd.DataFrame:
    """
    Lee una worksheet como DataFrame sin asumir encabezado.
    Si la primera fila luce como encabezado, la usa; si no, genera col_1, col_2, ...
    """
    gc = _client()
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet_name)
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    # ¿la primera fila es 'header'?: muchas celdas no vacías y sin repetidos
    first = values[0]
    header_like = (len(first) == len(set(first))) and any(first)
    data = values[1:] if header_like else values

    if header_like:
        cols = [str(c).strip() if str(c).strip() else f"col_{i+1}" for i,c in enumerate(first)]
    else:
        # genera encabezados genéricos
        max_cols = max(len(r) for r in values)
        cols = [f"col_{i+1}" for i in range(max_cols)]
        # normaliza filas a igual tamaño
        data = [r + [""]*(max_cols-len(r)) for r in values]

    df = pd.DataFrame(data, columns=cols)

    # tipado ligero: si hay col 'fecha' o que contenga 'date' la convertimos
    for c in list(df.columns):
        label = str(c).lower()
        if label == "fecha" or "date" in label:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True).dt.date
    return df


