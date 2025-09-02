# app.py
# =============================================================================
# Mel-IA — Plan táctico (diario por SVC)
# Tabs: FCST, SPR, Capacity, Rentals, Crowd, SRM (MLP caps).
# Encabezado autodetectado (busca SVC/SVCs/LC/Facility) + encabezado de 2 filas (Crowd).
# Robusto ante vacíos/alias y pestañas ausentes.
# =============================================================================
import os, json, re, unicodedata, textwrap, traceback
from datetime import date
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 0) Credenciales y configuración
# -----------------------------------------------------------------------------
if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
elif "gcp_service_account" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps(dict(st.secrets["gcp_service_account"]))

SERVICE_EMAIL = "planificacion@planificacion.iam.gserviceaccount.com"

DEFAULT_SVCS = ["SGD1", "SMT1", "SMX9", "SPB1"]
DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1UBjU3-ftGCow3EzTD0NB6UaYwMUYUARbn9QjD7SlxtY/edit?gid=148917403#gid=148917403"

def sanitize_sheet_id(text: str | None) -> str | None:
    if not text: return None
    text = text.strip()
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", text)
    if m: return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9-_]{20,}", text): return text
    return text

SHEET_ID = sanitize_sheet_id(
    st.secrets.get("SHEET_ID")
    or os.environ.get("SHEET_ID")
    or os.environ.get("PROJECT_SHEET_ID")
    or st.secrets.get("PROJECT_SHEET_ID")
    or DEFAULT_SHEET_URL
)

SHEET_TABS = {
    "fcst":     "FCST",
    "spr":      "SPR",
    "capacity": "Capacity",
    "rentals":  "Rentals",
    "crowd":    "Crowd",
    "srm":      "SRM",     # <--- NUEVO (capacidad MLP)
}

# -----------------------------------------------------------------------------
# 1) Normalización / coerción
# -----------------------------------------------------------------------------
def _canon_name(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    return re.sub(r"[ \-_/\.]", "", s).lower()

def find_and_rename(df: pd.DataFrame, candidates: List[str], new_name: str,
                    required: bool = True, source_label: str = "") -> Optional[str]:
    cmap = {_canon_name(c): c for c in df.columns}
    for cand in candidates:
        key = _canon_name(cand)
        if key in cmap:
            real = cmap[key]
            if real != new_name: df.rename(columns={real: new_name}, inplace=True)
            return new_name
    if required:
        raise ValueError(f"{source_label}: falta columna equivalente a {candidates}. Encabezados: {list(df.columns)}")
    return None

def ensure_columns(df: pd.DataFrame, defaults: Dict[str, object]) -> pd.DataFrame:
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v
    return df

def _clean_svc_values(series_like) -> list[str]:
    """
    Devuelve una lista de SVCs válidos, eliminando None/nan/'' y
    cadenas 'none'/'nan' y cosas raras.
    """
    import pandas as pd, re
    if series_like is None:
        return []
    s = pd.Series(series_like)
    # quita nulos reales
    s = s[~s.isna()]
    # a str para normalizar
    s = s.astype(str).str.strip()
    # quita valores vacíos o 'none'/'nan'
    s = s[~s.str.lower().isin(["", "none", "nan", "(none)"])]
    # patrón simple de SVC (alfa-num, guion/guion_bajo)
    s = s[s.str.match(r"^[A-Za-z0-9_\-]{2,}$", na=False)]
    return sorted(s.unique().tolist())


def _as_str_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def _find_units_like_column(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    def canon(x: str) -> str:
        return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", str(x).lower()).encode("ascii","ignore").decode("ascii"))
    cands = [canon(c) for c in df.columns]
    real  = list(df.columns)
    targets = [r"^unidades", r"^units?$", r"^cantidad$", r"^qty$", r"^count$"]
    for i, can in enumerate(cands):
        for pat in targets:
            if re.search(pat, can):
                return real[i]
    return None

_NUM_SEP_RE = re.compile(r"[ ,\u00A0]")

def _maybe_to_numeric(series_like):
    import pandas as pd
    if isinstance(series_like, pd.DataFrame):
        for sub in series_like.columns:
            series_like[sub] = _maybe_to_numeric(series_like[sub])
        return series_like
    s = series_like
    if getattr(s, "dtype", None) is not None and s.dtype.kind in "iufc":
        return s
    sample = s.dropna().astype(str).head(60) if hasattr(s, "dropna") else []
    if len(sample) == 0:
        try:    return pd.to_numeric(s, errors="coerce")
        except Exception: return s
    looks_numeric = 0
    for v in sample:
        v2 = _NUM_SEP_RE.sub("", v).replace("%","").replace("−","-")
        if re.fullmatch(r"-?\d+(\.\d+)?", v2): looks_numeric += 1
    if looks_numeric / max(1, len(sample)) >= 0.75:
        s2 = s.astype(str).str.replace("%","", regex=False)
        s2 = s2.str.replace("−","-", regex=False)
        s2 = s2.apply(lambda x: _NUM_SEP_RE.sub("", x) if x is not None else x)
        return pd.to_numeric(s2, errors="coerce")
    return s

def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = _maybe_to_numeric(df[c])
    return df

def coerce_date_column(df: pd.DataFrame, candidates: List[str], new_name: str,
                       source_label: str, required: bool = False) -> Optional[str]:
    col = find_and_rename(df, candidates, new_name, required=required, source_label=source_label)
    if col:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True, infer_datetime_format=True).dt.date
        if df[col].notna().sum() == 0:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=False, infer_datetime_format=True).dt.date
    return col

def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: List[str], how="left", suffixes=("_x","_y")):
    if right is None or right.empty:
        return left.copy()
    return left.merge(right, how=how, on=on, suffixes=suffixes)

def show_exception(e: Exception, title: str):
    with st.expander(f"⚠️ {title}", expanded=False):
        st.code("".join(traceback.format_exception(None, e, e.__traceback__)))

# -----------------------------------------------------------------------------
# 2) Header único + 2 filas + AUTODETECCIÓN
# -----------------------------------------------------------------------------
def _make_unique_headers(headers):
    clean = [(h or "").strip() for h in headers]
    out, seen = [], {}
    for i, h in enumerate(clean, start=1):
        base = h if h else f"col_{i}"
        if base in seen:
            seen[base] += 1
            out.append(f"{base}__{seen[base]}")
        else:
            seen[base] = 1
            out.append(base)
    return out

def _combine_two_header_rows(r1: List[str], r2: List[str]) -> List[str]:
    n = max(len(r1), len(r2))
    out = []
    for i in range(n):
        a = (r1[i] if i < len(r1) else "").strip()
        b = (r2[i] if i < len(r2) else "").strip()
        name = (a + " " + b).strip() if (a or b) else ""
        out.append(name)
    return out

def _looks_group_header(row_lower: List[str]) -> bool:
    words = ("base", "e1", "spot", "back", "up", "sdd")
    hits = sum(1 for c in row_lower if any(w in c for w in words))
    return hits >= max(2, len(row_lower)//6)

# -----------------------------------------------------------------------------
# 3) Carga desde Google Sheets con header autodetectado
# -----------------------------------------------------------------------------
def _get_gspread_client():
    import gspread
    from google.oauth2.service_account import Credentials
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw: raise RuntimeError("Faltan credenciales en GOOGLE_SERVICE_ACCOUNT_JSON.")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(json.loads(raw), scopes=scopes)
    return gspread.authorize(creds)

@st.cache_data(show_spinner=False, ttl=300)
def read_sheet(sheet_id: str, tab_name: str) -> pd.DataFrame:
    import gspread
    gc = _get_gspread_client()
    sh = gc.open_by_key(sanitize_sheet_id(sheet_id))
    try:
        ws = sh.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        return pd.DataFrame()

    values = ws.get_all_values()
    if not values: return pd.DataFrame()

    def _is_header_row(row: list[str]) -> bool:
        row_can = [_canon_name(c) for c in row]
        header_keys = {"svc", "svcs", "logisticcenterid", "facility", "lc"}
        return any(c in header_keys for c in row_can)

    header_idx = None
    limit = min(50, len(values))
    for i in range(limit):
        if _is_header_row(values[i]):
            header_idx = i
            break
    if header_idx is None:
        for i in range(limit):
            nonempty = sum(1 for x in values[i] if x.strip())
            if nonempty >= 2:
                header_idx = i
                break
    if header_idx is None:
        header_idx = 0

    r1 = values[header_idx]
    r1_lower = [c.strip().lower() for c in r1]

    combine = False
    if header_idx + 1 < len(values):
        r2 = values[header_idx + 1]
        r2_nonempty = sum(1 for x in r2 if x.strip())
        if _looks_group_header(r1_lower) and r2_nonempty >= max(2, len(r2)//4):
            combine = True

    if combine:
        header = _combine_two_header_rows(values[header_idx], values[header_idx+1])
        data_rows = values[header_idx+2:]
    else:
        header = values[header_idx]
        data_rows = values[header_idx+1:]

    header = _make_unique_headers(header)
    df = pd.DataFrame(data_rows, columns=header)
    return coerce_numeric_df(df)

def quick_healthcheck(sheet_id: str) -> Dict[str, str]:
    out = {"sheet_id": sanitize_sheet_id(sheet_id) or "", "ok": "false", "note": ""}
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(sanitize_sheet_id(sheet_id))
        titles = [w.title for w in sh.worksheets()]
        out["ok"] = "true"
        out["note"] = f"Pestañas: {', '.join(titles[:12])}" + ("…" if len(titles) > 12 else "")
    except Exception as e:
        out["note"] = f"{e}"
    return out

# -----------------------------------------------------------------------------
# 4) Loaders
# -----------------------------------------------------------------------------
def _finalize(df: pd.DataFrame, wanted: List[str]) -> pd.DataFrame:
    df = ensure_columns(df, {"FECHA": pd.NaT, "SVC": None})
    cols = [c for c in wanted if c in df.columns]
    return df[cols].copy() if cols else pd.DataFrame(columns=wanted)

def load_fcst() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["fcst"])
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","FCST"])
    coerce_date_column(df, ["FECHA","Fecha","DATE","OP_DT"], "FECHA", "FCST", required=False)
    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","FACILITY","LC"], "SVC", False, "FCST")
    find_and_rename(df, ["Shipments","SHIPMENTS","FCST","Forecast","PLAN"], "FCST", False, "FCST")
    df = ensure_columns(df, {"FCST":0})
    hoy = date.today()
    if "FECHA" in df.columns:
        df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce").dt.date
        df = df[df["FECHA"].notna()]
        df = df[df["FECHA"] <= hoy] if (df["FECHA"] <= hoy).any() else df
        if not df.empty:
            idx = df.groupby("SVC")["FECHA"].idxmax()
            df = df.loc[idx]
    return _finalize(df, ["FECHA","SVC","FCST"])

def load_spr() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["spr"])
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","SPR_OBJ","SPR_PEAK","SPR_PROM"])
    coerce_date_column(df, ["FECHA","Fecha","DATE","OP_DT"], "FECHA", "SPR", required=False)
    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","FACILITY","LC"], "SVC", False, "SPR")
    find_and_rename(df, ["SPR","Spr"], "SPR_VAL", False, "SPR")
    df = ensure_columns(df, {"SPR_VAL": np.nan})
    g = df.groupby("SVC")["SPR_VAL"]
    out = g.agg(SPR_PROM="mean", SPR_PEAK=lambda x: np.nanpercentile(x.dropna(), 95) if x.notna().any() else np.nan).reset_index()
    out["SPR_OBJ"] = out["SPR_PROM"]
    out["FECHA"] = date.today()
    return _finalize(out, ["FECHA","SVC","SPR_OBJ","SPR_PEAK","SPR_PROM"])

# ---- Rentals (homologación + SPR histórico ponderado) ----
def _norm_txt(x: str) -> str:
    x = (x or "").strip().lower()
    rep = {"eléctrica":"electric","eléctrico":"electric","electrica":"electric","electrico":"electric"}
    for a,b in rep.items(): x = x.replace(a,b)
    x = re.sub(r"\brental(s)?\b", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def homologar_vehicle_type(raw: str) -> str:
    t = _norm_txt(str(raw))
    if re.search(r"\blarge\s+van\s+electric\b", t): return "Large Van Electric"
    if re.search(r"\blarge\s+van\b", t):           return "Large Van"
    if re.search(r"\bsmall\s+van\s+electric\b", t):return "Small Van Electric"
    if re.search(r"\bsmall\s+van\b", t):           return "Small Van"
    if re.search(r"\blarge.*electric\b", t):       return "Large Van Electric"
    if re.search(r"\bsmall.*electric\b", t):       return "Small Van Electric"
    if "large" in t:                               return "Large Van"
    if "small" in t:                               return "Small Van"
    return "Large Van"

def load_spr_hist_from_sheet() -> pd.DataFrame:
    spr = read_sheet(SHEET_ID, SHEET_TABS["spr"])
    if spr.empty:
        return pd.DataFrame(columns=["SVC", "VEHICULO_TIPO_HOM", "SPR_HIST"])
    find_and_rename(spr, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", False, "SPR")
    find_and_rename(spr, ["SHP_LG_VEHICLE_TYPE","Vehicle type","Tipo de vehículo","Tipo de vehiculo"], "SHP_LG_VEHICLE_TYPE", False, "SPR")
    find_and_rename(spr, ["SPR","spr","Ships per route"], "SPR", False, "SPR")
    spr = ensure_columns(spr, {"SVC": None, "SHP_LG_VEHICLE_TYPE":"", "SPR": np.nan})
    spr["SPR"] = pd.to_numeric(spr["SPR"], errors="coerce")

    spr = _as_str_cols(spr, ["SVC", "SHP_LG_VEHICLE_TYPE"])
    spr["VEHICULO_TIPO_HOM"] = spr["SHP_LG_VEHICLE_TYPE"].map(homologar_vehicle_type)

    grp_local = spr.groupby(["SVC","VEHICULO_TIPO_HOM"], dropna=False)["SPR"].median().reset_index().rename(columns={"SPR":"SPR_HIST"})
    grp_glob  = spr.groupby("VEHICULO_TIPO_HOM")["SPR"].median().rename("SPR_GLOBAL_TIPO").reset_index()

    out = grp_local.merge(grp_glob, on="VEHICULO_TIPO_HOM", how="right")
    out["SVC"] = out["SVC"].fillna("__GLOBAL__")
    out["SPR_HIST"] = out["SPR_HIST"].fillna(out["SPR_GLOBAL_TIPO"])
    out = out.drop(columns=["SPR_GLOBAL_TIPO"])
    out = _as_str_cols(out, ["SVC","VEHICULO_TIPO_HOM"])
    return out

def load_rentals_caps_from_sheet() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["rentals"])
    if df.empty:
        return pd.DataFrame(columns=["SVC","RUTAS_RENTALS","SPR_RENTALS"])

    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", False, "Rentals")
    find_and_rename(df, ["Tipo de vehiculo","Tipo de vehículo","Vehicle type","Tipo"], "TIPO_VEHICULO", False, "Rentals")
    df = ensure_columns(df, {"SVC": None, "TIPO_VEHICULO":"", })

    # Columna UNIDADES (exactos + fuzzy)
    units_col = None
    for keys in [["Unidades","Unidades dispon","Unidades disponibles"],
                 ["Units","Cantidad","Qty","QTY","COUNT"]]:
        for k in keys:
            col = find_and_rename(df, [k], "UNIDADES", required=False, source_label="Rentals")
            if col: units_col = "UNIDADES"; break
        if units_col: break
    if not units_col:
        guessed = _find_units_like_column(df)
        if guessed:
            if guessed != "UNIDADES":
                df.rename(columns={guessed:"UNIDADES"}, inplace=True)
            units_col = "UNIDADES"
    if not units_col:
        return pd.DataFrame(columns=["SVC","RUTAS_RENTALS","SPR_RENTALS"])

    df["UNIDADES"] = pd.to_numeric(df["UNIDADES"], errors="coerce").fillna(0)
    df = _as_str_cols(df, ["SVC","TIPO_VEHICULO"])
    df["VEHICULO_TIPO_HOM"] = df["TIPO_VEHICULO"].map(homologar_vehicle_type)

    by_type = df.groupby(["SVC","VEHICULO_TIPO_HOM"], dropna=False)["UNIDADES"].sum().reset_index()
    by_type = _as_str_cols(by_type, ["SVC","VEHICULO_TIPO_HOM"])

    spr_hist = load_spr_hist_from_sheet()
    spr_loc  = _as_str_cols(spr_hist[spr_hist["SVC"] != "__GLOBAL__"], ["SVC","VEHICULO_TIPO_HOM"])
    spr_glob = _as_str_cols(spr_hist[spr_hist["SVC"] == "__GLOBAL__"].drop(columns=["SVC"]).rename(columns={"SPR_HIST":"SPR_GLOBAL_TIPO"}), ["VEHICULO_TIPO_HOM"])

    by_type = by_type.merge(spr_loc,  on=["SVC","VEHICULO_TIPO_HOM"], how="left") \
                     .merge(spr_glob, on=["VEHICULO_TIPO_HOM"],      how="left")
    by_type["SPR_HIST"] = by_type["SPR_HIST"].fillna(by_type["SPR_GLOBAL_TIPO"])
    by_type.drop(columns=["SPR_GLOBAL_TIPO"], inplace=True)

    rentals_sum = by_type.groupby("SVC", dropna=False)["UNIDADES"].sum().rename("RUTAS_RENTALS").reset_index()

    by_type["PESO"] = by_type["UNIDADES"]
    by_type["POND"] = by_type["UNIDADES"] * by_type["SPR_HIST"]
    spr_r = by_type.groupby("SVC", dropna=False)[["POND","PESO"]].sum().reset_index()
    spr_r["SPR_RENTALS"] = (spr_r["POND"] / spr_r["PESO"]).replace([np.inf,-np.inf], np.nan)

    out = rentals_sum.merge(spr_r[["SVC","SPR_RENTALS"]], on="SVC", how="left")
    out = _as_str_cols(out, ["SVC"])

    # Rellena con fallback simple por SVC si existiera
    rents_fb = load_rentals_fallback()
    if not rents_fb.empty:
        rents_fb = _as_str_cols(rents_fb, ["SVC"])
        out = out.merge(rents_fb[["SVC","RUTAS_RENTALS"]].rename(columns={"RUTAS_RENTALS":"RUTAS_RENTALS_FB"}),
                        on="SVC", how="outer")
        out["RUTAS_RENTALS"] = out["RUTAS_RENTALS"].fillna(out["RUTAS_RENTALS_FB"]).fillna(0)
        out.drop(columns=["RUTAS_RENTALS_FB"], inplace=True)

    return out[["SVC","RUTAS_RENTALS","SPR_RENTALS"]]

def load_rentals_fallback() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["rentals"])
    target_cols = ["FECHA","SVC","RUTAS_RENTALS"]
    if df.empty:
        return pd.DataFrame(columns=target_cols)

    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","FACILITY","LC"], "SVC", required=False, source_label="Rentals")
    if "SVC" not in df.columns:
        out = pd.DataFrame(columns=target_cols)
        out["FECHA"] = date.today()
        return out

    find_and_rename(df, ["Unidades disponibles","Unidades dispon","Units","Cantidad","Qty","QTY","COUNT"], "CANT", required=False, source_label="Rentals")
    df = ensure_columns(df, {"CANT":0})
    df["CANT"] = pd.to_numeric(df["CANT"], errors="coerce").fillna(0)

    out = df.groupby("SVC", as_index=False)["CANT"].sum().rename(columns={"CANT":"RUTAS_RENTALS"})
    out["FECHA"] = date.today()
    return _finalize(out, target_cols)

def load_crowd_caps() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["crowd"])
    if df.empty:
        return pd.DataFrame(columns=["FECHA","SVC","CROWD_E1_CAP"])
    try:
        find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","FACILITY","LC"], "SVC", True, "Crowd")
    except Exception:
        return pd.DataFrame(columns=["FECHA","SVC","CROWD_E1_CAP"])

    base_sem_keys = ["Base entre semana","Base entre sem","Base semana","Base entre sem."]
    base_sab_keys = ["Base sabado","Base sábado"]
    base_dom_keys = ["Base domingo"]
    holg_sem_keys = ["Holgura entre semana","Holgura entre sem","Holgura semana"]
    holg_sab_keys = ["Holgura sabado","Holgura sábado"]
    holg_dom_keys = ["Holgura domingo"]

    def pick(df, keys, name):
        find_and_rename(df, keys, name, required=False, source_label="Crowd")

    pick(df, base_sem_keys, "BASE_SEM")
    pick(df, base_sab_keys, "BASE_SAB")
    pick(df, base_dom_keys, "BASE_DOM")
    pick(df, holg_sem_keys, "HOLG_SEM")
    pick(df, holg_sab_keys, "HOLG_SAB")
    pick(df, holg_dom_keys, "HOLG_DOM")

    for c in ["BASE_SEM","BASE_SAB","BASE_DOM","HOLG_SEM","HOLG_SAB","HOLG_DOM"]:
        if c not in df.columns: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["CROWD_E1_CAP"] = df[["HOLG_SEM","HOLG_SAB","HOLG_DOM"]].max(axis=1)

    if "SVC" not in df.columns:
        return pd.DataFrame(columns=["FECHA","SVC","CROWD_E1_CAP"])

    out = df.groupby("SVC", as_index=False)["CROWD_E1_CAP"].max()
    out["FECHA"] = date.today()
    return _finalize(out, ["FECHA","SVC","CROWD_E1_CAP"])

def load_crowd_pct_from_capacity() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["capacity"])
    if df.empty:
        return pd.DataFrame(columns=["SVC", "CROWD_PCT"])

    find_and_rename(df, ["Delivery model","Deliverymodel","Model","DM"], "DELIVERY_MODEL", False, "Capacity")
    find_and_rename(df, ["Tipo","Type","Category"], "TIPO", False, "Capacity")
    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","FACILITY","LC"], "SVC", False, "Capacity")
    find_and_rename(df, ["Tipo DM","TipoDM","DM Type"], "TIPO_DM", False, "Capacity")
    coerce_date_column(df, ["Fecha","FECHA","Date","OP_DT"], "FECHA", "Capacity", required=False)
    find_and_rename(df, ["Cantidad","Qty","Quantity","COUNT","QTY"], "CANT", False, "Capacity")

    df = ensure_columns(df, {"DELIVERY_MODEL":"", "TIPO":"", "TIPO_DM":"", "SVC":None, "FECHA": pd.NaT, "CANT":0})
    df["CANT"] = pd.to_numeric(df["CANT"], errors="coerce").fillna(0)
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce").dt.date
    _as_str_cols(df, ["SVC", "DELIVERY_MODEL", "TIPO", "TIPO_DM"])  # <- asegura str

    # toma último día por SVC si hay fechas
    if df["FECHA"].notna().any():
        last_by_svc = df.groupby("SVC")["FECHA"].transform("max")
        df = df[df["FECHA"] == last_by_svc]

    # normaliza textos
    dm     = df["DELIVERY_MODEL"].fillna("").astype(str)
    tipo   = df["TIPO"].fillna("").astype(str)
    tipodm = df["TIPO_DM"].fillna("").astype(str)

    is_shipments = tipo.str.contains("ship", case=False, regex=False)
    is_crowd = (
        dm.str.contains("crowd", case=False, regex=False) |
        tipodm.str.contains("crowd", case=False, regex=False) |
        tipo.str.contains("crowd", case=False, regex=False)
    )
    is_crowd = is_shipments & is_crowd

    tot = df[is_shipments].groupby("SVC", dropna=False)["CANT"].sum().rename("SHIP_TOT")
    crd = df[is_crowd].groupby("SVC", dropna=False)["CANT"].sum().rename("SHIP_CROWD")

    out = pd.concat([tot, crd], axis=1).reset_index()
    _as_str_cols(out, ["SVC"])
    out["SHIP_TOT"]   = pd.to_numeric(out["SHIP_TOT"], errors="coerce").fillna(0)
    out["SHIP_CROWD"] = pd.to_numeric(out["SHIP_CROWD"], errors="coerce").fillna(0)
    out["CROWD_PCT"]  = (out["SHIP_CROWD"] / out["SHIP_TOT"]).replace([np.inf,-np.inf], 0).fillna(0)
    return out[["SVC", "CROWD_PCT"]]


def load_spr_crowd() -> pd.DataFrame:
    spr = read_sheet(SHEET_ID, SHEET_TABS["spr"])
    if spr.empty:
        return pd.DataFrame(columns=["SVC","SPR_CROWD"])
    find_and_rename(spr, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", False, "SPR")
    find_and_rename(spr, ["SPR","spr","Ships per route"], "SPR", False, "SPR")
    find_and_rename(spr, ["Delivery model","Deliverymodel","Model","DM"], "DELIVERY_MODEL", False, "SPR")
    find_and_rename(spr, ["Tipo","Type","Category"], "TIPO", False, "SPR")
    find_and_rename(spr, ["SHP_LG_VEHICLE_TYPE","Vehicle type","Tipo de vehículo","Tipo de vehiculo"], "VEH_TYPE", False, "SPR")

    spr = ensure_columns(spr, {"SVC":None, "SPR":np.nan, "DELIVERY_MODEL":"", "TIPO":"", "VEH_TYPE":""})
    spr["SPR"] = pd.to_numeric(spr["SPR"], errors="coerce")
    spr = _as_str_cols(spr, ["SVC","DELIVERY_MODEL","TIPO","VEH_TYPE"])

    is_crowd = spr["DELIVERY_MODEL"].str.lower().str.contains("crowd", regex=False) \
               | spr["TIPO"].str.lower().str.contains("crowd", regex=False) \
               | spr["VEH_TYPE"].str.lower().str.contains("crowd", regex=False)

    spr_crowd = spr[is_crowd].copy()
    if spr_crowd.empty:
        return pd.DataFrame(columns=["SVC","SPR_CROWD"])

    grp_local = spr_crowd.groupby("SVC")["SPR"].median().rename("SPR_CROWD").reset_index()
    global_val = spr_crowd["SPR"].median()
    grp_local["SPR_CROWD"] = grp_local["SPR_CROWD"].fillna(global_val)
    return _as_str_cols(grp_local, ["SVC"])

# ---- Capacity caps (MLP/Crowd caps + Shipments DC/SP) ----
def load_capacity_caps() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["capacity"])
    wanted = ["FECHA","SVC","RUTAS_MLP_SDD","RUTAS_MLP_SPOT","RUTAS_RENTALS","RUTAS_CROWD_CAP","SHIPMENTS_DC","SHIPMENTS_SP"]
    if df.empty:
        return pd.DataFrame(columns=wanted)

    find_and_rename(df, ["Delivery model","Deliverymodel","Model","DM"], "DELIVERY_MODEL", False, "Capacity")
    find_and_rename(df, ["Tipo","Type","Category"], "TIPO", False, "Capacity")
    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","FACILITY","LC"], "SVC", False, "Capacity")
    find_and_rename(df, ["Tipo DM","TipoDM","DM Type"], "TIPO_DM", False, "Capacity")
    coerce_date_column(df, ["Fecha","FECHA","Date","OP_DT"], "FECHA", "Capacity", required=False)
    find_and_rename(df, ["Cantidad","Qty","Quantity","COUNT","QTY"], "CANT", False, "Capacity")
    df = ensure_columns(df, {"DELIVERY_MODEL":"", "TIPO":"", "TIPO_DM":"", "SVC":None, "FECHA": pd.NaT, "CANT":0})
    df["CANT"] = pd.to_numeric(df["CANT"], errors="coerce").fillna(0)

    hoy = date.today()
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce").dt.date
    if df["FECHA"].notna().any():
        sub = df[df["FECHA"].notna()]
        sub = sub[sub["FECHA"] <= hoy]
        target = sub["FECHA"].max() if not sub.empty else df["FECHA"].max()
        df = df[df["FECHA"] == target]
    else:
        df["FECHA"] = hoy

    dm_norm     = df["DELIVERY_MODEL"].fillna("").astype(str).str.lower()
    tipo_norm   = df["TIPO"].fillna("").astype(str).str.lower()
    tipodm_norm = df["TIPO_DM"].fillna("").astype(str).str.lower()

    is_rentals = dm_norm.str.contains("rent",  regex=False)
    is_crowd_routes = dm_norm.str.contains("crowd", regex=False) & (
                        tipo_norm.str.contains("route", regex=False) |
                        tipodm_norm.str.contains("route", case=False, regex=False)
                      )
    is_mlp_spot = dm_norm.str.contains("mlp", regex=False) & tipodm_norm.str.contains("spot", regex=False)
    is_mlp_sdd  = dm_norm.str.contains("mlp", regex=False) & (~is_mlp_spot) & (
                        tipodm_norm.str.contains("mlp", regex=False) |
                        tipodm_norm.str.contains("sdd", regex=False) |
                        (tipodm_norm == "")
                  )

    is_shipments = tipo_norm.str.contains("ship", regex=False)
    is_dc = (
        (dm_norm.str.contains("delivery", regex=False) & dm_norm.str.contains("cell", regex=False)) |
        tipodm_norm.str.contains("delivery cell", regex=False) |
        tipodm_norm.str.contains(r"^(dc|cell)$", case=False, regex=True)
    )
    is_sp = (
        dm_norm.str.contains(r"^(s\.?p\.?|sp)$", case=False, regex=True) |
        (dm_norm.str.contains("service", regex=False) & dm_norm.str.contains("partner", regex=False)) |
        tipodm_norm.str.contains(r"\bsp\b|service partner", case=False, regex=True)
    )
    is_dc_ship = is_shipments & is_dc
    is_sp_ship = is_shipments & is_sp

    g = df.groupby(["FECHA","SVC"])["CANT"]
    agg = pd.DataFrame({
        "RUTAS_MLP_SDD":   g.apply(lambda s: s[is_mlp_sdd.loc[s.index]].sum()),
        "RUTAS_MLP_SPOT":  g.apply(lambda s: s[is_mlp_spot.loc[s.index]].sum()),
        "RUTAS_RENTALS":   g.apply(lambda s: s[is_rentals.loc[s.index]].sum()),
        "RUTAS_CROWD_CAP": g.apply(lambda s: s[is_crowd_routes.loc[s.index]].sum()),
        "SHIPMENTS_DC":    g.apply(lambda s: s[is_dc_ship.loc[s.index]].sum()),
        "SHIPMENTS_SP":    g.apply(lambda s: s[is_sp_ship.loc[s.index]].sum()),
    }).reset_index()

    return _finalize(agg, wanted)

# ---- NUEVO: MLP caps desde SRM (SDD/SPOT/BACKLOG) — SIN “TOTAL” y con desglose por tipo ----

# ---- MLP caps desde SRM (robusto: SDD/SPOT/Back Up; ignora "Total") ----
def load_mlp_caps_from_srm() -> pd.DataFrame:
    """
    Lee la pestaña SRM y arma capacidades MLP por SVC:
      SDD:  MLP_SDD_LV, MLP_SDD_SV, MLP_SDD_CAR, MLP_SDD_CAP
      SPOT: MLP_SPOT_LV, MLP_SPOT_SV, MLP_SPOT_CAR, MLP_SPOT_CAP
      BACK: MLP_BACK_CAP  (Back/Backup/BU/Backlog)
    Ignora columnas 'Total ...' para no contar doble.
    También arrastra la columna textual 'MLP' (proveedor) como referencia.
    """
    df = read_sheet(SHEET_ID, SHEET_TABS["srm"])
    out_cols = [
        "SVC",
        "MLP",  # referencia textual (proveedor)
        "MLP_SDD_LV","MLP_SDD_SV","MLP_SDD_CAR","MLP_SDD_CAP",
        "MLP_SPOT_LV","MLP_SPOT_SV","MLP_SPOT_CAR","MLP_SPOT_CAP",
        "MLP_BACK_CAP",
    ]
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    # --- SVC (con fallback fuzzy: acepta "SVC SCV1", "SVC (ID)", etc.) ---
    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", required=False, source_label="SRM")
    if "SVC" not in df.columns:
        cmap = {_canon_name(c): c for c in df.columns}
        for key, real in cmap.items():
            if key.startswith("svc"):
                if real != "SVC":
                    df.rename(columns={real: "SVC"}, inplace=True)
                break
    if "SVC" not in df.columns:
        return pd.DataFrame(columns=out_cols)
    df = _as_str_cols(df, ["SVC"])

    # --- Proveedor MLP (solo referencia textual) ---
    find_and_rename(df, ["MLP","Proveedor","Carrier","Proveedor MLP","Partner"], "MLP", required=False, source_label="SRM")
    if "MLP" not in df.columns:
        df["MLP"] = ""
    else:
        df["MLP"] = df["MLP"].astype(str).str.strip()

    # ---------- Canon de columnas para matching ----------
    def canon_col(name: str) -> str:
        c = _canon_name(name)           # quita espacios/acentos
        c = c.replace("h&b", "hb")
        c = re.sub(r"w\d+", "", c)      # quita W36, W37...
        c = re.sub(r"\d+$", "", c)      # quita número suelto final (“ … 3”)
        return c

    canon = {c: canon_col(c) for c in df.columns}

    def is_not_svc(cc: str) -> bool:
        return cc not in ("svc", "svcs", "logisticcenterid", "facility", "lc", "mlp")

    def has(cc: str, token: str) -> bool:
        return token in cc

    def has_any(cc: str, tokens: list[str]) -> bool:
        return any(t in cc for t in tokens)

    def pick_cols(type_tokens: list[str], family_tokens: list[str], exclude_tokens: list[str]) -> list[str]:
        sel = []
        for col, cc in canon.items():
            if not is_not_svc(cc):
                continue
            if all(has(cc, ft) for ft in family_tokens) and has_any(cc, type_tokens) and not has_any(cc, exclude_tokens):
                sel.append(col)
        return sel

    def pick_cols_any(family_tokens: list[str], include_any: list[str], exclude_tokens: list[str]) -> list[str]:
        sel = []
        for col, cc in canon.items():
            if not is_not_svc(cc):
                continue
            if all(has(cc, ft) for ft in family_tokens) and has_any(cc, include_any) and not has_any(cc, exclude_tokens):
                sel.append(col)
        return sel

    # Tokens (amplios) para detectar tipo/familia
    LV  = ["largevan","largev","large","lv","xlarge","xlv","heavybulky","hb"]
    SV  = ["smallvan","small","sv"]
    CAR = ["car","auto"]

    EXC_TOTAL = ["total"]
    EXC_BACK  = ["bu","backup","back","backlog"]
    BACK_ANY  = ["bu","backup","back","backlog"]

    # Columnas por familia/tipo
    sdd_lv_cols   = pick_cols(LV,  ["sdd"], EXC_TOTAL + [])
    sdd_sv_cols   = pick_cols(SV,  ["sdd"], EXC_TOTAL + [])
    sdd_car_cols  = pick_cols(CAR, ["sdd"], EXC_TOTAL + [])

    spot_lv_cols  = pick_cols(LV,  ["spot"], EXC_TOTAL + EXC_BACK)
    spot_sv_cols  = pick_cols(SV,  ["spot"], EXC_TOTAL + EXC_BACK)
    spot_car_cols = pick_cols(CAR, ["spot"], EXC_TOTAL + EXC_BACK)

    back_cols     = pick_cols_any(["spot"], BACK_ANY, EXC_TOTAL)

    # Asegura numérico en todas las columnas sumables
    for c in set(sdd_lv_cols + sdd_sv_cols + sdd_car_cols +
                 spot_lv_cols + spot_sv_cols + spot_car_cols + back_cols):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    grp = df.groupby("SVC", dropna=False)

    def sum_cols(cols: list[str]) -> pd.Series:
        if not cols:
            return grp.size().mul(0)
        # suma por columna dentro de cada grupo y luego por fila
        return grp[cols].sum().sum(axis=1)

    # Agrega referencia MLP (concatena nombres únicos por SVC)
    mlp_ref = grp["MLP"].apply(
        lambda s: ", ".join(pd.Series(s).astype(str).str.strip().replace("", np.nan).dropna().unique()[:5])
    ).rename("MLP").reset_index()

    out = pd.DataFrame({"SVC": grp.size().index}).reset_index(drop=True)

    out["MLP_SDD_LV"]   = sum_cols(sdd_lv_cols).values
    out["MLP_SDD_SV"]   = sum_cols(sdd_sv_cols).values
    out["MLP_SDD_CAR"]  = sum_cols(sdd_car_cols).values
    out["MLP_SDD_CAP"]  = (out["MLP_SDD_LV"] + out["MLP_SDD_SV"] + out["MLP_SDD_CAR"]).astype(int)

    out["MLP_SPOT_LV"]  = sum_cols(spot_lv_cols).values
    out["MLP_SPOT_SV"]  = sum_cols(spot_sv_cols).values
    out["MLP_SPOT_CAR"] = sum_cols(spot_car_cols).values
    out["MLP_SPOT_CAP"] = (out["MLP_SPOT_LV"] + out["MLP_SPOT_SV"] + out["MLP_SPOT_CAR"]).astype(int)

    out["MLP_BACK_CAP"] = sum_cols(back_cols).astype(int).values

    # Tipado final
    for c in ["MLP_SDD_LV","MLP_SDD_SV","MLP_SDD_CAR",
              "MLP_SPOT_LV","MLP_SPOT_SV","MLP_SPOT_CAR",
              "MLP_SDD_CAP","MLP_SPOT_CAP","MLP_BACK_CAP"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).round(0).astype(int)

    # Adjunta MLP (texto)
    out = out.merge(mlp_ref, on="SVC", how="left")
    out["MLP"] = out["MLP"].fillna("")

    return out[out_cols]


# ---- NUEVO: SPR de MLP ----
def load_spr_mlp() -> pd.DataFrame:
    spr = read_sheet(SHEET_ID, SHEET_TABS["spr"])
    if spr.empty:
        return pd.DataFrame(columns=["SVC","SPR_MLP"])
    find_and_rename(spr, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", False, "SPR")
    find_and_rename(spr, ["SPR","spr","Ships per route"], "SPR", False, "SPR")
    find_and_rename(spr, ["Delivery model","Deliverymodel","Model","DM"], "DELIVERY_MODEL", False, "SPR")
    find_and_rename(spr, ["Tipo","Type","Category"], "TIPO", False, "SPR")
    find_and_rename(spr, ["SHP_LG_VEHICLE_TYPE","Vehicle type","Tipo de vehículo","Tipo de vehiculo"], "VEH_TYPE", False, "SPR")

    spr = ensure_columns(spr, {"SVC":None, "SPR":np.nan, "DELIVERY_MODEL":"", "TIPO":"", "VEH_TYPE":""})
    spr["SPR"] = pd.to_numeric(spr["SPR"], errors="coerce")
    spr = _as_str_cols(spr, ["SVC","DELIVERY_MODEL","TIPO","VEH_TYPE"])

    is_mlp = (
        spr["DELIVERY_MODEL"].str.lower().str.contains("mlp|sdd|spot|back", regex=True)
        | spr["TIPO"].str.lower().str.contains("mlp|sdd|spot|back", regex=True)
        | spr["VEH_TYPE"].str.lower().str.contains("mlp|sdd|spot|back", regex=True)
    )
    mlp_rows = spr[is_mlp]
    if mlp_rows.empty:
        return pd.DataFrame(columns=["SVC","SPR_MLP"])
    grp_local = mlp_rows.groupby("SVC")["SPR"].median().rename("SPR_MLP").reset_index()
    global_val = mlp_rows["SPR"].median()
    grp_local["SPR_MLP"] = grp_local["SPR_MLP"].fillna(global_val)
    return grp_local

# -----------------------------------------------------------------------------
# 5) Cálculo del plan
# -----------------------------------------------------------------------------

# ----------------------------- REEMPLAZO COMPLETO -----------------------------
def apply_output_adjustments(resumen: pd.DataFrame) -> pd.DataFrame:
    """
    Esta función define el **orden final** de columnas que verás en la tabla.
    Si quieres ocultar/mostrar columnas, edítalo aquí.
    """
    # limpia nombres heredados
    resumen = resumen.drop(columns=["Demanda esperada", "DEMANDA_ESPERADA"], errors="ignore")

    orden = [
        # Identidad
        "SVC","FECHA",

        # Demanda y SPR base
        "FCST","SHIPMENTS_DC","SHIPMENTS_SP","FCST (sin DC & sin SP)","DEMANDA_AJUSTADA",
        "SPR_USADO","RUTAS_SPR_BASE",

        # Rentals
        "RUTAS_RENTALS","SPR_RENTALS","SHIP_RENTALS",

        # Crowd objetivo + desglose
        "CROWD_PCT","SPR_CROWD","SHIP_OBJ_CROWD","RUTAS_CROWD_OBJ",
        "RUTAS_CROWD_CAP","CROWD_E1_CAP",
        "RUTAS_CROWD_BASE_ASIGNADO","RUTAS_CROWD_E1_ASIGNADO","SHIP_CROWD",

        # Restantes antes de MLP
        "SHIP_RESTANTES_PRE_MLP",

        # MLP (capacidad por tipo y totales SRM)
        "SPR_MLP",
        "MLP_SDD_LV","MLP_SDD_SV","MLP_SDD_CAR","MLP_SDD_CAP",
        "MLP_SPOT_LV","MLP_SPOT_SV","MLP_SPOT_CAR","MLP_SPOT_CAP",
        "MLP_BACK_CAP",

        # Asignación MLP por familia
        "RUTAS_MLP_NEEDED",
        "RUTAS_MLP_SDD","RUTAS_MLP_SPOT","RUTAS_MLP_BACKLOG",

        # Remanentes/Faltantes
        "RUTAS_RESTANTES","RUTAS_POST_MLP","RUTAS_FALTANTES",

        # Capacidad total y riesgo
        "CAP_TOTAL","CAP_VS_FCST","CAP_DIFF_ABS","RIESGO",
    ]

    # Aplica orden dejando al final cualquier columna no listada
    cols = [c for c in orden if c in resumen.columns] + [c for c in resumen.columns if c not in orden]
    return resumen[cols]
# --------------------------- /REEMPLAZO COMPLETO ------------------------------



# ----------------------------- REEMPLAZO COMPLETO -----------------------------
def compute_plan(spr_mode: str, sel_svcs: Optional[List[str]] = None) -> pd.DataFrame:
    # --- Carga de datos ---
    fcst      = load_fcst()
    spr       = load_spr()
    caps      = load_capacity_caps()
    crowdc    = load_crowd_caps()
    rents     = load_rentals_caps_from_sheet()
    rents_fb  = load_rentals_fallback()
    crowd_pct = load_crowd_pct_from_capacity()
    spr_crowd = load_spr_crowd()
    mlp_caps  = load_mlp_caps_from_srm()
    spr_mlp   = load_spr_mlp()

    # Normaliza SVC a str donde aplique
    for d in (fcst, spr, caps, crowdc, rents, rents_fb, crowd_pct, spr_crowd, mlp_caps, spr_mlp):
        if not d.empty and "SVC" in d.columns:
            _as_str_cols(d, ["SVC"])

    hoy = date.today()

    # Base de SVCs
    bases = []
    for d in [fcst, spr, caps, crowdc, rents, rents_fb, crowd_pct, mlp_caps]:
        if "SVC" in d.columns and not d.empty:
            bases.append(d[["SVC"]])
    base = pd.concat(bases, axis=0).drop_duplicates() if bases else pd.DataFrame(columns=["SVC"])
    base = _as_str_cols(base, ["SVC"])

    out = base.copy()
    out["FECHA"] = hoy
    out = _as_str_cols(out, ["SVC"])

    # FCST y SPR objetivo/promedio/peak
    if not fcst.empty:
        out = safe_merge(out, fcst[["SVC","FCST"]], ["SVC"])
    if not spr.empty:
        out = safe_merge(out, spr[["SVC","SPR_OBJ","SPR_PEAK","SPR_PROM"]], ["SVC"])

    spr_mode_col = {"promedio":"SPR_PROM", "peak":"SPR_PEAK", "plan":"SPR_OBJ"}.get(spr_mode, "SPR_PROM")
    out = ensure_columns(out, {"SPR_OBJ":np.nan, "SPR_PEAK":np.nan, "SPR_PROM":np.nan})
    spr_usado = out[spr_mode_col].where(out[spr_mode_col].notna(), out["SPR_OBJ"]).fillna(20)
    out["SPR_USADO"] = pd.to_numeric(spr_usado, errors="coerce").fillna(20).clip(lower=1)

    # Capacity (DC/SP, crowd cap, rentals placeholders)
    if not caps.empty:
        out = safe_merge(out, caps[["SVC","RUTAS_MLP_SDD","RUTAS_MLP_SPOT","RUTAS_RENTALS","RUTAS_CROWD_CAP","SHIPMENTS_DC","SHIPMENTS_SP"]], ["SVC"])
    else:
        out = ensure_columns(out, {"RUTAS_MLP_SDD":0, "RUTAS_MLP_SPOT":0, "RUTAS_CROWD_CAP":0, "SHIPMENTS_DC":0, "SHIPMENTS_SP":0})

    # Rentals — sobrescribe lo de capacity si hay cálculo de rentals más fino
    out = out.drop(columns=["RUTAS_RENTALS"], errors="ignore")
    if not rents.empty:
        out = safe_merge(out, rents[["SVC","RUTAS_RENTALS","SPR_RENTALS"]], ["SVC"])
    elif not rents_fb.empty:
        out = safe_merge(out, rents_fb[["SVC","RUTAS_RENTALS"]], ["SVC"])
        out["SPR_RENTALS"] = np.nan
    else:
        out["RUTAS_RENTALS"] = 0
        out["SPR_RENTALS"]   = np.nan
    out["RUTAS_RENTALS"] = pd.to_numeric(out.get("RUTAS_RENTALS", 0), errors="coerce").fillna(0).astype(int)
    out["SPR_RENTALS"]   = pd.to_numeric(out.get("SPR_RENTALS", np.nan), errors="coerce")
    out["SPR_RENTALS"]   = out["SPR_RENTALS"].fillna(out["SPR_USADO"])

    # CROWD: capacidad E1
    if not crowdc.empty:
        out = safe_merge(out, crowdc[["SVC","CROWD_E1_CAP"]], ["SVC"])
    out = ensure_columns(out, {"CROWD_E1_CAP":0})

    # % crowd desde Capacity + SPR crowd
    if not crowd_pct.empty:
        out = safe_merge(out, crowd_pct, ["SVC"])
    else:
        out["CROWD_PCT"] = 0.0

    if not spr_crowd.empty:
        out = safe_merge(out, spr_crowd, ["SVC"])
    out["SPR_CROWD"] = pd.to_numeric(out.get("SPR_CROWD", np.nan), errors="coerce")
    out["SPR_CROWD"] = out["SPR_CROWD"].fillna(out["SPR_USADO"]).clip(lower=1)
    out["CROWD_PCT"] = pd.to_numeric(out.get("CROWD_PCT", 0), errors="coerce").fillna(0).clip(0,1)

    # FCST (sin DC & sin SP) = Demanda ajustada base para rutas
    out = ensure_columns(out, {"FCST":0, "SHIPMENTS_DC":0, "SHIPMENTS_SP":0})
    out["FCST (sin DC & sin SP)"] = (
        pd.to_numeric(out["FCST"], errors="coerce").fillna(0)
        - pd.to_numeric(out["SHIPMENTS_DC"], errors="coerce").fillna(0)
        - pd.to_numeric(out["SHIPMENTS_SP"], errors="coerce").fillna(0)
    ).clip(lower=0)
    out["DEMANDA_AJUSTADA"] = out["FCST (sin DC & sin SP)"]

    # Rutas base por SPR
    out["RUTAS_SPR_BASE"] = np.ceil(out["DEMANDA_AJUSTADA"] / out["SPR_USADO"]).astype(int)

    # Asegura numéricos de caps
    for c in ["RUTAS_MLP_SDD","RUTAS_MLP_SPOT","RUTAS_RENTALS","RUTAS_CROWD_CAP","CROWD_E1_CAP"]:
        out[c] = pd.to_numeric(out.get(c, 0), errors="coerce").fillna(0)

    # ----------------- CROWD: objetivo, base y escalado (E1) -----------------
    out["SHIP_OBJ_CROWD"]  = pd.to_numeric(out["FCST"], errors="coerce").fillna(0) * out["CROWD_PCT"]
    out["RUTAS_CROWD_OBJ"] = np.ceil(out["SHIP_OBJ_CROWD"] / out["SPR_CROWD"]).astype(int)

    base_crowd = np.minimum.reduce([
        np.maximum(out["RUTAS_SPR_BASE"] - out["RUTAS_RENTALS"], 0),
        out["RUTAS_CROWD_CAP"],
        out["RUTAS_CROWD_OBJ"]
    ]).astype(int)

    exceso_obj = (out["RUTAS_CROWD_OBJ"] - base_crowd).clip(lower=0)
    rem_despues_base = (np.maximum(out["RUTAS_SPR_BASE"] - out["RUTAS_RENTALS"], 0) - base_crowd).clip(lower=0)
    escalado_crowd = np.minimum.reduce([exceso_obj, out["CROWD_E1_CAP"], rem_despues_base]).astype(int)

    # columnas explícitas de crowd
    out["RUTAS_CROWD_BASE_ASIGNADO"] = base_crowd
    out["RUTAS_CROWD_E1_ASIGNADO"]   = escalado_crowd
    out["RUTAS_CROWDE1_USADAS"]      = escalado_crowd  # compat

    # Shipments cubiertos por Rentals y Crowd
    out["SHIP_RENTALS"] = out["RUTAS_RENTALS"] * out["SPR_RENTALS"]
    out["SHIP_CROWD"]   = (base_crowd + escalado_crowd) * out["SPR_CROWD"]

    # ----------------- RESTANTES PRE MLP -----------------
    base_otros = (
        pd.to_numeric(out["SHIPMENTS_DC"], errors="coerce").fillna(0) +
        pd.to_numeric(out["SHIPMENTS_SP"], errors="coerce").fillna(0)
    )

    out["SHIP_RESTANTES_PRE_MLP"] = (
        pd.to_numeric(out["FCST"], errors="coerce").fillna(0)
        - base_otros
        - pd.to_numeric(out["SHIP_RENTALS"], errors="coerce").fillna(0)
        - pd.to_numeric(out["SHIP_CROWD"],   errors="coerce").fillna(0)
    ).clip(lower=0)

    # ----------------- MLP: capacidades (SRM) y SPR_MLP -----------------
    if not mlp_caps.empty:
        out = safe_merge(out, mlp_caps, ["SVC"])
    else:
        out = ensure_columns(out, {
            "MLP_SDD_LV":0,"MLP_SDD_SV":0,"MLP_SDD_CAR":0,"MLP_SDD_CAP":0,
            "MLP_SPOT_LV":0,"MLP_SPOT_SV":0,"MLP_SPOT_CAR":0,"MLP_SPOT_CAP":0,
            "MLP_BACK_CAP":0
        })

    if not spr_mlp.empty:
        out = safe_merge(out, spr_mlp, ["SVC"])

    out["SPR_MLP"] = pd.to_numeric(out.get("SPR_MLP", np.nan), errors="coerce")
    out["SPR_MLP"] = out["SPR_MLP"].fillna(out["SPR_USADO"]).clip(lower=1)

    # Rutas MLP necesarias y asignación por SDD → SPOT → Backlog
    out["RUTAS_MLP_NEEDED"] = np.ceil(out["SHIP_RESTANTES_PRE_MLP"] / out["SPR_MLP"]).astype(int)

    need  = out["RUTAS_MLP_NEEDED"]
    use_sdd  = np.minimum(need,  pd.to_numeric(out["MLP_SDD_CAP"],  errors="coerce").fillna(0))
    need2    = (need - use_sdd).clip(lower=0)
    use_spot = np.minimum(need2, pd.to_numeric(out["MLP_SPOT_CAP"], errors="coerce").fillna(0))
    need3    = (need2 - use_spot).clip(lower=0)
    use_back = np.minimum(need3, pd.to_numeric(out["MLP_BACK_CAP"], errors="coerce").fillna(0))

    out["RUTAS_MLP_SDD"]     = use_sdd.astype(int)
    out["RUTAS_MLP_SPOT"]    = use_spot.astype(int)
    out["RUTAS_MLP_BACKLOG"] = use_back.astype(int)

    out["RUTAS_RESTANTES"] = (need3 - use_back).clip(lower=0).astype(int)
    out["RUTAS_POST_MLP"]  = out["RUTAS_RESTANTES"]
    out["RUTAS_FALTANTES"] = out["RUTAS_RESTANTES"]

    # ----------------- Capacidad total y riesgo (incluye DC y SP) -----------------
    # Capacidad (shipments) aportada por cada bloque
    cap_dc_sp    = pd.to_numeric(out["SHIPMENTS_DC"], errors="coerce").fillna(0) \
                 + pd.to_numeric(out["SHIPMENTS_SP"], errors="coerce").fillna(0)
    cap_rentals  = pd.to_numeric(out["RUTAS_RENTALS"], errors="coerce").fillna(0) * out["SPR_RENTALS"]
    cap_crowd    = (out["RUTAS_CROWD_BASE_ASIGNADO"] + out["RUTAS_CROWD_E1_ASIGNADO"]) * out["SPR_CROWD"]
    cap_mlp      = (out["RUTAS_MLP_SDD"] + out["RUTAS_MLP_SPOT"] + out["RUTAS_MLP_BACKLOG"]) * out["SPR_MLP"]

    out["CAP_TOTAL"]   = (cap_dc_sp + cap_rentals + cap_crowd + cap_mlp).round(2)
    out["CAP_VS_FCST"] = (out["CAP_TOTAL"] / pd.to_numeric(out["FCST"], errors="coerce").replace(0, np.nan)).round(2)
    out["CAP_DIFF_ABS"] = (out["CAP_TOTAL"] - pd.to_numeric(out["FCST"], errors="coerce").fillna(0)).round(2)
    out["RIESGO"] = np.where(out["CAP_TOTAL"] + 1e-6 >= pd.to_numeric(out["FCST"], errors="coerce").fillna(0), "OK", "RIESGO")

    # Filtro por SVC (si el usuario seleccionó)
    if sel_svcs:
        out = out[out["SVC"].isin(sel_svcs)]

    # Orden/visibilidad final
    out = apply_output_adjustments(out).fillna(0).sort_values("SVC").reset_index(drop=True)
    return out
# --------------------------- /REEMPLAZO COMPLETO ------------------------------




# -----------------------------------------------------------------------------
# 6) UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Mel-IA — Plan táctico (diario por SVC)", layout="wide")

st.sidebar.markdown("## 🗂️ Proyecto")
raw_input = st.sidebar.text_input("SHEET_ID (puede ser URL o ID)", value=SHEET_ID or "", placeholder="pega aquí la URL o el ID del Sheet")
new_sheet_id = sanitize_sheet_id(raw_input)
if new_sheet_id != SHEET_ID:
    SHEET_ID = new_sheet_id
    st.cache_data.clear()
    st.session_state["sheet_id"] = SHEET_ID
if SHEET_ID:
    st.sidebar.markdown(f"**Sheet (ID):** `{SHEET_ID}`")
else:
    st.sidebar.error("Configura/Pega `SHEET_ID`.")

st.sidebar.markdown("## 🔐 Credenciales")
st.sidebar.code(f"Comparte el Sheet con:\n{SERVICE_EMAIL}")

with st.sidebar.expander("Estado de conexión", expanded=False):
    try:
        if SHEET_ID:
            hc = quick_healthcheck(SHEET_ID)
            ok = hc.get("ok") == "true"
            st.write("OK ✅" if ok else "Fallo ❌")
            st.caption(hc.get("note", ""))
        else:
            st.info("Proporciona SHEET_ID para checar acceso.")
    except Exception as e:
        st.error("No se pudo validar acceso.")
        st.caption(str(e))

st.title("Mel-IA — Plan táctico (diario por SVC)")
spr_mode = st.radio("SPR objetivo", options=["promedio","peak","plan"], horizontal=True, index=0)

run_btn = False
auto_run = False
sel_svcs: List[str] = []

with st.expander("▶️ Cargando datos...", expanded=True):
    try:
        if not SHEET_ID:
            st.warning("Falta `SHEET_ID`. Pégalo en la barra lateral.")
            svc_list = []
        else:
            fcst_svcs    = load_fcst()[["SVC"]]
            caps         = load_capacity_caps()
            cap_svcs     = caps[["SVC"]] if "SVC" in caps.columns else caps.to_frame(name="SVC")
            crowd_svcs   = load_crowd_caps()[["SVC"]]
            rents_svcs   = load_rentals_caps_from_sheet()[["SVC"]]
            rent_fb_svcs = load_rentals_fallback()[["SVC"]]
            mlp_svcs     = load_mlp_caps_from_srm()[["SVC"]]
            base_svcs = pd.concat([fcst_svcs, cap_svcs, crowd_svcs, rents_svcs, rent_fb_svcs, mlp_svcs], axis=0).drop_duplicates()
            base_svcs = _as_str_cols(base_svcs, ["SVC"])
            svc_list = sorted(base_svcs["SVC"].dropna().astype(str).unique().tolist())

        default_sel = [s for s in DEFAULT_SVCS if s in svc_list] or svc_list[:4]
        sel_svcs = st.multiselect("Filtrar SVC", options=svc_list, default=default_sel, placeholder="Selecciona SVCs")
        st.write(" ")
        run_btn = st.button("Calcular plan", type="primary")
    except Exception as e:
        st.error("No se pudieron preparar los filtros.")
        show_exception(e, "Detalles (filtros)")


if 'auto_run_once' not in st.session_state:
    st.session_state['auto_run_once'] = True
    auto_run = True

try:
    if run_btn or auto_run:
        if not SHEET_ID:
            st.warning("Proporciona `SHEET_ID` para calcular.")
        else:
            plan = compute_plan(spr_mode, sel_svcs or DEFAULT_SVCS)
            if plan.empty:
                st.warning("No hay datos para mostrar con los filtros seleccionados.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("SVCs", plan["SVC"].nunique())
                c2.metric("Demanda ajustada", int(plan["DEMANDA_AJUSTADA"].sum()))
                c3.metric("Rutas (SPR base)", int(plan["RUTAS_SPR_BASE"].sum()))
                c4.metric("Rutas faltantes", int(plan["RUTAS_FALTANTES"].sum()))
                st.dataframe(plan, use_container_width=True, hide_index=True)
except Exception as e:
    st.error("Ocurrió un error durante el cálculo.")
    show_exception(e, "Traceback completo")

with st.expander("ℹ️ Notas de esta versión"):
    st.markdown(textwrap.dedent("""
    - Rentals desde **Rentals** (fuzzy en “Unidades dispon…”) con **SPR_RENTALS** ponderado; siempre se usa 100% antes de Crowd/MLP.
    - Crowd por % de **Capacity**: **CROWD_PCT**, **SHIP_OBJ_CROWD**, **SPR_CROWD**, base y escalado (E1).
    - **MLP** (SRM): se ignoran columnas **Total** y se suma por tipo de vehículo (**Large/Small/Car**) para **SDD** y **SPOT**.
      Se muestran columnas de desglose y se asignan rutas por prioridad **SDD → SPOT → Backlog**.
    """))
