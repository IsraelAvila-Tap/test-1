# app.py
# =============================================================================
# Mel-IA — Plan táctico (diario por SVC)
# Tabs: FCST, SPR, Capacity, Rentals, Crowd, SRM (MLP caps).
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
    "srm":      "SRM",     # MLP caps
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

def _as_str_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def _find_units_like_column(df: pd.DataFrame) -> str | None:
    if df is None or df.empty: return None
    def canon(x: str) -> str:
        return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", str(x).lower()).encode("ascii","ignore").decode("ascii"))
    real = list(df.columns)
    cands = [canon(c) for c in real]
    pats = [r"^unidades", r"^units?$", r"^cantidad$", r"^qty$", r"^count$"]
    for i, can in enumerate(cands):
        for pat in pats:
            if re.search(pat, can): return real[i]
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

def _clean_svc_values(series_like) -> list[str]:
    """
    Devuelve una lista de SVCs válidos, eliminando None/nan/'' y
    cadenas 'none'/'nan' y cosas raras.
    """
    import pandas as pd, numpy as np, re
    if series_like is None:
        return []
    s = pd.Series(series_like)
    # quita nulos reales
    s = s[~s.isna()]
    # a str para normalizar
    s = s.astype(str).str.strip()
    # quita valores vacíos o 'none'/'nan'
    s = s[~s.str.lower().isin(["", "none", "nan", "(none)"])]
    # opcional: aplica un patrón simple de SVC (alfa-num, guion/guion_bajo)
    pat = re.compile(r"^[A-Za-z0-9_\-]{2,}$")
    s = s[s.apply(lambda x: bool(pat.match(x)))]
    return sorted(s.unique().tolist())

def _empty_plan_for(svcs: List[str]) -> pd.DataFrame:
    """Plan mínimo para renderizar UI aunque no haya datos."""
    svcs = _clean_svc_values(svcs or [])
    if not svcs:
        return pd.DataFrame()

    cols = [
        "SVC","FECHA",
        "DEMANDA_AJUSTADA","RUTAS_SPR_BASE","RUTAS_FALTANTES",
        # columnas comunes para no romper orden ni sumas
        "FCST","SHIPMENTS_DC","SHIPMENTS_SP","SPR_USADO",
        "RUTAS_RENTALS","SPR_RENTALS",
        "CROWD_PCT","SPR_CROWD","SHIP_OBJ_CROWD","RUTAS_CROWD_OBJ",
        "RUTAS_CROWD_CAP","RUTAS_CROWD_BASE","RUTAS_CROWD_ESCALADO",
        "SHIP_RENTALS","SHIP_CROWD","SHIP_RESTANTES_PRE_MLP",
        "SPR_MLP",
        "MLP_SDD_LV","MLP_SDD_SV","MLP_SDD_CAR","MLP_SDD_CAP",
        "MLP_SPOT_LV","MLP_SPOT_SV","MLP_SPOT_CAR","MLP_SPOT_CAP",
        "MLP_BACK_CAP",
        "RUTAS_MLP_NEEDED","RUTAS_MLP_SDD_USADAS","RUTAS_MLP_SPOT_USADAS","RUTAS_MLP_BACKLOG_USADAS",
        "CROWD_E1_CAP","RUTAS_CROWDE1_USADAS","RUTAS_RESTANTES","RUTAS_POST_MLP"
    ]
    df = pd.DataFrame({"SVC": svcs})
    df["FECHA"] = date.today()
    for c in cols:
        if c not in ("SVC","FECHA"):
            df[c] = 0
    # valores razonables por defecto
    df["SPR_USADO"] = 20
    df["SPR_MLP"] = 20
    df["DEMANDA_AJUSTADA"] = 0
    df["RUTAS_SPR_BASE"] = 0
    df["RUTAS_FALTANTES"] = 0
    return df[cols]


# -----------------------------------------------------------------------------
# 2) Header único + 2 filas + autodetección
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
    """
    Detecta headers 'agrupados' (fila1=familias, fila2=tipos),
    ampliando tokens típicos de SRM y relajando el umbral.
    """
    tokens = ("mlp","sdd","spot","back","backup","bu","total",
              "lv","sv","car","hb","heavy","bulky")
    hits = sum(1 for c in row_lower if any(t in c for t in tokens) and c != "")
    return hits >= 1


# -----------------------------------------------------------------------------
# 3) Lectura de Google Sheets
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
        keys = {"svc", "svcs", "logisticcenterid", "facility", "lc"}
        return any(c in keys for c in row_can)

    header_idx = None
    limit = min(50, len(values))
    for i in range(limit):
        if _is_header_row(values[i]):
            header_idx = i
            break
    if header_idx is None:
        for i in range(limit):
            if sum(1 for x in values[i] if x.strip()) >= 2:
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
        # ➜ más laxo: si la fila1 “parece grupo” y la fila2 tiene ≥ 2 celdas, combinamos
        if _looks_group_header(r1_lower) and r2_nonempty >= 2:
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
# 4) Loaders de cada pestaña
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
    target_cols = ["FECHA", "SVC", "SPR_OBJ", "SPR_PEAK", "SPR_PROM"]

    if df.empty:
        return pd.DataFrame(columns=target_cols)

    # Fecha (opcional)
    coerce_date_column(df, ["FECHA","Fecha","DATE","OP_DT"], "FECHA", "SPR", required=False)

    # 1) Intento estándar
    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","FACILITY","LC"], "SVC", required=False, source_label="SPR")

    # 2) Fuzzy fallback: si tras combinar headers la col quedó como "SVC algo"
    if "SVC" not in df.columns:
        cmap = {_canon_name(c): c for c in df.columns}
        # busca cualquier columna cuyo nombre canónico empiece por "svc" o sea un alias común
        for key, real in cmap.items():
            if key.startswith("svc") or key in ("svcs", "logisticcenterid", "facility", "lc"):
                if real != "SVC":
                    df.rename(columns={real: "SVC"}, inplace=True)
                break

    # Si de plano no hay SVC, devolvemos vacío para no romper groupby
    if "SVC" not in df.columns:
        return pd.DataFrame(columns=target_cols)

    # SPR
    find_and_rename(df, ["SPR","Spr","Ships per route"], "SPR_VAL", required=False, source_label="SPR")
    df = ensure_columns(df, {"SPR_VAL": np.nan})

    # Agregados por SVC (promedio y percentil 95)
    try:
        g = df.groupby("SVC", dropna=False)["SPR_VAL"]
    except TypeError:
        # pandas viejos no tienen dropna= en groupby
        g = df.groupby("SVC")["SPR_VAL"]

    out = g.agg(
        SPR_PROM="mean",
        SPR_PEAK=lambda x: np.nanpercentile(x.dropna(), 95) if x.notna().any() else np.nan
    ).reset_index()

    out["SPR_OBJ"] = out["SPR_PROM"]
    out["FECHA"] = date.today()

    # Orden y columnas finales
    return _finalize(out, target_cols)

# ---- Rentals: homologación + SPR ponderado ----
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
    return _as_str_cols(out, ["SVC","VEHICULO_TIPO_HOM"])

def load_rentals_caps_from_sheet() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["rentals"])
    if df.empty:
        return pd.DataFrame(columns=["SVC","RUTAS_RENTALS","SPR_RENTALS"])

    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", False, "Rentals")
    find_and_rename(df, ["Tipo de vehiculo","Tipo de vehículo","Vehicle type","Tipo"], "TIPO_VEHICULO", False, "Rentals")
    df = ensure_columns(df, {"SVC": None, "TIPO_VEHICULO":"", })

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

    # fallback simple
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
        out = pd.DataFrame(columns=target_cols); out["FECHA"] = date.today(); return out
    find_and_rename(df, ["Unidades disponibles","Unidades dispon","Units","Cantidad","Qty","QTY","COUNT"], "CANT", required=False, source_label="Rentals")
    df = ensure_columns(df, {"CANT":0})
    df["CANT"] = pd.to_numeric(df["CANT"], errors="coerce").fillna(0)
    out = df.groupby("SVC", as_index=False)["CANT"].sum().rename(columns={"CANT":"RUTAS_RENTALS"})
    out["FECHA"] = date.today()
    return _finalize(out, target_cols)

# ---- Crowd (cap E1) y Crowd% desde Capacity + SPR_CROWD ----
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
    df = _as_str_cols(df, ["SVC"])

    if df["FECHA"].notna().any():
        last_by_svc = df.groupby("SVC")["FECHA"].transform("max")
        df = df[df["FECHA"] == last_by_svc]

    tipo   = df["TIPO"].astype(str).str.lower()
    dm     = df["DELIVERY_MODEL"].astype(str).str.lower()
    tipodm = df["TIPO_DM"].astype(str).str.lower()

    is_shipments = tipo.str.contains("ship", regex=False)
    is_crowd = is_shipments & (dm.str.contains("crowd", regex=False) | tipodm.str.contains("crowd", regex=False) | tipo.str.contains("crowd", regex=False))

    tot = df[is_shipments].groupby("SVC", dropna=False)["CANT"].sum().rename("SHIP_TOT")
    crd = df[is_crowd].groupby("SVC", dropna=False)["CANT"].sum().rename("SHIP_CROWD")
    out = pd.concat([tot, crd], axis=1).reset_index()
    out = _as_str_cols(out, ["SVC"])
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

# ---- MLP desde SRM (por tipo de vehículo) ----
def load_mlp_caps_from_srm() -> pd.DataFrame:
    """
    Capacidad MLP por SVC leyendo la pestaña SRM, desglosada por tipo:
      - SDD:  MLP_SDD_LV, MLP_SDD_SV, MLP_SDD_CAR  + MLP_SDD_CAP (suma)
      - SPOT: MLP_SPOT_LV, MLP_SPOT_SV, MLP_SPOT_CAR + MLP_SPOT_CAP (suma)
      - BACKLOG/BU/Back Up: MLP_BACK_CAP
    Se ignoran columnas 'total' para no contar doble.
    Incluye fallback cuando el header viene en 2 filas (familia / tipo).
    """
    df = read_sheet(SHEET_ID, SHEET_TABS["srm"])
    if df.empty:
        return pd.DataFrame(columns=[
            "SVC",
            "MLP_SDD_LV","MLP_SDD_SV","MLP_SDD_CAR","MLP_SDD_CAP",
            "MLP_SPOT_LV","MLP_SPOT_SV","MLP_SPOT_CAR","MLP_SPOT_CAP",
            "MLP_BACK_CAP"
        ])

    # Normaliza SVC
    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", False, "SRM")
    df = _as_str_cols(df, ["SVC"])
    if "SVC" not in df.columns:
        return pd.DataFrame(columns=[
            "SVC",
            "MLP_SDD_LV","MLP_SDD_SV","MLP_SDD_CAR","MLP_SDD_CAP",
            "MLP_SPOT_LV","MLP_SPOT_SV","MLP_SPOT_CAR","MLP_SPOT_CAP",
            "MLP_BACK_CAP"
        ])

    # ------- Fallback: cuando familia y tipo están en filas distintas -------
    canon = {c: _canon_name(c) for c in df.columns}
    fam_tokens = ("sdd", "spot")
    tipo_tokens = ("lv", "sv", "car", "hb", "heavy", "bulky", "large", "small", "xlarge", "xlv")
    cols_raw = list(df.columns)

    def _is_family(c):
        cc = _canon_name(c)
        return any(t in cc for t in fam_tokens) and not any(t in cc for t in tipo_tokens)

    def _is_tipo(c):
        cc = _canon_name(c)
        return any(t in cc for t in tipo_tokens) and not any(t in cc for t in fam_tokens)

    fam_cols = [c for c in cols_raw if _is_family(c)]
    tipo_cols = [c for c in cols_raw if _is_tipo(c)]

    # Si detectamos patrón familia→tipos a la derecha, construimos columnas "familia tipo"
    if fam_cols and tipo_cols:
        for c in cols_raw:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        new_cols = {}
        for fam in fam_cols:
            idx = cols_raw.index(fam)
            tail = cols_raw[idx+1: idx+1+6]  # miramos unas cuantas columnas a la derecha
            tipos_here = [t for t in tail if _is_tipo(t)]
            for t in tipos_here:
                name = f"{fam} {t}".strip()
                if name not in df.columns:
                    new_cols[name] = pd.to_numeric(df.get(t, 0), errors="coerce").fillna(0)
        for k, s in new_cols.items():
            df[k] = s
        canon = {c: _canon_name(c) for c in df.columns}
    # -----------------------------------------------------------------------

    # Helpers de matching
    def is_not_svc(cc: str) -> bool:
        return cc not in ("svc", "svcs", "logisticcenterid", "facility", "lc")

    def has(cc: str, token: str) -> bool:
        return token in cc

    def has_any(cc: str, tokens: list[str]) -> bool:
        return any(t in cc for t in tokens)

    def pick_cols(type_tokens: list[str], family_tokens: list[str], exclude_tokens: list[str]) -> list[str]:
        selected = []
        for col, cc_raw in canon.items():
            cc = cc_raw
            if not is_not_svc(cc): 
                continue
            if all(has(cc, ft) for ft in family_tokens) and has_any(cc, type_tokens) and not has_any(cc, exclude_tokens):
                selected.append(col)
        return selected

    def pick_cols_any(family_tokens: list[str], include_any: list[str], exclude_tokens: list[str]) -> list[str]:
        selected = []
        for col, cc_raw in canon.items():
            cc = cc_raw
            if not is_not_svc(cc): 
                continue
            if all(has(cc, ft) for ft in family_tokens) and has_any(cc, include_any) and not has_any(cc, exclude_tokens):
                selected.append(col)
        return selected

    # Tokens de tipo
    LV  = ["largevan", "large", "lv", "xlarge", "xlv", "heavybulky", "hb"]
    SV  = ["smallvan", "small", "sv"]
    CAR = ["car", "auto"]

    EXC_TOTAL = ["total"]
    EXC_BACK  = ["bu", "backup", "back", "backlog"]
    BACK_ANY  = ["bu", "backup", "back", "backlog"]

    # Columnas por familia/tipo
    sdd_lv_cols   = pick_cols(LV,  ["sdd"], EXC_TOTAL + [])
    sdd_sv_cols   = pick_cols(SV,  ["sdd"], EXC_TOTAL + [])
    sdd_car_cols  = pick_cols(CAR, ["sdd"], EXC_TOTAL + [])

    spot_lv_cols  = pick_cols(LV,  ["spot"], EXC_TOTAL + EXC_BACK)
    spot_sv_cols  = pick_cols(SV,  ["spot"], EXC_TOTAL + EXC_BACK)
    spot_car_cols = pick_cols(CAR, ["spot"], EXC_TOTAL + EXC_BACK)

    back_cols     = pick_cols_any(["spot"], BACK_ANY, EXC_TOTAL)

    # Asegura numérico
    for c in set(sdd_lv_cols + sdd_sv_cols + sdd_car_cols +
                 spot_lv_cols + spot_sv_cols + spot_car_cols +
                 back_cols):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    grp = df.groupby("SVC", dropna=False)

    def sum_cols(cols: list[str]) -> pd.Series:
        if not cols:
            return grp.size().mul(0)  # serie de 0s indexada por SVC
        return grp[cols].sum().sum(axis=1)

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

    for c in ["MLP_SDD_LV","MLP_SDD_SV","MLP_SDD_CAR",
              "MLP_SPOT_LV","MLP_SPOT_SV","MLP_SPOT_CAR",
              "MLP_SDD_CAP","MLP_SPOT_CAP","MLP_BACK_CAP"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).round(0).astype(int)

    return out

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

    is_mlp = (spr["DELIVERY_MODEL"].str.lower().str.contains("mlp|sdd|spot|back", regex=True)
              | spr["TIPO"].str.lower().str.contains("mlp|sdd|spot|back", regex=True)
              | spr["VEH_TYPE"].str.lower().str.contains("mlp|sdd|spot|back", regex=True))
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
def apply_output_adjustments(resumen: pd.DataFrame) -> pd.DataFrame:
    resumen = resumen.drop(columns=["Demanda esperada", "DEMANDA_ESPERADA"], errors="ignore")
    orden = [
        "SVC","FECHA",
        "FCST","SHIPMENTS_DC","SHIPMENTS_SP","FCST (sin DC & sin SP)","DEMANDA_AJUSTADA",
        "SPR_USADO","RUTAS_SPR_BASE",
        "RUTAS_RENTALS","SPR_RENTALS",
        "CROWD_PCT","SPR_CROWD","SHIP_OBJ_CROWD","RUTAS_CROWD_OBJ","RUTAS_CROWD_CAP","RUTAS_CROWD_BASE","RUTAS_CROWD_ESCALADO",
        "SHIP_RENTALS","SHIP_CROWD","SHIP_RESTANTES_PRE_MLP",
        "SPR_MLP",
        # Capacidades MLP por tipo y totales
        "MLP_SDD_LV","MLP_SDD_SV","MLP_SDD_CAR","MLP_SDD_CAP",
        "MLP_SPOT_LV","MLP_SPOT_SV","MLP_SPOT_CAR","MLP_SPOT_CAP",
        "MLP_BACK_CAP",
        # Asignación de rutas MLP
        "RUTAS_MLP_NEEDED","RUTAS_MLP_SDD_USADAS","RUTAS_MLP_SPOT_USADAS","RUTAS_MLP_BACKLOG_USADAS",
        # Comparativo legacy
        "RUTAS_MLP_SDD","RUTAS_MLP_SPOT",
        # E1/MLP y faltantes
        "CROWD_E1_CAP","RUTAS_CROWDE1_USADAS",
        "RUTAS_RESTANTES","RUTAS_FALTANTES",
    ]
    cols = [c for c in orden if c in resumen.columns] + [c for c in resumen.columns if c not in orden]
    return resumen[cols]


def compute_plan(spr_mode: str, sel_svcs: Optional[List[str]] = None) -> pd.DataFrame:
    # --- carga de datos ---
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

    # normaliza SVC = str sin espacios
    for d in (fcst, spr, caps, crowdc, rents, rents_fb, crowd_pct, spr_crowd, mlp_caps, spr_mlp):
        if not d.empty and "SVC" in d.columns:
            _as_str_cols(d, ["SVC"])

    hoy = date.today()

    # ----------------- BASE DE SVCs -----------------
    # 1) intenta construir base a partir de los tabs
    bases = []
    for d in [fcst, spr, caps, crowdc, rents, rents_fb, crowd_pct, mlp_caps]:
        if "SVC" in d.columns and not d.empty:
            bases.append(d[["SVC"]])
    base = pd.concat(bases, axis=0).drop_duplicates() if bases else pd.DataFrame(columns=["SVC"])

    # 2) si no hay nada en tabs, construye base desde selección o defaults
    if base.empty:
        chosen = _clean_svc_values(sel_svcs or DEFAULT_SVCS)
        base = pd.DataFrame({"SVC": chosen})

    base = _as_str_cols(base, ["SVC"])

    out = base.copy()
    out["FECHA"] = hoy
    out = _as_str_cols(out, ["SVC"])

    # ----------------- MERGES PRINCIPALES -----------------
    if not fcst.empty:
        out = safe_merge(out, fcst[["SVC","FCST"]], ["SVC"])

    if not spr.empty:
        out = safe_merge(out, spr[["SVC","SPR_OBJ","SPR_PEAK","SPR_PROM"]], ["SVC"])

    # SPR a usar según modo (promedio/peak/plan), con fallback y clip
    spr_mode_col = {"promedio":"SPR_PROM", "peak":"SPR_PEAK", "plan":"SPR_OBJ"}.get(spr_mode, "SPR_PROM")
    out = ensure_columns(out, {"SPR_OBJ":np.nan, "SPR_PEAK":np.nan, "SPR_PROM":np.nan})
    spr_usado = out[spr_mode_col].where(out[spr_mode_col].notna(), out["SPR_OBJ"]).fillna(20)
    out["SPR_USADO"] = pd.to_numeric(spr_usado, errors="coerce").fillna(20).clip(lower=1)

    # Capacity (MLP legacy caps, rentals, crowd cap, shipments DC/SP)
    if not caps.empty:
        out = safe_merge(out, caps[["SVC","RUTAS_MLP_SDD","RUTAS_MLP_SPOT","RUTAS_RENTALS","RUTAS_CROWD_CAP","SHIPMENTS_DC","SHIPMENTS_SP"]], ["SVC"])
    else:
        out = ensure_columns(out, {"RUTAS_MLP_SDD":0, "RUTAS_MLP_SPOT":0, "RUTAS_CROWD_CAP":0, "SHIPMENTS_DC":0, "SHIPMENTS_SP":0})

    # Rentals: sobrescribe lo que venga de Capacity con el cálculo de Rentals
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

    # Crowd caps (E1) y % crowd desde Capacity + SPR_CROWD
    if not crowdc.empty:
        out = safe_merge(out, crowdc[["SVC","CROWD_E1_CAP"]], ["SVC"])
    out = ensure_columns(out, {"CROWD_E1_CAP":0})

    if not crowd_pct.empty:
        out = safe_merge(out, crowd_pct, ["SVC"])
    else:
        out["CROWD_PCT"] = 0.0

    if not spr_crowd.empty:
        out = safe_merge(out, spr_crowd, ["SVC"])

    out["SPR_CROWD"] = pd.to_numeric(out.get("SPR_CROWD", np.nan), errors="coerce")
    out["SPR_CROWD"] = out["SPR_CROWD"].fillna(out["SPR_USADO"]).clip(lower=1)
    out["CROWD_PCT"] = pd.to_numeric(out.get("CROWD_PCT", 0), errors="coerce").fillna(0).clip(0,1)

    # ----------------- DEMANDA & SPR BASE -----------------
    out = ensure_columns(out, {"FCST":0, "SHIPMENTS_DC":0, "SHIPMENTS_SP":0})
    out["FCST (sin DC & sin SP)"] = (
        pd.to_numeric(out["FCST"], errors="coerce").fillna(0)
        - pd.to_numeric(out["SHIPMENTS_DC"], errors="coerce").fillna(0)
        - pd.to_numeric(out["SHIPMENTS_SP"], errors="coerce").fillna(0)
    ).clip(lower=0)
    out["DEMANDA_AJUSTADA"] = out["FCST (sin DC & sin SP)"]

    out["RUTAS_SPR_BASE"] = np.ceil(out["DEMANDA_AJUSTADA"] / out["SPR_USADO"]).astype(int)

    for c in ["RUTAS_MLP_SDD","RUTAS_MLP_SPOT","RUTAS_RENTALS","RUTAS_CROWD_CAP","CROWD_E1_CAP"]:
        out[c] = pd.to_numeric(out.get(c, 0), errors="coerce").fillna(0)

    # ----------------- CROWD ASIGNACIÓN -----------------
    out["SHIP_OBJ_CROWD"]  = pd.to_numeric(out["FCST"], errors="coerce").fillna(0) * out["CROWD_PCT"]
    out["RUTAS_CROWD_OBJ"] = np.ceil(out["SHIP_OBJ_CROWD"] / out["SPR_CROWD"]).astype(int)

    out["RUTAS_CROWD_BASE"] = np.minimum.reduce([
        np.maximum(out["RUTAS_SPR_BASE"] - out["RUTAS_RENTALS"], 0),
        out["RUTAS_CROWD_CAP"],
        out["RUTAS_CROWD_OBJ"]
    ]).astype(int)

    exceso_obj = (out["RUTAS_CROWD_OBJ"] - out["RUTAS_CROWD_BASE"]).clip(lower=0)
    rem_despues_base = (np.maximum(out["RUTAS_SPR_BASE"] - out["RUTAS_RENTALS"], 0) - out["RUTAS_CROWD_BASE"]).clip(lower=0)

    out["RUTAS_CROWD_ESCALADO"] = np.minimum.reduce([exceso_obj, out["CROWD_E1_CAP"], rem_despues_base]).astype(int)
    out["RUTAS_CROWDE1_USADAS"] = out["RUTAS_CROWD_ESCALADO"]

    # Shipments cubiertos por Rentals y Crowd
    out["SHIP_RENTALS"] = out["RUTAS_RENTALS"] * out["SPR_RENTALS"]
    out["SHIP_CROWD"]   = (out["RUTAS_CROWD_BASE"] + out["RUTAS_CROWD_ESCALADO"]) * out["SPR_CROWD"]

    base_otros = pd.to_numeric(out["SHIPMENTS_DC"], errors="coerce").fillna(0) + pd.to_numeric(out["SHIPMENTS_SP"], errors="coerce").fillna(0)
    out["SHIP_RESTANTES_PRE_MLP"] = (
        pd.to_numeric(out["FCST"], errors="coerce").fillna(0) - base_otros - out["SHIP_RENTALS"] - out["SHIP_CROWD"]
    ).clip(lower=0)

    # ----------------- MLP (caps SRM + SPR_MLP) -----------------
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

    out["RUTAS_MLP_NEEDED"] = np.ceil(out["SHIP_RESTANTES_PRE_MLP"] / out["SPR_MLP"]).astype(int)

    # Asignación priorizada: SDD → SPOT → BACKLOG
    need     = out["RUTAS_MLP_NEEDED"]
    use_sdd  = np.minimum(need, out.get("MLP_SDD_CAP", 0))
    need2    = (need - use_sdd).clip(lower=0)
    use_spot = np.minimum(need2, out.get("MLP_SPOT_CAP", 0))
    need3    = (need2 - use_spot).clip(lower=0)
    use_back = np.minimum(need3, out.get("MLP_BACK_CAP", 0))

    out["RUTAS_MLP_SDD_USADAS"]     = use_sdd.astype(int)
    out["RUTAS_MLP_SPOT_USADAS"]    = use_spot.astype(int)
    out["RUTAS_MLP_BACKLOG_USADAS"] = use_back.astype(int)

    out["RUTAS_RESTANTES"] = (need3 - use_back).clip(lower=0).astype(int)
    out["RUTAS_POST_MLP"]  = out["RUTAS_RESTANTES"]
    out["RUTAS_FALTANTES"] = out["RUTAS_RESTANTES"]

    # ----------------- FILTRO FINAL POR SELECCIÓN -----------------
    if sel_svcs:
        sel_svcs = _clean_svc_values(sel_svcs)
        if sel_svcs:
            out = out[out["SVC"].isin(sel_svcs)]

    # ----------------- ORDEN & LIMPIEZA -----------------
    out = apply_output_adjustments(out).fillna(0).sort_values("SVC").reset_index(drop=True)
    return out

# -----------------------------------------------------------------------------
# 6) UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Mel-IA — Plan táctico (diario por SVC)", layout="wide")

# ----- Sidebar -----
st.sidebar.markdown("## 🗂️ Proyecto")
raw_input = st.sidebar.text_input(
    "SHEET_ID (puede ser URL o ID)",
    value=SHEET_ID or "",
    placeholder="pega aquí la URL o el ID del Sheet",
)
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

# ----- Main -----
st.title("Mel-IA — Plan táctico (diario por SVC)")
spr_mode = st.radio("SPR objetivo", options=["promedio", "peak", "plan"], horizontal=True, index=0)

run_btn = False
auto_run = False
sel_svcs: List[str] = []

# Selector de SVCs (robusto; usa defaults si no hay datos en Sheets)
with st.expander("▶️ Cargando datos...", expanded=True):
    try:
        if not SHEET_ID:
            st.warning("Falta `SHEET_ID`. Pégalo en la barra lateral.")
            svc_list = []
        else:
            fcst_svcs    = load_fcst()[["SVC"]]
            caps         = load_capacity_caps()
            cap_svcs     = caps[["SVC"]] if "SVC" in caps.columns else pd.DataFrame(columns=["SVC"])
            crowd_svcs   = load_crowd_caps()[["SVC"]]
            rents_svcs   = load_rentals_caps_from_sheet()[["SVC"]]
            rent_fb_svcs = load_rentals_fallback()[["SVC"]]
            mlp_svcs     = load_mlp_caps_from_srm()[["SVC"]]

            base_svcs = pd.concat(
                [fcst_svcs, cap_svcs, crowd_svcs, rents_svcs, rent_fb_svcs, mlp_svcs],
                axis=0, ignore_index=True
            )
            svc_list = _clean_svc_values(base_svcs.get("SVC", pd.Series(dtype=object)))

        # si no hay nada en sheets, usa los defaults
        if not svc_list:
            svc_list = sorted(set(DEFAULT_SVCS))

        # muestra también defaults aunque no estén en sheets
        options_union = sorted(set(svc_list) | set(DEFAULT_SVCS))

        default_sel = [s for s in DEFAULT_SVCS if s in options_union] or options_union[:4]
        sel_svcs = st.multiselect(
            "Filtrar SVC",
            options=options_union,
            default=default_sel,
            placeholder="Selecciona SVCs"
        )
        st.write(" ")
        run_btn = st.button("Calcular plan", type="primary")

    except Exception as e:
        st.error("No se pudieron preparar los filtros.")
        show_exception(e, "Detalles (filtros)")

# primer render automático 1 vez
if "auto_run_once" not in st.session_state:
    st.session_state["auto_run_once"] = True
    auto_run = True

# Cálculo y render (con placeholder si queda vacío)
try:
    if run_btn or auto_run:
        if not SHEET_ID:
            st.warning("Proporciona `SHEET_ID` para calcular.")
        else:
            chosen_svcs = sel_svcs or DEFAULT_SVCS

            plan = compute_plan(spr_mode, chosen_svcs)
            if plan.empty:
                plan = _empty_plan_for(chosen_svcs)

            if plan.empty:
                st.warning("No hay datos para mostrar con los filtros seleccionados.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("SVCs", int(plan["SVC"].nunique()))
                c2.metric(
                    "Demanda ajustada",
                    int(pd.to_numeric(plan.get("DEMANDA_AJUSTADA", 0), errors="coerce").fillna(0).sum())
                )
                c3.metric(
                    "Rutas (SPR base)",
                    int(pd.to_numeric(plan.get("RUTAS_SPR_BASE", 0), errors="coerce").fillna(0).sum())
                )
                c4.metric(
                    "Rutas faltantes",
                    int(pd.to_numeric(plan.get("RUTAS_FALTANTES", 0), errors="coerce").fillna(0).sum())
                )
                st.dataframe(plan, use_container_width=True, hide_index=True)
except Exception as e:
    st.error("Ocurrió un error durante el cálculo.")
    show_exception(e, "Traceback completo")

with st.expander("ℹ️ Notas de esta versión"):
    st.markdown(textwrap.dedent("""
    - Rentals desde **Rentals** con **SPR_RENTALS** ponderado; se usa 100% antes de Crowd/MLP.
    - Crowd por % de **Capacity**: **CROWD_PCT**, **SHIP_OBJ_CROWD**, **SPR_CROWD**, base y escalado (E1).
    - **MLP** desde **SRM**:
        - Sumo **solo por tipo** (LV / SV / Car) y **excluyo** columnas “Total …” en SDD/SPOT.
        - **Backlog** = columnas con `back|backlog|bu`.
        - Asigno rutas con prioridad **SDD → SPOT → Backlog**.
    """))
