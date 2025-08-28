# app.py
# =============================================================================
# Mel-IA — Plan táctico (diario por SVC)
# Tabs: FCST, SPR, Capacity, Rentals, Crowd.
# Encabezado autodetectado (busca SVC) + encabezado de 2 filas (Crowd).
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

    header_idx = None
    limit = min(50, len(values))
    for i in range(limit):
        row_lower = [c.strip().lower() for c in values[i]]
        if any(c == "svc" for c in row_lower):
            header_idx = i
            break
    if header_idx is None: header_idx = 0

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

def load_capacity_caps() -> pd.DataFrame:
    """Devuelve caps por rutas y la columna SHIPMENTS_DC_SP para quitar a FCST."""
    df = read_sheet(SHEET_ID, SHEET_TABS["capacity"])
    wanted = ["FECHA","SVC","RUTAS_MLP_SDD","RUTAS_MLP_SPOT","RUTAS_RENTALS","RUTAS_CROWD_CAP","SHIPMENTS_DC_SP"]
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

    # Normaliza a minúsculas y asegura tipo texto
    dm_norm     = df["DELIVERY_MODEL"].fillna("").astype(str).str.lower()
    tipo_norm   = df["TIPO"].fillna("").astype(str).str.lower()
    tipodm_norm = df["TIPO_DM"].fillna("").astype(str).str.lower()

    # Máscaras
    is_rentals      = dm_norm.str.contains("rent",  regex=False)
    is_crowd_routes = dm_norm.str.contains("crowd", regex=False) & (
                        tipo_norm.str.contains("route",  regex=False) |
                        tipodm_norm.str.contains("route", regex=False)
                      )
    is_mlp_spot     = dm_norm.str.contains("mlp",   regex=False) & tipodm_norm.str.contains("spot", regex=False)
    is_mlp_sdd      = dm_norm.str.contains("mlp",   regex=False) & (~is_mlp_spot) & (
                        tipodm_norm.str.contains("mlp",  regex=False) |
                        tipodm_norm.str.contains("sdd",  regex=False) |
                        (tipodm_norm == "")
                      )
    # NUEVO: shipments (Delivery Cells + Service Partners) a restar del FCST
    is_shipments    = tipo_norm.str.contains("ship", regex=False)

    g = df.groupby(["FECHA","SVC"])["CANT"]
    agg = pd.DataFrame({
        "RUTAS_MLP_SDD":   g.apply(lambda s: s[is_mlp_sdd.loc[s.index]].sum()),
        "RUTAS_MLP_SPOT":  g.apply(lambda s: s[is_mlp_spot.loc[s.index]].sum()),
        "RUTAS_RENTALS":   g.apply(lambda s: s[is_rentals.loc[s.index]].sum()),
        "RUTAS_CROWD_CAP": g.apply(lambda s: s[is_crowd_routes.loc[s.index]].sum()),
        "SHIPMENTS_DC_SP": g.apply(lambda s: s[is_shipments.loc[s.index]].sum()),
    }).reset_index()

    return _finalize(agg, wanted)

def load_rentals_fallback() -> pd.DataFrame:
    """Si la pestaña Rentals no tiene SVC o está vacía, devolvemos un DF seguro sin agrupar."""
    df = read_sheet(SHEET_ID, SHEET_TABS["rentals"])
    target_cols = ["FECHA","SVC","RUTAS_RENTALS"]
    if df.empty:
        return pd.DataFrame(columns=target_cols)

    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","FACILITY","LC"], "SVC", required=False, source_label="Rentals")
    if "SVC" not in df.columns:
        out = pd.DataFrame(columns=target_cols)
        out["FECHA"] = date.today()
        return out

    find_and_rename(df, ["Unidades disponibles","Units","Cantidad","Qty","QTY","COUNT"], "CANT", required=False, source_label="Rentals")
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

# -----------------------------------------------------------------------------
# 5) Cálculo del plan
# -----------------------------------------------------------------------------
def compute_plan(spr_mode: str, sel_svcs: Optional[List[str]] = None) -> pd.DataFrame:
    fcst   = load_fcst()
    spr    = load_spr()
    caps   = load_capacity_caps()
    crowdc = load_crowd_caps()
    rents_fb = load_rentals_fallback()

    hoy = date.today()

    bases = []
    for d in [fcst, spr, caps, crowdc, rents_fb]:
        if "SVC" in d.columns and not d.empty:
            bases.append(d[["SVC"]])
    base = pd.concat(bases, axis=0).drop_duplicates() if bases else pd.DataFrame(columns=["SVC"])
    out = base.copy()
    out["FECHA"] = hoy

    if not fcst.empty:
        out = safe_merge(out, fcst[["SVC","FCST"]], ["SVC"])
    if not spr.empty:
        out = safe_merge(out, spr[["SVC","SPR_OBJ","SPR_PEAK","SPR_PROM"]], ["SVC"])

    # SPR usado
    spr_mode_col = {"promedio":"SPR_PROM", "peak":"SPR_PEAK", "plan":"SPR_OBJ"}.get(spr_mode, "SPR_PROM")
    out = ensure_columns(out, {"SPR_OBJ":np.nan, "SPR_PEAK":np.nan, "SPR_PROM":np.nan})
    spr_usado = out[spr_mode_col].where(out[spr_mode_col].notna(), out["SPR_OBJ"]).fillna(20)
    out["SPR_USADO"] = pd.to_numeric(spr_usado, errors="coerce").fillna(20).clip(lower=1)

    # Caps y shipments a descontar del FCST
    if not caps.empty:
        out = safe_merge(out, caps[["SVC","RUTAS_MLP_SDD","RUTAS_MLP_SPOT","RUTAS_RENTALS","RUTAS_CROWD_CAP","SHIPMENTS_DC_SP"]], ["SVC"])
    else:
        out = ensure_columns(out, {"RUTAS_MLP_SDD":0, "RUTAS_MLP_SPOT":0, "RUTAS_CROWD_CAP":0, "SHIPMENTS_DC_SP":0})

    if "RUTAS_RENTALS" not in out.columns or out["RUTAS_RENTALS"].isna().all():
        if not rents_fb.empty:
            out = safe_merge(out, rents_fb[["SVC","RUTAS_RENTALS"]], ["SVC"])
    out = ensure_columns(out, {"RUTAS_RENTALS":0})

    if not crowdc.empty:
        out = safe_merge(out, crowdc[["SVC","CROWD_E1_CAP"]], ["SVC"])
    out = ensure_columns(out, {"CROWD_E1_CAP":0})

    # DEMANDA = FCST - (Delivery Cells + Service Partners) shipments
    out = ensure_columns(out, {"FCST":0, "SHIPMENTS_DC_SP":0})
    out["FCST_NETO"] = (pd.to_numeric(out["FCST"], errors="coerce").fillna(0)
                        - pd.to_numeric(out["SHIPMENTS_DC_SP"], errors="coerce").fillna(0)).clip(lower=0)

    out["DEMANDA_AJUSTADA"] = out["FCST_NETO"]

    # Rutas base por SPR
    out["RUTAS_SPR_BASE"]   = np.ceil(out["DEMANDA_AJUSTADA"] / out["SPR_USADO"]).astype(int)

    # Deducciones
    for c in ["RUTAS_MLP_SDD","RUTAS_MLP_SPOT","RUTAS_RENTALS","RUTAS_CROWD_CAP","CROWD_E1_CAP"]:
        out[c] = pd.to_numeric(out.get(c, 0), errors="coerce").fillna(0)

    out["RUTAS_POST_RENTALS"] = (out["RUTAS_SPR_BASE"] - out["RUTAS_RENTALS"]).clip(lower=0)
    out["RUTAS_CROWD_BASE"]   = np.minimum(out["RUTAS_POST_RENTALS"], out["RUTAS_CROWD_CAP"]).astype(int)
    out["RUTAS_RESTANTES"]    = (out["RUTAS_POST_RENTALS"] - out["RUTAS_CROWD_BASE"]).clip(lower=0)

    out["RUTAS_POST_MLP"] = (out["RUTAS_RESTANTES"]
                             - out["RUTAS_MLP_SDD"]
                             - out["RUTAS_MLP_SPOT"]).clip(lower=0)

    out["RUTAS_CROWDE1_USADAS"] = np.minimum(out["RUTAS_POST_MLP"], out["CROWD_E1_CAP"]).astype(int)
    out["RUTAS_FALTANTES"]      = (out["RUTAS_POST_MLP"] - out["RUTAS_CROWDE1_USADAS"]).clip(lower=0)

    if sel_svcs:
        out = out[out["SVC"].isin(sel_svcs)]

    cols = ["SVC","FECHA","FCST","SHIPMENTS_DC_SP","FCST_NETO","DEMANDA_AJUSTADA","SPR_USADO","RUTAS_SPR_BASE",
            "RUTAS_RENTALS","RUTAS_CROWD_CAP","RUTAS_CROWD_BASE",
            "RUTAS_MLP_SDD","RUTAS_MLP_SPOT","CROWD_E1_CAP",
            "RUTAS_CROWDE1_USADAS","RUTAS_FALTANTES"]
    out = out.reindex(columns=cols).fillna(0).sort_values("SVC").reset_index(drop=True)
    return out

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
            cap_svcs     = load_capacity_caps()[["SVC"]]
            crowd_svcs   = load_crowd_caps()[["SVC"]]
            rent_fb_svcs = load_rentals_fallback()[["SVC"]]
            base_svcs = pd.concat([fcst_svcs, cap_svcs, crowd_svcs, rent_fb_svcs], axis=0).drop_duplicates()
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
    - FCST_NETO = FCST – **SHIPMENTS_DC_SP** (Delivery Cells + Service Partners detectados como *Shipments* en “Capacity”).
    - Se reporta `SHIPMENTS_DC_SP` y `FCST_NETO` en la tabla.
    - Resto de la lógica y robustecimientos se mantienen (headers dobles, alias, coerción numérica/fechas, etc.).
    """))
