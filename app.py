# app.py
# =============================================================================
# Mel-IA — Plan táctico (diario por SVC)
# Carga estable (gspread), headers únicos, soporte a encabezados combinados (2 filas),
# normalización de columnas y pipeline robusto (sin caídas por columnas faltantes).
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

# Tabs FIJAS (overrideables en Secrets)
SHEET_TABS = {
    "fcst":     st.secrets.get("TAB_FCST", "FCST"),
    "dc":       st.secrets.get("TAB_DC", "DC"),
    "sp":       st.secrets.get("TAB_SP", "SP"),
    "spr":      st.secrets.get("TAB_SPR", "SPR"),
    "rentals":  st.secrets.get("TAB_RENTALS", "Rentals"),
    "crowd":    st.secrets.get("TAB_CROWD", "Crowd"),
    "mlp_sdd":  st.secrets.get("TAB_MLP_SDD", "MLP_SDD"),
    "crowd_e1": st.secrets.get("TAB_CROWD_E1", "Crowd_E1"),
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

_NUM_SEP_RE = re.compile(r"[ ,\u00A0]")  # espacio, NBSP, coma

def _maybe_to_numeric(series_like):
    """Soporta Series o DataFrame (por si hay headers duplicados)."""
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
        try:
            return pd.to_numeric(s, errors="coerce")
        except Exception:
            return s

    looks_numeric = 0
    for v in sample:
        v2 = _NUM_SEP_RE.sub("", v).replace("%","").replace("−","-")
        if re.fullmatch(r"-?\d+(\.\d+)?", v2):
            looks_numeric += 1
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
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=False, infer_datetime_format=True).dt.date
        if df[col].notna().sum() == 0:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True, infer_datetime_format=True).dt.date
    return col

def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: List[str], how="left", suffixes=("_x","_y")):
    if right is None or right.empty:
        return left.copy()
    return left.merge(right, how=how, on=on, suffixes=suffixes)

def show_exception(e: Exception, title: str):
    with st.expander(f"⚠️ {title}", expanded=False):
        st.code("".join(traceback.format_exception(None, e, e.__traceback__)))

# -----------------------------------------------------------------------------
# 2) Headers únicos + encabezados combinados (2 filas)
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

# -----------------------------------------------------------------------------
# 3) Carga ESTABLE desde Google Sheets
# -----------------------------------------------------------------------------
def _get_gspread_client():
    import gspread
    from google.oauth2.service_account import Credentials
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise RuntimeError("Faltan credenciales en GOOGLE_SERVICE_ACCOUNT_JSON.")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(json.loads(raw), scopes=scopes)
    return gspread.authorize(creds)

@st.cache_data(show_spinner=False, ttl=300)
def read_sheet(sheet_id: str, tab_name: str) -> pd.DataFrame:
    """Lee una pestaña. Si detecta encabezados combinados (fila1 con muchos vacíos),
    usa fila1+fila2 para armar header. Siempre hace headers únicos y coerciona números."""
    gc = _get_gspread_client()
    sh = gc.open_by_key(sanitize_sheet_id(sheet_id))
    ws = sh.worksheet(tab_name)
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    # Detecta encabezados combinados
    if len(values) >= 2:
        r1 = values[0]
        empty_ratio = sum(1 for x in r1 if not x.strip()) / max(1, len(r1))
        if empty_ratio > 0.4:
            header = _combine_two_header_rows(values[0], values[1])
            data_rows = values[2:]
        else:
            header = values[0]
            data_rows = values[1:]
    else:
        header = values[0]
        data_rows = values[1:]

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
        out["note"] = f"Pestañas: {', '.join(titles[:8])}" + ("…" if len(titles) > 8 else "")
    except Exception as e:
        out["note"] = f"{e}"
    return out

# -----------------------------------------------------------------------------
# 4) Loaders (garantizan FECHA y columnas presentes)
# -----------------------------------------------------------------------------
def _finalize(df: pd.DataFrame, wanted: List[str]) -> pd.DataFrame:
    """Asegura FECHA y SVC existan; devuelve solo columnas existentes para evitar KeyError."""
    df = ensure_columns(df, {"FECHA": pd.NaT, "SVC": None})
    cols = [c for c in wanted if c in df.columns]
    return df[cols].copy() if cols else pd.DataFrame(columns=wanted)

def load_fcst() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["fcst"])
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","FCST"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT","SHP_DATE_DISPATCHED_ID"], "FECHA", "FCST", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID","CENTRO","FACILITY","LC"], "SVC", False, "FCST")
    find_and_rename(df, ["FCST","FORECAST","PRONOSTICO","PRONÓSTICO","VOLUMEN_PLAN","PLAN","VOL_PLAN"], "FCST", False, "FCST")
    df = ensure_columns(df, {"FCST":0})
    return _finalize(df, ["FECHA","SVC","FCST"])

def load_dc() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["dc"])
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","DC"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", "DC", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID"], "SVC", False, "DC")
    find_and_rename(df, ["DC","AJUSTE","CORRECCION","CORRECCIÓN","DEMAND_CORRECTION"], "DC", False, "DC")
    df = ensure_columns(df, {"DC":0})
    return _finalize(df, ["FECHA","SVC","DC"])

def load_sp() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["sp"])
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","SP"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", "SP", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID"], "SVC", False, "SP")
    find_and_rename(df, ["SP","SERVICE_PARTNER","CAP_SP","CAPACIDAD_SP","CAPACITY_SP"], "SP", False, "SP")
    df = ensure_columns(df, {"SP":0})
    return _finalize(df, ["FECHA","SVC","SP"])

def load_spr() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["spr"])
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","SPR_OBJ","SPR_PEAK","SPR_PROM"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", "SPR", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID"], "SVC", False, "SPR")
    find_and_rename(df, ["SPR","SPR_OBJ","SPR objetivo","SPR plan","OBJ_SPR"], "SPR_OBJ", False, "SPR")
    find_and_rename(df, ["SPR_PEAK","SPR_PICO","PICO"], "SPR_PEAK", False, "SPR")
    find_and_rename(df, ["SPR_PROM","SPR_AVG","SPR_PROMEDIO","PROMEDIO"], "SPR_PROM", False, "SPR")
    return _finalize(df, ["FECHA","SVC","SPR_OBJ","SPR_PEAK","SPR_PROM"])

def load_rentals() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["rentals"])
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","RUTAS_RENTALS"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", "Rentals", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID","FACILITY","LC","CENTRO"], "SVC", False, "Rentals")
    find_and_rename(df, ["RUTAS","RUTAS_PLAN","ROUTES_PLAN","CAP_RUTAS","RENTALS_ROUTES","RUTAS_RENTALS"], "RUTAS_RENTALS", False, "Rentals")
    df = ensure_columns(df, {"RUTAS_RENTALS":0})
    return _finalize(df, ["FECHA","SVC","RUTAS_RENTALS"])

def load_crowd() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["crowd"])
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","CROWD_BASE_PCT"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", "Crowd", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID","CENTRO","FACILITY","LC"], "SVC", False, "Crowd")
    find_and_rename(df, ["CROWD_BASE","CROWD_BASE_%","%CROWD","CROWD_PCT_PLAN","CROWD","BASE_CROWD","PCT_CROWD"], "CROWD_BASE_PCT", False, "Crowd")
    df["CROWD_BASE_PCT"] = pd.to_numeric(df.get("CROWD_BASE_PCT", 0), errors="coerce").fillna(0)
    if (df["CROWD_BASE_PCT"] > 1).mean() > 0.7:
        df["CROWD_BASE_PCT"] = (df["CROWD_BASE_PCT"]/100).clip(0,1)
    return _finalize(df, ["FECHA","SVC","CROWD_BASE_PCT"])

def load_mlp_sdd() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["mlp_sdd"])
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","RUTAS_MLP_SDD","RUTAS_MLP_SPOT"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", "MLP_SDD", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID","CENTRO","FACILITY","LC"], "SVC", False, "MLP_SDD")
    find_and_rename(df, ["SDD","RUTAS_SDD","MLP_SDD","RUTAS_MLP_SDD"], "RUTAS_MLP_SDD", False, "MLP_SDD")
    find_and_rename(df, ["SPOT","RUTAS_SPOT","MLP_SPOT","RUTAS_MLP_SPOT"], "RUTAS_MLP_SPOT", False, "MLP_SDD")
    df = ensure_columns(df, {"RUTAS_MLP_SDD":0, "RUTAS_MLP_SPOT":0})
    return _finalize(df, ["FECHA","SVC","RUTAS_MLP_SDD","RUTAS_MLP_SPOT"])

def load_crowd_e1() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["crowd_e1"])
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","CROWD_E1"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", "Crowd_E1", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID","CENTRO","FACILITY","LC"], "SVC", False, "Crowd_E1")
    find_and_rename(df, ["E1","CROWD_E1","CROWD_SUPLEMENTO","CROWD_EXTRA","EXTRA"], "CROWD_E1", False, "Crowd_E1")
    df = ensure_columns(df, {"CROWD_E1":0})
    return _finalize(df, ["FECHA","SVC","CROWD_E1"])

# -----------------------------------------------------------------------------
# 5) Cálculo del plan
# -----------------------------------------------------------------------------
def compute_plan(spr_mode: str, sel_svcs: Optional[List[str]] = None) -> pd.DataFrame:
    fcst     = load_fcst()
    dc       = load_dc()
    sp       = load_sp()
    spr      = load_spr()
    rentals  = load_rentals()
    crowd    = load_crowd()
    mlp_sdd  = load_mlp_sdd()
    crowd_e1 = load_crowd_e1()

    hoy = date.today()
    frames = [fcst, dc, sp, spr, rentals, crowd, mlp_sdd, crowd_e1]
    fixed = []
    for d in frames:
        if "FECHA" in d.columns and not d.empty:
            d = d.dropna(subset=["FECHA"])
            d = d[d["FECHA"] <= hoy]
        fixed.append(d)
    fcst, dc, sp, spr, rentals, crowd, mlp_sdd, crowd_e1 = fixed

    bases = [x[["SVC"]].drop_duplicates() for x in fixed if "SVC" in x.columns and not x.empty]
    base = pd.concat(bases, axis=0).drop_duplicates() if bases else pd.DataFrame(columns=["SVC"])
    out = base.copy()
    out["FECHA"] = hoy

    if not fcst.empty:
        out = safe_merge(out, fcst.groupby("SVC", as_index=False)["FCST"].sum(), ["SVC"])
    if not dc.empty:
        out = safe_merge(out, dc.groupby("SVC", as_index=False)["DC"].sum(), ["SVC"])
    if not sp.empty:
        out = safe_merge(out, sp.groupby("SVC", as_index=False)["SP"].sum(), ["SVC"])

    spr_mode_col = {"promedio":"SPR_PROM", "peak":"SPR_PEAK", "plan":"SPR_OBJ"}[spr_mode]
    if not spr.empty:
        spr_tmp = spr.groupby("SVC", as_index=False).agg({"SPR_OBJ":"max","SPR_PEAK":"max","SPR_PROM":"max"})
        out = safe_merge(out, spr_tmp, ["SVC"])

    if not rentals.empty:
        out = safe_merge(out, rentals.groupby("SVC", as_index=False)["RUTAS_RENTALS"].sum(), ["SVC"])
    if not crowd.empty:
        out = safe_merge(out, crowd.groupby("SVC", as_index=False)["CROWD_BASE_PCT"].max(), ["SVC"])
    if not mlp_sdd.empty:
        out = safe_merge(out, mlp_sdd.groupby("SVC", as_index=False)[["RUTAS_MLP_SDD","RUTAS_MLP_SPOT"]].sum(), ["SVC"])
    if not crowd_e1.empty:
        out = safe_merge(out, crowd_e1.groupby("SVC", as_index=False)["CROWD_E1"].sum(), ["SVC"])

    out = ensure_columns(out, {
        "FCST":0, "DC":0, "SP":0,
        "SPR_OBJ":np.nan, "SPR_PEAK":np.nan, "SPR_PROM":np.nan,
        "RUTAS_RENTALS":0, "CROWD_BASE_PCT":0, "RUTAS_MLP_SDD":0, "RUTAS_MLP_SPOT":0, "CROWD_E1":0
    })

    out["DEMANDA_AJUSTADA"] = (pd.to_numeric(out["FCST"], errors="coerce").fillna(0)
                               - pd.to_numeric(out["DC"], errors="coerce").fillna(0)
                               - pd.to_numeric(out["SP"], errors="coerce").fillna(0)).clip(lower=0)

    spr_usado = out[spr_mode_col].where(out[spr_mode_col].notna(), out["SPR_OBJ"]).fillna(20)
    out["SPR_USADO"] = pd.to_numeric(spr_usado, errors="coerce").fillna(20).clip(lower=1)

    out["RUTAS_SPR_BASE"]    = np.ceil(out["DEMANDA_AJUSTADA"] / out["SPR_USADO"]).astype(int)
    out["RUTAS_POST_RENTALS"] = (out["RUTAS_SPR_BASE"]
                                 - pd.to_numeric(out["RUTAS_RENTALS"], errors="coerce").fillna(0)).clip(lower=0)

    pct = pd.to_numeric(out["CROWD_BASE_PCT"], errors="coerce").fillna(0)
    pct = np.where(pct > 1, pct/100.0, pct)
    pct = np.clip(pct, 0, 1)
    out["RUTAS_CROWD_BASE"] = np.ceil(out["RUTAS_POST_RENTALS"] * pct).astype(int)

    out["RUTAS_RESTANTES"] = (out["RUTAS_POST_RENTALS"]
                              - pd.to_numeric(out["RUTAS_CROWD_BASE"], errors="coerce").fillna(0)).clip(lower=0)

    out["RUTAS_POST_MLP"] = (out["RUTAS_RESTANTES"]
                              - pd.to_numeric(out["RUTAS_MLP_SDD"], errors="coerce").fillna(0)
                              - pd.to_numeric(out["RUTAS_MLP_SPOT"], errors="coerce").fillna(0)).clip(lower=0)

    e1 = pd.to_numeric(out["CROWD_E1"], errors="coerce").fillna(0)
    out["RUTAS_CROWDE1_USADAS"] = np.minimum(out["RUTAS_POST_MLP"], e1).astype(int)
    out["RUTAS_FALTANTES"] = (out["RUTAS_POST_MLP"] - out["RUTAS_CROWDE1_USADAS"]).clip(lower=0)

    if sel_svcs:
        out = out[out["SVC"].isin(sel_svcs)]

    cols = ["SVC","FECHA","FCST","DC","SP","DEMANDA_AJUSTADA","SPR_USADO","RUTAS_SPR_BASE",
            "RUTAS_RENTALS","CROWD_BASE_PCT","RUTAS_CROWD_BASE","RUTAS_MLP_SDD","RUTAS_MLP_SPOT",
            "CROWD_E1","RUTAS_CROWDE1_USADAS","RUTAS_FALTANTES"]
    out = out.reindex(columns=cols).fillna(0).sort_values("SVC").reset_index(drop=True)
    return out

# -----------------------------------------------------------------------------
# 6) UI — estable, con filtros por defecto y auto-run
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

# Previene NameError y fija defaults de SVC
run_btn = False
auto_run = False
sel_svcs: List[str] = []

with st.expander("▶️ Cargando datos...", expanded=True):
    try:
        if not SHEET_ID:
            st.warning("Falta `SHEET_ID`. Pégalo en la barra lateral.")
            svc_list = []
        else:
            rentals_svcs = load_rentals()[["SVC"]]
            fcst_svcs    = load_fcst()[["SVC"]]
            crowd_svcs   = load_crowd()[["SVC"]]
            mlp_svcs     = load_mlp_sdd()[["SVC"]]
            base_svcs = pd.concat([rentals_svcs, fcst_svcs, crowd_svcs, mlp_svcs], axis=0).drop_duplicates()
            svc_list = sorted(base_svcs["SVC"].dropna().astype(str).unique().tolist())

        default_sel = [s for s in DEFAULT_SVCS if s in svc_list] or svc_list[:4]
        sel_svcs = st.multiselect("Filtrar SVC", options=svc_list, default=default_sel, placeholder="Selecciona SVCs")
        st.write(" ")
        run_btn = st.button("Calcular plan", type="primary")
    except Exception as e:
        st.error("No se pudieron preparar los filtros.")
        show_exception(e, "Detalles (filtros)")

# Auto-run 1a vez
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
    - Carga estable con gspread + saneo de URL/ID.
    - Headers únicos y soporte a encabezados combinados (fila1+fila2) — útil para Crowd.
    - Normalización de columnas; coerción de números/porcentajes/fechas.
    - Fix: todos los loaders garantizan **FECHA** y devuelven solo columnas existentes (adiós `KeyError: ['FECHA']`).
    - Filtro inicial: **SGD1, SMT1, SMX9, SPB1** y auto-run al inicio.
    """))
