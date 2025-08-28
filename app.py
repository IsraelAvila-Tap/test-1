# app.py
# =============================================================================
# Mel-IA — Plan táctico (diario por SVC)
# Pipeline: FCST − DC − SP → SPR → Rentals → Crowd base (% plan) → MLP SDD/Spot → Crowd E1
# Blindado con:
# - Normalización de columnas (SVC, fechas, modelos, etc.)
# - Evita caídas: valida, completa faltantes, avisa y sigue
# - Cache de lectura (Streamlit 1.49+)
# - Parametrización por Secrets/ENV
# =============================================================================
import os, json, io, textwrap, unicodedata, traceback
from datetime import datetime, date
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 0) Credenciales GCP desde Secrets o ENV (ambos soportados)
# -----------------------------------------------------------------------------
if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
elif "gcp_service_account" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps(dict(st.secrets["gcp_service_account"]))

# Sheet ID (obligatorio)
SHEET_ID = (
    st.secrets.get("SHEET_ID")
    or os.environ.get("SHEET_ID")
    or os.environ.get("PROJECT_SHEET_ID")
    or st.secrets.get("PROJECT_SHEET_ID")
)

# Nombre de worksheet por dataset (personalízalo si tus tabs tienen otros nombres)
SHEET_TABS = {
    "fcst": st.secrets.get("TAB_FCST", "FCST"),
    "dc": st.secrets.get("TAB_DC", "DC"),
    "sp": st.secrets.get("TAB_SP", "SP"),
    "spr": st.secrets.get("TAB_SPR", "SPR"),
    "rentals": st.secrets.get("TAB_RENTALS", "Rentals"),
    "crowd": st.secrets.get("TAB_CROWD", "Crowd"),
    "mlp_sdd": st.secrets.get("TAB_MLP_SDD", "MLP_SDD"),
    "crowd_e1": st.secrets.get("TAB_CROWD_E1", "Crowd_E1"),
}

# -----------------------------------------------------------------------------
# 1) Utilidades de normalización y renombres tolerantes
# -----------------------------------------------------------------------------
def _canon_name(s: str) -> str:
    """Normaliza nombres: sin acentos, minúsculas, sin espacios/_/-/./'/'."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    for ch in [" ", "_", "-", ".", "/"]:
        s = s.replace(ch, "")
    return s.lower()

def find_and_rename(df: pd.DataFrame,
                    candidates: List[str],
                    new_name: str,
                    required: bool = True,
                    source_label: str = "") -> Optional[str]:
    """
    Busca en df.columns cualquiera de los 'candidates' (acepta variantes como SVC/SVCs, svcs, etc.)
    y renombra la primera que encuentre a `new_name`. Si no encuentra:
      - required=True: levanta ValueError (con etiqueta de fuente)
      - required=False: no hace nada (devuelve None)
    """
    canon_map = {_canon_name(c): c for c in df.columns}
    for cand in candidates:
        cn = _canon_name(cand)
        if cn in canon_map:
            real = canon_map[cn]
            if real != new_name:
                df.rename(columns={real: new_name}, inplace=True)
            return new_name
    if required:
        msg = f"{source_label}: falta columna equivalente a '{'/'.join(candidates)}'. Columnas disponibles: {list(df.columns)}"
        raise ValueError(msg)
    return None

def ensure_columns(df: pd.DataFrame, cols: Dict[str, float | int | str]) -> pd.DataFrame:
    """
    Crea columnas que falten con un valor por defecto.
    cols = {"RUTAS_PLAN": 0, "SPR_OBJ": np.nan, ...}
    """
    for c, v in cols.items():
        if c not in df.columns:
            df[c] = v
    return df

def coerce_date_column(df: pd.DataFrame, candidates: List[str], new_name: str, source_label: str,
                       required: bool = False) -> Optional[str]:
    """
    Renombra columna de fecha y la parsea a date. Admite varios nombres.
    """
    col = find_and_rename(df, candidates, new_name, required=required, source_label=source_label)
    if col:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return col

def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: List[str], how: str = "left", suffixes=("_x", "_y")):
    """Merge tolerante: si right está vacío, devuelve left intacto."""
    if right is None or right.empty:
        return left.copy()
    return left.merge(right, how=how, on=on, suffixes=suffixes)

def show_exception(e: Exception, title: str = "Error"):
    with st.expander(f"⚠️ {title}", expanded=False):
        st.code("".join(traceback.format_exception(None, e, e.__traceback__)))

# -----------------------------------------------------------------------------
# 2) Lectura Google Sheets (gspread + pandas)
# -----------------------------------------------------------------------------
def _get_gspread_client():
    import gspread
    from google.oauth2.service_account import Credentials
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise RuntimeError("No hay credenciales en GOOGLE_SERVICE_ACCOUNT_JSON.")
    info = json.loads(raw)
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)

@st.cache_data(show_spinner=False, ttl=300)
def read_sheet(sheet_id: str, tab_name: str) -> pd.DataFrame:
    """Lee una pestaña de Google Sheets a DataFrame."""
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(sheet_id)
        ws = sh.worksheet(tab_name)
        values = ws.get_all_values()
        if not values:
            return pd.DataFrame()
        header, rows = values[0], values[1:]
        df = pd.DataFrame(rows, columns=header)
        # Try to infer numeric cols
        for c in df.columns:
            # Si toda la col es numérica o vacía, conviértela
            if df[c].str.match(r"^-?\d+(\.\d+)?$").fillna(False).all():
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception as e:
        raise RuntimeError(f"Fallo al leer '{tab_name}': {e}")

# -----------------------------------------------------------------------------
# 3) Loaders por dataset con normalización de columnas
#    Todos devuelven al menos: ['FECHA','SVC', ...] cuando aplica
# -----------------------------------------------------------------------------
def load_fcst() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["fcst"])
    if df.empty:
        return pd.DataFrame(columns=["FECHA","SVC","FCST"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT","SHP_DATE_DISPATCHED_ID"], "FECHA", "FCST", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID"], "SVC", source_label="FCST")
    # Pronóstico esperado en col FCST (o similar)
    find_and_rename(df, ["FCST","FORECAST","PRONOSTICO","VOLUMEN_PLAN"], "FCST", required=False, source_label="FCST")
    df = ensure_columns(df, {"FCST": 0})
    return df[["FECHA","SVC","FCST"]].copy()

def load_dc() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["dc"])
    if df.empty:
        return pd.DataFrame(columns=["FECHA","SVC","DC"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", "DC", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID"], "SVC", source_label="DC")
    find_and_rename(df, ["DC","DESCARGA","DEMAND_CORRECTION","AJUSTE"], "DC", required=False, source_label="DC")
    df = ensure_columns(df, {"DC": 0})
    return df[["FECHA","SVC","DC"]].copy()

def load_sp() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["sp"])
    if df.empty:
        return pd.DataFrame(columns=["FECHA","SVC","SP"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", "SP", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID"], "SVC", source_label="SP")
    find_and_rename(df, ["SP","SERVICE_PARTNER","CAP_SP","CAPACIDAD_SP"], "SP", required=False, source_label="SP")
    df = ensure_columns(df, {"SP": 0})
    return df[["FECHA","SVC","SP"]].copy()

def load_spr() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["spr"])
    if df.empty:
        return pd.DataFrame(columns=["FECHA","SVC","SPR_OBJ"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", "SPR", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID"], "SVC", source_label="SPR")
    find_and_rename(df, ["SPR","SPR_OBJ","SPR objetivo","SPR objetivo (plan)"], "SPR_OBJ", required=False, source_label="SPR")
    df = ensure_columns(df, {"SPR_OBJ": np.nan})
    return df[["FECHA","SVC","SPR_OBJ"]].copy()

def load_rentals() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["rentals"])
    if df.empty:
        return pd.DataFrame(columns=["FECHA","SVC","RUTAS_RENTALS"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", "Rentals", required=False)
    find_and_rename(
        df,
        ["SVC","SVCs","SVC/SVCs","SVC SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID","FACILITY","LC","CENTRO_LOGISTICO"],
        "SVC",
        source_label="Rentals"
    )
    find_and_rename(df, ["RUTAS","RUTAS_PLAN","ROUTES_PLAN","CAP_RUTAS","RENTALS_ROUTES"], "RUTAS_RENTALS",
                    required=False, source_label="Rentals")
    df = ensure_columns(df, {"RUTAS_RENTALS": 0})
    return df[["FECHA","SVC","RUTAS_RENTALS"]].copy()

def load_crowd() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["crowd"])
    if df.empty:
        return pd.DataFrame(columns=["FECHA","SVC","CROWD_BASE_PCT"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", "Crowd", required=False)
    find_and_rename(df, ["SVC","svc","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID"], "SVC", source_label="Crowd")
    find_and_rename(df, ["CROWD_BASE","CROWD_BASE_%","%CROWD","CROWD_PCT_PLAN"], "CROWD_BASE_PCT",
                    required=False, source_label="Crowd")
    df["CROWD_BASE_PCT"] = pd.to_numeric(df.get("CROWD_BASE_PCT", 0), errors="coerce").fillna(0).clip(0, 1)
    return df[["FECHA","SVC","CROWD_BASE_PCT"]].copy()

def load_mlp_sdd() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["mlp_sdd"])
    if df.empty:
        return pd.DataFrame(columns=["FECHA","SVC","RUTAS_MLP_SDD","RUTAS_MLP_SPOT"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", "MLP_SDD", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID"], "SVC", source_label="MLP_SDD")
    find_and_rename(df, ["SDD","RUTAS_SDD","MLP_SDD","RUTAS_MLP_SDD"], "RUTAS_MLP_SDD", required=False, source_label="MLP_SDD")
    find_and_rename(df, ["SPOT","RUTAS_SPOT","MLP_SPOT","RUTAS_MLP_SPOT"], "RUTAS_MLP_SPOT", required=False, source_label="MLP_SDD")
    df = ensure_columns(df, {"RUTAS_MLP_SDD": 0, "RUTAS_MLP_SPOT": 0})
    return df[["FECHA","SVC","RUTAS_MLP_SDD","RUTAS_MLP_SPOT"]].copy()

def load_crowd_e1() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["crowd_e1"])
    if df.empty:
        return pd.DataFrame(columns=["FECHA","SVC","CROWD_E1"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", "Crowd_E1", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID"], "SVC", source_label="Crowd_E1")
    find_and_rename(df, ["E1","CROWD_E1","CROWD_SUPLEMENTO","CROWD_EXTRA"], "CROWD_E1", required=False, source_label="Crowd_E1")
    df = ensure_columns(df, {"CROWD_E1": 0})
    return df[["FECHA","SVC","CROWD_E1"]].copy()

# -----------------------------------------------------------------------------
# 4) Lógica de cálculo de plan
# -----------------------------------------------------------------------------
def compute_plan(spr_mode: str, sel_svcs: Optional[List[str]] = None) -> pd.DataFrame:
    """
    spr_mode: 'promedio' | 'peak' | 'plan'
    Junta datasets y calcula rutas objetivo por SVC.
    """
    # Carga datasets
    fcst = load_fcst()
    dc = load_dc()
    sp = load_sp()
    spr = load_spr()
    rentals = load_rentals()
    crowd = load_crowd()
    mlp_sdd = load_mlp_sdd()
    crowd_e1 = load_crowd_e1()

    # Día objetivo = hoy (ajusta si usas otra fecha en tus tabs)
    hoy = date.today()

    # Filtra a hoy si FECHA existe
    for df in [fcst, dc, sp, spr, rentals, crowd, mlp_sdd, crowd_e1]:
        if "FECHA" in df.columns and not df.empty:
            df.dropna(subset=["FECHA"], inplace=True)
            df = df[df["FECHA"] <= hoy]  # permitir histórico <= hoy
        # Reasigna al nombre original
        if df is fcst: fcst = df
        elif df is dc: dc = df
        elif df is sp: sp = df
        elif df is spr: spr = df
        elif df is rentals: rentals = df
        elif df is crowd: crowd = df
        elif df is mlp_sdd: mlp_sdd = df
        elif df is crowd_e1: crowd_e1 = df

    # Base de SVC con FCST (si no hay, usa la union de lo que exista)
    bases = [d[["SVC"]].drop_duplicates() for d in [fcst, rentals, crowd, mlp_sdd, crowd_e1] if not d.empty]
    base = pd.concat(bases, axis=0).drop_duplicates() if bases else pd.DataFrame(columns=["SVC"])

    # Merge incremental
    out = base.copy()
    out["FECHA"] = hoy
    out = safe_merge(out, fcst.groupby("SVC", as_index=False)["FCST"].sum(), on=["SVC"])
    out = safe_merge(out, dc.groupby("SVC", as_index=False)["DC"].sum(), on=["SVC"])
    out = safe_merge(out, sp.groupby("SVC", as_index=False)["SP"].sum(), on=["SVC"])
    # SPR objetivo: según modo
    spr_mode_col = {"promedio": "SPR_PROM", "peak": "SPR_PEAK", "plan": "SPR_OBJ"}[spr_mode]
    # Si tu hoja solo trae SPR_OBJ, las otras se quedarán NaN y usaremos fallback
    spr_tmp = spr.copy()
    find_and_rename(spr_tmp, ["SPR_PEAK","SPR_PICO"], "SPR_PEAK", required=False, source_label="SPR")
    find_and_rename(spr_tmp, ["SPR_PROM","SPR_AVG","SPR_PROMEDIO"], "SPR_PROM", required=False, source_label="SPR")
    spr_tmp = spr_tmp.groupby("SVC", as_index=False).agg({"SPR_OBJ":"max","SPR_PEAK":"max","SPR_PROM":"max"})
    out = safe_merge(out, spr_tmp, on=["SVC"])

    # Rentals, Crowd, MLP_SDD, Crowd E1
    out = safe_merge(out, rentals.groupby("SVC", as_index=False)["RUTAS_RENTALS"].sum(), on=["SVC"])
    out = safe_merge(out, crowd.groupby("SVC", as_index=False)["CROWD_BASE_PCT"].max(), on=["SVC"])
    out = safe_merge(out, mlp_sdd.groupby("SVC", as_index=False)[["RUTAS_MLP_SDD","RUTAS_MLP_SPOT"]].sum(), on=["SVC"])
    out = safe_merge(out, crowd_e1.groupby("SVC", as_index=False)["CROWD_E1"].sum(), on=["SVC"])

    # Defaults
    out = ensure_columns(out, {
        "FCST": 0, "DC": 0, "SP": 0,
        "SPR_OBJ": np.nan, "SPR_PEAK": np.nan, "SPR_PROM": np.nan,
        "RUTAS_RENTALS": 0, "CROWD_BASE_PCT": 0, "RUTAS_MLP_SDD": 0, "RUTAS_MLP_SPOT": 0, "CROWD_E1": 0
    })

    # Ajuste demanda base: FCST − DC − SP
    out["DEMANDA_AJUSTADA"] = (out["FCST"] - out["DC"] - out["SP"]).clip(lower=0)

    # Selección de SPR objetivo según modo, con fallback a SPR_OBJ
    out["SPR_USADO"] = out[spr_mode_col].where(out[spr_mode_col].notna(), out["SPR_OBJ"])
    # Si sigue NaN, asume 20 (fallback ultra conservador)
    out["SPR_USADO"] = out["SPR_USADO"].fillna(20)

    # Rutas base por SPR
    out["RUTAS_SPR_BASE"] = np.ceil(out["DEMANDA_AJUSTADA"] / out["SPR_USADO"]).astype(int)

    # Suma Rentals primero
    out["RUTAS_POST_RENTALS"] = (out["RUTAS_SPR_BASE"] - out["RUTAS_RENTALS"]).clip(lower=0)

    # Aplicar base de Crowd como porcentaje del plan (sobre rutas restantes)
    out["RUTAS_CROWD_BASE"] = np.ceil(out["RUTAS_POST_RENTALS"] * out["CROWD_BASE_PCT"]).astype(int)
    out["RUTAS_RESTANTES"] = (out["RUTAS_POST_RENTALS"] - out["RUTAS_CROWD_BASE"]).clip(lower=0)

    # MLP SDD y Spot (si están definidos, se sostienen con prioridad)
    out["RUTAS_POST_MLP"] = (out["RUTAS_RESTANTES"] - out["RUTAS_MLP_SDD"] - out["RUTAS_MLP_SPOT"]).clip(lower=0)

    # Crowd E1 como colchón final si sobran
    out["RUTAS_CROWDE1_USADAS"] = np.minimum(out["RUTAS_POST_MLP"], out["CROWD_E1"]).astype(int)
    out["RUTAS_FALTANTES"] = (out["RUTAS_POST_MLP"] - out["RUTAS_CROWDE1_USADAS"]).clip(lower=0)

    # Resultado ordenado
    out.fillna(0, inplace=True)
    out = out[[
        "SVC","FECHA",
        "FCST","DC","SP","DEMANDA_AJUSTADA",
        "SPR_USADO","RUTAS_SPR_BASE",
        "RUTAS_RENTALS",
        "CROWD_BASE_PCT","RUTAS_CROWD_BASE",
        "RUTAS_MLP_SDD","RUTAS_MLP_SPOT",
        "CROWD_E1","RUTAS_CROWDE1_USADAS",
        "RUTAS_FALTANTES"
    ]].sort_values("SVC")

    # Filtrado por SVC (chips)
    if sel_svcs:
        out = out[out["SVC"].isin(sel_svcs)]

    return out.reset_index(drop=True)

# -----------------------------------------------------------------------------
# 5) UI Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Mel-IA — Plan táctico (diario por SVC)", layout="wide")

st.sidebar.markdown("## 🗂️ Proyecto")
if SHEET_ID:
    st.sidebar.markdown(f"**Sheet:** `{SHEET_ID}`")
else:
    st.sidebar.error("Configura `SHEET_ID` en Secrets/ENV.")

st.sidebar.markdown("## 🔐 Credenciales")
st.sidebar.code("Comparte el Sheet con:\nplanificacion@planificacion.iam.gserviceaccount.com")

st.title("Mel-IA — Plan táctico (diario por SVC)")

spr_mode = st.radio("SPR objetivo", options=["promedio","peak","plan"], horizontal=True, index=0)

with st.expander("▶️ Cargando datos...", expanded=True):
    try:
        # Cargar minimal para filtros (SVCs desde cualquiera)
        rentals_svcs = load_rentals()[["SVC"]]
        fcst_svcs = load_fcst()[["SVC"]]
        crowd_svcs = load_crowd()[["SVC"]]
        mlp_svcs = load_mlp_sdd()[["SVC"]]
        base_svcs = pd.concat([rentals_svcs, fcst_svcs, crowd_svcs, mlp_svcs], axis=0).drop_duplicates()
        svc_list = sorted(base_svcs["SVC"].dropna().unique().tolist())
        default_sel = [s for s in ["SGD1","SMT1","SMX9","SPB1"] if s in svc_list] or svc_list[:4]

        sel_svcs = st.multiselect("Filtrar SVC", options=svc_list, default=default_sel)

        st.write(" ")
        run_btn = st.button("Calcular plan", type="primary")
    except Exception as e:
        st.error("No se pudieron preparar los filtros.")
        show_exception(e, "Detalles (filtros)")

result_placeholder = st.empty()

if 'auto_run_once' not in st.session_state:
    st.session_state['auto_run_once'] = True
    auto_run = True
else:
    auto_run = False

if run_btn or auto_run:
    try:
        plan = compute_plan(spr_mode, sel_svcs or None)
        if plan.empty:
            st.warning("No hay datos para mostrar con los filtros seleccionados.")
        else:
            # Métricas rápidas
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("SVCs", value=plan["SVC"].nunique())
            with col2:
                st.metric("Demanda ajustada", value=int(plan["DEMANDA_AJUSTADA"].sum()))
            with col3:
                st.metric("Rutas (SPR base)", value=int(plan["RUTAS_SPR_BASE"].sum()))
            with col4:
                st.metric("Rutas faltantes", value=int(plan["RUTAS_FALTANTES"].sum()))

            st.dataframe(plan, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(str(e))
        show_exception(e, "Traceback completo")

# -----------------------------------------------------------------------------
# 6) Notas / Ayuda
# -----------------------------------------------------------------------------
with st.expander("ℹ️ Notas de esta versión"):
    st.markdown(textwrap.dedent("""
    - **Normalización de columnas** automática para evitar errores por encabezados como `SVC/SVCs`, `SVCs`, `SHP_LG_FACILITY_ID`, etc.
    - Si falta un dataset o una columna secundaria, se completa con `0`/`NaN` y **no se cae** la app.
    - `SPR objetivo` usa: **promedio** → `SPR_PROM`, **peak** → `SPR_PEAK`, **plan** → `SPR_OBJ`.
      Si no existe esa columna, hace *fallback* a `SPR_OBJ` y, si tampoco existe, usa `20`.
    - El orden del pipeline es: **SPR base** → **Rentals** → **Crowd base** → **MLP SDD/Spot** → **Crowd E1**.
    """))

