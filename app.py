# app.py — Mel-IA Plan táctico (diario por SVC)
# Incluye: autodetección de pestañas, normalización de headers, coerción robusta,
# acepta URL o ID, healthcheck, no-crash, y filtro inicial en SGD1/SMT1/SMX9/SPB1.

import os, json, re, unicodedata, textwrap, traceback
from datetime import date
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# =======================
# Configuración / Secrets
# =======================
if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
elif "gcp_service_account" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps(dict(st.secrets["gcp_service_account"]))

SERVICE_EMAIL = "planificacion@planificacion.iam.gserviceaccount.com"

# 👉 SVCs preseleccionados
DEFAULT_SVCS = ["SGD1", "SMT1", "SMX9", "SPB1"]

# 👉 Tu Sheet por defecto (puedes pegar otro en la barra lateral)
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
    or DEFAULT_SHEET_URL  # ⬅️ usamos tu URL como valor por defecto
)

# =========
# Aliases
# =========
TAB_ALIASES: Dict[str, List[str]] = {
    "fcst":     ["FCST","Forecast","Pronostico","Pronóstico","Demanda","Demanda FCST","FCST diario"],
    "dc":       ["DC","Ajuste","Demand Correction","Correccion","Corrección"],
    "sp":       ["SP","Service Partner","Capacidad SP","Cap_SP","Partners"],
    "spr":      ["SPR","SPR objetivo","SPR plan","SPR obj","Objetivo SPR"],
    "rentals":  ["Rentals","Rentas","Rutas Rentals","Cap Rentals","MM Rentals"],
    "crowd":    ["Crowd","Base Crowd","Crowd base","% Crowd plan","Crowd %"],
    "mlp_sdd":  ["MLP_SDD","MLP SDD","SDD","Rutas MLP","MLP"],
    "crowd_e1": ["Crowd_E1","E1","Crowd extra","Crowd suplemento","Suplemento E1"],
}

# ==========================
# Normalización / Coerción
# ==========================
def _canon_name(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    return re.sub(r"[ \-_/\.]", "", s).lower()

def _col_heuristic_match(col: str, bucket: str) -> bool:
    c = _canon_name(col)
    if bucket == "SVC": return any(k in c for k in ["svc","svcs","facility","centro","logistic","lc"])
    if bucket == "FECHA": return any(k in c for k in ["fecha","date","opdt","dia","day","dispatch"])
    if bucket == "FCST": return any(k in c for k in ["fcst","forecast","pronostic","demanda","volumenplan","planvol"])
    if bucket == "DC": return any(k in c for k in ["dc","correction","ajuste","corr"])
    if bucket == "SP": return any(k in c for k in ["sp","servicepartner","capsp","partner"])
    if bucket in {"SPR_OBJ","SPR_PEAK","SPR_PROM"}:
        return "spr" in c or "objetivo" in c or "avg" in c or "prom" in c or "peak" in c or "pico" in c
    if bucket == "RUTAS_RENTALS": return any(k in c for k in ["renta","rentals","routes","rutas","caprutas"])
    if bucket in {"RUTAS_MLP_SDD","RUTAS_MLP_SPOT"}: return any(k in c for k in ["sdd","spot","mlp"])
    if bucket == "CROWD_BASE_PCT": return any(k in c for k in ["crowd","%","pct","porc","base"])
    if bucket == "CROWD_E1": return any(k in c for k in ["e1","extra","suplemento"])
    return False

def find_and_rename(df, candidates, new_name, required=True, source_label=""):
    cmap = {_canon_name(c): c for c in df.columns}
    for cand in candidates:
        key = _canon_name(cand)
        if key in cmap:
            real = cmap[key]
            if real != new_name: df.rename(columns={real: new_name}, inplace=True)
            return new_name
    for col in df.columns:  # heurística
        if _col_heuristic_match(col, new_name):
            if col != new_name: df.rename(columns={col: new_name}, inplace=True)
            return new_name
    if required:
        raise ValueError(f"{source_label}: falta '{new_name}'. Encabezados: {list(df.columns)}")
    return None

def ensure_columns(df: pd.DataFrame, defaults: Dict[str, object]) -> pd.DataFrame:
    for c, v in defaults.items():
        if c not in df.columns: df[c] = v
    return df

_NUM_SEP_RE = re.compile(r"[ ,\u00A0]")
def _maybe_to_numeric(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "iufc": return s
    sample = s.dropna().astype(str).head(60)
    if sample.empty: return pd.to_numeric(s, errors="coerce")
    looks = 0
    for v in sample:
        v2 = _NUM_SEP_RE.sub("", v).replace("%","").replace("−","-")
        if re.fullmatch(r"-?\d+(\.\d+)?", v2): looks += 1
    if looks/ max(1,len(sample)) >= 0.75:
        s2 = s.astype(str).str.replace("%","", regex=False).str.replace("−","-", regex=False)
        s2 = s2.apply(lambda x: _NUM_SEP_RE.sub("", x) if x is not None else x)
        return pd.to_numeric(s2, errors="coerce")
    return s
def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns: df[c] = _maybe_to_numeric(df[c])
    return df

def coerce_date_column(df, candidates, new_name, source_label, required=False):
    col = find_and_rename(df, candidates, new_name, required=required, source_label=source_label)
    if col:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=False, infer_datetime_format=True).dt.date
        if df[col].notna().sum() == 0:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True, infer_datetime_format=True).dt.date
    return col

def safe_merge(left, right, on, how="left", suffixes=("_x","_y")):
    if right is None or right.empty: return left.copy()
    return left.merge(right, how=how, on=on, suffixes=suffixes)

def show_exception(e: Exception, title: str):
    with st.expander(f"⚠️ {title}", expanded=False):
        st.code("".join(traceback.format_exception(None, e, e.__traceback__)))

# ======================
# Google Sheets helpers
# ======================
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
def _list_worksheets(sheet_id: str) -> List[str]:
    import gspread
    gc = _get_gspread_client()
    sh = gc.open_by_key(sheet_id)
    return [w.title for w in sh.worksheets()]

def _open_ws(sheet_id: str, tab_name: str):
    gc = _get_gspread_client()
    sh = gc.open_by_key(sheet_id)
    return sh.worksheet(tab_name)

def _discover_tab(sheet_id: str, key: str, aliases: List[str], column_targets: Dict[str, List[str]]) -> Optional[str]:
    titles = _list_worksheets(sheet_id)
    for a in aliases:
        if a in titles: return a
    # heurística por encabezados
    for t in titles:
        try:
            ws = _open_ws(sheet_id, t)
            vals = ws.get_all_values()
            if not vals: continue
            header = [h.strip() for h in (vals[0] if vals else [])]
            hits = 0
            for bucket, cands in column_targets.items():
                ok = False
                for h in header:
                    if _canon_name(h) in {_canon_name(x) for x in cands} or _col_heuristic_match(h, bucket):
                        ok = True; break
                if ok: hits += 1
            if hits >= max(2, len(column_targets)-1):
                return t
        except Exception:
            continue
    return None

@st.cache_data(show_spinner=False, ttl=300)
def read_sheet(sheet_id: str, tab_name: str) -> pd.DataFrame:
    sheet_id = sanitize_sheet_id(sheet_id)
    ws = _open_ws(sheet_id, tab_name)
    values = ws.get_all_values()
    if not values: return pd.DataFrame()
    df = pd.DataFrame(values[1:], columns=values[0])
    return coerce_numeric_df(df)

def quick_healthcheck(sheet_id: str) -> Dict[str, str]:
    sheet_id = sanitize_sheet_id(sheet_id)
    out = {"sheet_id": sheet_id or "", "ok": "false", "note": ""}
    try:
        titles = _list_worksheets(sheet_id)
        out["ok"] = "true"
        out["note"] = f"Pestañas: {', '.join(titles[:10])}" + ("…" if len(titles) > 10 else "")
    except Exception as e:
        out["note"] = f"{e}"
    return out

# ==============
# Resolver tabs
# ==============
def resolve_tab(sheet_id: str, key: str) -> Optional[str]:
    titles = _list_worksheets(sheet_id)
    aliases = TAB_ALIASES.get(key, [key])
    targets = {
        "fcst":     {"SVC":["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID"], "FECHA":["FECHA","DATE","OP_DT","SHP_DATE_DISPATCHED_ID"], "FCST":["FCST","FORECAST","PRONOSTICO","PRONÓSTICO","VOLUMEN_PLAN","PLAN"]},
        "dc":       {"SVC":["SVC","LOGISTIC_CENTER_ID"], "FECHA":["FECHA","DATE","OP_DT"], "DC":["DC","AJUSTE","CORRECCION","CORRECCIÓN","DEMAND_CORRECTION"]},
        "sp":       {"SVC":["SVC","LOGISTIC_CENTER_ID"], "FECHA":["FECHA","DATE","OP_DT"], "SP":["SP","SERVICE_PARTNER","CAP_SP","CAPACIDAD_SP"]},
        "spr":      {"SVC":["SVC","LOGISTIC_CENTER_ID"], "FECHA":["FECHA","DATE","OP_DT"], "SPR_OBJ":["SPR","SPR_OBJ","SPR objetivo","SPR plan"], "SPR_PEAK":["SPR_PEAK","SPR_PICO"], "SPR_PROM":["SPR_PROM","SPR_AVG","SPR_PROMEDIO"]},
        "rentals":  {"SVC":["SVC","SVCs","SVC/SVCs","LOGISTIC_CENTER_ID","SHP_LG_FACILITY_ID","FACILITY","LC"], "FECHA":["FECHA","DATE","OP_DT"], "RUTAS_RENTALS":["RUTAS","RUTAS_PLAN","ROUTES_PLAN","CAP_RUTAS","RENTALS_ROUTES"]},
        "crowd":    {"SVC":["SVC","LOGISTIC_CENTER_ID"], "FECHA":["FECHA","DATE","OP_DT"], "CROWD_BASE_PCT":["CROWD_BASE","CROWD_BASE_%","%CROWD","CROWD_PCT_PLAN","CROWD"]},
        "mlp_sdd":  {"SVC":["SVC","LOGISTIC_CENTER_ID"], "FECHA":["FECHA","DATE","OP_DT"], "RUTAS_MLP_SDD":["SDD","RUTAS_SDD","MLP_SDD","RUTAS_MLP_SDD"], "RUTAS_MLP_SPOT":["SPOT","RUTAS_SPOT","MLP_SPOT","RUTAS_MLP_SPOT"]},
        "crowd_e1": {"SVC":["SVC","LOGISTIC_CENTER_ID"], "FECHA":["FECHA","DATE","OP_DT"], "CROWD_E1":["E1","CROWD_E1","CROWD_SUPLEMENTO","CROWD_EXTRA"]},
    }
    for a in aliases:
        if a in titles: return a
    return _discover_tab(sheet_id, key, aliases, targets[key])

# =========
# Loaders
# =========
def load_fcst() -> pd.DataFrame:
    tab = resolve_tab(SHEET_ID, "fcst")
    if not tab: return pd.DataFrame(columns=["FECHA","SVC","FCST"])
    df = read_sheet(SHEET_ID, tab)
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","FCST"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT","SHP_DATE_DISPATCHED_ID"], "FECHA", f"FCST[{tab}]", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID","CENTRO","FACILITY","LC"], "SVC", False, f"FCST[{tab}]")
    find_and_rename(df, ["FCST","FORECAST","PRONOSTICO","PRONÓSTICO","VOLUMEN_PLAN","PLAN","VOL_PLAN"], "FCST", False, f"FCST[{tab}]")
    df = ensure_columns(df, {"SVC":None, "FCST":0})
    return df[["FECHA","SVC","FCST"]].copy()

def load_dc() -> pd.DataFrame:
    tab = resolve_tab(SHEET_ID, "dc")
    if not tab: return pd.DataFrame(columns=["FECHA","SVC","DC"])
    df = read_sheet(SHEET_ID, tab)
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","DC"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", f"DC[{tab}]", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID"], "SVC", False, f"DC[{tab}]")
    find_and_rename(df, ["DC","AJUSTE","CORRECCION","CORRECCIÓN","DEMAND_CORRECTION"], "DC", False, f"DC[{tab}]")
    df = ensure_columns(df, {"SVC":None, "DC":0})
    return df[["FECHA","SVC","DC"]].copy()

def load_sp() -> pd.DataFrame:
    tab = resolve_tab(SHEET_ID, "sp")
    if not tab: return pd.DataFrame(columns=["FECHA","SVC","SP"])
    df = read_sheet(SHEET_ID, tab)
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","SP"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", f"SP[{tab}]", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID"], "SVC", False, f"SP[{tab}]")
    find_and_rename(df, ["SP","SERVICE_PARTNER","CAP_SP","CAPACIDAD_SP","CAPACITY_SP"], "SP", False, f"SP[{tab}]")
    df = ensure_columns(df, {"SVC":None, "SP":0})
    return df[["FECHA","SVC","SP"]].copy()

def load_spr() -> pd.DataFrame:
    tab = resolve_tab(SHEET_ID, "spr")
    if not tab: return pd.DataFrame(columns=["FECHA","SVC","SPR_OBJ","SPR_PEAK","SPR_PROM"])
    df = read_sheet(SHEET_ID, tab)
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","SPR_OBJ","SPR_PEAK","SPR_PROM"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", f"SPR[{tab}]", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID"], "SVC", False, f"SPR[{tab}]")
    find_and_rename(df, ["SPR","SPR_OBJ","SPR objetivo","SPR plan","OBJ_SPR"], "SPR_OBJ", False, f"SPR[{tab}]")
    find_and_rename(df, ["SPR_PEAK","SPR_PICO","PICO"], "SPR_PEAK", False, f"SPR[{tab}]")
    find_and_rename(df, ["SPR_PROM","SPR_AVG","SPR_PROMEDIO","PROMEDIO"], "SPR_PROM", False, f"SPR[{tab}]")
    df = ensure_columns(df, {"SVC":None, "SPR_OBJ":np.nan, "SPR_PEAK":np.nan, "SPR_PROM":np.nan})
    return df[["FECHA","SVC","SPR_OBJ","SPR_PEAK","SPR_PROM"]].copy()

def load_rentals() -> pd.DataFrame:
    tab = resolve_tab(SHEET_ID, "rentals")
    if not tab: return pd.DataFrame(columns=["FECHA","SVC","RUTAS_RENTALS"])
    df = read_sheet(SHEET_ID, tab)
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","RUTAS_RENTALS"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", f"Rentals[{tab}]", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID","FACILITY","LC","CENTRO"], "SVC", False, f"Rentals[{tab}]")
    find_and_rename(df, ["RUTAS","RUTAS_PLAN","ROUTES_PLAN","CAP_RUTAS","RENTALS_ROUTES","RUTAS_RENTALS"], "RUTAS_RENTALS", False, f"Rentals[{tab}]")
    df = ensure_columns(df, {"SVC":None, "RUTAS_RENTALS":0})
    return df[["FECHA","SVC","RUTAS_RENTALS"]].copy()

def load_crowd() -> pd.DataFrame:
    tab = resolve_tab(SHEET_ID, "crowd")
    if not tab: return pd.DataFrame(columns=["FECHA","SVC","CROWD_BASE_PCT"])
    df = read_sheet(SHEET_ID, tab)
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","CROWD_BASE_PCT"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", f"Crowd[{tab}]", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID"], "SVC", False, f"Crowd[{tab}]")
    find_and_rename(df, ["CROWD_BASE","CROWD_BASE_%","%CROWD","CROWD_PCT_PLAN","CROWD","BASE_CROWD","PCT_CROWD"], "CROWD_BASE_PCT", False, f"Crowd[{tab}]")
    df["CROWD_BASE_PCT"] = pd.to_numeric(df.get("CROWD_BASE_PCT", 0), errors="coerce").fillna(0)
    if (df["CROWD_BASE_PCT"] > 1).mean() > 0.7: df["CROWD_BASE_PCT"] = (df["CROWD_BASE_PCT"]/100).clip(0,1)
    df = ensure_columns(df, {"SVC":None})
    return df[["FECHA","SVC","CROWD_BASE_PCT"]].copy()

def load_mlp_sdd() -> pd.DataFrame:
    tab = resolve_tab(SHEET_ID, "mlp_sdd")
    if not tab: return pd.DataFrame(columns=["FECHA","SVC","RUTAS_MLP_SDD","RUTAS_MLP_SPOT"])
    df = read_sheet(SHEET_ID, tab)
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","RUTAS_MLP_SDD","RUTAS_MLP_SPOT"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", f"MLP_SDD[{tab}]", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID"], "SVC", False, f"MLP_SDD[{tab}]")
    find_and_rename(df, ["SDD","RUTAS_SDD","MLP_SDD","RUTAS_MLP_SDD"], "RUTAS_MLP_SDD", False, f"MLP_SDD[{tab}]")
    find_and_rename(df, ["SPOT","RUTAS_SPOT","MLP_SPOT","RUTAS_MLP_SPOT"], "RUTAS_MLP_SPOT", False, f"MLP_SDD[{tab}]")
    df = ensure_columns(df, {"SVC":None, "RUTAS_MLP_SDD":0, "RUTAS_MLP_SPOT":0})
    return df[["FECHA","SVC","RUTAS_MLP_SDD","RUTAS_MLP_SPOT"]].copy()

def load_crowd_e1() -> pd.DataFrame:
    tab = resolve_tab(SHEET_ID, "crowd_e1")
    if not tab: return pd.DataFrame(columns=["FECHA","SVC","CROWD_E1"])
    df = read_sheet(SHEET_ID, tab)
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","CROWD_E1"])
    coerce_date_column(df, ["FECHA","DATE","OP_DT"], "FECHA", f"Crowd_E1[{tab}]", required=False)
    find_and_rename(df, ["SVC","SVCs","SVC/SVCs","SHP_LG_FACILITY_ID","LOGISTIC_CENTER_ID"], "SVC", False, f"Crowd_E1[{tab}]")
    find_and_rename(df, ["E1","CROWD_E1","CROWD_SUPLEMENTO","CROWD_EXTRA","EXTRA"], "CROWD_E1", False, f"Crowd_E1[{tab}]")
    df = ensure_columns(df, {"SVC":None, "CROWD_E1":0})
    return df[["FECHA","SVC","CROWD_E1"]].copy()

# ==============
# Cálculo plan
# ==============
def compute_plan(spr_mode: str, sel_svcs: Optional[List[str]] = None) -> pd.DataFrame:
    fcst, dc, sp = load_fcst(), load_dc(), load_sp()
    spr, rentals, crowd = load_spr(), load_rentals(), load_crowd()
    mlp_sdd, crowd_e1 = load_mlp_sdd(), load_crowd_e1()
    hoy = date.today()

    frames = [fcst, dc, sp, spr, rentals, crowd, mlp_sdd, crowd_e1]
    fixed = []
    for d in frames:
        if "FECHA" in d.columns and not d.empty:
            d = d.dropna(subset=["FECHA"]); d = d[d["FECHA"] <= hoy]
        fixed.append(d)
    fcst, dc, sp, spr, rentals, crowd, mlp_sdd, crowd_e1 = fixed

    bases = [x[["SVC"]].drop_duplicates() for x in fixed if "SVC" in x.columns and not x.empty]
    base = pd.concat(bases, axis=0).drop_duplicates() if bases else pd.DataFrame(columns=["SVC"])
    out = base.copy(); out["FECHA"] = hoy

    if not fcst.empty: out = safe_merge(out, fcst.groupby("SVC", as_index=False)["FCST"].sum(), ["SVC"])
    if not dc.empty:   out = safe_merge(out, dc.groupby("SVC", as_index=False)["DC"].sum(), ["SVC"])
    if not sp.empty:   out = safe_merge(out, sp.groupby("SVC", as_index=False)["SP"].sum(), ["SVC"])

    spr_mode_col = {"promedio":"SPR_PROM","peak":"SPR_PEAK","plan":"SPR_OBJ"}[spr_mode]
    if not spr.empty:
        spr_tmp = spr.groupby("SVC", as_index=False).agg({"SPR_OBJ":"max","SPR_PEAK":"max","SPR_PROM":"max"})
        out = safe_merge(out, spr_tmp, ["SVC"])

    if not rentals.empty:  out = safe_merge(out, rentals.groupby("SVC", as_index=False)["RUTAS_RENTALS"].sum(), ["SVC"])
    if not crowd.empty:    out = safe_merge(out, crowd.groupby("SVC", as_index=False)["CROWD_BASE_PCT"].max(), ["SVC"])
    if not mlp_sdd.empty:  out = safe_merge(out, mlp_sdd.groupby("SVC", as_index=False)[["RUTAS_MLP_SDD","RUTAS_MLP_SPOT"]].sum(), ["SVC"])
    if not crowd_e1.empty: out = safe_merge(out, crowd_e1.groupby("SVC", as_index=False)["CROWD_E1"].sum(), ["SVC"])

    out = ensure_columns(out, {"FCST":0,"DC":0,"SP":0,"SPR_OBJ":np.nan,"SPR_PEAK":np.nan,"SPR_PROM":np.nan,
                               "RUTAS_RENTALS":0,"CROWD_BASE_PCT":0,"RUTAS_MLP_SDD":0,"RUTAS_MLP_SPOT":0,"CROWD_E1":0})

    out["DEMANDA_AJUSTADA"] = (pd.to_numeric(out["FCST"], errors="coerce").fillna(0)
                               - pd.to_numeric(out["DC"], errors="coerce").fillna(0)
                               - pd.to_numeric(out["SP"], errors="coerce").fillna(0)).clip(lower=0)

    spr_usado = out[spr_mode_col].where(out[spr_mode_col].notna(), out["SPR_OBJ"]).fillna(20)
    out["SPR_USADO"] = pd.to_numeric(spr_usado, errors="coerce").fillna(20).clip(lower=1)

    out["RUTAS_SPR_BASE"]    = np.ceil(out["DEMANDA_AJUSTADA"] / out["SPR_USADO"]).astype(int)
    out["RUTAS_POST_RENTALS"] = (out["RUTAS_SPR_BASE"] - pd.to_numeric(out["RUTAS_RENTALS"], errors="coerce").fillna(0)).clip(lower=0)

    pct = pd.to_numeric(out["CROWD_BASE_PCT"], errors="coerce").fillna(0); pct = np.where(pct>1, pct/100.0, pct); pct = np.clip(pct,0,1)
    out["RUTAS_CROWD_BASE"] = np.ceil(out["RUTAS_POST_RENTALS"] * pct).astype(int)

    out["RUTAS_RESTANTES"] = (out["RUTAS_POST_RENTALS"] - pd.to_numeric(out["RUTAS_CROWD_BASE"], errors="coerce").fillna(0)).clip(lower=0)
    out["RUTAS_POST_MLP"] = (out["RUTAS_RESTANTES"]
                             - pd.to_numeric(out["RUTAS_MLP_SDD"], errors="coerce").fillna(0)
                             - pd.to_numeric(out["RUTAS_MLP_SPOT"], errors="coerce").fillna(0)).clip(lower=0)
    e1 = pd.to_numeric(out["CROWD_E1"], errors="coerce").fillna(0)
    out["RUTAS_CROWDE1_USADAS"] = np.minimum(out["RUTAS_POST_MLP"], e1).astype(int)
    out["RUTAS_FALTANTES"] = (out["RUTAS_POST_MLP"] - out["RUTAS_CROWDE1_USADAS"]).clip(lower=0)

    if sel_svcs: out = out[out["SVC"].isin(sel_svcs)]
    cols = ["SVC","FECHA","FCST","DC","SP","DEMANDA_AJUSTADA","SPR_USADO","RUTAS_SPR_BASE",
            "RUTAS_RENTALS","CROWD_BASE_PCT","RUTAS_CROWD_BASE","RUTAS_MLP_SDD","RUTAS_MLP_SPOT",
            "CROWD_E1","RUTAS_CROWDE1_USADAS","RUTAS_FALTANTES"]
    return out.reindex(columns=cols).fillna(0).sort_values("SVC").reset_index(drop=True)

# =========
#   UI
# =========
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

# Previene NameError
run_btn = False
auto_run = False
sel_svcs: List[str] = []

with st.expander("▶️ Cargando datos...", expanded=True):
    try:
        if not SHEET_ID:
            st.warning("Falta SHEET_ID. Pégalo en la barra lateral.")
            svc_list = []
        else:
            rentals_svcs = load_rentals()[["SVC"]]
            fcst_svcs    = load_fcst()[["SVC"]]
            crowd_svcs   = load_crowd()[["SVC"]]
            mlp_svcs     = load_mlp_sdd()[["SVC"]]
            base_svcs = pd.concat([rentals_svcs, fcst_svcs, crowd_svcs, mlp_svcs], axis=0).drop_duplicates()
            svc_list = sorted(base_svcs["SVC"].dropna().astype(str).unique().tolist())

        # 👉 por defecto deja seleccionados SGD1, SMT1, SMX9, SPB1 si existen
        default_sel = [s for s in DEFAULT_SVCS if s in svc_list] or svc_list[:4]
        sel_svcs = st.multiselect("Filtrar SVC", options=svc_list, default=default_sel, placeholder="Selecciona SVCs")

        st.write(" ")
        run_btn = st.button("Calcular plan", type="primary")
    except Exception as e:
        st.error("No se pudieron preparar los filtros.")
        show_exception(e, "Detalles (filtros)")

# Auto-ejecuta en el primer render con esos 4 SVC si existen
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
    - Acepta URL o ID de Google Sheet (se extrae automáticamente).
    - Autodescubre pestañas por alias y por encabezados.
    - Normalización de `SVC` (SVC/SVCs/LOGISTIC_CENTER_ID, etc.) y del resto de columnas.
    - Coerción de números/fechas/porcentajes; defaults si faltan columnas.
    - Filtro inicial fijo: **SGD1, SMT1, SMX9, SPB1**.
    """))
