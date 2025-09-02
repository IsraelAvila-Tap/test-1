# app.py
# =============================================================================
# Mel-IA — Plan táctico (diario por SVC)  — versión SIN sidebar
# - Pestañas: FCST, SPR, Capacity, Rentals, Crowd, SRM
# - Crowd determinista: UNA columna según (día: entre semana/sábado/domingo) y (escenario: Base o E1).
# - Muestra en la tabla las columnas crudas de Crowd y la columna usada.
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
    "srm":      "SRM",     # capacidad MLP
}

# -----------------------------------------------------------------------------
# 1) Normalización / utilidades
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

def show_exception(e: Exception, title: str):
    with st.expander(f"⚠️ {title}", expanded=False):
        st.code("".join(traceback.format_exception(None, e, e.__traceback__)))

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
# 2) Lectura Sheets (con autodetección de encabezado y posible 2 filas)
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
    if not values:
        return pd.DataFrame()

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
            if sum(1 for x in values[i] if x.strip()) >= 2:
                header_idx = i
                break
    if header_idx is None:
        header_idx = 0

    # ¿combinar con la fila anterior o siguiente?
    r1 = values[header_idx]
    r1_lower = [c.strip().lower() for c in r1]
    combine_prev = False
    combine_next = False

    if header_idx > 0:
        prev = values[header_idx - 1]
        prev_lower = [c.strip().lower() for c in prev]
        if _looks_group_header(prev_lower):
            combine_prev = True

    if not combine_prev and (header_idx + 1 < len(values)):
        r2 = values[header_idx + 1]
        r2_nonempty = sum(1 for x in r2 if x.strip())
        if _looks_group_header(r1_lower) and r2_nonempty >= max(2, len(r2)//4):
            combine_next = True

    if combine_prev:
        header = _combine_two_header_rows(values[header_idx - 1], values[header_idx])
        data_rows = values[header_idx + 1:]
    elif combine_next:
        header = _combine_two_header_rows(values[header_idx], values[header_idx + 1])
        data_rows = values[header_idx + 2:]
    else:
        header = values[header_idx]
        data_rows = values[header_idx + 1:]

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
# 3) Loaders (FCST, SPR, Rentals, Capacity, SRM, etc.)
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

def coerce_date_column(df: pd.DataFrame, candidates: List[str], new_name: str,
                       source_label: str, required: bool = False) -> Optional[str]:
    col = find_and_rename(df, candidates, new_name, required=required, source_label=source_label)
    if col:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True, infer_datetime_format=True).dt.date
        if df[col].notna().sum() == 0:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=False, infer_datetime_format=True).dt.date
    return col

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

# ---- Rentals (hist ponderado si hay vehículo; si no, suma simple) ----------
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
        # heurística suave
        def _find_units_like_column(df: pd.DataFrame) -> str | None:
            if df is None or df.empty: return None
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

# ---- Capacity caps (MLP/Crowd caps + Shipments DC/SP) ----
def load_capacity_caps() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["capacity"])
    wanted = ["FECHA","SVC","RUTAS_MLP_SDD","RUTAS_MLP_SPOT","RUTAS_RENTALS","RUTAS_CROWD_CAP","SHIPMENTS_DC","SHIPMENTS_SP"]
    if df.empty:
        return pd.DataFrame(columns=wanted)

    find_and_rename(df, ["Delivery model","Deliverymodel","Model","DM"], "DELIVERY_MODEL", False, "Capacity")
    find_and_rename(df, ["Tipo","Type","Category"], "TIPO", False, "Capacity")
    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","FACILITY","LC"], "SVC", False, "Capacity")
    find_and_rename(df, ["Tipo DM","TipoDM","DM Type"], "TIPO_DM", False, "Capacity)
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

# ---- SRM MLP caps ----
def load_mlp_caps_from_srm() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["srm"])
    out_cols = [
        "SVC",
        "MLP_SDD_LV","MLP_SDD_SV","MLP_SDD_CAR","MLP_SDD_CAP",
        "MLP_SPOT_LV","MLP_SPOT_SV","MLP_SPOT_CAR","MLP_SPOT_CAP",
        "MLP_BACK_CAP",
    ]
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", required=False, source_label="SRM")

    def canon_col(name: str) -> str:
        c = _canon_name(name).replace("h&b","hb")
        c = re.sub(r"w\d+", "", c)
        c = re.sub(r"\d+$", "", c)
        return c

    canon = {c: canon_col(c) for c in df.columns}

    def pick_cols(type_tokens: list[str], family_tokens: list[str], exclude_tokens: list[str]) -> list[str]:
        sel = []
        for col, cc in canon.items():
            if cc in ("svc","svcs","logisticcenterid","facility","lc"): continue
            if all(ft in cc for ft in family_tokens) and any(t in cc for t in type_tokens) and not any(t in cc for t in exclude_tokens):
                sel.append(col)
        return sel

    def pick_cols_any(family_tokens: list[str], include_any: list[str], exclude_tokens: list[str]) -> list[str]:
        sel = []
        for col, cc in canon.items():
            if cc in ("svc","svcs","logisticcenterid","facility","lc"): continue
            if all(ft in cc for ft in family_tokens) and any(t in cc for t in include_any) and not any(t in cc for t in exclude_tokens):
                sel.append(col)
        return sel

    LV  = ["largevan","largev","large","lv","xlarge","xlv","heavybulky","hb"]
    SV  = ["smallvan","small","sv"]
    CAR = ["car","auto"]
    EXC_TOTAL = ["total"]
    EXC_BACK  = ["bu","backup","back","backlog"]
    BACK_ANY  = ["bu","backup","back","backlog"]

    sdd_lv_cols   = pick_cols(LV,  ["sdd"], EXC_TOTAL + [])
    sdd_sv_cols   = pick_cols(SV,  ["sdd"], EXC_TOTAL + [])
    sdd_car_cols  = pick_cols(CAR, ["sdd"], EXC_TOTAL + [])

    spot_lv_cols  = pick_cols(LV,  ["spot"], EXC_TOTAL + EXC_BACK)
    spot_sv_cols  = pick_cols(SV,  ["spot"], EXC_TOTAL + EXC_BACK)
    spot_car_cols = pick_cols(CAR, ["spot"], EXC_TOTAL + EXC_BACK)

    back_cols     = pick_cols_any(["spot"], BACK_ANY, EXC_TOTAL)

    for c in set(sdd_lv_cols + sdd_sv_cols + sdd_car_cols +
                 spot_lv_cols + spot_sv_cols + spot_car_cols + back_cols):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    grp = df.groupby("SVC", dropna=False)

    def sum_cols(cols: list[str]) -> pd.Series:
        if not cols:
            return grp.size().mul(0)
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

    return out[out_cols]

# ---- SPR de MLP ----
def load_spr_mlp() -> pd.DataFrame:
    spr = read_sheet(SHEET_ID, SHEET_TABS["spr"])
    if spr.empty:
        return pd.DataFrame(columns=["SVC","SPR_MLP"])
    find_and_rename(spr, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", False, "SPR")
    find_and_rename(spr, ["SPR","spr","Ships per route"], "SPR", False, "SPR")

    spr = ensure_columns(spr, {"SVC":None, "SPR":np.nan})
    spr["SPR"] = pd.to_numeric(spr["SPR"], errors="coerce")
    spr = _as_str_cols(spr, ["SVC"])

    # detecta filas que correspondan a MLP/SDD/SPOT/BACK
    cols_texto = [c for c in spr.columns if c not in ["SVC","SPR"]]
    mask = pd.Series(False, index=spr.index)
    for c in cols_texto:
        mask |= spr[c].astype(str).str.lower().str.contains("mlp|sdd|spot|back", regex=True)
    mlp_rows = spr[mask]
    if mlp_rows.empty:
        return pd.DataFrame(columns=["SVC","SPR_MLP"])
    grp_local = mlp_rows.groupby("SVC")["SPR"].median().rename("SPR_MLP").reset_index()
    grp_local["SPR_MLP"] = grp_local["SPR_MLP"].fillna(mlp_rows["SPR"].median())
    return grp_local

# ---- Crowd % (objetivo) desde Capacity ----
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
    _as_str_cols(df, ["SVC", "DELIVERY_MODEL", "TIPO", "TIPO_DM"])

    if df["FECHA"].notna().any():
        last_by_svc = df.groupby("SVC")["FECHA"].transform("max")
        df = df[df["FECHA"] == last_by_svc]

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

# ---- Crowd determinista: toma UNA columna según día & escenario --------------
def load_crowd_caps_for(planning_date: date, escenario: str) -> pd.DataFrame:
    """
    Espera layout fijo de Crowd:
      SVC | Base entre semana | Base sabado | Base domingo | Holgura entre semana | Holgura sabado | Holgura domingo | [...]
    Si escenario == "Base" -> usa Base (weekday/sab/dom según fecha).
    Si escenario == "E1"   -> usa Holgura (weekday/sab/dom según fecha).
    Devuelve además las 6 columnas crudas (para mostrarlas en la tabla).
    """
    df = read_sheet(SHEET_ID, SHEET_TABS["crowd"])
    cols_out = ["SVC","BASE_ENTRE_SEM","BASE_SAB","BASE_DOM","E1_ENTRE_SEM","E1_SAB","E1_DOM","RUTAS_CROWD_CAP"]
    if df.empty: return pd.DataFrame(columns=cols_out)

    # SVC
    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","FACILITY","LC"], "SVC", False, "Crowd")
    _as_str_cols(df, ["SVC"])

    # Renombres exactos (con y sin acentos)
    mapping = {
        "BASE_ENTRE_SEM": ["Base entre semana","Base entre sem","Base weekday","Base entre"],
        "BASE_SAB":       ["Base sabado","Base sábado","Base sab"],
        "BASE_DOM":       ["Base domingo","Base dom"],
        "E1_ENTRE_SEM":   ["Holgura entre semana","E1 entre semana","Holgura entre","E1 entre"],
        "E1_SAB":         ["Holgura sabado","Holgura sábado","E1 sabado","E1 sábado","E1 sab"],
        "E1_DOM":         ["Holgura domingo","E1 domingo","Holgura dom","E1 dom"],
    }
    for new, cands in mapping.items():
        find_and_rename(df, cands, new, required=False, source_label="Crowd")

    # Asegura numérico y rellena faltantes
    for c in ["BASE_ENTRE_SEM","BASE_SAB","BASE_DOM","E1_ENTRE_SEM","E1_SAB","E1_DOM"]:
        if c not in df.columns: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Selección determinista
    dow = planning_date.weekday()  # 0=Lunes ... 6=Domingo
    esc = (escenario or "Base").strip().lower()
    if esc == "e1":
        col_usada = "E1_ENTRE_SEM" if dow < 5 else ("E1_SAB" if dow == 5 else "E1_DOM")
    else:
        col_usada = "BASE_ENTRE_SEM" if dow < 5 else ("BASE_SAB" if dow == 5 else "BASE_DOM")

    out = df[["SVC","BASE_ENTRE_SEM","BASE_SAB","BASE_DOM","E1_ENTRE_SEM","E1_SAB","E1_DOM"]].copy()
    out["RUTAS_CROWD_CAP"] = pd.to_numeric(df[col_usada], errors="coerce").fillna(0).astype(int)
    return out

# -----------------------------------------------------------------------------
# 4) Cálculo del plan
# -----------------------------------------------------------------------------
def apply_output_adjustments(resumen: pd.DataFrame) -> pd.DataFrame:
    resumen = resumen.drop(columns=["Demanda esperada", "DEMANDA_ESPERADA"], errors="ignore")
    orden = [
        "SVC","FECHA",
        "FCST","SHIPMENTS_DC","SHIPMENTS_SP","FCST (sin DC & sin SP)","DEMANDA_AJUSTADA",
        "SPR_USADO","RUTAS_SPR_BASE",
        "RUTAS_RENTALS","SPR_RENTALS","SHIP_RENTALS",

        # --- Crowd (mostrar crudo + usado) ---
        "BASE_ENTRE_SEM","BASE_SAB","BASE_DOM","E1_ENTRE_SEM","E1_SAB","E1_DOM",
        "CROWD_PCT","SPR_CROWD","SHIP_OBJ_CROWD","RUTAS_CROWD_OBJ",
        "RUTAS_CROWD_CAP","RUTAS_CROWD_ASIGNADAS","SHIP_CROWD",

        # Antes de MLP
        "SHIP_RESTANTES_PRE_MLP",

        # MLP (capacidad por tipo y totales SRM)
        "SPR_MLP",
        "MLP_SDD_LV","MLP_SDD_SV","MLP_SDD_CAR","MLP_SDD_CAP",
        "MLP_SPOT_LV","MLP_SPOT_SV","MLP_SPOT_CAR","MLP_SPOT_CAP",
        "MLP_BACK_CAP",

        # Asignación MLP
        "RUTAS_MLP_NEEDED","RUTAS_MLP_SDD","RUTAS_MLP_SPOT","RUTAS_MLP_BACKLOG",

        # Remanentes/Faltantes
        "RUTAS_RESTANTES","RUTAS_FALTANTES",

        # Capacidad total y riesgo
        "CAP_TOTAL","CAP_VS_FCST","CAP_DIFF_ABS","RIESGO",
    ]
    cols = [c for c in orden if c in resumen.columns] + [c for c in resumen.columns if c not in orden]
    return resumen[cols]

def compute_plan(spr_mode: str, planning_date: date, crowd_escenario: str, sel_svcs: Optional[List[str]] = None) -> pd.DataFrame:
    fcst       = load_fcst()
    spr        = load_spr()
    caps       = load_capacity_caps()
    rents      = load_rentals_caps_from_sheet()
    rents_fb   = load_rentals_fallback()
    crowd_pct  = load_crowd_pct_from_capacity()     # % objetivo Crowd
    spr_crowd  = load_spr_crowd()                   # SPR específico para Crowd
    crowd_det  = load_crowd_caps_for(planning_date, crowd_escenario)  # determinista
    mlp_caps   = load_mlp_caps_from_srm()
    spr_mlp    = load_spr_mlp()

    for d in (fcst, spr, caps, rents, rents_fb, crowd_pct, spr_crowd, crowd_det, mlp_caps, spr_mlp):
        if not d.empty and "SVC" in d.columns:
            _as_str_cols(d, ["SVC"])

    # universo base SVCs
    bases = []
    for d in [fcst, spr, caps, rents, rents_fb, crowd_pct, crowd_det, mlp_caps]:
        if "SVC" in d.columns and not d.empty:
            bases.append(d[["SVC"]])
    base = pd.concat(bases, axis=0).drop_duplicates() if bases else pd.DataFrame(columns=["SVC"])
    base = _as_str_cols(base, ["SVC"])

    out = base.copy()
    out["FECHA"] = planning_date

    # FCST + SPR base
    if not fcst.empty: out = out.merge(fcst[["SVC","FCST"]], on="SVC", how="left")
    if not spr.empty:  out = out.merge(spr[["SVC","SPR_OBJ","SPR_PEAK","SPR_PROM"]], on="SVC", how="left")
    spr_mode_col = {"promedio":"SPR_PROM", "peak":"SPR_PEAK", "plan":"SPR_OBJ"}.get(spr_mode, "SPR_PROM")
    out = ensure_columns(out, {"SPR_OBJ":np.nan, "SPR_PEAK":np.nan, "SPR_PROM":np.nan})
    spr_usado = out[spr_mode_col].where(out[spr_mode_col].notna(), out["SPR_OBJ"]).fillna(20)
    out["SPR_USADO"] = pd.to_numeric(spr_usado, errors="coerce").fillna(20).clip(lower=1)

    # DC / SP desde Capacity
    if not caps.empty:
        out = out.merge(caps[["SVC","SHIPMENTS_DC","SHIPMENTS_SP"]], on="SVC", how="left")
    out = ensure_columns(out, {"SHIPMENTS_DC":0, "SHIPMENTS_SP":0})

    # Rentals
    if not rents.empty:
        out = out.merge(rents[["SVC","RUTAS_RENTALS","SPR_RENTALS"]], on="SVC", how="left")
    elif not rents_fb.empty:
        out = out.merge(rents_fb[["SVC","RUTAS_RENTALS"]], on="SVC", how="left")
        out["SPR_RENTALS"] = np.nan
    else:
        out["RUTAS_RENTALS"] = 0
        out["SPR_RENTALS"]   = np.nan
    out["RUTAS_RENTALS"] = pd.to_numeric(out.get("RUTAS_RENTALS", 0), errors="coerce").fillna(0).astype(int)
    out["SPR_RENTALS"]   = pd.to_numeric(out.get("SPR_RENTALS", np.nan), errors="coerce")
    out["SPR_RENTALS"]   = out["SPR_RENTALS"].fillna(out["SPR_USADO"]).clip(lower=1)

    # Crowd objetivo (%)
    if not crowd_pct.empty: out = out.merge(crowd_pct, on="SVC", how="left")
    out["CROWD_PCT"] = pd.to_numeric(out.get("CROWD_PCT", 0), errors="coerce").fillna(0).clip(0,1)

    # SPR específico crowd
    if not spr_crowd.empty: out = out.merge(spr_crowd, on="SVC", how="left")
    out["SPR_CROWD"] = pd.to_numeric(out.get("SPR_CROWD", np.nan), errors="coerce")
    out["SPR_CROWD"] = out["SPR_CROWD"].fillna(out["SPR_USADO"]).clip(lower=1)

    # FCST (sin DC & SP) y DEMANDA_AJUSTADA
    out = ensure_columns(out, {"FCST":0, "SHIPMENTS_DC":0, "SHIPMENTS_SP":0})
    out["FCST (sin DC & sin SP)"] = (
        pd.to_numeric(out["FCST"], errors="coerce").fillna(0)
        - pd.to_numeric(out["SHIPMENTS_DC"], errors="coerce").fillna(0)
        - pd.to_numeric(out["SHIPMENTS_SP"], errors="coerce").fillna(0)
    ).clip(lower=0)
    out["DEMANDA_AJUSTADA"] = out["FCST (sin DC & sin SP)"]

    # RUTAS_SPR_BASE
    out["RUTAS_SPR_BASE"] = np.ceil(out["DEMANDA_AJUSTADA"] / out["SPR_USADO"]).astype(int)

    # Crowd determinista: unir columnas crudas + cap usada
    if not crowd_det.empty:
        out = out.merge(crowd_det, on="SVC", how="left")
    for c in ["BASE_ENTRE_SEM","BASE_SAB","BASE_DOM","E1_ENTRE_SEM","E1_SAB","E1_DOM","RUTAS_CROWD_CAP"]:
        out[c] = pd.to_numeric(out.get(c, 0), errors="coerce").fillna(0).astype(int)

    out["SHIP_OBJ_CROWD"]  = pd.to_numeric(out["FCST"], errors="coerce").fillna(0) * out["CROWD_PCT"]
    out["RUTAS_CROWD_OBJ"] = np.ceil(out["SHIP_OBJ_CROWD"] / out["SPR_CROWD"]).astype(int)
    out["RUTAS_CROWD_ASIGNADAS"] = np.minimum(out["RUTAS_CROWD_OBJ"], out["RUTAS_CROWD_CAP"]).astype(int)
    out["SHIP_CROWD"] = out["RUTAS_CROWD_ASIGNADAS"] * out["SPR_CROWD"]

    # Shipments restantes pre-MLP
    base_otros = pd.to_numeric(out["SHIPMENTS_DC"], errors="coerce").fillna(0) + pd.to_numeric(out["SHIPMENTS_SP"], errors="coerce").fillna(0)
    out["SHIP_RENTALS"] = out["RUTAS_RENTALS"] * out["SPR_RENTALS"]
    out["SHIP_RESTANTES_PRE_MLP"] = (
        pd.to_numeric(out["FCST"], errors="coerce").fillna(0) - base_otros - out["SHIP_RENTALS"] - out["SHIP_CROWD"]
    ).clip(lower=0)

    # MLP caps + SPR_MLP
    if not mlp_caps.empty: out = out.merge(mlp_caps, on="SVC", how="left")
    if not spr_mlp.empty:  out = out.merge(spr_mlp,  on="SVC", how="left")
    out["SPR_MLP"] = pd.to_numeric(out.get("SPR_MLP", np.nan), errors="coerce").fillna(out["SPR_USADO"]).clip(lower=1)

    # Rutas MLP necesarias
    out["RUTAS_MLP_NEEDED"] = np.ceil(out["SHIP_RESTANTES_PRE_MLP"] / out["SPR_MLP"]).astype(int)

    # Asignación por tipo: SDD → SPOT → Backlog
    def alloc_types(need, lv, sv, car):
        use_lv = int(min(need, lv));   need -= use_lv
        use_sv = int(min(need, sv));   need -= use_sv
        use_car= int(min(need, car));  need -= use_car
        total  = use_lv + use_sv + use_car
        return total, int(max(0, need))

    uses = out.apply(lambda r: alloc_types(
        r["RUTAS_MLP_NEEDED"],
        int(r.get("MLP_SDD_LV",0)), int(r.get("MLP_SDD_SV",0)), int(r.get("MLP_SDD_CAR",0))
    ), axis=1, result_type="expand")
    out["RUTAS_MLP_SDD"], rem1 = uses[0], uses[1]

    uses2 = out.apply(lambda r: alloc_types(
        rem1.loc[r.name],
        int(r.get("MLP_SPOT_LV",0)), int(r.get("MLP_SPOT_SV",0)), int(r.get("MLP_SPOT_CAR",0))
    ), axis=1, result_type="expand")
    out["RUTAS_MLP_SPOT"], rem2 = uses2[0], uses2[1]

    out["RUTAS_MLP_BACKLOG"] = np.minimum(rem2, pd.to_numeric(out.get("MLP_BACK_CAP",0), errors="coerce").fillna(0)).astype(int)

    # Restantes / faltantes
    out["RUTAS_RESTANTES"] = (rem2 - out["RUTAS_MLP_BACKLOG"]).clip(lower=0).astype(int)
    out["RUTAS_FALTANTES"] = out["RUTAS_RESTANTES"]

    # Capacidad total vs FCST
    cap_mlp_ship = (out["RUTAS_MLP_SDD"] + out["RUTAS_MLP_SPOT"] + out["RUTAS_MLP_BACKLOG"]) * out["SPR_MLP"]
    out["CAP_TOTAL"] = (base_otros + out["SHIP_RENTALS"] + out["SHIP_CROWD"] + cap_mlp_ship)
    fcst_series = pd.to_numeric(out["FCST"], errors="coerce")
    out["CAP_VS_FCST"] = (out["CAP_TOTAL"] / fcst_series.replace(0, np.nan)).fillna(0)
    out["CAP_DIFF_ABS"] = (fcst_series.fillna(0) - out["CAP_TOTAL"]).abs()
    out["RIESGO"] = np.where(out["CAP_TOTAL"] + 1e-9 >= fcst_series.fillna(0), "OK", "RIESGO")

    if sel_svcs:
        out = out[out["SVC"].isin(sel_svcs)]

    out = apply_output_adjustments(out).fillna(0).sort_values("SVC").reset_index(drop=True)
    return out

# -----------------------------------------------------------------------------
# 5) UI (SIN sidebar): controles arriba + tabla
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Mel-IA — Plan táctico (diario por SVC)", layout="wide")
st.title("Mel-IA — Plan táctico (diario por SVC)")
st.caption(f"Comparte el Sheet con: `{SERVICE_EMAIL}` · Sheet ID detectado: `{SHEET_ID or '—'}`")

# Controles superiores
c1, c2, c3 = st.columns(3)
with c1:
    spr_mode = st.radio("SPR objetivo", options=["promedio","peak","plan"], horizontal=True, index=0)
with c2:
    planning_date = st.date_input("Fecha a planear", value=date.today(), format="YYYY-MM-DD")
with c3:
    crowd_esc = st.radio("Escenario Crowd", options=["Base","E1"], horizontal=True, index=0)

# Descubre SVCs disponibles de las pestañas (para multiselect)
try:
    fcst_svcs    = load_fcst()[["SVC"]]
    caps         = load_capacity_caps()
    cap_svcs     = caps[["SVC"]] if "SVC" in caps.columns else caps.to_frame(name="SVC")
    crowd_svcs   = load_crowd_caps_for(planning_date, crowd_esc)[["SVC"]]
    rents_svcs   = load_rentals_caps_from_sheet()[["SVC"]]
    rent_fb_svcs = load_rentals_fallback()[["SVC"]]
    mlp_svcs     = load_mlp_caps_from_srm()[["SVC"]]
    base_svcs = pd.concat([fcst_svcs, cap_svcs, crowd_svcs, rents_svcs, rent_fb_svcs, mlp_svcs], axis=0).drop_duplicates()
    base_svcs = _as_str_cols(base_svcs, ["SVC"])
    svc_list = sorted(base_svcs["SVC"].dropna().astype(str).unique().tolist())
except Exception:
    svc_list = []

default_sel = [s for s in DEFAULT_SVCS if s in svc_list] or svc_list[:4]
sel_svcs = st.multiselect("Filtrar SVC", options=svc_list, default=default_sel, placeholder="Selecciona SVCs")

run_btn = st.button("Calcular plan", type="primary")

try:
    if run_btn:
        plan = compute_plan(spr_mode, planning_date, crowd_esc, sel_svcs or None)
        if plan.empty:
            st.warning("No hay datos para mostrar con los filtros seleccionados.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("SVCs", plan["SVC"].nunique())
            c2.metric("Demanda ajustada", int(plan["DEMANDA_AJUSTADA"].sum()))
            c3.metric("Rutas (SPR base)", int(plan["RUTAS_SPR_BASE"].sum()))
            c4.metric("Rutas faltantes", int(plan["RUTAS_FALTANTES"].sum()))
            st.dataframe(plan, use_container_width=True, hide_index=True)
    else:
        st.info("Selecciona fecha y escenario Crowd, luego presiona **Calcular plan**.")
except Exception as e:
    st.error("Ocurrió un error durante el cálculo.")
    show_exception(e, "Traceback completo")
