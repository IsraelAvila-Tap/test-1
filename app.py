# app.py
# =============================================================================
# Mel-IA — Plan táctico (diario por SVC)
# Lee Google Sheets y planifica rutas por Delivery Model con lógica:
# FCST − DC − SP → rutas (SPR) → Rentals → Crowd base (% plan) → MLP SDD/Spot → Crowd E1
# =============================================================================
import os, json, yaml
import numpy as np
import pandas as pd
from math import ceil
from datetime import timedelta, date, datetime
from typing import List, Optional
import streamlit as st

# ---------------------------------------------------------------------
# 0) Patch credenciales GCP desde Secrets (2 formatos soportados)
# ---------------------------------------------------------------------
if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
elif "gcp_service_account" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps(dict(st.secrets["gcp_service_account"]))

if "PROJECT_KEY" in st.secrets:
    os.environ["PROJECT_KEY"] = st.secrets["PROJECT_KEY"]

# ---------------------------------------------------------------------
# 1) Imports locales
# ---------------------------------------------------------------------
from utils_gsheets import read_ws, _client, get_service_account_email

# ---------------------------------------------------------------------
# 2) Config Streamlit
# ---------------------------------------------------------------------
st.set_page_config(page_title="Mel-IA — Plan táctico", layout="wide")

@st.cache_resource
def load_cfg():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
CFG = load_cfg()
PROJECT = list(CFG["projects"].keys())[0]
SHEET_ID = CFG["projects"][PROJECT]["sheet_id"]

# ---------------------------------------------------------------------
# 3) Utilidades robustas (nunca fallan con .str / Series / DF)
# ---------------------------------------------------------------------
def _ensure_series(x) -> pd.Series:
    """Devuelve siempre una Series; tolera DataFrame/array/list/escalares."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return pd.Series(dtype=float)
        return _ensure_series(x.iloc[:, 0])
    if isinstance(x, (list, tuple, np.ndarray)):
        return pd.Series(x)
    return pd.Series([x])

def _to_num(s) -> pd.Series:
    s = _ensure_series(s).astype(str)
    s = s.str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")

def _safe_to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Convierte a numérico sólo si la columna existe; evita usar 'c' fuera de scope."""
    for col in cols:
        if col in df.columns:
            df[col] = _to_num(df[col]).fillna(0.0)
    return df

def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # normaliza nombres únicos (si hay duplicados, agrega sufijo)
    new = []
    seen = {}
    for c in df.columns:
        base = str(c).strip().lower()
        count = seen.get(base, 0)
        seen[base] = count + 1
        new.append(base if count == 0 else f"{base}_{count}")
    df.columns = new
    return df

def _norm_date_col(df: pd.DataFrame, col: str = "fecha") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True).dt.date
    return df

def _weekday(d: date) -> int:
    return pd.Timestamp(d).weekday()  # 0..6 = L..D

def _iso_yw(d: date):
    iso = pd.Timestamp(d).isocalendar()
    return int(iso.year), int(iso.week)

def _safe_mean(vals):
    x = [float(v) for v in vals if pd.notna(v)]
    return float(np.mean(x)) if x else np.nan

# ---------------------------------------------------------------------
# 4) Lecturas de Hojas
# ---------------------------------------------------------------------
def _read(tab: str) -> pd.DataFrame:
    df = read_ws(SHEET_ID, tab)
    return _lower_cols(df)

def load_fcst() -> pd.DataFrame:
    raw = _read("FCST")

    # --- 1) Detectar fila de header en las primeras 10 filas
    header_row = None
    ship_tokens = {"shipments","shipment","fcst","forecast","envios","envíos","q","qty","cantidad","volume","volumen","ship","demand"}
    date_tokens = {"fecha","date","día","dia","day"}

    for i in range(min(10, len(raw))):
        row = [str(x).strip().lower() for x in raw.iloc[i].tolist()]
        has_svc   = any(("svc" in cell) for cell in row)
        has_ship  = any(any(tok in cell for tok in ship_tokens) for cell in row)
        has_date  = any(any(tok in cell for tok in date_tokens) for cell in row)
        if has_svc and has_ship and has_date:
            header_row = i
            break

    # --- 2) Fijar encabezados y limpiar
    if header_row is not None:
        df = raw.copy()
        df.columns = [str(x).strip().lower() for x in raw.iloc[header_row].tolist()]
        df = df.iloc[header_row+1:].reset_index(drop=True)
        df = _lower_cols(df)
    else:
        df = _lower_cols(raw)

    # --- 3) Detectar columnas por sinónimos
    # svc
    svc_col = None
    for c in df.columns:
        if "svc" in c:
            svc_col = c; break

    # fecha
    date_col = None
    for c in df.columns:
        cc = c.lower()
        if any(tok in cc for tok in ("fecha","date","día","dia","day")):
            date_col = c; break

    # shipments / forecast / envíos / cantidad...
    ship_col = None
    for c in df.columns:
        cc = c.lower()
        if any(tok in cc for tok in ship_tokens):
            ship_col = c; break

    if not (svc_col and date_col and ship_col):
        cols = list(df.columns)[:30]
        raise ValueError(f"FCST: faltan columnas ['svc','fecha','shipments']. Detectadas: {cols}")

    # --- 4) Normalizar nombres y tipos
    df = df.rename(columns={svc_col:"svc", date_col:"fecha", ship_col:"shipments"})
    df = _norm_date_col(df, "fecha")
    df["svc"] = _ensure_series(df["svc"]).astype(str).str.upper().str.strip()
    df["shipments"] = _to_num(df["shipments"]).fillna(0.0)

    out = df[["fecha","svc","shipments"]].dropna(subset=["fecha","svc"])
    return out


def load_spr_real() -> pd.DataFrame:
    df = _read("SPR")
    # Detectar columnas flexibles
    svc_col = "svc" if "svc" in df.columns else ("svcs" if "svcs" in df.columns else None)
    if not svc_col:
        raise ValueError("SPR: falta columna 'svc'.")

    date_col = "fecha" if "fecha" in df.columns else ("date" if "date" in df.columns else None)
    if not date_col:
        # intenta detectar por tokens
        for c in df.columns:
            if any(tok in c for tok in ["fecha","date","día","dia","day"]):
                date_col = c; break
    if not date_col:
        raise ValueError("SPR: falta columna de fecha ('fecha' o 'date').")

    spr_col = None
    for c in df.columns:
        if "spr" in c.lower():
            spr_col = c; break
    if not spr_col:
        raise ValueError("SPR: no se encontró columna con 'spr' en el nombre.")

    # Renombres
    if svc_col != "svc":
        df = df.rename(columns={svc_col: "svc"})
    if date_col != "fecha":
        df = df.rename(columns={date_col: "fecha"})
    if spr_col != "spr":
        df = df.rename(columns={spr_col: "spr"})

    # Tipados
    df = _norm_date_col(df, "fecha")
    df["spr"] = _to_num(df["spr"])
    df = df.dropna(subset=["fecha","svc","spr"])
    df["svc"] = _ensure_series(df["svc"]).astype(str).str.upper().str.strip()

    df["dow"] = df["fecha"].apply(_weekday)
    iso = df["fecha"].apply(lambda d: pd.Timestamp(d).isocalendar())
    df["iso_year"] = [int(x.year) for x in iso]
    df["iso_week"] = [int(x.week) for x in iso]
    return df[["fecha","svc","spr","dow","iso_year","iso_week"]]


def load_capacity() -> pd.DataFrame:
    raw = _read("Capacity")

    # --- 1) Detectar encabezado en las primeras 10 filas
    header_row = None
    for i in range(min(10, len(raw))):
        row = [str(x).strip().lower() for x in raw.iloc[i].tolist()]
        has_dm    = any(("delivery" in c) or ("modelo" in c) or ("tipo dm" in c) or (c.strip() == "dm") for c in row)
        has_tipo  = any(("tipo" in c) or ("type" in c) for c in row)
        has_svc   = any("svc" in c for c in row)
        has_fecha = any(("fecha" in c) or ("date" in c) or ("día" in c) or ("dia" in c) or ("day" in c) for c in row)
        has_qty   = any(
            ("cantidad" in c) or ("qty" in c) or ("quantity" in c) or ("capacidad" in c) or
            ("units" in c) or ("count" in c) or ("routes" in c) or ("rutas" in c) or
            ("volume" in c) or ("volumen" in c)
        )
        if sum([has_dm, has_tipo, has_svc, has_fecha, has_qty]) >= 3:
            header_row = i
            break

    if header_row is not None:
        df = raw.copy()
        df.columns = [str(x).strip().lower() for x in raw.iloc[header_row].tolist()]
        df = df.iloc[header_row + 1 :].reset_index(drop=True)
        df = _lower_cols(df)
    else:
        df = _lower_cols(raw)

    # --- 2) Resolver nombres por sinónimos
    def find_col(tokens: List[str]) -> Optional[str]:
        for c in df.columns:
            cc = c.strip().lower()
            if any(tok in cc for tok in tokens):
                return c
        return None

    dm_col    = find_col(["delivery model","delivery","modelo","modelo de entrega","tipo dm"," dm "])
    tipo_col  = find_col(["tipo","type","categoria","category"])
    svc_col   = find_col(["svc","svcs","svc "])
    fecha_col = find_col(["fecha","date","día","dia","day"])
    qty_col   = find_col(["cantidad","qty","quantity","capacidad","units","count","rutas","routes","volume","volumen"])

    faltan = []
    if not dm_col:    faltan.append("delivery model")
    if not tipo_col:  faltan.append("tipo")
    if not svc_col:   faltan.append("svc")
    if not fecha_col: faltan.append("fecha")
    if not qty_col:   faltan.append("cantidad")
    if faltan:
        raise ValueError(f"Capacity: faltan columnas {faltan}. Encabezados vistos: {list(df.columns)[:30]}")

    # --- 3) Normalizar y tipar
    df = df.rename(columns={
        dm_col: "delivery model",
        tipo_col: "tipo",
        svc_col: "svc",
        fecha_col: "fecha",
        qty_col: "cantidad",
    })

    df = _norm_date_col(df, "fecha")
    df["svc"] = _ensure_series(df["svc"]).astype(str).str.upper().str.strip()
    df["tipo"] = _ensure_series(df["tipo"]).astype(str).str.strip().str.lower()
    df["delivery model"] = _ensure_series(df["delivery model"]).astype(str).str.strip().str.lower()
    df["cantidad"] = _to_num(df["cantidad"]).fillna(0.0)

    # Salida consistente
    return df[["fecha", "svc", "delivery model", "tipo", "cantidad"]].dropna(subset=["fecha","svc"])

def _safe_to_numeric_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Convierte a numérico (robusto) sólo las columnas presentes."""
    for col in cols:
        if col in df.columns:
            df[col] = _to_num(df[col]).fillna(0.0)
    return df


def load_srm() -> pd.DataFrame:
    """
    Lee la pestaña SRM (encabezado puede estar en fila 5 y con muchas columnas).
    Detecta columnas SDD/SPOT por nombre y agrega por SVC.
    """
    raw = _read("SRM")  # wrapper que ya normaliza columnas a minúsculas y trim

    # Detectar fila de encabezado (primera que contenga 'svc')
    header_row = None
    for i in range(min(10, len(raw))):
        row = [str(x).strip().lower() for x in raw.iloc[i].tolist()]
        if any(x == "svc" or x == "svcs" for x in row):
            header_row = i
            break
    if header_row is None:
        header_row = 4  # fallback

    # Aplicar encabezados
    cols = [str(x).strip().lower() for x in raw.iloc[header_row].tolist()]
    df = raw.iloc[header_row + 1:].copy()
    df.columns = cols

    # Columna SVC
    svc_col = "svc" if "svc" in df.columns else ("svcs" if "svcs" in df.columns else None)
    if not svc_col:
        raise ValueError("SRM: no se encontró columna 'SVC/SVCs'.")

    # Detectar columnas SDD y SPOT por patrón en el nombre
    sdd_cols  = [c for c in df.columns if "sdd"  in c]
    spot_cols = [c for c in df.columns if "spot" in c]
    if not sdd_cols and not spot_cols:
        first_cols = list(df.columns)[:30]
        raise ValueError(f"SRM: no se hallaron columnas con 'sdd' o 'spot'. Encabezados: {first_cols}")

    # Numéricos robustos
    num_cols = list(set(sdd_cols + spot_cols))
    df = _safe_to_numeric_cols(df, num_cols)

    # Agregación por SVC
    g = df.copy()
    g["svc"] = _ensure_series(g[svc_col]).astype(str).str.upper().str.strip()

    agg = g.groupby("svc", as_index=False)[num_cols].sum()
    agg["sdd_routes_max"]  = agg[sdd_cols].sum(axis=1)  if sdd_cols  else 0.0
    agg["spot_routes_max"] = agg[spot_cols].sum(axis=1) if spot_cols else 0.0

    return agg[["svc", "sdd_routes_max", "spot_routes_max"]]


def load_rentals() -> pd.DataFrame:
    df = _read("Rentals")
    svc_col = "svc" if "svc" in df.columns else ("svcs" if "svcs" in df.columns else None)
    if not svc_col:
        raise ValueError("Rentals: falta columna 'SVC/SVCs'.")
    # columna unidades (cualquier que empiece con 'unidades')
    qty_col = None
    for c in df.columns:
        if str(c).lower().startswith("unidades"):
            qty_col = c; break
    if not qty_col:
        raise ValueError("Rentals: falta columna de unidades disponibles (que empiece por 'unidades').")
    df["rentals_routes_max"] = _to_num(df[qty_col]).fillna(0.0).astype(float)
    out = (df.groupby(svc_col, as_index=False)["rentals_routes_max"]
             .sum()
             .rename(columns={svc_col:"svc"}))
    out["svc"] = _ensure_series(out["svc"]).astype(str).str.upper().str.strip()
    return out

def load_crowd_caps() -> pd.DataFrame:
    df = _read("Crowd")
    # SVC
    svc_col = "svc" if "svc" in df.columns else ("svcs" if "svcs" in df.columns else None)
    if not svc_col:
        raise ValueError("Crowd: falta columna 'svc'.")
    # layout detallado
    def _pick(patterns: List[str]) -> Optional[str]:
        for c in df.columns:
            cc = str(c).lower()
            if any(p in cc for p in patterns):
                return c
        return None

    c_base_wd = _pick(["base entre"])
    c_base_sa = _pick(["base sab"])
    c_base_su = _pick(["base dom"])
    c_e1_wd   = _pick(["holgura entre","e1 entre"])
    c_e1_sa   = _pick(["holgura sab","e1 sab"])
    c_e1_su   = _pick(["holgura dom","e1 dom"])

    if all(x is not None for x in [c_base_wd,c_base_sa,c_base_su,c_e1_wd,c_e1_sa,c_e1_su]):
        out = df[[svc_col, c_base_wd,c_base_sa,c_base_su,c_e1_wd,c_e1_sa,c_e1_su]].copy()
        out = out.rename(columns={
            svc_col:"svc", c_base_wd:"base_wd", c_base_sa:"base_sa", c_base_su:"base_su",
            c_e1_wd:"e1_wd", c_e1_sa:"e1_sa", c_e1_su:"e1_su"
        })
        for coln in ["base_wd","base_sa","base_su","e1_wd","e1_sa","e1_su"]:
            out[coln] = _to_num(out[coln]).fillna(0.0).astype(float)
        out["svc"] = _ensure_series(out["svc"]).astype(str).str.upper().str.strip()
        return out

    # layout compacto: 3 columnas que empiezan con base y 3 con e1/holgura
    base_cols = [c for c in df.columns if str(c).lower().startswith("base")]
    e1_cols   = [c for c in df.columns if (str(c).lower().startswith("e1") or "holgura" in str(c).lower())]
    if len(base_cols) == 3 and len(e1_cols) == 3:
        b1,b2,b3 = base_cols
        e1,e2,e3 = e1_cols
        out = df[[svc_col,b1,b2,b3,e1,e2,e3]].copy()
        out.columns = ["svc","base_wd","base_sa","base_su","e1_wd","e1_sa","e1_su"]
        for coln in ["base_wd","base_sa","base_su","e1_wd","e1_sa","e1_su"]:
            out[coln] = _to_num(out[coln]).fillna(0.0).astype(float)
        out["svc"] = _ensure_series(out["svc"]).astype(str).str.upper().str.strip()
        return out

    # último intento: una 'base' y un 'e1' (mismo valor para todos los días)
    if ("base" in df.columns) and ("e1" in df.columns):
        out = df[[svc_col,"base","e1"]].copy()
        for coln in ["base","e1"]:
            out[coln] = _to_num(out[coln]).fillna(0.0).astype(float)
        out = out.rename(columns={"base":"base_wd", "e1":"e1_wd"})
        out["base_sa"] = out["base_wd"]
        out["base_su"] = out["base_wd"]
        out["e1_sa"] = out["e1_wd"]
        out["e1_su"] = out["e1_wd"]
        out["svc"] = _ensure_series(out["svc"]).astype(str).str.upper().str.strip()
        return out

    raise ValueError(f"Crowd: layout no reconocido. Encabezados: {list(df.columns)}")

# ---------------------------------------------------------------------
# 5) Cálculo de SPR objetivo
# ---------------------------------------------------------------------
def compute_spr_targets(fcst: pd.DataFrame, spr_real: pd.DataFrame, capacity: pd.DataFrame, mode: str) -> pd.DataFrame:
    target = fcst[["fecha","svc"]].drop_duplicates().copy()
    target["dow"] = target["fecha"].apply(_weekday)
    target["iso_year"] = target["fecha"].apply(lambda d: int(pd.Timestamp(d).isocalendar().year))

    spr_exec_map = spr_real.set_index(["fecha","svc"])["spr"]

    def avg_last4(row):
        d, s = row["fecha"], row["svc"]
        vals = []
        for k in (7,14,21,28):
            dk = d - timedelta(days=k)
            v = spr_exec_map.get((dk,s), np.nan)
            if pd.notna(v): vals.append(float(v))
        if not vals:
            m = (spr_real["svc"].eq(s) & spr_real["fecha"].between(d - timedelta(days=28), d - timedelta(days=1)))
            vals = list(spr_real.loc[m,"spr"])
        return _safe_mean(vals)

    def avg_peak(row):
        d, s, yr, dow = row["fecha"], row["svc"], row["iso_year"], row["dow"]
        m = (spr_real["svc"].eq(s) & spr_real["iso_year"].eq(yr) &
             spr_real["iso_week"].isin([20,21,22]) & spr_real["dow"].eq(dow))
        vals = list(spr_real.loc[m,"spr"])
        if not vals:
            m = (spr_real["svc"].eq(s) & spr_real["iso_year"].eq(yr) &
                 spr_real["iso_week"].between(19,23) & spr_real["dow"].eq(dow))
            vals = list(spr_real.loc[m,"spr"])
        return _safe_mean(vals)

    if mode == "promedio":
        target["spr_objetivo"] = target.apply(avg_last4, axis=1)
    elif mode == "peak":
        target["spr_objetivo"] = target.apply(avg_peak, axis=1)
    else:
        # 'plan': desde Capacity (tipo == 'spr')
        cap = capacity.copy()
        m = _ensure_series(cap["tipo"]).astype(str).str.lower().str.strip().eq("spr")
        plan = cap.loc[m, ["svc","fecha","cantidad"]].rename(columns={"cantidad":"spr_plan"})
        if plan.empty:
            by_svc = cap.loc[m].groupby("svc", as_index=False)["cantidad"].mean().rename(columns={"cantidad":"spr_plan"})
            target = target.merge(by_svc, on="svc", how="left")
        else:
            target = target.merge(plan, on=["fecha","svc"], how="left")
        target = target.rename(columns={"spr_plan":"spr_objetivo"})

    return target[["fecha","svc","spr_objetivo"]]

# ---------------------------------------------------------------------
# 6) Share crowd objetivo (desde Shipments por día)
# ---------------------------------------------------------------------
def compute_crowd_share(capacity: pd.DataFrame) -> pd.DataFrame:
    cap = capacity.copy()
    cap["tipo"] = _ensure_series(cap["tipo"]).astype(str).str.lower().str.strip()
    cap["delivery model"] = _ensure_series(cap["delivery model"]).astype(str).str.lower().str.strip()
    if "tipo dm" in cap.columns:
        cap["tipo dm"] = _ensure_series(cap["tipo dm"]).astype(str).str.lower().str.strip()
    else:
        cap["tipo dm"] = ""  # asegurar columna para filtros

    # Solo 'Shipments' para el share
    m_ship = cap["tipo"].eq("shipments")
    ship = cap.loc[m_ship, ["fecha","svc","delivery model","tipo dm","cantidad"]].copy()

    # total por día
    tot = ship.groupby(["fecha","svc"], as_index=False)["cantidad"].sum().rename(columns={"cantidad":"ship_total"})
    # crowd del día (delivery model crowd, o tipo dm contenga 'crowd')
    is_crowd = ship["delivery model"].str.contains("crowd", case=False) | ship["tipo dm"].str.contains("crowd", case=False)
    crw = ship.loc[is_crowd].groupby(["fecha","svc"], as_index=False)["cantidad"].sum().rename(columns={"cantidad":"ship_crowd"})

    out = tot.merge(crw, on=["fecha","svc"], how="left").fillna({"ship_crowd":0.0})
    out["share_crowd_obj"] = np.where(out["ship_total"]>0, (out["ship_crowd"]/out["ship_total"]).clip(0,1), 0.0)

    # DC / SP para restar del FCST
    is_dc = ship["tipo dm"].str.contains("dc", case=False) | ship["delivery model"].str.contains("dc", case=False)
    is_sp = (ship["tipo dm"].str.contains("sp", case=False) |
             ship["delivery model"].str.contains("service partner", case=False) |
             ship["delivery model"].str.contains("sp", case=False))

    dc = ship.loc[is_dc].groupby(["fecha","svc"], as_index=False)["cantidad"].sum().rename(columns={"cantidad":"ship_dc"})
    sp = ship.loc[is_sp].groupby(["fecha","svc"], as_index=False)["cantidad"].sum().rename(columns={"cantidad":"ship_sp"})

    out = out.merge(dc, on=["fecha","svc"], how="left").merge(sp, on=["fecha","svc"], how="left")
    out[["ship_dc","ship_sp"]] = out[["ship_dc","ship_sp"]].fillna(0.0)

    return out[["fecha","svc","share_crowd_obj","ship_total","ship_crowd","ship_dc","ship_sp"]]

# ---------------------------------------------------------------------
# 7) Map de Crowd por fecha (base/e1 según día de la semana)
# ---------------------------------------------------------------------
def map_crowd_by_date(target_days: pd.DataFrame, crowd_caps: pd.DataFrame) -> pd.DataFrame:
    def f(row):
        s, d = row["svc"], row["fecha"]
        dow = _weekday(d)
        r = crowd_caps.loc[crowd_caps["svc"]==s]
        if r.empty:
            return pd.Series({"crowd_base_routes":0.0,"crowd_e1_routes":0.0})
        r = r.iloc[0]
        if dow <= 4:
            base, e1 = r["base_wd"], r["e1_wd"]
        elif dow == 5:
            base, e1 = r["base_sa"], r["e1_sa"]
        else:
            base, e1 = r["base_su"], r["e1_su"]
        return pd.Series({"crowd_base_routes":float(base), "crowd_e1_routes":float(e1)})
    tmp = target_days.apply(f, axis=1)
    return pd.concat([target_days.reset_index(drop=True), tmp], axis=1)

# ---------------------------------------------------------------------
# 8) Scheduler MLP descansos semanales
# ---------------------------------------------------------------------
def schedule_mlp_rest(df_day: pd.DataFrame) -> pd.DataFrame:
    out = df_day.copy()
    out["week_key"] = out["fecha"].apply(lambda d: f"{_iso_yw(d)[0]}-{_iso_yw(d)[1]:02d}")
    out["sdd_trabaja"]  = 1
    out["spot_trabaja"] = 1

    def proc(g):
        n = len(g)
        need_days = int((g["routes_mlp_need"]>0).sum())
        # SDD: 6x7 (no más que días con necesidad)
        work_sdd = min(6, need_days)
        rest_sdd = max(n - work_sdd, 0)
        # Spot: 5x7; si hay déficit en >=6 días, 6x7; si <5 días con necesidad, sólo esos
        work_spot = 5
        if need_days >= 6: work_spot = 6
        elif need_days < 5: work_spot = need_days
        rest_spot = max(n - work_spot, 0)

        g_sorted = g.sort_values(["routes_mlp_need","fecha"], ascending=[True,True])
        if rest_sdd > 0:
            g.loc[g_sorted.head(rest_sdd).index, "sdd_trabaja"] = 0
        g_sorted2 = g.sort_values(["routes_mlp_need","fecha"], ascending=[True,True])
        if rest_spot > 0:
            g.loc[g_sorted2.head(rest_spot).index, "spot_trabaja"] = 0
        return g

    out = out.groupby(["svc","week_key"], group_keys=False).apply(proc)
    return out.drop(columns=["week_key"])

# ---------------------------------------------------------------------
# 9) Motor principal
# ---------------------------------------------------------------------
def compute_plan(spr_mode: str, sel_svcs: Optional[List[str]] = None) -> pd.DataFrame:
    # Lecturas
    fcst       = load_fcst()
    spr_real   = load_spr_real()
    capacity   = load_capacity()
    srm        = load_srm()
    rentals    = load_rentals()
    crowd_caps = load_crowd_caps()

    if sel_svcs:
        fcst = fcst[fcst["svc"].isin(sel_svcs)]

    # SPR objetivo
    spr_tbl = compute_spr_targets(fcst, spr_real, capacity, spr_mode)

    # Share Crowd y DC/SP
    share_tbl = compute_crowd_share(capacity)

    # Crowd por día
    days = fcst[["fecha","svc"]].drop_duplicates()
    crowd_daily = map_crowd_by_date(days, crowd_caps)

    # Merge base
    df = (fcst
          .merge(share_tbl, on=["fecha","svc"], how="left")
          .merge(crowd_daily, on=["fecha","svc"], how="left")
          .merge(srm, on="svc", how="left")
          .merge(rentals, on="svc", how="left")
          .merge(spr_tbl, on=["fecha","svc"], how="left")
         )

    # Limpieza
    for coln in ["share_crowd_obj","crowd_base_routes","crowd_e1_routes","sdd_routes_max","spot_routes_max","rentals_routes_max","ship_dc","ship_sp"]:
        if coln in df.columns:
            df[coln] = _to_num(df[coln]).fillna(0.0)
        else:
            df[coln] = 0.0
    df["spr_objetivo"] = _to_num(df["spr_objetivo"])

    # FCST neto (descuenta DC/SP)
    df["ship_fcst_neto"] = (df["shipments"] - df["ship_dc"] - df["ship_sp"]).clip(lower=0)

    # Rutas requeridas por SPR
    df["routes_need_total"] = np.where(
        (df["ship_fcst_neto"]>0) & (df["spr_objetivo"]>0),
        np.ceil(df["ship_fcst_neto"] / df["spr_objetivo"]).astype(int),
        0
    )
    df["alerta_spr_missing"] = ((df["ship_fcst_neto"]>0) & (df["spr_objetivo"].isna() | (df["spr_objetivo"]<=0)))

    # 1) Crowd base según % plan
    df["routes_crowd_target"] = np.ceil(df["routes_need_total"] * df["share_crowd_obj"]).astype(int)
    df["routes_crowd_base"]   = np.minimum(df["routes_crowd_target"], df["crowd_base_routes"]).astype(int)

    # 2) Rentals directo
    rem1 = (df["routes_need_total"] - df["routes_crowd_base"]).clip(lower=0)
    df["routes_rentals_alloc"] = np.minimum(rem1, df["rentals_routes_max"]).astype(int)

    # 3) Necesidad MLP
    rem2 = (rem1 - df["routes_rentals_alloc"]).clip(lower=0).astype(int)
    rest_base = pd.DataFrame({
        "fecha": df["fecha"],
        "svc": df["svc"],
        "routes_mlp_need": rem2,
        "sdd_routes_max": df["sdd_routes_max"],
        "spot_routes_max": df["spot_routes_max"],
    })
    rest_sched = schedule_mlp_rest(rest_base)
    df = df.merge(rest_sched[["fecha","svc","sdd_trabaja","spot_trabaja"]], on=["fecha","svc"], how="left")
    df["routes_mlp_cap_day"] = (df["sdd_routes_max"]*df["sdd_trabaja"] + df["spot_routes_max"]*df["spot_trabaja"]).fillna(0).astype(int)
    df["routes_mlp_alloc"]   = np.minimum(rem2, df["routes_mlp_cap_day"]).astype(int)

    # 4) Crowd extra (E1) si aún falta
    rem_def = (rem2 - df["routes_mlp_alloc"]).clip(lower=0).astype(int)
    df["routes_crowd_e1"] = np.minimum(rem_def, df["crowd_e1_routes"]).astype(int)

    # Totales, déficit, métricas
    df["routes_total_alloc"] = (df["routes_crowd_base"] + df["routes_rentals_alloc"] + df["routes_mlp_alloc"] + df["routes_crowd_e1"]).astype(int)
    df["routes_deficit"]     = (df["routes_need_total"] - df["routes_total_alloc"]).clip(lower=0).astype(int)
    df["shipments_plan"]     = np.where(df["spr_objetivo"]>0, df["routes_total_alloc"]*df["spr_objetivo"], 0.0)
    df["spr_logrado"]        = np.where(df["routes_total_alloc"]>0, df["ship_fcst_neto"]/df["routes_total_alloc"], np.nan)
    df["share_crowd_real"]   = np.where(df["routes_need_total"]>0, (df["routes_crowd_base"]+df["routes_crowd_e1"]) / df["routes_need_total"], 0.0)
    df["risk_flag"]          = (df["routes_deficit"]>0) | df["alerta_spr_missing"]

    cols = [
        "fecha","svc",
        "shipments","ship_dc","ship_sp","ship_fcst_neto",
        "spr_objetivo",
        "routes_need_total",
        "rentals_routes_max","routes_rentals_alloc",
        "crowd_base_routes","crowd_e1_routes","share_crowd_obj","routes_crowd_target","routes_crowd_base","routes_crowd_e1",
        "sdd_routes_max","spot_routes_max","sdd_trabaja","spot_trabaja","routes_mlp_cap_day","routes_mlp_alloc",
        "routes_total_alloc","routes_deficit","shipments_plan","spr_logrado","share_crowd_real",
        "alerta_spr_missing","risk_flag",
    ]
    return df[cols].sort_values(["fecha","svc"]).reset_index(drop=True)

# ---------------------------------------------------------------------
# 10) UI
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Proyecto")
    st.write(f"Sheet: `{SHEET_ID}`")
    st.subheader("🔐 Credenciales")
    svc_email = get_service_account_email()
    if svc_email:
        st.caption("Comparte el Sheet con:")
        st.code(svc_email, language="text")
    else:
        st.warning("No se detectó Service Account.")

st.title("Mel-IA — Plan táctico (diario por SVC)")

spr_mode = st.radio("SPR objetivo", ["promedio","peak","plan"], index=0, horizontal=True)

with st.expander("▶ Cargando datos…", expanded=True):
    try:
        # Carga mínima para saber SVCs disponibles (FCST)
        fcst_preview = load_fcst()
        all_svcs = sorted(fcst_preview["svc"].dropna().astype(str).str.upper().unique().tolist())
        sel_svcs = st.multiselect("Filtrar SVC", all_svcs, default=all_svcs[:4])
    except Exception as e:
        st.error(f"Error al leer FCST: {e}")
        sel_svcs = []

    try:
        plan = compute_plan(spr_mode, sel_svcs or None)
        st.success("Datos listos ✅")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# Tabla principal (svc,fecha) × Delivery Model desglosado en columnas
st.subheader("Tabla principal — (svc, fecha) × Delivery model")
st.dataframe(plan, use_container_width=True, hide_index=True)

# Resumen de riesgos por fecha
st.subheader("Riesgos por fecha")
resumen = (plan.groupby("fecha", as_index=False)
           .agg(
               svcs_con_deficit=("routes_deficit", lambda s: int((s>0).sum())),
               rutas_deficit=("routes_deficit","sum"),
               svcs_sin_spr=("alerta_spr_missing", "sum"),
           ))
st.dataframe(resumen, use_container_width=True, hide_index=True)

# Vistas agregadas por Delivery Model (columnas)
st.subheader("Vistas agrupadas por Delivery Model")
agg_cols = [
    "routes_crowd_base","routes_crowd_e1","routes_rentals_alloc",
    "routes_mlp_alloc","routes_deficit","routes_total_alloc"
]
vista = (plan.groupby(["svc","fecha"], as_index=False)[agg_cols].sum())
st.dataframe(vista.sort_values(["svc","fecha"]), use_container_width=True, hide_index=True)
