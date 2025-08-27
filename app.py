# app.py — Mel-IA Plan táctico (diario por SVC)

import os, json, yaml, math, traceback
from datetime import timedelta
from math import ceil
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------- Credenciales -------------------------------
if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
elif "gcp_service_account" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps(dict(st.secrets["gcp_service_account"]))
if "PROJECT_KEY" in st.secrets:
    os.environ["PROJECT_KEY"] = st.secrets["PROJECT_KEY"]

st.set_page_config(page_title="Mel-IA — Plan táctico", layout="wide")
st.title("Mel-IA — Plan táctico (diario por SVC)")
spr_mode = st.radio("SPR objetivo", ["promedio","peak","plan"], index=0, horizontal=True)

# ------------------------------- Utils -------------------------------
def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace("\n"," ") for c in df.columns]
    return df

def _ensure_series(x) -> pd.Series:
    return x if isinstance(x, pd.Series) else pd.Series(x)

def _upper_series(s: pd.Series) -> pd.Series:
    s = _ensure_series(s)
    return s.astype(str).str.strip().str.upper()

def _to_num(s: pd.Series) -> pd.Series:
    s = _ensure_series(s).astype(str)
    s = (s.str.replace(",", "", regex=False)
          .str.replace("%","", regex=False)
          .str.strip())
    return pd.to_numeric(s, errors="coerce")

def _weekday(d) -> int:
    return pd.to_datetime(d, errors="coerce", dayfirst=True).dayofweek

def _safe_mean(vals):
    vals = [float(v) for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan

from utils_gsheets import read_ws, _client, get_service_account_email

@st.cache_resource
def load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()
proj_key = list(cfg["projects"].keys())[0]
SHEET_ID = cfg["projects"][proj_key]["sheet_id"]

def _reheader(raw: pd.DataFrame, prefer=("svc","fecha","shipments"), scan_rows=15) -> pd.DataFrame:
    if raw.empty:
        return raw
    header_row = None
    pref = set(x.lower() for x in prefer)

    for i in range(min(scan_rows, len(raw))):
        row = [str(x).strip().lower() for x in raw.iloc[i,:].tolist()]
        if pref.issubset(set(row)):
            header_row = i
            break
    if header_row is None:
        for i in range(min(scan_rows, len(raw))):
            row = set(str(x).strip().lower() for x in raw.iloc[i,:].tolist())
            if len(pref.intersection(row)) >= 2:
                header_row = i
                break

    if header_row is not None:
        cols = [(str(x).strip().lower() if str(x).strip() else f"col_{j+1}")
                for j,x in enumerate(raw.iloc[header_row,:].tolist())]
        df = raw.iloc[header_row+1:].reset_index(drop=True)
        df.columns = cols
        return _lower_cols(df)

    return _lower_cols(raw.copy())

# ------------------------------- Loaders -------------------------------
def load_fcst() -> pd.DataFrame:
    raw = read_ws(SHEET_ID, "FCST")
    df = _reheader(raw, prefer=("svc","fecha","shipments"))
    need = {"svc","fecha","shipments"}
    if not need.issubset(df.columns):
        raise ValueError(f"FCST: faltan columnas {sorted(list(need))}")
    out = pd.DataFrame({
        "svc": _upper_series(df["svc"]),
        "fecha": pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True).dt.date,
        "shipments": _to_num(df["shipments"]).fillna(0.0)
    }).dropna(subset=["fecha","svc"])
    return out

def load_spr_real() -> pd.DataFrame:
    raw = read_ws(SHEET_ID, "SPR")
    df = _reheader(raw, prefer=("svc","fecha","spr"))
    need = {"svc","fecha","spr"}
    if not need.issubset(df.columns):
        raise ValueError("SPR: faltan columnas 'svc','fecha','spr'")
    df = df.rename(columns={"spr":"spr_exec"})
    df["svc"] = _upper_series(df["svc"])
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True).dt.date
    df["spr_exec"] = _to_num(df["spr_exec"])
    day = (df.dropna(subset=["fecha"])
             .groupby(["fecha","svc"], as_index=False)["spr_exec"].mean())
    iso = pd.to_datetime(day["fecha"]).dt.isocalendar()
    day["dow"] = pd.to_datetime(day["fecha"]).dt.dayofweek
    day["iso_year"] = iso["year"].astype(int)
    day["iso_week"] = iso["week"].astype(int)
    return day

def load_capacity() -> pd.DataFrame:
    raw = read_ws(SHEET_ID, "Capacity")
    df = _reheader(raw, prefer=("delivery model","tipo","svc","fecha","cantidad"))
    df = df.rename(columns={
        "delivery_model":"delivery model","tipo_dm":"tipo dm","tipo_dm ":"tipo dm",
        "cantidad ":"cantidad"
    })
    need = {"delivery model","tipo","svc","fecha","cantidad"}
    if not need.issubset(df.columns):
        raise ValueError(f"Capacity: faltan columnas {sorted(list(need))}")
    df["svc"] = _upper_series(df["svc"])
    df["tipo"] = df["tipo"].astype(str).str.strip().str.lower()
    df["delivery model"] = df["delivery model"].astype(str).str.strip().str.lower()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True).dt.date
    df["cantidad"] = _to_num(df["cantidad"]).fillna(0.0)
    return df.dropna(subset=["fecha","svc"])

def load_rentals() -> pd.DataFrame:
    raw = read_ws(SHEET_ID, "Rentals")
    df = _reheader(raw, prefer=("svcs","tipo de vehículo","unidades disponibles"))
    svc_col = None
    for c in df.columns:
        if c in ("svc","svcs","svcs "):
            svc_col = c; break
    if not svc_col:
        cand = [c for c in df.columns if "svc" in c]
        svc_col = cand[0] if cand else None
    qty_col = None
    for c in df.columns:
        if ("unidades" in c) and ("dispon" in c):
            qty_col = c; break
    if not svc_col:
        raise ValueError("Rentals: falta columna 'SVC/SVCs'.")
    if not qty_col:
        raise ValueError("Rentals: falta columna de cantidad (ej. 'Unidades disponibles').")
    out = (df.assign(svc=_upper_series(df[svc_col]), qty=_to_num(df[qty_col]).fillna(0.0))
             .groupby("svc", as_index=False)["qty"].sum()
             .rename(columns={"qty":"rentals_routes_max"}))
    return out

def load_srm() -> pd.DataFrame:
    raw = read_ws(SHEET_ID, "SRM")
    header_row = None
    for i in range(min(10, len(raw))):
        row = [str(x).strip().lower() for x in raw.iloc[i,:].tolist()]
        if any("svc" == x for x in row):
            header_row = i; break
    if header_row is None: header_row = 4
    cols = [(str(x).strip().lower() if str(x).strip() else f"col_{j+1}")
            for j,x in enumerate(raw.iloc[header_row,:].tolist())]
    df = raw.iloc[header_row+1:].reset_index(drop=True)
    df.columns = cols
    df = _lower_cols(df)
    svc_col = None
    for c in df.columns:
        if c in ("svc","svcs"): svc_col = c; break
    if not svc_col:
        cand = [c for c in df.columns if "svc" in c]
        svc_col = cand[0] if cand else None
    if not svc_col:
        raise ValueError("SRM: no se encontró columna SVC.")
    sdd_cols = [c for c in df.columns if ("total" in c and "sdd" in c)]
    spot_cols = [c for c in df.columns if ("total" in c and "spot" in c)]
    if not sdd_cols or not spot_cols:
        raise ValueError("SRM: no se hallaron columnas con 'sdd' o 'spot'.")
    for c in sdd_cols+spot_cols:
        df[c] = _to_num(df[c]).fillna(0.0)
    out = (df.assign(svc=_upper_series(df[svc_col]))
             .groupby("svc", as_index=False)[sdd_cols+spot_cols].sum())
    out["sdd_routes_max"]  = out[sdd_cols].sum(axis=1)
    out["spot_routes_max"] = out[spot_cols].sum(axis=1)
    return out[["svc","sdd_routes_max","spot_routes_max"]]

def load_crowd_caps() -> pd.DataFrame:
    raw = read_ws(SHEET_ID, "Crowd")
    df = _lower_cols(raw.copy())
    svc_col = None
    for c in df.columns:
        if c in ("svc","svcs"): svc_col = c; break
    if not svc_col:
        cand = [c for c in df.columns if "svc" in c]
        svc_col = cand[0] if cand else None
    if not svc_col:
        raise ValueError("Crowd: falta columna 'svc'.")
    def _pick(opts):
        for c in df.columns:
            for o in opts:
                if o in c: return c
        return None
    c_base_wd = _pick(["base entre","base semana"])
    c_base_sa = _pick(["base sab"])
    c_base_su = _pick(["base dom"])
    c_e1_wd   = _pick(["holgura entre","e1 entre"])
    c_e1_sa   = _pick(["holgura sab","e1 sab"])
    c_e1_su   = _pick(["holgura dom","e1 dom"])
    if not all([c_base_wd,c_base_sa,c_base_su,c_e1_wd,c_e1_sa,c_e1_su]):
        cols = list(df.columns)
        if "base" in cols and "e1" in cols:
            i_base = cols.index("base"); i_e1 = cols.index("e1")
            base_candidates = [cols[i_base]] + [c for c in cols[i_base+1:i_base+4]]
            e1_candidates   = [cols[i_e1]]   + [c for c in cols[i_e1+1:i_e1+4]]
            base_candidates = base_candidates[:3]
            e1_candidates   = e1_candidates[:3]
            if len(base_candidates)==3 and len(e1_candidates)==3:
                c_base_wd,c_base_sa,c_base_su = base_candidates
                c_e1_wd,  c_e1_sa,  c_e1_su   = e1_candidates
            else:
                raise ValueError("Crowd: no se reconoció layout compacto (base/e1).")
        else:
            raise ValueError("Crowd: no se reconoció layout. Encabezados: "+str(list(df.columns)))
    for c in [c_base_wd,c_base_sa,c_base_su,c_e1_wd,c_e1_sa,c_e1_su]:
        df[c] = _to_num(df[c]).fillna(0.0)
    out = (df.assign(svc=_upper_series(df[svc_col])).rename(columns={
        c_base_wd:"base_wd", c_base_sa:"base_sa", c_base_su:"base_su",
        c_e1_wd:"e1_wd",     c_e1_sa:"e1_sa",     c_e1_su:"e1_su"
    }))
    return out[["svc","base_wd","base_sa","base_su","e1_wd","e1_sa","e1_su"]]

# ------------------------------- Aux cálculos -------------------------------
def compute_spr_scenarios(fcst: pd.DataFrame, spr_real: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    target = fcst[["fecha","svc"]].drop_duplicates().copy()
    iso = pd.to_datetime(target["fecha"]).dt.isocalendar()
    target["dow"] = pd.to_datetime(target["fecha"]).dt.dayofweek
    target["iso_year"] = iso["year"].astype(int)

    spr_map = spr_real.set_index(["fecha","svc"])["spr_exec"]
    def last4(row):
        d,s = row["fecha"], row["svc"]
        vals=[]
        for k in [7,14,21,28]:
            dk = d - timedelta(days=k)
            v = spr_map.get((dk,s), np.nan)
            if pd.notna(v): vals.append(float(v))
        if not vals:
            mask = (spr_real["svc"].eq(s) & spr_real["fecha"].between(d - timedelta(days=28), d - timedelta(days=1)))
            vals = list(spr_real.loc[mask,"spr_exec"])
        return _safe_mean(vals)
    target["spr_promedio"] = target.apply(last4, axis=1)

    def peak(row):
        d,s,yr,dow = row["fecha"], row["svc"], row["iso_year"], row["dow"]
        m = (spr_real["svc"].eq(s) & spr_real["iso_year"].eq(yr) &
             spr_real["iso_week"].isin([20,21,22]) & spr_real["dow"].eq(dow))
        vals = list(spr_real.loc[m,"spr_exec"])
        if not vals:
            m = (spr_real["svc"].eq(s) & spr_real["iso_year"].eq(yr) &
                 spr_real["iso_week"].between(19,23) & spr_real["dow"].eq(dow))
            vals = list(spr_real.loc[m,"spr_exec"])
        return _safe_mean(vals)
    target["spr_peak"] = target.apply(peak, axis=1)

    cap = capacity.copy()
    m_spr = cap["tipo"].eq("spr")
    spr_plan = cap.loc[m_spr, ["svc","fecha","cantidad"]].rename(columns={"cantidad":"spr_plan"})
    if spr_plan.empty:
        by_svc = (cap.loc[m_spr].groupby("svc", as_index=False)["cantidad"]
                    .mean().rename(columns={"cantidad":"spr_plan"}))
        spr_plan = target[["fecha","svc"]].merge(by_svc, on="svc", how="left")
    target = target.merge(spr_plan, on=["fecha","svc"], how="left")
    return target[["fecha","svc","spr_promedio","spr_peak","spr_plan"]]

def compute_crowd_share(capacity: pd.DataFrame) -> pd.DataFrame:
    cap = capacity.copy()
    cap["tipo"] = cap["tipo"].str.strip().str.lower()
    ship = cap.loc[cap["tipo"].eq("shipments"), ["fecha","svc","delivery model","cantidad"]].copy()
    ship["delivery model"] = ship["delivery model"].str.strip().str.lower()
    tot = (ship.groupby(["fecha","svc"], as_index=False)["cantidad"]
              .sum().rename(columns={"cantidad":"ship_total"}))
    crw = (ship.loc[ship["delivery model"].eq("crowd")]
              .groupby(["fecha","svc"], as_index=False)["cantidad"]
              .sum().rename(columns={"cantidad":"ship_crowd"}))
    out = tot.merge(crw, on=["fecha","svc"], how="left").fillna({"ship_crowd":0.0})
    out["share_crowd_obj"] = np.where(out["ship_total"]>0,
                                      (out["ship_crowd"]/out["ship_total"]).clip(0,1), 0.0)
    return out

def map_crowd_capacity_by_date(target_days: pd.DataFrame, crowd_caps: pd.DataFrame) -> pd.DataFrame:
    def cap_for(row):
        s, d = row["svc"], row["fecha"]
        dow = _weekday(d)
        r = crowd_caps.loc[crowd_caps["svc"]==s]
        if r.empty: return pd.Series({"crowd_base_routes":0,"crowd_e1_routes":0})
        r = r.iloc[0]
        if dow <= 4: base,e1 = r["base_wd"], r["e1_wd"]
        elif dow==5: base,e1 = r["base_sa"], r["e1_sa"]
        else:        base,e1 = r["base_su"], r["e1_su"]
        return pd.Series({"crowd_base_routes":int(base), "crowd_e1_routes":int(e1)})
    tmp = target_days.apply(cap_for, axis=1)
    return pd.concat([target_days.reset_index(drop=True), tmp], axis=1)

def schedule_mlp_rest(df_day: pd.DataFrame) -> pd.DataFrame:
    out = df_day.copy()
    iso = pd.to_datetime(out["fecha"]).dt.isocalendar()
    out["week_key"] = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
    out["sdd_trabaja"]  = 1
    out["spot_trabaja"] = 1
    def proc(g):
        n = len(g)
        need_days = int((g["routes_mlp_need"]>0).sum())
        work_sdd  = min(6, need_days)
        rest_sdd  = max(n - work_sdd, 0)
        work_spot = 5 if need_days < 6 else 6
        if need_days < 5: work_spot = need_days
        rest_spot  = max(n - work_spot, 0)
        g_sorted = g.sort_values(["routes_mlp_need","fecha"])
        if rest_sdd>0:
            idx = g_sorted.head(rest_sdd).index
            g.loc[idx,"sdd_trabaja"] = 0
        g_sorted2 = g.sort_values(["routes_mlp_need","fecha"])
        if rest_spot>0:
            idx2 = g_sorted2.head(rest_spot).index
            g.loc[idx2,"spot_trabaja"] = 0
        return g
    out = out.groupby(["svc","week_key"], group_keys=False).apply(proc)
    return out.drop(columns=["week_key"])

# ------------------------------- Plan engine -------------------------------
def compute_plan(spr_mode: str, sel_svcs=None) -> pd.DataFrame:
    # Load con etiquetas de paso para debug fino
    try:
        fcst = load_fcst()
    except Exception as e:
        raise RuntimeError(f"[1/6 FCST] {e}") from e
    try:
        spr_real = load_spr_real()
    except Exception as e:
        raise RuntimeError(f"[2/6 SPR (real)] {e}") from e
    try:
        capacity = load_capacity()
    except Exception as e:
        raise RuntimeError(f"[3/6 Capacity] {e}") from e
    try:
        srm = load_srm()
    except Exception as e:
        raise RuntimeError(f"[4/6 SRM] {e}") from e
    try:
        rentals = load_rentals()
    except Exception as e:
        raise RuntimeError(f"[5/6 Rentals] {e}") from e
    try:
        crowd_caps = load_crowd_caps()
    except Exception as e:
        raise RuntimeError(f"[6/6 Crowd] {e}") from e

    if sel_svcs:
        fcst = fcst[fcst["svc"].isin(sel_svcs)]
        spr_real = spr_real[spr_real["svc"].isin(sel_svcs)]
        capacity = capacity[capacity["svc"].isin(sel_svcs)]
        srm = srm[srm["svc"].isin(sel_svcs)]
        rentals = rentals[rentals["svc"].isin(sel_svcs)]
        crowd_caps = crowd_caps[crowd_caps["svc"].isin(sel_svcs)]

    spr_tbl = compute_spr_scenarios(fcst, spr_real, capacity)
    spr_col = {"promedio":"spr_promedio","peak":"spr_peak","plan":"spr_plan"}[spr_mode]

    share_tbl = compute_crowd_share(capacity)

    cap_ship = capacity[capacity["tipo"].eq("shipments")]
    ship_dc = (cap_ship[cap_ship["delivery model"].str.contains("delivery cell|\\bdc\\b", regex=True)]
               .groupby(["fecha","svc"], as_index=False)["cantidad"].sum()
               .rename(columns={"cantidad":"ship_dc"}))
    ship_sp = (cap_ship[cap_ship["delivery model"].str.contains("service partner|\\bsp\\b", regex=True)]
               .groupby(["fecha","svc"], as_index=False)["cantidad"].sum()
               .rename(columns={"cantidad":"ship_sp"}))

    target_days = fcst[["fecha","svc"]].drop_duplicates()
    crowd_daily = map_crowd_capacity_by_date(target_days, crowd_caps)

    df = (fcst
          .merge(ship_dc, on=["fecha","svc"], how="left")
          .merge(ship_sp, on=["fecha","svc"], how="left")
          .merge(share_tbl, on=["fecha","svc"], how="left")
          .merge(crowd_daily, on=["fecha","svc"], how="left")
          .merge(srm, on="svc", how="left")
          .merge(rentals, on="svc", how="left")
          .merge(spr_tbl[["fecha","svc",spr_col]], on=["fecha","svc"], how="left"))

    for c in ["ship_dc","ship_sp","share_crowd_obj","crowd_base_routes","crowd_e1_routes",
              "sdd_routes_max","spot_routes_max","rentals_routes_max"]:
        df[c] = _to_num(df.get(c, 0)).fillna(0.0)

    df["spr_objetivo"] = _to_num(df[spr_col])
    df.drop(columns=[spr_col], inplace=True)

    df["ship_fcst_neto"] = (df["shipments"] - df["ship_dc"] - df["ship_sp"]).clip(lower=0.0)

    df["routes_need_total"] = np.where(
        (df["ship_fcst_neto"]>0) & (df["spr_objetivo"]>0),
        np.ceil(df["ship_fcst_neto"]/df["spr_objetivo"]).astype(int),
        0
    )
    df["alerta_spr_missing"] = ((df["ship_fcst_neto"]>0) & (df["spr_objetivo"].isna() | (df["spr_objetivo"]<=0)))

    df["routes_crowd_target"] = np.ceil(df["routes_need_total"] * df["share_crowd_obj"]).astype(int)

    df["routes_crowd_base"] = np.minimum(df["routes_crowd_target"], df["crowd_base_routes"]).astype(int)
    df["routes_crowd_e1"] = np.minimum(
        (df["routes_crowd_target"] - df["routes_crowd_base"]).clip(lower=0),
        df["crowd_e1_routes"]
    ).astype(int)
    df["routes_crowd_alloc"] = df["routes_crowd_base"] + df["routes_crowd_e1"]
    df["alerta_crowd_high"] = df["routes_crowd_e1"] > 0

    df["routes_after_crowd"] = (df["routes_need_total"] - df["routes_crowd_alloc"]).clip(lower=0).astype(int)

    df["routes_rentals_alloc"] = np.minimum(df["routes_after_crowd"], df["rentals_routes_max"]).astype(int)

    df["routes_mlp_need"] = (df["routes_after_crowd"] - df["routes_rentals_alloc"]).clip(lower=0).astype(int)

    rest_base = df[["fecha","svc","routes_mlp_need","sdd_routes_max","spot_routes_max"]].copy()
    rest_sched = schedule_mlp_rest(rest_base)
    df = df.merge(rest_sched[["fecha","svc","sdd_trabaja","spot_trabaja"]], on=["fecha","svc"], how="left")

    df["routes_mlp_cap_day"] = (df["sdd_routes_max"]*df["sdd_trabaja"] + df["spot_routes_max"]*df["spot_trabaja"]).fillna(0).astype(int)
    df["routes_mlp_alloc"]   = np.minimum(df["routes_mlp_need"], df["routes_mlp_cap_day"]).astype(int)

    df["routes_deficit_pre_extra"] = (df["routes_mlp_need"] - df["routes_mlp_alloc"]).clip(lower=0).astype(int)
    df["crowd_e1_remaining"] = (df["crowd_e1_routes"] - df["routes_crowd_e1"]).clip(lower=0).astype(int)
    df["routes_crowd_extra"] = np.minimum(df["routes_deficit_pre_extra"], df["crowd_e1_remaining"]).astype(int)

    df["routes_total_alloc"] = (df["routes_crowd_alloc"] + df["routes_rentals_alloc"] +
                                df["routes_mlp_alloc"] + df["routes_crowd_extra"]).astype(int)

    df["shipments_plan"] = np.where(df["spr_objetivo"]>0, df["routes_total_alloc"]*df["spr_objetivo"], 0.0)
    df["routes_deficit"] = (df["routes_need_total"] - df["routes_total_alloc"]).clip(lower=0).astype(int)
    df["alerta_deficit"] = df["shipments_plan"] + 1e-6 < df["ship_fcst_neto"]

    df["spr_logrado"] = np.where(df["routes_total_alloc"]>0, df["ship_fcst_neto"]/df["routes_total_alloc"], np.nan)
    df["share_crowd_real"] = np.where(df["routes_need_total"]>0,
                                      (df["routes_crowd_alloc"] + df["routes_crowd_extra"]) / df["routes_need_total"], 0.0)
    df["risk_flag"] = df["alerta_deficit"] | df["alerta_spr_missing"]

    cols = [
        "fecha","svc",
        "shipments","ship_dc","ship_sp","ship_fcst_neto","spr_objetivo",
        "routes_need_total",
        "share_crowd_obj","routes_crowd_target","routes_crowd_base","routes_crowd_e1","routes_crowd_extra","routes_crowd_alloc","alerta_crowd_high",
        "rentals_routes_max","routes_rentals_alloc",
        "sdd_routes_max","spot_routes_max","sdd_trabaja","spot_trabaja","routes_mlp_need","routes_mlp_cap_day","routes_mlp_alloc",
        "routes_deficit","routes_total_alloc","shipments_plan","spr_logrado","share_crowd_real",
        "alerta_spr_missing","alerta_deficit","risk_flag"
    ]
    return df[cols].sort_values(["fecha","svc"]).reset_index(drop=True)

# ------------------------------- Sidebar -------------------------------
with st.sidebar:
    st.header("📁 Proyecto")
    st.write(f"**Sheet:** `{SHEET_ID}`")
    st.subheader("🔐 Credenciales")
    svc_email = get_service_account_email()
    if svc_email:
        st.caption("Comparte el Sheet con:")
        st.code(svc_email, language="text")
    else:
        st.warning("No se detectó Service Account.")

# ------------------------------- UI run -------------------------------
with st.expander("➤ Cargando datos...", expanded=True):
    try:
        # Para selector de SVC (rápido)
        _fcst = load_fcst()
        svc_list = sorted(_fcst["svc"].dropna().astype(str).unique().tolist())
        sel_svcs = st.multiselect("Filtrar SVC", svc_list, default=svc_list)

        plan = compute_plan(spr_mode, sel_svcs=sel_svcs)

        st.subheader("Tabla principal — (svc, fecha) × Delivery model")
        st.dataframe(plan, use_container_width=True, hide_index=True)

        st.subheader("Riesgos por fecha")
        resumen = (plan.groupby("fecha", as_index=False)
                        .agg(
                            svcs_con_deficit=("alerta_deficit","sum"),
                            rutas_deficit=("routes_deficit","sum"),
                            svcs_sin_spr=("alerta_spr_missing","sum"),
                        ))
        st.dataframe(resumen, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error: {e}")
        st.caption("Detalle de la excepción:")
        st.code(traceback.format_exc())
