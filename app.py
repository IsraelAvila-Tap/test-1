# -*- coding: utf-8 -*-
import os, json, yaml
import pandas as pd
import numpy as np
from math import ceil
from datetime import timedelta
import streamlit as st

# ---------------- Credenciales desde Secrets -> variable de entorno ----------------
if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
elif "gcp_service_account" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps(dict(st.secrets["gcp_service_account"]))

if "PROJECT_KEY" in st.secrets:
    os.environ["PROJECT_KEY"] = st.secrets["PROJECT_KEY"]

from utils_gsheets import read_ws, _client, get_service_account_email

# ---------------- UI config ----------------
st.set_page_config(page_title="Mel-IA — Plan táctico", layout="wide")

# ---------------- Helpers ----------------
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("%", "", regex=False)
         .str.strip(),
        errors="coerce"
    )

def _weekday(d) -> int:
    return pd.Timestamp(d).weekday()  # 0..6 (L..D)

def _safe_mean(vals):
    vals = [float(v) for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan

@st.cache_resource
def load_config():
    with open("config.yaml","r",encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()
proj_key = list(cfg["projects"].keys())[0]
proj = cfg["projects"][proj_key]
SHEET_ID = proj["sheet_id"]

def _norm_cols(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def _iso_year_week(d):
    iso = pd.Timestamp(d).isocalendar()
    return int(iso.year), int(iso.week)

# ---------------- Loaders ----------------
@st.cache_data(ttl=300)
def load_fcst() -> pd.DataFrame:
    df = _norm_cols(read_ws(SHEET_ID, "FCST"))
    need = {"svc","fecha","shipments"}
    if not need.issubset(df.columns):
        raise ValueError(f"FCST: faltan columnas {need - set(df.columns)}")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
    df["shipments"] = _to_num(df["shipments"]).fillna(0.0)
    return df[["svc","fecha","shipments"]]

@st.cache_data(ttl=300)
def load_spr_real() -> pd.DataFrame:
    df = _norm_cols(read_ws(SHEET_ID, "SPR"))
    need = {"svc","fecha","spr"}
    if not need.issubset(df.columns):
        raise ValueError(f"SPR: faltan columnas {need - set(df.columns)}")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
    df["spr"]   = _to_num(df["spr"])
    df = df.dropna(subset=["spr"])
    day = df.groupby(["fecha","svc"], as_index=False)["spr"].mean().rename(columns={"spr":"spr_exec"})
    day["dow"] = day["fecha"].apply(_weekday)
    iso = day["fecha"].apply(lambda d: pd.Timestamp(d).isocalendar())
    day["iso_year"] = [int(x.year) for x in iso]
    day["iso_week"] = [int(x.week) for x in iso]
    return day

@st.cache_data(ttl=300)
def load_capacity() -> pd.DataFrame:
    df = _norm_cols(read_ws(SHEET_ID, "Capacity"))
    need = {"delivery model","tipo","svc","fecha","cantidad"}
    if not need.issubset(df.columns):
        raise ValueError(f"Capacity: faltan columnas {need - set(df.columns)}")
    df["fecha"]    = pd.to_datetime(df["fecha"], errors="coerce").dt.date
    df["cantidad"] = _to_num(df["cantidad"]).fillna(0.0)
    return df

@st.cache_data(ttl=300)
def load_rentals() -> pd.DataFrame:
    df = _norm_cols(read_ws(SHEET_ID, "Rentals"))
    svc_col = "svc" if "svc" in df.columns else ("svcs" if "svcs" in df.columns else None)
    qty_col = [c for c in df.columns if "unidades dispon" in c]
    if not svc_col or not qty_col:
        raise ValueError("Rentals: se esperan columnas 'SVC/SVCs' y 'Unidades disponibles'.")
    qty_col = qty_col[0]
    out = df.groupby(svc_col, as_index=False)[qty_col].sum().rename(columns={svc_col:"svc", qty_col:"rentals_routes_max"})
    out["rentals_routes_max"] = _to_num(out["rentals_routes_max"]).fillna(0).astype(int)
    return out

@st.cache_data(ttl=300)
def load_crowd_caps() -> pd.DataFrame:
    df = _norm_cols(read_ws(SHEET_ID, "Crowd"))
    if "svc" not in df.columns:
        # modo compacto: col_2 es svc, 'base' y 'e1' con wd/sa/su en filas
        # pero para operatividad exigimos layout detallado:
        raise ValueError("Crowd: falta columna 'svc'. Usa layout detallado con columnas base/e1 por día (entre semana/sábado/domingo).")
    # localizar columnas
    def pick(*opts):
        for c in df.columns:
            for o in opts:
                if o in c:
                    return c
        return None
    c_base_wd = pick("base entre")
    c_base_sa = pick("base sab")
    c_base_su = pick("base domingo","base dom")
    c_e1_wd   = pick("holgura entre","e1 entre")
    c_e1_sa   = pick("holgura sab","e1 sab")
    c_e1_su   = pick("holgura domingo","e1 dom")

    need_cols = [c_base_wd,c_base_sa,c_base_su,c_e1_wd,c_e1_sa,c_e1_su]
    if any(c is None for c in need_cols):
        raise ValueError("Crowd: no se encontraron columnas esperadas: base entre semana, base sábado, base domingo, E1 entre semana, E1 sábado, E1 domingo.")

    for c in need_cols:
        df[c] = _to_num(df[c]).fillna(0).astype(int)

    out = df[["svc", c_base_wd,c_base_sa,c_base_su,c_e1_wd,c_e1_sa,c_e1_su]].copy()
    out.columns = ["svc","base_wd","base_sa","base_su","e1_wd","e1_sa","e1_su"]
    return out

# --- SRM dinámico (muchas columnas) ---
@st.cache_data(ttl=300)
def load_srm() -> pd.DataFrame:
    HEADER_ROW = int(os.getenv("SRM_HEADER_ROW", "5"))
    MAX_ROWS   = int(os.getenv("SRM_MAX_ROWS", "2000"))
    MAX_COLS   = int(os.getenv("SRM_MAX_COLS", "2000"))

    def col_to_a1(n: int) -> str:
        s = ""
        while n > 0:
            n, r = divmod(n - 1, 26)
            s = chr(65 + r) + s
        return s

    gc = _client(); sh = gc.open_by_key(SHEET_ID); ws = sh.worksheet("SRM")
    try:
        col_count = int(ws.col_count or 0)
    except Exception:
        col_count = MAX_COLS
    col_count = min(col_count if col_count>0 else MAX_COLS, MAX_COLS)

    rng = f"A{HEADER_ROW}:{col_to_a1(col_count)}{HEADER_ROW+MAX_ROWS}"
    cells = ws.get(rng, value_render_option="UNFORMATTED_VALUE")
    if not cells or len(cells)<2: raise ValueError(f"SRM vacío en rango {rng}")

    header = cells[0]; rows = cells[1:]
    seen={}; cols=[]
    for j,h in enumerate(header):
        base = (h or "").replace("\n"," ").strip() or f"col_{j+1}"
        name=base; k=1
        while name in seen:
            k+=1; name=f"{base}_{k}"
        seen[name]=1; cols.append(name)
    df = pd.DataFrame(rows, columns=[c.strip().lower() for c in cols])

    svc_cols = [c for c in df.columns if "svc" in c.replace(" ","")]
    if not svc_cols: raise ValueError(f"SRM: no hay SVC en {list(df.columns)}")
    svc_col = svc_cols[0]

    sdd_cols  = [c for c in df.columns if "sdd"  in c]
    spot_cols = [c for c in df.columns if "spot" in c]
    if not sdd_cols and not spot_cols:
        raise ValueError(f"SRM: no hay columnas con 'sdd' o 'spot' en {list(df.columns)}")

    for c in sdd_cols+spot_cols:
        df[c] = _to_num(df[c]).fillna(0)

    out = df[[svc_col]+sdd_cols+spot_cols].copy().rename(columns={svc_col:"svc"})
   
    out["svc"] = out["svc"].astype(str).str.strip().str.upper()
    out["sdd_routes_max"]  = out[sdd_cols].sum(axis=1) if sdd_cols else 0
    out["spot_routes_max"] = out[spot_cols].sum(axis=1) if spot_cols else 0
    out = out.groupby("svc", as_index=False)[["sdd_routes_max","spot_routes_max"]].sum()

    st.caption(f"SRM leído en rango {rng} · columnas={col_count} (tope={MAX_COLS}) · SVC='svc' · SDD={len(sdd_cols)} · SPOT={len(spot_cols)}")
    return out

# ---------------- KPIs auxiliares ----------------
def compute_spr_scenarios(fcst: pd.DataFrame, spr_real: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    target = fcst[["fecha","svc"]].drop_duplicates().copy()
    target["dow"] = target["fecha"].apply(_weekday)
    target["iso_year"] = target["fecha"].apply(lambda d: int(pd.Timestamp(d).isocalendar().year))

    # SPR_promedio: últimas 4 semanas, mismo día de semana
    sr_map = spr_real.set_index(["fecha","svc"])["spr_exec"]
    def avg_last4(row):
        d, s = row["fecha"], row["svc"]; vals=[]
        for k in (7,14,21,28):
            v = sr_map.get((d - timedelta(days=k), s), np.nan)
            if pd.notna(v): vals.append(float(v))
        if not vals:
            mask = (spr_real["svc"].eq(s) &
                    spr_real["fecha"].between(d - timedelta(days=28), d - timedelta(days=1)))
            vals = list(spr_real.loc[mask,"spr_exec"])
        return _safe_mean(vals)
    target["spr_promedio"] = target.apply(avg_last4, axis=1)

    # SPR_peak: sem 20-22 del mismo año, mismo DOW (fallback 19..23)
    def avg_peak(row):
        d, s, yr, dow = row["fecha"], row["svc"], row["iso_year"], row["dow"]
        m = (spr_real["svc"].eq(s)&spr_real["iso_year"].eq(yr)&
             spr_real["iso_week"].isin([20,21,22])&spr_real["dow"].eq(dow))
        vals = list(spr_real.loc[m,"spr_exec"])
        if not vals:
            m = (spr_real["svc"].eq(s)&spr_real["iso_year"].eq(yr)&
                 spr_real["iso_week"].between(19,23)&spr_real["dow"].eq(dow))
            vals = list(spr_real.loc[m,"spr_exec"])
        return _safe_mean(vals)
    target["spr_peak"] = target.apply(avg_peak, axis=1)

    # SPR_plan desde Capacity (Tipo == 'SPR'), AGRUPADO por fecha+svc
    cap = capacity.copy()
    m_spr = cap["tipo"].str.strip().str.lower().eq("spr")
    spr_plan = (cap.loc[m_spr]
                  .groupby(["svc","fecha"], as_index=False)["cantidad"]
                  .mean().rename(columns={"cantidad":"spr_plan"}))
    if spr_plan.empty:
        spr_by_svc = (cap.loc[m_spr]
                        .groupby("svc", as_index=False)["cantidad"]
                        .mean().rename(columns={"cantidad":"spr_plan"}))
        spr_plan = target[["fecha","svc"]].merge(spr_by_svc, on="svc", how="left")

    target = target.merge(spr_plan, on=["fecha","svc"], how="left")
    return target[["fecha","svc","spr_promedio","spr_peak","spr_plan"]]

def _dm_key(x:str) -> str:
    x = (x or "").strip().lower()
    if "delivery" in x and "cell" in x: return "dc"
    if "service" in x and "partner" in x: return "sp"
    if "crowd" in x: return "crowd"
    if "rental" in x: return "rentals"
    if "mlp" in x: return "mlp"
    return x

@st.cache_data(ttl=300)
def compute_crowd_share(capacity: pd.DataFrame) -> pd.DataFrame:
    cap = capacity.copy()
    cap["tipo"] = cap["tipo"].str.strip().str.lower()
    m_ship = cap["tipo"].eq("shipments")
    ship = cap.loc[m_ship, ["fecha","svc","delivery model","cantidad"]].copy()
    ship["dm"] = ship["delivery model"].apply(_dm_key)

    tot = ship.groupby(["fecha","svc"], as_index=False)["cantidad"].sum().rename(columns={"cantidad":"ship_total"})
    crw = (ship.loc[ship["dm"].eq("crowd")]
                  .groupby(["fecha","svc"], as_index=False)["cantidad"]
                  .sum().rename(columns={"cantidad":"ship_crowd"}))
    out = tot.merge(crw, on=["fecha","svc"], how="left").fillna({"ship_crowd":0.0})
    out["share_crowd_obj"] = np.where(out["ship_total"]>0, (out["ship_crowd"]/out["ship_total"]).clip(0,1), 0.0)
    return out[["fecha","svc","share_crowd_obj","ship_total","ship_crowd"]]

@st.cache_data(ttl=300)
def compute_dc_sp_shipments(capacity: pd.DataFrame) -> pd.DataFrame:
    cap = capacity.copy()
    cap["tipo"] = cap["tipo"].str.strip().str.lower()
    m_ship = cap["tipo"].eq("shipments")
    ship = cap.loc[m_ship, ["fecha","svc","delivery model","cantidad"]].copy()
    ship["dm"] = ship["delivery model"].apply(_dm_key)
    dc = ship.loc[ship["dm"].eq("dc")].groupby(["fecha","svc"], as_index=False)["cantidad"].sum().rename(columns={"cantidad":"ship_dc"})
    sp = ship.loc[ship["dm"].eq("sp")].groupby(["fecha","svc"], as_index=False)["cantidad"].sum().rename(columns={"cantidad":"ship_sp"})
    out = dc.merge(sp, on=["fecha","svc"], how="outer").fillna(0.0)
    return out

def map_crowd_capacity_by_date(target_days: pd.DataFrame, crowd_caps: pd.DataFrame) -> pd.DataFrame:
    # devuelve capacidad crowd diaria (base y e1) por svc+fecha
    def cap_for(row):
        s, d = row["svc"], row["fecha"]
        dow = _weekday(d)
        r = crowd_caps.loc[crowd_caps["svc"]==s]
        if r.empty: return pd.Series({"crowd_base_routes":0,"crowd_e1_routes":0})
        r = r.iloc[0]
        if   dow <= 4: base, e1 = r["base_wd"], r["e1_wd"]
        elif dow == 5: base, e1 = r["base_sa"],  r["e1_sa"]
        else:          base, e1 = r["base_su"],  r["e1_su"]
        return pd.Series({"crowd_base_routes":int(base), "crowd_e1_routes":int(e1)})
    tmp = target_days.apply(cap_for, axis=1)
    return pd.concat([target_days.reset_index(drop=True), tmp], axis=1)

def schedule_mlp_rest(df_day: pd.DataFrame) -> pd.DataFrame:
    out = df_day.copy()
    out["week_key"] = out["fecha"].apply(lambda d: f"{_iso_year_week(d)[0]}-{_iso_year_week(d)[1]:02d}")
    out["sdd_trabaja"]  = 1
    out["spot_trabaja"] = 1

    def proc(g):
        n = len(g)
        need_days = int((g["routes_mlp_need"]>0).sum())
        work_sdd = min(6, need_days)  # 6x7
        work_spot = 5                 # 5x7 (si ≥6 días con necesidad sube a 6, si <5 usa solo los días necesarios)
        if need_days >= 6: work_spot = 6
        elif need_days < 5: work_spot = need_days
        rest_sdd = max(n - work_sdd, 0)
        rest_spot = max(n - work_spot, 0)

        g_sorted = g.sort_values(["routes_mlp_need","fecha"], ascending=[True,True])
        if rest_sdd>0:  g.loc[g_sorted.head(rest_sdd).index,"sdd_trabaja"]=0
        g_sorted2 = g.sort_values(["routes_mlp_need","fecha"], ascending=[True,True])
        if rest_spot>0: g.loc[g_sorted2.head(rest_spot).index,"spot_trabaja"]=0
        return g

    out = out.groupby(["svc","week_key"], group_keys=False).apply(proc)
    return out.drop(columns=["week_key"])

# ---------------- Motor principal ----------------
def compute_plan(spr_mode: str, sel_svcs=None):
    with st.status("Cargando datos...", expanded=True) as status:
        st.write("1/6 FCST…");       fcst       = load_fcst()
        st.write("2/6 SPR (real)…"); spr_real   = load_spr_real()
        st.write("3/6 Capacity…");   capacity   = load_capacity()
        st.write("4/6 SRM…");        srm        = load_srm()
        st.write("5/6 Rentals…");    rentals    = load_rentals()
        st.write("6/6 Crowd…");      crowd_caps = load_crowd_caps()
        status.update(label="Datos listos ✅", state="complete")

    # Filtro SVC (chips de la izquierda)
    if sel_svcs:
        sel_svcs = set(str(s).strip().upper() for s in sel_svcs if s)
        fcst       = fcst[fcst["svc"].isin(sel_svcs)]
        spr_real   = spr_real[spr_real["svc"].isin(sel_svcs)]
        capacity   = capacity[capacity["svc"].isin(sel_svcs)]
        srm        = srm[srm["svc"].isin(sel_svcs)]
        rentals    = rentals[rentals["svc"].isin(sel_svcs)]
        crowd_caps = crowd_caps[crowd_caps["svc"].isin(sel_svcs)]

    # SPRs
    spr_tbl = compute_spr_scenarios(fcst, spr_real, capacity)
    spr_col = {"promedio":"spr_promedio","peak":"spr_peak","plan":"spr_plan"}[spr_mode]

    # Share crowd + Shipments DC/SP
    share_tbl = compute_crowd_share(capacity)
    dcspship  = compute_dc_sp_shipments(capacity)

    # Capacidades diarias de Crowd
    target_days = fcst[["fecha","svc"]].drop_duplicates()
    crowd_daily = map_crowd_capacity_by_date(target_days, crowd_caps)

    # Merge base
    df = (fcst
          .merge(share_tbl, on=["fecha","svc"], how="left")
          .merge(dcspship, on=["fecha","svc"], how="left")
          .merge(crowd_daily, on=["fecha","svc"], how="left")
          .merge(srm, on="svc", how="left")
          .merge(rentals, on="svc", how="left")
          .merge(spr_tbl[["fecha","svc",spr_col]], on=["fecha","svc"], how="left")
         )
    df = df.fillna({"ship_dc":0.0,"ship_sp":0.0})
    df = df.drop_duplicates(subset=["fecha","svc"]).copy()

    # Limpieza
    for c in ["share_crowd_obj","crowd_base_routes","crowd_e1_routes",
              "sdd_routes_max","spot_routes_max","rentals_routes_max"]:
        df[c] = pd.to_numeric(df.get(c,0), errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0)
    df["spr_objetivo"] = pd.to_numeric(df[spr_col], errors="coerce")
    df.drop(columns=[spr_col], inplace=True)

    # FCST neto para flota (desc. DC y SP)
    df["ship_fcst_neto"] = (df["shipments"] - df["ship_dc"] - df["ship_sp"]).clip(lower=0)

    # Rutas requeridas
    df["routes_need_total"] = np.where(
        (df["ship_fcst_neto"]>0) & (df["spr_objetivo"]>0),
        np.ceil(df["ship_fcst_neto"]/df["spr_objetivo"]),
        0.0
    )
    df["alerta_spr_missing"] = ((df["ship_fcst_neto"]>0) & (df["spr_objetivo"].isna() | (df["spr_objetivo"]<=0)))

    # Objetivo crowd por share
    df["routes_crowd_target"] = np.ceil(df["routes_need_total"] * df["share_crowd_obj"])

    # Asignación crowd objetivo (base y E1 hasta target)
    df["routes_crowd_base_obj"] = np.minimum(df["routes_crowd_target"], df["crowd_base_routes"])
    df["routes_crowd_e1_obj"]   = np.minimum(
        (df["routes_crowd_target"] - df["routes_crowd_base_obj"]).clip(lower=0),
        df["crowd_e1_routes"]
    )
    df["routes_crowd_alloc"] = df["routes_crowd_base_obj"] + df["routes_crowd_e1_obj"]

    # Rentals directo
    df["routes_after_crowd"] = (df["routes_need_total"] - df["routes_crowd_alloc"]).clip(lower=0)
    df["routes_rentals_alloc"] = np.minimum(df["routes_after_crowd"], df["rentals_routes_max"])

    # Necesidad para MLP
    df["routes_mlp_need"] = (df["routes_after_crowd"] - df["routes_rentals_alloc"]).clip(lower=0)

    # Descansos MLP + capacidad día
    base_rest = df[["fecha","svc","routes_mlp_need","sdd_routes_max","spot_routes_max"]].copy()
    rest_sched = schedule_mlp_rest(base_rest)
    df = df.merge(rest_sched[["fecha","svc","sdd_trabaja","spot_trabaja"]], on=["fecha","svc"], how="left")
    df["sdd_cap_day"]  = (df["sdd_routes_max"]  * df["sdd_trabaja"]).fillna(0.0)
    df["spot_cap_day"] = (df["spot_routes_max"] * df["spot_trabaja"]).fillna(0.0)

    # Asignación MLP (primero SDD, luego Spot)
    df["routes_mlp_sdd_alloc"]  = np.minimum(df["routes_mlp_need"], df["sdd_cap_day"])
    df["routes_mlp_spot_alloc"] = np.minimum((df["routes_mlp_need"] - df["routes_mlp_sdd_alloc"]).clip(lower=0), df["spot_cap_day"])
    df["routes_mlp_alloc"]      = df["routes_mlp_sdd_alloc"] + df["routes_mlp_spot_alloc"]

    # Déficit después de MLP
    df["routes_deficit"] = (df["routes_mlp_need"] - df["routes_mlp_alloc"]).clip(lower=0)

    # -------- Escalamiento Crowd si persiste déficit (sin rebasar cap total Crowd) --------
    # capacidad restante de Crowd base/E1 no usada en el objetivo inicial
    df["crowd_base_rem"] = (df["crowd_base_routes"] - df["routes_crowd_base_obj"]).clip(lower=0)
    df["crowd_e1_rem"]   = (df["crowd_e1_routes"]   - df["routes_crowd_e1_obj"]).clip(lower=0)

    extra_base = np.minimum(df["routes_deficit"], df["crowd_base_rem"])
    df["routes_crowd_base_extra"] = extra_base
    extra_e1 = np.minimum((df["routes_deficit"] - extra_base).clip(lower=0), df["crowd_e1_rem"])
    df["routes_crowd_e1_extra"] = extra_e1

    df["routes_crowd_base"] = df["routes_crowd_base_obj"] + df["routes_crowd_base_extra"]
    df["routes_crowd_e1"]   = df["routes_crowd_e1_obj"]   + df["routes_crowd_e1_extra"]
    df["routes_crowd_total"] = df["routes_crowd_base"] + df["routes_crowd_e1"]

    # Recalcular totales y déficit final
    df["routes_total_alloc"] = (df["routes_crowd_total"] + df["routes_rentals_alloc"] +
                                df["routes_mlp_sdd_alloc"] + df["routes_mlp_spot_alloc"])
    df["routes_deficit_final"] = (df["routes_need_total"] - df["routes_total_alloc"]).clip(lower=0)

    # Shipments por modelo (con el SPR elegido)
    df["ship_crowd_base"] = df["routes_crowd_base"]   * df["spr_objetivo"]
    df["ship_crowd_e1"]   = df["routes_crowd_e1"]     * df["spr_objetivo"]
    df["ship_rentals"]    = df["routes_rentals_alloc"]* df["spr_objetivo"]
    df["ship_mlp_sdd"]    = df["routes_mlp_sdd_alloc"]* df["spr_objetivo"]
    df["ship_mlp_spot"]   = df["routes_mlp_spot_alloc"]* df["spr_objetivo"]

    df["ship_fleet_plan"] = (df["ship_crowd_base"] + df["ship_crowd_e1"] + df["ship_rentals"] +
                             df["ship_mlp_sdd"] + df["ship_mlp_spot"])
    df["ship_total_plan"] = (df["ship_dc"] + df["ship_sp"] + df["ship_fleet_plan"])

    # Flags
    df["alerta_crowd_high"] = (df["routes_crowd_e1"] > 0)
    df["alerta_deficit"]    = (df["ship_fleet_plan"] + 1e-6 < df["ship_fcst_neto"])
    df["risk_flag"]         = df["alerta_deficit"] | df["alerta_spr_missing"]

    # ---- Tabla 1: FILAS = (svc, fecha) y COLUMNAS = Delivery model ----
    cols = [
        "fecha","svc",
        # FCST y netos
        "shipments","ship_dc","ship_sp","ship_fcst_neto","spr_objetivo",
        # Rutas requeridas y totales
        "routes_need_total","routes_total_alloc","routes_deficit_final",
        # Crowd
        "routes_crowd_base","routes_crowd_e1",
        # Rentals
        "routes_rentals_alloc",
        # MLP
        "routes_mlp_sdd_alloc","routes_mlp_spot_alloc",
        # Shipments logrados por modelo
        "ship_crowd_base","ship_crowd_e1","ship_rentals","ship_mlp_sdd","ship_mlp_spot",
        "ship_fleet_plan","ship_total_plan",
        # Señales
        "alerta_crowd_high","alerta_spr_missing","alerta_deficit","risk_flag"
    ]
    t1 = df[cols].sort_values(["fecha","svc"]).reset_index(drop=True)

    return t1

# ---------------- Sidebar / UI ----------------
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
    st.markdown("---")

st.title("Mel-IA — Plan táctico (diario por SVC)")

spr_mode = st.radio("SPR objetivo", ["promedio","peak","plan"], index=0, horizontal=True)

# Filtro rápido por SVC
try:
    all_fcst = load_fcst()
    svc_list = sorted(all_fcst["svc"].astype(str).unique().tolist())
except Exception:
    svc_list = []

with st.sidebar:
    st.markdown("### Filtrar SVC")
    sel_svcs = st.multiselect("", svc_list, default=svc_list[:4])

# --------- RUN ----------
try:
    tabla = compute_plan(spr_mode, sel_svcs=sel_svcs)

    st.subheader("Tabla principal — (svc, fecha) × Delivery model")
    st.dataframe(tabla, use_container_width=True, hide_index=True)

    st.subheader("Riesgos por fecha")
    resumen = (tabla.groupby("fecha", as_index=False)
               .agg(
                   svcs_con_deficit=("alerta_deficit","sum"),
                   rutas_deficit=("routes_deficit_final","sum"),
                   svcs_sin_spr=("alerta_spr_missing","sum"),
               ))
    st.dataframe(resumen, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Error: {e}")




