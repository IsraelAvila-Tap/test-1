# app.py
# Mel-IA — Plan táctico (diario por SVC)
# Streamlit + Google Sheets (service account vía st.secrets)
# Flujo: FCST – DC – SP – Rentals*SPR – Crowd (% objetivo)*SPR – MLP SDD(6x7)*SPR – MLP Spot(5x7)*SPR – Crowd extra (sin rebasar nube)

import os, json, yaml, math
from math import ceil
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# 0) CREDENCIALES (dos formatos) y PROJECT_KEY opcional
# -------------------------------------------------------------------
if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
elif "gcp_service_account" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps(dict(st.secrets["gcp_service_account"]))

if "PROJECT_KEY" in st.secrets:
    os.environ["PROJECT_KEY"] = st.secrets["PROJECT_KEY"]

# -------------------------------------------------------------------
# 1) UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Mel-IA — Plan táctico", layout="wide")
st.title("Mel-IA — Plan táctico (diario por SVC)")
spr_mode = st.radio("SPR objetivo", ["promedio", "peak", "plan"], index=0, horizontal=True)

# -------------------------------------------------------------------
# 2) UTILS
# -------------------------------------------------------------------
def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas a minúsculas, sin espacios."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace("\n", " ") for c in df.columns]
    return df

def _ensure_series(x) -> pd.Series:
    return x if isinstance(x, pd.Series) else pd.Series(x)

def _upper_series(s: pd.Series) -> pd.Series:
    s = _ensure_series(s)
    return s.astype(str).str.strip().str.upper()

def _to_num(s: pd.Series) -> pd.Series:
    s = _ensure_series(s).astype(str)
    s = (
        s.str.replace(",", "", regex=False)
         .str.replace("%", "", regex=False)
         .str.strip()
    )
    return pd.to_numeric(s, errors="coerce")

def _weekday(d) -> int:
    """0=Lunes ... 6=Domingo; soporta str/date/ts."""
    return pd.to_datetime(d, errors="coerce", dayfirst=True).dayofweek

def _iso_yr_week(d):
    ts = pd.to_datetime(d, errors="coerce", dayfirst=True)
    iso = ts.dt.isocalendar() if isinstance(ts, pd.Series) else ts.isocalendar()
    if isinstance(iso, pd.DataFrame):  # Series-like (pandas >= 1.4)
        return (iso["year"], iso["week"])
    return (iso.year, iso.week)

def _safe_mean(vals):
    vals = [float(v) for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan

# -------------------------------------------------------------------
# 3) Google Sheets utils (tu helper local)
# -------------------------------------------------------------------
from utils_gsheets import read_ws, _client, get_service_account_email

@st.cache_resource
def load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()
proj_key = list(cfg["projects"].keys())[0]
SHEET_ID = cfg["projects"][proj_key]["sheet_id"]

# -------------------------------------------------------------------
# 4) LOADERS robustos (detectan encabezados en filas 1..15)
# -------------------------------------------------------------------
def _reheader(raw: pd.DataFrame, prefer=("svc","fecha","shipments"), scan_rows=15) -> pd.DataFrame:
    """Busca una fila de encabezado con preferencias y re-encabeza."""
    if raw.empty:
        return raw
    header_row = None
    pref = set(x.lower() for x in prefer)

    # criterio fuerte (todas)
    for i in range(min(scan_rows, len(raw))):
        row = [str(x).strip().lower() for x in raw.iloc[i, :].tolist()]
        if pref.issubset(set(row)):
            header_row = i
            break

    # criterio laxo (≥2 coincidencias)
    if header_row is None:
        for i in range(min(scan_rows, len(raw))):
            row = set(str(x).strip().lower() for x in raw.iloc[i, :].tolist())
            if len(pref.intersection(row)) >= 2:
                header_row = i
                break

    if header_row is not None:
        cols = [
            (str(x).strip().lower() if str(x).strip() else f"col_{j+1}")
            for j, x in enumerate(raw.iloc[header_row, :].tolist())
        ]
        df = raw.iloc[header_row + 1 :].reset_index(drop=True)
        df.columns = cols
        return _lower_cols(df)

    return _lower_cols(raw.copy())

# 4.1 FCST ------------------------------------------------------------
def load_fcst() -> pd.DataFrame:
    raw = read_ws(SHEET_ID, "FCST")
    df = _reheader(raw, prefer=("svc","fecha","shipments"))
    need = {"svc","fecha","shipments"}
    if not need.issubset(df.columns):
        raise ValueError(f"FCST: faltan columnas {sorted(list(need))}")

    out = pd.DataFrame({
        "svc": _upper_series(df["svc"]),
        "fecha": pd.to_datetime(_ensure_series(df["fecha"]), errors="coerce", dayfirst=True).dt.date,
        "shipments": _to_num(df["shipments"]).fillna(0.0)
    }).dropna(subset=["fecha","svc"])
    return out

# 4.2 SPR (histórico real) --------------------------------------------
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
    # derivadas
    day["dow"] = day["fecha"].apply(_weekday)
    iso = pd.to_datetime(day["fecha"]).dt.isocalendar()
    day["iso_year"] = iso["year"].astype(int)
    day["iso_week"] = iso["week"].astype(int)
    return day

# 4.3 Capacity (para share crowd y SPR plan + DC/SP si existieran) ----
def load_capacity() -> pd.DataFrame:
    raw = read_ws(SHEET_ID, "Capacity")
    df = _reheader(raw, prefer=("delivery model","tipo","svc","fecha","cantidad"))

    # normaliza nombres típicos
    rename = {
        "delivery model":"delivery model",
        "delivery_model":"delivery model",
        "tipo dm":"tipo dm",
        "tipo_dm":"tipo dm",
        "cantidad":"cantidad",
    }
    df = df.rename(columns={k:_lower_cols(pd.DataFrame(columns=[k])).columns[0] for k in df.columns})
    df = df.rename(columns=rename)

    # campos mínimos
    need_any = {"delivery model","tipo","svc","fecha","cantidad"}
    missing = need_any - set(df.columns)
    if missing:
        raise ValueError(f"Capacity: faltan columnas {sorted(list(missing))}")

    df["svc"] = _upper_series(df["svc"])
    df["tipo"] = _ensure_series(df["tipo"]).astype(str).str.strip().str.lower()
    df["delivery model"] = _ensure_series(df["delivery model"]).astype(str).str.strip().str.lower()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True).dt.date
    df["cantidad"] = _to_num(df["cantidad"]).fillna(0.0)

    return df.dropna(subset=["fecha","svc"])

# 4.4 Rentals (capacidad directa) -------------------------------------
def load_rentals() -> pd.DataFrame:
    raw = read_ws(SHEET_ID, "Rentals")
    df = _reheader(raw, prefer=("svcs","tipo de vehículo","unidades disponibles"))

    # detecta columna svc/svcs
    svc_col = None
    for c in df.columns:
        if c in ("svc","svcs","svcs "):
            svc_col = c
            break
    if svc_col is None:
        # búsqueda laxa (contenga 'svc')
        cand = [c for c in df.columns if "svc" in c]
        svc_col = cand[0] if cand else None

    qty_col = None
    for c in df.columns:
        if ("unidades" in c) and ("dispon" in c):
            qty_col = c
            break

    if not svc_col:
        raise ValueError("Rentals: falta columna 'SVC/SVCs'.")
    if not qty_col:
        raise ValueError("Rentals: falta columna de cantidad (ej. 'Unidades disponibles').")

    out = (df
        .assign(svc=_upper_series(df[svc_col]),
                qty=_to_num(df[qty_col]).fillna(0.0))
        .groupby("svc", as_index=False)["qty"].sum()
        .rename(columns={"qty":"rentals_routes_max"})
    )
    # asumimos capacidad diaria (si tu lógica requiere otra cosa, aquí se ajusta)
    return out

# 4.5 SRM (SDD 6x7 y SPOT 5x7/6x7) ------------------------------------
def load_srm() -> pd.DataFrame:
    raw = read_ws(SHEET_ID, "SRM")
    # SRM puede tener encabezado en fila 5 con muchas columnas
    # Detectar fila header (la primera que contenga 'svc')
    header_row = None
    top = min(10, len(raw))
    for i in range(top):
        row = [str(x).strip().lower() for x in raw.iloc[i, :].tolist()]
        if any("svc" == x for x in row):
            header_row = i
            break
    if header_row is None:
        header_row = 4  # fallback típico en tus hojas

    cols = [
        (str(x).strip().lower() if str(x).strip() else f"col_{j+1}")
        for j, x in enumerate(raw.iloc[header_row, :].tolist())
    ]
    df = raw.iloc[header_row + 1 :].reset_index(drop=True)
    df.columns = cols
    df = _lower_cols(df)

    # detecta SVC
    svc_col = None
    for c in df.columns:
        if c == "svc" or c == "svcs":
            svc_col = c
            break
    if not svc_col:
        cand = [c for c in df.columns if "svc" in c]
        svc_col = cand[0] if cand else None
    if not svc_col:
        raise ValueError("SRM: no se encontró columna SVC.")

    # columnas totales de SDD / SPOT (sumamos todas las que contengan ambos tokens)
    sdd_cols = [c for c in df.columns if ("total" in c and "sdd" in c)]
    spot_cols = [c for c in df.columns if ("total" in c and "spot" in c)]
    if not sdd_cols or not spot_cols:
        # intenta también columnas que terminen en "w.." con sdd/spot
        sdd_cols = sdd_cols + [c for c in df.columns if c.endswith(" sdd") or c.endswith(" sdd w36")]
        spot_cols = spot_cols + [c for c in df.columns if c.endswith(" spot") or c.endswith(" spot w36")]

    if not sdd_cols or not spot_cols:
        raise ValueError("SRM: no se hallaron columnas con 'sdd' o 'spot'.")

    for c in sdd_cols + spot_cols:
        df[c] = _to_num(df[c]).fillna(0.0)

    out = (df
        .assign(svc=_upper_series(df[svc_col]))
        .groupby("svc", as_index=False)[sdd_cols + spot_cols].sum()
    )
    out["sdd_routes_max"] = out[sdd_cols].sum(axis=1)
    out["spot_routes_max"] = out[spot_cols].sum(axis=1)

    return out[["svc","sdd_routes_max","spot_routes_max"]]

# 4.6 Crowd caps (base y E1) ------------------------------------------
def load_crowd_caps() -> pd.DataFrame:
    raw = read_ws(SHEET_ID, "Crowd")
    df = _lower_cols(raw.copy())

    # localizar SVC
    svc_col = None
    for c in df.columns:
        if c in ("svc","svcs"):
            svc_col = c
            break
    if not svc_col:
        cand = [c for c in df.columns if "svc" in c]
        svc_col = cand[0] if cand else None
    if not svc_col:
        raise ValueError("Crowd: falta columna 'svc'.")

    # layout detallado (nombres con día)
    def _pick(name_opts):
        for c in df.columns:
            for opt in name_opts:
                if opt in c:
                    return c
        return None

    c_base_wd = _pick(["base entre", "base semana"])
    c_base_sa = _pick(["base sab"])
    c_base_su = _pick(["base dom"])
    c_e1_wd   = _pick(["holgura entre", "e1 entre"])
    c_e1_sa   = _pick(["holgura sab", "e1 sab"])
    c_e1_su   = _pick(["holgura dom", "e1 dom"])

    if not all([c_base_wd, c_base_sa, c_base_su, c_e1_wd, c_e1_sa, c_e1_su]):
        # layout compacto (ej. 'base', ... columnas vecinas; 'e1', ... vecinas)
        cols = list(df.columns)
        if "base" in cols and "e1" in cols:
            i_base = cols.index("base")
            i_e1   = cols.index("e1")
            # tomamos base + dos siguientes numéricas
            base_candidates = [cols[i_base]] + [c for c in cols[i_base+1 : i_base+4]]
            e1_candidates   = [cols[i_e1]]   + [c for c in cols[i_e1+1 : i_e1+4]]

            base_candidates = base_candidates[:3]
            e1_candidates   = e1_candidates[:3]

            if len(base_candidates) == 3 and len(e1_candidates) == 3:
                c_base_wd, c_base_sa, c_base_su = base_candidates
                c_e1_wd,   c_e1_sa,   c_e1_su   = e1_candidates
            else:
                raise ValueError("Crowd: no se reconoció layout compacto (base/e1).")
        else:
            raise ValueError("Crowd: no se reconoció layout. Encabezados: " + str(list(df.columns)))

    for c in [c_base_wd, c_base_sa, c_base_su, c_e1_wd, c_e1_sa, c_e1_su]:
        df[c] = _to_num(df[c]).fillna(0.0)

    out = (df
        .assign(svc=_upper_series(df[svc_col]))
        .rename(columns={
            c_base_wd:"base_wd", c_base_sa:"base_sa", c_base_su:"base_su",
            c_e1_wd:"e1_wd",     c_e1_sa:"e1_sa",     c_e1_su:"e1_su"
        })
    )

    return out[["svc","base_wd","base_sa","base_su","e1_wd","e1_sa","e1_su"]]

# -------------------------------------------------------------------
# 5) CÁLCULOS AUX: SPR escenarios, share crowd, crowd por día, descansos
# -------------------------------------------------------------------
def compute_spr_scenarios(fcst: pd.DataFrame, spr_real: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    target = fcst[["fecha","svc"]].drop_duplicates().copy()
    target["dow"] = target["fecha"].apply(_weekday)
    year = pd.to_datetime(target["fecha"]).dt.isocalendar()["year"].astype(int)
    target["iso_year"] = year

    # 5.1 promedio últimas 4 semanas, mismo DOW
    spr_map = spr_real.set_index(["fecha","svc"])["spr_exec"]

    def last4_same_dow(row):
        d, s = row["fecha"], row["svc"]
        vals = []
        for k in [7,14,21,28]:
            dk = d - timedelta(days=k)
            v = spr_map.get((dk, s), np.nan)
            if pd.notna(v):
                vals.append(float(v))
        if not vals:
            mask = (spr_real["svc"]==s) & (spr_real["fecha"].between(d - timedelta(days=28), d - timedelta(days=1)))
            vals = list(spr_real.loc[mask, "spr_exec"])
        return _safe_mean(vals)

    target["spr_promedio"] = target.apply(last4_same_dow, axis=1)

    # 5.2 peak: semanas 20–22 del mismo año, mismo DOW (fallback 19–23)
    def peak_20_22(row):
        d, s, yr, dow = row["fecha"], row["svc"], row["iso_year"], row["dow"]
        m = (spr_real["svc"].eq(s) & spr_real["iso_year"].eq(yr) &
             spr_real["iso_week"].isin([20,21,22]) & spr_real["dow"].eq(dow))
        vals = list(spr_real.loc[m,"spr_exec"])
        if not vals:
            m = (spr_real["svc"].eq(s) & spr_real["iso_year"].eq(yr) &
                 spr_real["iso_week"].between(19,23) & spr_real["dow"].eq(dow))
            vals = list(spr_real.loc[m,"spr_exec"])
        return _safe_mean(vals)

    target["spr_peak"] = target.apply(peak_20_22, axis=1)

    # 5.3 plan: desde Capacity tipo 'spr' por fecha o promedio por svc
    cap = capacity.copy()
    m_spr = cap["tipo"].eq("spr")
    spr_plan = cap.loc[m_spr, ["svc","fecha","cantidad"]].rename(columns={"cantidad":"spr_plan"})
    if spr_plan.empty:
        by_svc = (cap.loc[m_spr]
                    .groupby("svc", as_index=False)["cantidad"]
                    .mean()
                    .rename(columns={"cantidad":"spr_plan"}))
        spr_plan = target[["fecha","svc"]].merge(by_svc, on="svc", how="left")
    target = target.merge(spr_plan, on=["fecha","svc"], how="left")

    return target[["fecha","svc","spr_promedio","spr_peak","spr_plan"]]

def compute_crowd_share(capacity: pd.DataFrame) -> pd.DataFrame:
    cap = capacity.copy()
    cap["tipo"] = cap["tipo"].str.strip().str.lower()
    m = cap["tipo"].eq("shipments")
    ship = cap.loc[m, ["fecha","svc","delivery model","cantidad"]].copy()
    ship["delivery model"] = ship["delivery model"].str.strip().str.lower()

    tot = (ship
        .groupby(["fecha","svc"], as_index=False)["cantidad"]
        .sum()
        .rename(columns={"cantidad":"ship_total"})
    )
    crw = (ship
        .loc[ship["delivery model"].eq("crowd")]
        .groupby(["fecha","svc"], as_index=False)["cantidad"]
        .sum()
        .rename(columns={"cantidad":"ship_crowd"})
    )
    out = tot.merge(crw, on=["fecha","svc"], how="left").fillna({"ship_crowd":0.0})
    out["share_crowd_obj"] = np.where(out["ship_total"]>0,
                                      (out["ship_crowd"]/out["ship_total"]).clip(0,1),
                                      0.0)
    return out

def map_crowd_capacity_by_date(target_days: pd.DataFrame, crowd_caps: pd.DataFrame) -> pd.DataFrame:
    # asigna base/e1 por DOW
    def cap_for(row):
        s, d = row["svc"], row["fecha"]
        dow = _weekday(d)  # 0..6
        r = crowd_caps.loc[crowd_caps["svc"]==s]
        if r.empty:
            return pd.Series({"crowd_base_routes":0,"crowd_e1_routes":0})
        r = r.iloc[0]
        if dow <= 4:
            base, e1 = r["base_wd"], r["e1_wd"]
        elif dow == 5:
            base, e1 = r["base_sa"], r["e1_sa"]
        else:
            base, e1 = r["base_su"], r["e1_su"]
        return pd.Series({"crowd_base_routes":int(base), "crowd_e1_routes":int(e1)})

    tmp = target_days.apply(cap_for, axis=1)
    return pd.concat([target_days.reset_index(drop=True), tmp], axis=1)

def schedule_mlp_rest(df_day: pd.DataFrame) -> pd.DataFrame:
    """Aplica descansos por semana: SDD 6x7, Spot 5x7 (o 6x7 si necesidad >=6)."""
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
        if need_days < 5:
            work_spot = need_days
        rest_spot  = max(n - work_spot, 0)

        # asigna descanso a días con menor necesidad
        g_sorted = g.sort_values(["routes_mlp_need", "fecha"], ascending=[True, True])
        if rest_sdd > 0:
            idx = g_sorted.head(rest_sdd).index
            g.loc[idx, "sdd_trabaja"] = 0

        g_sorted2 = g.sort_values(["routes_mlp_need", "fecha"], ascending=[True, True])
        if rest_spot > 0:
            idx2 = g_sorted2.head(rest_spot).index
            g.loc[idx2, "spot_trabaja"] = 0

        return g

    out = out.groupby(["svc","week_key"], group_keys=False).apply(proc)
    return out.drop(columns=["week_key"])

# -------------------------------------------------------------------
# 6) MOTOR PRINCIPAL
# -------------------------------------------------------------------
def compute_plan(spr_mode: str, sel_svcs: list[str] | None = None) -> pd.DataFrame:
    # Carga fuentes
    fcst       = load_fcst()
    spr_real   = load_spr_real()
    capacity   = load_capacity()
    rentals    = load_rentals()
    srm        = load_srm()
    crowd_caps = load_crowd_caps()

    if sel_svcs:
        fcst = fcst[fcst["svc"].isin(sel_svcs)]
        spr_real = spr_real[spr_real["svc"].isin(sel_svcs)]
        capacity = capacity[capacity["svc"].isin(sel_svcs)]
        rentals = rentals[rentals["svc"].isin(sel_svcs)]
        srm = srm[srm["svc"].isin(sel_svcs)]
        crowd_caps = crowd_caps[crowd_caps["svc"].isin(sel_svcs)]

    # SPR escenarios
    spr_tbl = compute_spr_scenarios(fcst, spr_real, capacity)
    spr_col = {"promedio":"spr_promedio","peak":"spr_peak","plan":"spr_plan"}[spr_mode]

    # Share crowd objetivo
    share_tbl = compute_crowd_share(capacity)

    # DC y SP desde Capacity tipo 'Shipments' si existen
    cap_ship = capacity[capacity["tipo"].eq("shipments")]
    ship_dc = (cap_ship[cap_ship["delivery model"].str.contains("delivery cell|dc", regex=True)]
               .groupby(["fecha","svc"], as_index=False)["cantidad"].sum()
               .rename(columns={"cantidad":"ship_dc"}))
    ship_sp = (cap_ship[cap_ship["delivery model"].str.contains("service partner|sp", regex=True)]
               .groupby(["fecha","svc"], as_index=False)["cantidad"].sum()
               .rename(columns={"cantidad":"ship_sp"}))

    # CROWD caps por día
    target_days = fcst[["fecha","svc"]].drop_duplicates()
    crowd_daily = map_crowd_capacity_by_date(target_days, crowd_caps)

    # Merge base
    df = (fcst
          .merge(ship_dc, on=["fecha","svc"], how="left")
          .merge(ship_sp, on=["fecha","svc"], how="left")
          .merge(share_tbl, on=["fecha","svc"], how="left")
          .merge(crowd_daily, on=["fecha","svc"], how="left")
          .merge(srm, on="svc", how="left")
          .merge(rentals, on="svc", how="left")
          .merge(spr_tbl[["fecha","svc",spr_col]], on=["fecha","svc"], how="left")
         )

    # Limpieza / nulos
    for c in ["ship_dc","ship_sp","share_crowd_obj","crowd_base_routes","crowd_e1_routes",
              "sdd_routes_max","spot_routes_max","rentals_routes_max"]:
        df[c] = _to_num(df.get(c, 0)).fillna(0.0)

    df["spr_objetivo"] = _to_num(df[spr_col])
    df.drop(columns=[spr_col], inplace=True)

    # FCST neto (descuento DC + SP)
    df["ship_fcst_neto"] = (df["shipments"] - df["ship_dc"] - df["ship_sp"]).clip(lower=0.0)

    # Rutas requeridas por SPR
    df["routes_need_total"] = np.where(
        (df["ship_fcst_neto"]>0) & (df["spr_objetivo"]>0),
        np.ceil(df["ship_fcst_neto"]/df["spr_objetivo"]).astype(int),
        0
    )
    df["alerta_spr_missing"] = ((df["ship_fcst_neto"]>0) & (df["spr_objetivo"].isna() | (df["spr_objetivo"]<=0)))

    # Objetivo crowd (rutas)
    df["routes_crowd_target"] = np.ceil(df["routes_need_total"] * df["share_crowd_obj"]).astype(int)

    # Asignación Crowd base y E1 (no exceder caps)
    df["routes_crowd_base"] = np.minimum(df["routes_crowd_target"], df["crowd_base_routes"]).astype(int)
    df["routes_crowd_e1"] = np.minimum(
        (df["routes_crowd_target"] - df["routes_crowd_base"]).clip(lower=0),
        df["crowd_e1_routes"]
    ).astype(int)
    df["routes_crowd_alloc"] = df["routes_crowd_base"] + df["routes_crowd_e1"]
    df["alerta_crowd_high"] = df["routes_crowd_e1"] > 0

    # Restante tras crowd objetivo
    df["routes_after_crowd"] = (df["routes_need_total"] - df["routes_crowd_alloc"]).clip(lower=0).astype(int)

    # Rentals (capacidad diaria) — en rutas, no exceder
    df["routes_rentals_alloc"] = np.minimum(df["routes_after_crowd"], df["rentals_routes_max"]).astype(int)

    # Necesidad para MLP
    df["routes_mlp_need"] = (df["routes_after_crowd"] - df["routes_rentals_alloc"]).clip(lower=0).astype(int)

    # Descansos MLP por semana (SDD 6x7, Spot 5x7/6x7)
    rest_base = df[["fecha","svc","routes_mlp_need","sdd_routes_max","spot_routes_max"]].copy()
    rest_sched = schedule_mlp_rest(rest_base)
    df = df.merge(rest_sched[["fecha","svc","sdd_trabaja","spot_trabaja"]], on=["fecha","svc"], how="left")

    df["routes_mlp_cap_day"] = (df["sdd_routes_max"]*df["sdd_trabaja"] + df["spot_routes_max"]*df["spot_trabaja"]).fillna(0).astype(int)
    df["routes_mlp_alloc"]   = np.minimum(df["routes_mlp_need"], df["routes_mlp_cap_day"]).astype(int)

    # Déficit y crowd extra si aún falta (sin exceder e1 restante)
    df["routes_deficit_pre_extra"] = (df["routes_mlp_need"] - df["routes_mlp_alloc"]).clip(lower=0).astype(int)
    df["crowd_e1_remaining"] = (df["crowd_e1_routes"] - df["routes_crowd_e1"]).clip(lower=0).astype(int)
    df["routes_crowd_extra"] = np.minimum(df["routes_deficit_pre_extra"], df["crowd_e1_remaining"]).astype(int)

    # Total asignado
    df["routes_total_alloc"] = (df["routes_crowd_alloc"] + df["routes_rentals_alloc"] +
                                df["routes_mlp_alloc"] + df["routes_crowd_extra"]).astype(int)

    # Shipments plan y métricas
    df["shipments_plan"] = np.where(
        df["spr_objetivo"]>0,
        df["routes_total_alloc"] * df["spr_objetivo"],
        0.0
    )
    df["routes_deficit"] = (df["routes_need_total"] - df["routes_total_alloc"]).clip(lower=0).astype(int)
    df["alerta_deficit"] = df["shipments_plan"] + 1e-6 < df["ship_fcst_neto"]

    df["spr_logrado"] = np.where(
        df["routes_total_alloc"]>0,
        df["ship_fcst_neto"] / df["routes_total_alloc"],
        np.nan
    )
    df["share_crowd_real"] = np.where(
        df["routes_need_total"]>0,
        (df["routes_crowd_alloc"] + df["routes_crowd_extra"]) / df["routes_need_total"],
        0.0
    )
    df["risk_flag"] = df["alerta_deficit"] | df["alerta_spr_missing"]

    # Orden y salida compacta (una fila por SVC-fecha)
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

# -------------------------------------------------------------------
# 7) SIDEBAR de credenciales / filtro SVC
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# 8) RUN & UI
# -------------------------------------------------------------------
with st.expander("➤ Cargando datos...", expanded=True):
    try:
        # Cargamos rápido para el selector de SVC
        _fcst = load_fcst()
        svc_list = sorted(_fcst["svc"].dropna().astype(str).unique().tolist())
        sel_svcs = st.multiselect("Filtrar SVC", svc_list, default=svc_list, placeholder="Selecciona 1 o más SVC")

        plan = compute_plan(spr_mode, sel_svcs=sel_svcs)

        # Tabla principal (una fila por SVC-fecha)
        st.subheader("Tabla principal — (svc, fecha) × Delivery model")
        st.dataframe(plan, use_container_width=True, hide_index=True)

        # Riesgos por fecha
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

