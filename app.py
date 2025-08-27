# app.py — Mel-IA Plan táctico (diario por SVC)
import os, json, yaml
import numpy as np
import pandas as pd
from math import ceil
from datetime import timedelta
import streamlit as st

# ======================== CREDENCIALES (parche) ========================
# Acepta cualquiera de estos dos nombres de secret:
#  1) GOOGLE_SERVICE_ACCOUNT_JSON (string JSON de una línea)
#  2) gcp_service_account (bloque toml convertido a JSON)
if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
elif "gcp_service_account" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps(dict(st.secrets["gcp_service_account"]))

if "PROJECT_KEY" in st.secrets:
    os.environ["PROJECT_KEY"] = st.secrets["PROJECT_KEY"]

# ======================== CONFIG & UTILS ===============================
st.set_page_config(page_title="Mel-IA — Plan táctico", layout="wide")

@st.cache_resource
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()
proj_key = list(cfg["projects"].keys())[0]
proj = cfg["projects"][proj_key]
SHEET_ID = proj["sheet_id"]

from utils_gsheets import read_ws, _client, get_service_account_email

def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: str(c).strip().lower() for c in df.columns})

def _norm_date_col(df: pd.DataFrame, col: str) -> pd.Series:
    # Forzamos dayfirst=True por tus formatos DD/MM/YYYY
    return pd.to_datetime(df[col], errors="coerce", dayfirst=True).dt.date

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip(),
        errors="coerce"
    )

def _first_series(df: pd.DataFrame, colname: str) -> pd.Series:
    """Si hay columnas duplicadas con el mismo nombre, toma la PRIMERA como Series."""
    if colname not in df.columns:
        return pd.Series([np.nan] * len(df))
    obj = df[colname]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return obj

def _weekday(d) -> int:
    # Acepta date/str/NaT
    ts = pd.to_datetime(d, errors="coerce")
    return int(ts.weekday()) if not pd.isna(ts) else -1

def _iso_yr_week(d):
    ts = pd.to_datetime(d, errors="coerce")
    if pd.isna(ts):
        return (0, 0)
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)

def _safe_mean(vals):
    vals = [float(v) for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan

# ======================== CARGA DE HOJAS ================================
def load_fcst() -> pd.DataFrame:
    df = read_ws(SHEET_ID, "FCST")
    df = _lower_cols(df)
    need = {"svc", "fecha", "shipments"}
    if not need.issubset(df.columns):
        raise ValueError(f"FCST: faltan columnas {sorted(need - set(df.columns))}")
    out = df[["svc", "fecha", "shipments"]].copy()
    out["svc"] = out["svc"].astype(str).strip().str.upper()
    out["fecha"] = _norm_date_col(out, "fecha")
    out["shipments"] = _to_num(out["shipments"]).fillna(0.0)
    out = out.dropna(subset=["fecha", "svc"])
    return out

def load_spr_real() -> pd.DataFrame:
    df = read_ws(SHEET_ID, "SPR")
    df = _lower_cols(df)
    need = {"fecha", "svc", "spr"}
    if not need.issubset(df.columns):
        raise ValueError(f"SPR: faltan columnas {sorted(need - set(df.columns))}")

    out = df[["fecha", "svc", "spr"]].copy()
    out["svc"] = out["svc"].astype(str).str.strip().str.upper()
    out["fecha"] = _norm_date_col(out, "fecha")
    out["spr"] = _to_num(out["spr"])

    out = out.dropna(subset=["fecha", "svc", "spr"])
    out = (out.groupby(["fecha", "svc"], as_index=False)["spr"]
              .mean()
              .rename(columns={"spr": "spr_exec"}))

    # Derivados de fecha
    ts = pd.to_datetime(out["fecha"])
    iso = ts.dt.isocalendar()
    out["dow"] = ts.dt.weekday
    out["iso_year"] = iso["year"].astype(int)
    out["iso_week"] = iso["week"].astype(int)
    return out

def load_capacity() -> pd.DataFrame:
    df = read_ws(SHEET_ID, "Capacity")
    df = _lower_cols(df)

    tipo     = _first_series(df, "tipo")
    dm       = _first_series(df, "delivery model")
    svc      = _first_series(df, "svc")
    fecha    = _first_series(df, "fecha")
    cantidad = _first_series(df, "cantidad")

    if tipo.isna().all() or dm.isna().all() or svc.isna().all() or fecha.isna().all() or cantidad.isna().all():
        raise ValueError("Capacity: faltan columnas requeridas ('delivery model','tipo','svc','fecha','cantidad').")

    out = pd.DataFrame({
        "delivery model": dm.astype(str).str.strip().str.lower(),
        "tipo":           tipo.astype(str).str.strip().str.lower(),
        "svc":            svc.astype(str).str.strip().str.upper(),
        "fecha":          fecha,
        "cantidad":       cantidad
    })
    out["fecha"] = _norm_date_col(out, "fecha")
    out["cantidad"] = _to_num(out["cantidad"]).fillna(0.0)
    out = out.dropna(subset=["fecha", "svc"])
    return out

def load_rentals() -> pd.DataFrame:
    df = read_ws(SHEET_ID, "Rentals")
    df = _lower_cols(df)

    svc_col = None
    for c in ("svc", "svcs", "svc "):
        if c in df.columns:
            svc_col = c
            break
    if svc_col is None:
        raise ValueError("Rentals: falta columna 'SVC'/'SVCs'.")

    qty_col = None
    for c in df.columns:
        if "unidades" in c and "dispon" in c:
            qty_col = c
            break
    if qty_col is None:
        raise ValueError("Rentals: falta columna tipo 'Unidades disponibles'.")

    svc_series = _first_series(df, svc_col)
    qty_series = _first_series(df, qty_col)

    out = (pd.DataFrame({"svc": svc_series, "qty": qty_series})
             .assign(svc=lambda x: x["svc"].astype(str).str.strip().str.upper(),
                     qty=lambda x: _to_num(x["qty"]).fillna(0.0))
             .groupby("svc", as_index=False)["qty"].sum()
             .rename(columns={"qty": "rentals_routes_max"}))
    out["rentals_routes_max"] = out["rentals_routes_max"].astype(int)
    return out

def load_crowd_caps() -> pd.DataFrame:
    """
    Soporta dos layouts:
    1) Detallado: columnas explícitas por día (base entre semana/sábado/domingo + E1 idem)
    2) Compacto: una columna 'base' y una 'e1/holgura' (replicamos a wd/sa/su)
    Además, acepta nombres genéricos como col_7 / col_8 si ya están mapeados.
    """
    df = read_ws(SHEET_ID, "Crowd")
    df = _lower_cols(df)

    # Detecta SVC
    svc_col = None
    for c in ("svc", "svcs", "svc "):
        if c in df.columns:
            svc_col = c
            break
    if svc_col is None:
        # fallback muy usado: segunda columna suele ser 'SVC'
        if "col_2" in df.columns:
            svc_col = "col_2"
        else:
            raise ValueError("Crowd: falta columna 'svc'.")

    # Caso “detallado”
    def _pick(name_opts):
        for n in df.columns:
            for opt in name_opts:
                if opt in n:
                    return n
        return None

    base_wd = _pick(["base entre"])
    base_sa = _pick(["base sab"])
    base_su = _pick(["base domingo", "base dom"])
    e1_wd   = _pick(["holgura entre", "e1 entre"])
    e1_sa   = _pick(["holgura sab", "e1 sab"])
    e1_su   = _pick(["holgura domingo", "e1 dom"])

    if not all([base_wd, base_sa, base_su, e1_wd, e1_sa, e1_su]):
        # Caso “compacto”: existen 'base' y 'e1' (o equivalente)
        base_simple = None
        e1_simple = None
        # Busca literalmente base/e1
        if "base" in df.columns: base_simple = "base"
        if "e1" in df.columns:   e1_simple   = "e1"
        # Si además detectaste columnas “col_4/col_5/col_7/col_8” previas, las usamos:
        # (esto se ajusta a tu hoja real que ya mapeaste)
        if base_simple is None and "col_4" in df.columns: base_simple = "col_4"
        if base_simple is None and "col_5" in df.columns: base_simple = "col_5"
        if e1_simple   is None and "col_7" in df.columns: e1_simple   = "col_7"
        if e1_simple   is None and "col_8" in df.columns: e1_simple   = "col_8"

        if base_simple is None or e1_simple is None:
            raise ValueError(f"Crowd: layout no reconocido. Encabezados={list(df.columns)}")

        out = pd.DataFrame({
            "svc": df[svc_col].astype(str).str.strip().str.upper(),
            "base_wd": _to_num(_first_series(df, base_simple)).fillna(0).astype(int),
            "base_sa": _to_num(_first_series(df, base_simple)).fillna(0).astype(int),
            "base_su": _to_num(_first_series(df, base_simple)).fillna(0).astype(int),
            "e1_wd":   _to_num(_first_series(df, e1_simple)).fillna(0).astype(int),
            "e1_sa":   _to_num(_first_series(df, e1_simple)).fillna(0).astype(int),
            "e1_su":   _to_num(_first_series(df, e1_simple)).fillna(0).astype(int),
        })
        return out

    # Detallado
    out = df[[svc_col, base_wd, base_sa, base_su, e1_wd, e1_sa, e1_su]].copy()
    out = out.rename(columns={svc_col: "svc",
                              base_wd: "base_wd", base_sa: "base_sa", base_su: "base_su",
                              e1_wd: "e1_wd", e1_sa: "e1_sa", e1_su: "e1_su"})
    for c in ["base_wd", "base_sa", "base_su", "e1_wd", "e1_sa", "e1_su"]:
        out[c] = _to_num(out[c]).fillna(0).astype(int)
    out["svc"] = out["svc"].astype(str).str.strip().str.upper()
    return out

def load_srm() -> pd.DataFrame:
    """
    SRM tiene encabezado en fila 5 y muchas columnas.
    Sumamos todo lo que contenga 'sdd' (capacidad SDD) y 'spot' (capacidad spot).
    """
    raw = read_ws(SHEET_ID, "SRM")
    raw = _lower_cols(raw)

    # Detecta header row (primera fila con una celda que contenga 'svc')
    header_row = None
    for i in range(min(10, len(raw))):
        row_vals = [str(x).strip().lower() for x in raw.iloc[i, :].tolist()]
        if any(("svc" == x) for x in row_vals):
            header_row = i
            break
    if header_row is None:
        header_row = 4  # fallback a fila 5 (0-based)

    # Relee y setea encabezados correctos
    df = read_ws(SHEET_ID, "SRM")
    df.columns = [str(x).strip().lower() for x in df.iloc[header_row].tolist()]
    df = df.iloc[header_row + 1:].reset_index(drop=True)
    df = _lower_cols(df)

    # SVC
    svc_col = None
    for c in ("svc", "svcs", "svc "):
        if c in df.columns:
            svc_col = c
            break
    if svc_col is None:
        raise ValueError("SRM: no se encontró columna 'SVC'.")

    # Columnas SDD y SPOT (pueden ser muchísimas; sumamos todo)
    sdd_cols = [c for c in df.columns if "sdd" in c]
    spot_cols = [c for c in df.columns if "spot" in c]

    # Construye salida
    out = pd.DataFrame({"svc": df[svc_col].astype(str).str.strip().str.upper()})
    out["sdd_routes_max"] = 0
    out["spot_routes_max"] = 0
    if sdd_cols:
        out["sdd_routes_max"] = _to_num(df[sdd_cols]).fillna(0).sum(axis=1)
    if spot_cols:
        out["spot_routes_max"] = _to_num(df[spot_cols]).fillna(0).sum(axis=1)

    out = (out.groupby("svc", as_index=False)[["sdd_routes_max", "spot_routes_max"]].sum())
    # Redondea a int
    for c in ["sdd_routes_max", "spot_routes_max"]:
        out[c] = out[c].fillna(0).astype(int)
    return out

# ======================== CÁLCULOS AUXILIARES ==========================
def compute_spr_scenarios(fcst: pd.DataFrame, spr_real: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    target = fcst[["fecha", "svc"]].drop_duplicates().copy()
    ts = pd.to_datetime(target["fecha"])
    target["dow"] = ts.dt.weekday
    target["iso_year"] = ts.dt.isocalendar()["year"].astype(int)

    spr_exec_map = spr_real.set_index(["fecha", "svc"])["spr_exec"]

    def avg_last4(row):
        d, s = row["fecha"], row["svc"]
        vals = []
        for k in [7,14,21,28]:
            dk = pd.to_datetime(d) - pd.to_timedelta(k, unit="D")
            v = spr_exec_map.get((dk.date(), s), np.nan)
            if pd.notna(v):
                vals.append(float(v))
        if not vals:
            mask = (spr_real["svc"].eq(s) &
                    (pd.to_datetime(spr_real["fecha"]).between(pd.to_datetime(d)-pd.Timedelta(days=28),
                                                              pd.to_datetime(d)-pd.Timedelta(days=1))))
            vals = list(spr_real.loc[mask, "spr_exec"])
        return _safe_mean(vals)

    target["spr_promedio"] = target.apply(avg_last4, axis=1)

    def avg_peak(row):
        d, s, yr, dow = row["fecha"], row["svc"], row["iso_year"], row["dow"]
        m = (spr_real["svc"].eq(s) & spr_real["iso_year"].eq(yr) &
             spr_real["iso_week"].isin([20,21,22]) & spr_real["dow"].eq(dow))
        vals = list(spr_real.loc[m, "spr_exec"])
        if not vals:
            m = (spr_real["svc"].eq(s) & spr_real["iso_year"].eq(yr) &
                 spr_real["iso_week"].between(19,23) & spr_real["dow"].eq(dow))
            vals = list(spr_real.loc[m, "spr_exec"])
        return _safe_mean(vals)

    target["spr_peak"] = target.apply(avg_peak, axis=1)

    cap = capacity.copy()
    m_spr = cap["tipo"].str.strip().str.lower().eq("spr")
    spr_plan = cap.loc[m_spr, ["svc", "fecha", "cantidad"]].rename(columns={"cantidad": "spr_plan"})
    if spr_plan.empty:
        spr_by_svc = cap.loc[m_spr].groupby("svc", as_index=False)["cantidad"].mean().rename(columns={"cantidad": "spr_plan"})
        spr_plan = target[["fecha", "svc"]].merge(spr_by_svc, on="svc", how="left")
    target = target.merge(spr_plan, on=["fecha", "svc"], how="left")

    return target[["fecha", "svc", "spr_promedio", "spr_peak", "spr_plan"]]

def compute_crowd_share(capacity: pd.DataFrame) -> pd.DataFrame:
    cap = capacity.copy()
    cap["tipo"] = cap["tipo"].astype(str).str.strip().str.lower()
    cap["delivery model"] = cap["delivery model"].astype(str).str.strip().str.lower()

    m_ship = cap["tipo"].eq("shipments")
    ship = cap.loc[m_ship, ["fecha", "svc", "delivery model", "cantidad"]].copy()

    tot = ship.groupby(["fecha", "svc"], as_index=False)["cantidad"].sum().rename(columns={"cantidad":"ship_total"})
    crw = ship.loc[ship["delivery model"].eq("crowd")].groupby(["fecha", "svc"], as_index=False)["cantidad"].sum().rename(columns={"cantidad":"ship_crowd"})

    out = tot.merge(crw, on=["fecha", "svc"], how="left").fillna({"ship_crowd":0.0})
    out["share_crowd_obj"] = np.where(out["ship_total"]>0, (out["ship_crowd"]/out["ship_total"]).clip(0,1), 0.0)
    return out[["fecha","svc","share_crowd_obj","ship_total","ship_crowd"]]

def map_crowd_capacity_by_date(target_days: pd.DataFrame, crowd_caps: pd.DataFrame) -> pd.DataFrame:
    def cap_for(row):
        s, d = row["svc"], row["fecha"]
        dow = _weekday(d)
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
    out = df_day.copy()
    out["week_key"] = out["fecha"].apply(lambda d: f"{_iso_yr_week(d)[0]}-{_iso_yr_week(d)[1]:02d}")
    out["sdd_trabaja"]  = 1
    out["spot_trabaja"] = 1

    def proc(g):
        n = len(g)
        need_days = int((g["routes_mlp_need"]>0).sum())
        work_sdd  = min(6, need_days)   # 6x7
        rest_sdd  = max(n - work_sdd, 0)

        work_spot = 5                   # 5x7 base
        if need_days >= 6:
            work_spot = 6
        elif need_days < 5:
            work_spot = need_days
        rest_spot = max(n - work_spot, 0)

        g_sorted = g.sort_values(["routes_mlp_need","fecha"], ascending=[True,True])
        if rest_sdd > 0:
            idx = g_sorted.head(rest_sdd).index
            g.loc[idx,"sdd_trabaja"] = 0

        g_sorted2 = g.sort_values(["routes_mlp_need","fecha"], ascending=[True,True])
        if rest_spot > 0:
            idx2 = g_sorted2.head(rest_spot).index
            g.loc[idx2,"spot_trabaja"] = 0
        return g

    out = out.groupby(["svc","week_key"], group_keys=False).apply(proc)
    return out.drop(columns=["week_key"])

# ======================== MOTOR PRINCIPAL (lógica) =====================
def compute_dc_sp_shipments(capacity: pd.DataFrame) -> pd.DataFrame:
    cap = capacity.copy()
    cap["tipo"] = cap["tipo"].astype(str).str.strip().str.lower()
    cap["delivery model"] = cap["delivery model"].astype(str).str.strip().str.lower()

    m_ship = cap["tipo"].eq("shipments")
    df = cap.loc[m_ship, ["fecha","svc","delivery model","cantidad"]].copy()

    is_dc = df["delivery model"].str.contains("cell", na=False) | df["delivery model"].str.fullmatch("dc", case=False, na=False)
    is_sp = df["delivery model"].str.contains("partner", na=False) | df["delivery model"].str.fullmatch("sp", case=False, na=False)

    dc = (df.loc[is_dc].groupby(["fecha","svc"], as_index=False)["cantidad"].sum()
            .rename(columns={"cantidad":"ship_dc"}))
    sp = (df.loc[is_sp].groupby(["fecha","svc"], as_index=False)["cantidad"].sum()
            .rename(columns={"cantidad":"ship_sp"}))

    out = dc.merge(sp, on=["fecha","svc"], how="outer").fillna(0.0)
    return out

def compute_plan(spr_mode: str) -> pd.DataFrame:
    # Carga
    fcst       = load_fcst()
    spr_real   = load_spr_real()
    capacity   = load_capacity()
    rentals    = load_rentals()
    crowd_caps = load_crowd_caps()
    srm        = load_srm()

    # Derivados
    spr_tbl    = compute_spr_scenarios(fcst, spr_real, capacity)
    spr_col    = {"promedio":"spr_promedio","peak":"spr_peak","plan":"spr_plan"}[spr_mode]
    share_tbl  = compute_crowd_share(capacity)
    dcsp_tbl   = compute_dc_sp_shipments(capacity)

    target_days  = fcst[["fecha","svc"]].drop_duplicates()
    crowd_daily  = map_crowd_capacity_by_date(target_days, crowd_caps)

    # Merge base (UNA fila por svc-día)
    df = (fcst
          .merge(dcsp_tbl, on=["fecha","svc"], how="left")
          .merge(share_tbl, on=["fecha","svc"], how="left")
          .merge(crowd_daily, on=["fecha","svc"], how="left")
          .merge(srm, on="svc", how="left")
          .merge(rentals, on="svc", how="left")
          .merge(spr_tbl[["fecha","svc",spr_col]], on=["fecha","svc"], how="left")
         )

    for c in ["ship_dc","ship_sp","share_crowd_obj","crowd_base_routes","crowd_e1_routes",
              "sdd_routes_max","spot_routes_max","rentals_routes_max"]:
        if c not in df.columns: df[c] = 0
        df[c] = _to_num(df[c]).fillna(0)

    df["spr_objetivo"] = _to_num(df[spr_col])
    df.drop(columns=[spr_col], inplace=True, errors="ignore")

    # 1) FCST neto = FCST – DC – SP
    df["ship_fcst_neto"] = (df["shipments"] - df["ship_dc"] - df["ship_sp"]).clip(lower=0)

    # 2) Rutas requeridas iniciales (sobre FCST neto)
    df["routes_need_total"] = np.where(
        (df["ship_fcst_neto"]>0) & (df["spr_objetivo"]>0),
        np.ceil(df["ship_fcst_neto"] / df["spr_objetivo"]).astype(int),
        0
    )
    df["alerta_spr_missing"] = ((df["ship_fcst_neto"]>0) & (df["spr_objetivo"].isna() | (df["spr_objetivo"]<=0)))

    # 3) Rentals primero
    df["routes_rentals_alloc"] = np.minimum(df["routes_need_total"], df["rentals_routes_max"]).astype(int)
    df["routes_left_1"] = (df["routes_need_total"] - df["routes_rentals_alloc"]).clip(lower=0).astype(int)

    # 4) Crowd sobre el restante (usar % objetivo del plan)
    df["routes_crowd_target"] = np.ceil(df["routes_left_1"] * df["share_crowd_obj"]).astype(int)
    df["routes_crowd_base"]   = np.minimum(df["routes_crowd_target"], df["crowd_base_routes"]).astype(int)
    df["routes_left_2"]       = (df["routes_left_1"] - df["routes_crowd_base"]).clip(lower=0).astype(int)

    # 5) MLP SDD 6x7 (descanso por semana donde menos se necesita)
    rest_base = df[["fecha","svc","routes_mlp_need","sdd_routes_max","spot_routes_max"]].copy()
    rest_base["routes_mlp_need"] = df["routes_left_2"]
    rest_sched = schedule_mlp_rest(rest_base)
    df = df.merge(rest_sched[["fecha","svc","sdd_trabaja","spot_trabaja"]], on=["fecha","svc"], how="left")

    df["routes_mlp_sdd_cap_day"]   = (df["sdd_routes_max"] * df["sdd_trabaja"]).fillna(0).astype(int)
    df["routes_mlp_sdd_alloc"]     = np.minimum(df["routes_left_2"], df["routes_mlp_sdd_cap_day"]).astype(int)
    df["routes_left_3"]            = (df["routes_left_2"] - df["routes_mlp_sdd_alloc"]).clip(lower=0).astype(int)

    # 6) MLP Spot 5x7 (o 6x7 si la semana viene pesada)
    df["routes_mlp_spot_cap_day"]  = (df["spot_trabaja"] * df["spot_routes_max"]).fillna(0).astype(int)
    df["routes_mlp_spot_alloc"]    = np.minimum(df["routes_left_3"], df["routes_mlp_spot_cap_day"]).astype(int)
    df["routes_left_4"]            = (df["routes_left_3"] - df["routes_mlp_spot_alloc"]).clip(lower=0).astype(int)

    # 7) Crowd extra (E1) si sigue faltando, sin sobrepasar nube
    df["routes_crowd_e1"]          = np.minimum(df["routes_left_4"], df["crowd_e1_routes"]).astype(int)
    df["routes_left_5"]            = (df["routes_left_4"] - df["routes_crowd_e1"]).clip(lower=0).astype(int)

    # Totales, déficits y métricas
    df["routes_total_alloc"] = (df["routes_rentals_alloc"] +
                                df["routes_crowd_base"] +
                                df["routes_mlp_sdd_alloc"] +
                                df["routes_mlp_spot_alloc"] +
                                df["routes_crowd_e1"]).astype(int)

    df["routes_deficit"] = df["routes_left_5"].astype(int)
    df["shipments_plan"] = np.where(df["spr_objetivo"]>0, df["routes_total_alloc"] * df["spr_objetivo"], 0.0)
    df["alerta_deficit"] = df["routes_deficit"] > 0
    df["risk_flag"] = df["alerta_deficit"] | df["alerta_spr_missing"]

    # Tabla final por (svc, fecha) UNA FILA
    cols = [
        "fecha","svc",
        "shipments","ship_dc","ship_sp","ship_fcst_neto",
        "spr_objetivo","routes_need_total",
        "rentals_routes_max","routes_rentals_alloc",
        "share_crowd_obj","crowd_base_routes","routes_crowd_target","routes_crowd_base","crowd_e1_routes","routes_crowd_e1",
        "sdd_routes_max","spot_routes_max","sdd_trabaja","spot_trabaja","routes_mlp_sdd_cap_day","routes_mlp_spot_cap_day",
        "routes_mlp_sdd_alloc","routes_mlp_spot_alloc",
        "routes_total_alloc","routes_deficit","shipments_plan",
        "alerta_spr_missing","alerta_deficit","risk_flag"
    ]
    # Garantiza unicidad por (svc, fecha)
    out = df[cols].copy().sort_values(["fecha","svc"])
    out = out.drop_duplicates(subset=["fecha","svc"], keep="first")
    return out

# ======================== UI ===========================================
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

st.title("Mel-IA — Plan táctico (diario por SVC)")
spr_mode = st.radio("SPR objetivo", ["promedio","peak","plan"], index=0, horizontal=True)

with st.expander("Cargando datos…", expanded=True):
    ok = True
    try:
        plan = compute_plan(spr_mode)
    except Exception as e:
        ok = False
        st.error(f"Error: {e}")

if not ok:
    st.stop()

# Filtro SVC (desde el plan ya consolidado)
svc_list = sorted(plan["svc"].dropna().astype(str).unique().tolist())
sel_svcs = st.multiselect("Filtrar SVC", svc_list, default=svc_list)

if sel_svcs:
    plan = plan[plan["svc"].isin(sel_svcs)]

# TABLA PRINCIPAL (una fila por svc-día)
st.subheader("Tabla principal — (svc, fecha) × Delivery model")
main_cols = [
    "fecha","svc",
    "shipments","ship_dc","ship_sp","ship_fcst_neto",
    "spr_objetivo","routes_need_total",
    "rentals_routes_max","routes_rentals_alloc",
    "share_crowd_obj","crowd_base_routes","routes_crowd_target","routes_crowd_base","crowd_e1_routes","routes_crowd_e1",
    "sdd_routes_max","spot_routes_max","sdd_trabaja","spot_trabaja",
    "routes_mlp_sdd_cap_day","routes_mlp_spot_cap_day",
    "routes_mlp_sdd_alloc","routes_mlp_spot_alloc",
    "routes_total_alloc","routes_deficit","shipments_plan",
    "alerta_spr_missing","alerta_deficit","risk_flag"
]
st.dataframe(plan[main_cols], use_container_width=True, hide_index=True)

# RESUMEN DE RIESGOS
st.subheader("Riesgos por fecha")
resumen = (plan.groupby("fecha", as_index=False)
           .agg(
                svcs_con_deficit=("alerta_deficit","sum"),
                rutas_deficit=("routes_deficit","sum"),
                svcs_sin_spr=("alerta_spr_missing","sum"),
           ))
st.dataframe(resumen, use_container_width=True, hide_index=True)


