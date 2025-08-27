# app.py
# -*- coding: utf-8 -*-

import os, json, re
import pandas as pd
import numpy as np
from math import ceil
from datetime import timedelta, date, datetime
import streamlit as st

from utils_gsheets import read_ws, _client, get_service_account_email

# ================== Credenciales desde st.secrets ==================
if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
elif "gcp_service_account" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps(dict(st.secrets["gcp_service_account"]))
if "PROJECT_KEY" in st.secrets:
    os.environ["PROJECT_KEY"] = st.secrets["PROJECT_KEY"]

# ================== Config básica ==================
st.set_page_config(page_title="Mel-IA — Plan táctico", layout="wide")

# ------------------ Helpers generales ------------------
def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() if str(c).strip() else f"col_{i+1}" for i, c in enumerate(df.columns)]
    return df

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("%", "", regex=False)
         .str.strip(),
        errors="coerce"
    )

def _norm_date_col(df: pd.DataFrame, col: str) -> pd.Series:
    # intentamos día/mes/año y año-mes-día; devolvemos date
    s = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    s = s.fillna(pd.to_datetime(df[col], errors="coerce", dayfirst=False))
    return s.dt.date

def _weekday(d: date) -> int:
    return pd.Timestamp(d).weekday()  # 0=Lunes ... 6=Domingo

def _iso_year_week(d: date):
    iso = pd.Timestamp(d).isocalendar()
    return int(iso.year), int(iso.week)

def _safe_mean(seq) -> float:
    arr = [float(x) for x in seq if pd.notna(x)]
    return float(np.mean(arr)) if arr else np.nan

# ------------------ Config del proyecto ------------------
# Lee config.yaml si existe; si no, deja edición rápida en sidebar
def _load_sheet_id() -> str:
    try:
        import yaml
        with open("config.yaml","r",encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        key = list(cfg["projects"].keys())[0]
        return cfg["projects"][key]["sheet_id"]
    except Exception:
        return st.sidebar.text_input("Sheet ID", value="", placeholder="1UBjU3-...").strip()

SHEET_ID = _load_sheet_id()

# ================== Loaders ==================

# ---- FCST (fecha, svc, shipments) ----
def load_fcst() -> pd.DataFrame:
    df = read_ws(SHEET_ID, "FCST")
    df = _lower_cols(df)
    need = {"svc","fecha","shipments"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"FCST: faltan columnas {miss}")
    df["fecha"] = _norm_date_col(df, "fecha")
    df["shipments"] = _to_num(df["shipments"]).fillna(0.0).astype(float)
    df = df.dropna(subset=["fecha","svc"])
    return df[["fecha","svc","shipments"]]

# ---- SPR (ejecutado real por día) -> para promedio/peak ----
def load_spr_real() -> pd.DataFrame:
    df = read_ws(SHEET_ID, "SPR")
    df = _lower_cols(df)
    # Aceptamos columnas variadas pero requerimos al menos 'svc','fecha','spr'
    cand_svc = [c for c in df.columns if c in ("svc","svcs","svc ")]
    if not cand_svc: raise ValueError("SPR: falta columna 'svc'")
    sc = cand_svc[0]
    if "fecha" not in df.columns: raise ValueError("SPR: falta columna 'fecha'")
    if "spr" not in df.columns:   raise ValueError("SPR: falta columna 'spr'")

    out = df[[sc,"fecha","spr"]].copy()
    out["svc"]   = out[sc].astype(str).str.strip().str.upper()
    out["fecha"] = _norm_date_col(out, "fecha")
    out["spr"]   = _to_num(out["spr"])
    out = out.dropna(subset=["fecha","svc","spr"])

    out["dow"] = out["fecha"].apply(_weekday)
    iso = out["fecha"].apply(lambda d: pd.Timestamp(d).isocalendar())
    out["iso_year"] = [int(x.year) for x in iso]
    out["iso_week"] = [int(x.week) for x in iso]
    return out.rename(columns={"spr":"spr_exec"})[["fecha","svc","dow","iso_year","iso_week","spr_exec"]]

# ---- Capacity (shipments/routes/spr plan por Delivery model) ----
def load_capacity() -> pd.DataFrame:
    df = read_ws(SHEET_ID, "Capacity")
    df = _lower_cols(df)
    need = {"delivery model","tipo","svc","fecha","cantidad"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Capacity: faltan columnas {miss}")
    df["fecha"]    = _norm_date_col(df, "fecha")
    df["cantidad"] = _to_num(df["cantidad"]).fillna(0.0)
    df["tipo"]     = df["tipo"].astype(str).str.strip().str.lower()
    df["delivery model"] = df["delivery model"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["fecha","svc"])
    return df

# ---- SRM (SDD / SPOT) con header raro (fila 5, etc.) ----
def _find_header_row_by_svc_df(raw: pd.DataFrame, search_rows: int = 12) -> int | None:
    for i in range(min(search_rows, len(raw))):
        vals = (raw.iloc[i].astype(str)
                .str.replace(r"\s+"," ", regex=True)
                .str.strip()
                .str.lower())
        if any(v in ("svc","svcs","svc ") for v in vals):
            return i
    return None

def _apply_header_from_row(raw: pd.DataFrame, hdr_idx: int) -> pd.DataFrame:
    hdr = (raw.iloc[hdr_idx]
           .astype(str)
           .str.replace(r"[\r\n]+", " ", regex=True)
           .str.replace(r"\s+"," ", regex=True)
           .str.strip()
           .str.lower()
           .tolist())
    hdr = [h if h else f"col_{i+1}" for i, h in enumerate(hdr)]
    body = raw.iloc[hdr_idx+1:].copy()
    body.columns = hdr
    return _lower_cols(body).dropna(how="all", axis=1).reset_index(drop=True)

def load_srm() -> pd.DataFrame:
    raw = read_ws(SHEET_ID, "SRM")
    raw = _lower_cols(raw)
    hdr_idx = _find_header_row_by_svc_df(raw)
    df = _apply_header_from_row(raw, hdr_idx) if hdr_idx is not None else raw.copy()

    # buscamos col SVC
    svc_col = None
    for c in ("svc","svcs","svc "):
        if c in df.columns: svc_col = c; break
    if svc_col is None: raise ValueError("SRM: no se encontró columna 'SVC'.")

    # columnas que contengan totales SDD/Spot (muy ancho, así que sumamos todas las que empiecen con esos patrones)
    sdd_cols  = [c for c in df.columns if ("sdd" in c) and ("total" in c)]
    spot_cols = [c for c in df.columns if ("spot" in c) and ("total" in c)]
    if not sdd_cols and not spot_cols:
        # fallback: toma cualquier col que contenga 'sdd' o 'spot' y sea numérica
        sdd_cols  = [c for c in df.columns if "sdd"  in c]
        spot_cols = [c for c in df.columns if "spot" in c]

    for c in sdd_cols + spot_cols:
        df[c] = _to_num(df[c]).fillna(0.0)

    out = pd.DataFrame({"svc": df[svc_col].astype(str).str.strip().str.upper()})
    out["sdd_routes_max"]  = df[sdd_cols].sum(axis=1)  if sdd_cols  else 0.0
    out["spot_routes_max"] = df[spot_cols].sum(axis=1) if spot_cols else 0.0

    out = out.groupby("svc", as_index=False)[["sdd_routes_max","spot_routes_max"]].sum()
    out[["sdd_routes_max","spot_routes_max"]] = out[["sdd_routes_max","spot_routes_max"]].fillna(0.0).astype(int)
    return out

# ---- Rentals (svc, unidades disponibles) ----
def load_rentals() -> pd.DataFrame:
    df = read_ws(SHEET_ID, "Rentals")
    df = _lower_cols(df)
    # nombres flexibles
    svc_col = "svc" if "svc" in df.columns else ("svcs" if "svcs" in df.columns else None)
    if svc_col is None:
        raise ValueError("Rentals: falta columna 'SVC'/'SVCs'")
    qty_col = None
    for c in df.columns:
        if "unidades" in c and "dispon" in c:
            qty_col = c; break
    if qty_col is None:
        raise ValueError("Rentals: falta columna 'Unidades disponibles'")

    out = (df.groupby(svc_col, as_index=False)[qty_col].sum()
             .rename(columns={svc_col:"svc", qty_col:"rentals_units"}))
    out["svc"] = out["svc"].astype(str).str.strip().str.upper()
    out["rentals_units"] = _to_num(out["rentals_units"]).fillna(0.0).astype(int)
    # Interpretamos "unidades" como rutas máximas de rentals (1 unidad = 1 ruta)
    out = out.rename(columns={"rentals_units":"rentals_routes_max"})
    return out

# ---- Crowd (soporta layout detallado y layout 2-filas base/e1) ----
def _guess_svc_col(df: pd.DataFrame) -> str | None:
    svc_regex = re.compile(r"^[A-Z]{3}\d$")
    best_col, best_ratio = None, 0.0
    for c in df.columns:
        s = df[c].astype(str).str.strip().str.upper()
        valid = s[(s!="") & (s!="NAN")]
        if valid.empty: continue
        ratio = valid.map(lambda x: bool(svc_regex.match(x))).mean()
        if ratio > best_ratio:
            best_ratio, best_col = ratio, c
    return best_col if best_ratio >= 0.5 else None

def load_crowd() -> pd.DataFrame:
    raw = read_ws(SHEET_ID, "Crowd").dropna(how="all", axis=1)
    raw = _lower_cols(raw)

    hdr_idx = _find_header_row_by_svc_df(raw)
    df = _apply_header_from_row(raw, hdr_idx) if hdr_idx is not None else raw.copy()
    df = _lower_cols(df)

    # localizar/crear 'svc'
    svc_col = None
    for cand in ("svc","svcs","svc "):
        if cand in df.columns: svc_col = cand; break
    if svc_col is None:
        guess = _guess_svc_col(df)
        if guess:
            df = df.rename(columns={guess: "svc"})
            svc_col = "svc"
    if svc_col is None:
        raise ValueError("Crowd: falta columna 'svc'")

    # Layout A: columnas 'base' y 'e1' que agrupan tres subcolumnas (entre/sab/dom) en la primera fila
    if ("base" in df.columns) and ("e1" in df.columns):
        labrow = df.iloc[0].astype(str).str.strip().str.lower()

        cols = list(df.columns)
        def take3(idx): return cols[idx:idx+3]
        b_idx = cols.index("base")
        e_idx = cols.index("e1")
        base_cols = take3(b_idx); e1_cols = take3(e_idx)

        def reorder(group_cols):
            labs = [str(labrow.get(c,"")).lower() for c in group_cols]
            mapping = {}
            for c, lab in zip(group_cols, labs):
                if ("entre" in lab) or ("sem" in lab) or (lab in ("wd","entre semana")): mapping["wd"] = c
                elif "sab" in lab: mapping["sa"] = c
                elif "dom" in lab: mapping["su"] = c
            order = ["wd","sa","su"]
            out = [mapping.get(k) for k in order]
            rest = [c for c in group_cols if c not in out]
            for i in range(3):
                if out[i] is None and rest: out[i] = rest.pop(0)
            return out

        b_wd, b_sa, b_su = reorder(base_cols)
        e_wd, e_sa, e_su = reorder(e1_cols)

        df = df.iloc[1:].reset_index(drop=True)  # quita fila etiquetas
        for c in [b_wd, b_sa, b_su, e_wd, e_sa, e_su]:
            df[c] = _to_num(df[c]).fillna(0.0)

        out = pd.DataFrame({
            "svc": df[svc_col].astype(str).str.strip().str.upper(),
            "base_wd": df[b_wd].astype(int),
            "base_sa": df[b_sa].astype(int),
            "base_su": df[b_su].astype(int),
            "e1_wd":   df[e_wd].astype(int),
            "e1_sa":   df[e_sa].astype(int),
            "e1_su":   df[e_su].astype(int),
        })
        return out.groupby("svc", as_index=False).sum()

    # Layout B: detallado (nombres con 'base entre/sab/dom' y 'holgura/e1 entre/sab/dom')
    def pick(opts): 
        for n in df.columns:
            n2 = str(n).lower()
            if any(opt in n2 for opt in opts): return n
        return None

    c_base_wd = pick(["base entre","entre sem"])
    c_base_sa = pick(["base sab"])
    c_base_su = pick(["base dom"])
    c_e1_wd   = pick(["holgura entre","e1 entre"])
    c_e1_sa   = pick(["holgura sab","e1 sab"])
    c_e1_su   = pick(["holgura dom","e1 dom"])

    need = [c_base_wd,c_base_sa,c_base_su,c_e1_wd,c_e1_sa,c_e1_su]
    if any(c is None for c in need):
        raise ValueError(f"Crowd: no se reconoció layout. Encabezados: {list(df.columns)}")

    for c in need: df[c] = _to_num(df[c]).fillna(0.0)

    out = df[[svc_col,c_base_wd,c_base_sa,c_base_su,c_e1_wd,c_e1_sa,c_e1_su]].copy()
    out.columns = ["svc","base_wd","base_sa","base_su","e1_wd","e1_sa","e1_su"]
    out["svc"] = out["svc"].astype(str).str.strip().str.upper()
    return out.groupby("svc", as_index=False).sum()

# alias por compatibilidad si el resto del código lo usa
def load_crowd_caps() -> pd.DataFrame:
    return load_crowd()

# ================== Lógicas de negocio ==================

# SPR escenarios: promedio, peak, plan (diario por svc)
def compute_spr_scenarios(fcst: pd.DataFrame, spr_real: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    target = fcst[["fecha","svc"]].drop_duplicates().copy()
    target["dow"] = target["fecha"].apply(_weekday)
    target["iso_year"] = target["fecha"].apply(lambda d: int(pd.Timestamp(d).isocalendar().year))

    spr_map = spr_real.set_index(["fecha","svc"])["spr_exec"]

    def avg_last4(row):
        d, s = row["fecha"], row["svc"]
        vals = []
        for k in (7,14,21,28):
            dk = d - timedelta(days=k)
            v = spr_map.get((dk,s), np.nan)
            if pd.notna(v): vals.append(float(v))
        if not vals:
            m = (spr_real["svc"].eq(s) & spr_real["fecha"].between(d - timedelta(days=28), d - timedelta(days=1)))
            vals = list(spr_real.loc[m, "spr_exec"])
        return _safe_mean(vals)

    def avg_peak(row):
        d, s, yr, dow = row["fecha"], row["svc"], row["iso_year"], row["dow"]
        m = (spr_real["svc"].eq(s) & spr_real["iso_year"].eq(yr) &
             spr_real["iso_week"].isin([20,21,22]) & spr_real["dow"].eq(dow))
        vals = list(spr_real.loc[m,"spr_exec"])
        if not vals:
            m = (spr_real["svc"].eq(s) & spr_real["iso_year"].eq(yr) &
                 spr_real["iso_week"].between(19,23) & spr_real["dow"].eq(dow))
            vals = list(spr_real.loc[m,"spr_exec"])
        return _safe_mean(vals)

    target["spr_promedio"] = target.apply(avg_last4, axis=1)
    target["spr_peak"]     = target.apply(avg_peak,  axis=1)

    # SPR plan desde Capacity (tipo == 'spr')
    cap = capacity.copy()
    m_spr = cap["tipo"].eq("spr")
    spr_plan = cap.loc[m_spr, ["svc","fecha","cantidad"]].rename(columns={"cantidad":"spr_plan"})
    if spr_plan.empty:
        spr_by_svc = cap.loc[m_spr].groupby("svc", as_index=False)["cantidad"].mean().rename(columns={"cantidad":"spr_plan"})
        spr_plan = target[["fecha","svc"]].merge(spr_by_svc, on="svc", how="left")

    target = target.merge(spr_plan, on=["fecha","svc"], how="left")
    return target[["fecha","svc","spr_promedio","spr_peak","spr_plan"]]

# Shipments de DC/SP desde Capacity (tipo==shipments)
def compute_dc_sp_shipments(capacity: pd.DataFrame) -> pd.DataFrame:
    cap = capacity.copy()
    cap["tipo"] = cap["tipo"].str.strip().str.lower()
    cap["delivery model"] = cap["delivery model"].str.strip().str.lower()

    m_ship = cap["tipo"].eq("shipments")
    df = cap.loc[m_ship, ["fecha","svc","delivery model","cantidad"]].copy()

    # "dc" / "delivery cell" / "cells"
    is_dc = df["delivery model"].str.contains("cell", na=False) | df["delivery model"].str.fullmatch("dc", case=False, na=False)
    # "service partner" / "sp"
    is_sp = df["delivery model"].str.contains("partner", na=False) | df["delivery model"].str.fullmatch("sp", case=False, na=False)

    dc = (df.loc[is_dc].groupby(["fecha","svc"], as_index=False)["cantidad"].sum()
            .rename(columns={"cantidad":"ship_dc"}))
    sp = (df.loc[is_sp].groupby(["fecha","svc"], as_index=False)["cantidad"].sum()
            .rename(columns={"cantidad":"ship_sp"}))

    out = dc.merge(sp, on=["fecha","svc"], how="outer").fillna(0.0)
    return out

# Share crowd objetivo desde shipments (capacidad -> tipo shipments)
def compute_crowd_share(capacity: pd.DataFrame) -> pd.DataFrame:
    cap = capacity.copy()
    cap["tipo"] = cap["tipo"].str.strip().str.lower()
    cap["delivery model"] = cap["delivery model"].str.strip().str.lower()

    m_ship = cap["tipo"].eq("shipments")
    ship = cap.loc[m_ship, ["fecha","svc","delivery model","cantidad"]].copy()
    tot = ship.groupby(["fecha","svc"], as_index=False)["cantidad"].sum().rename(columns={"cantidad":"ship_total"})
    crw = ship.loc[ship["delivery model"].str.contains("crowd", na=False)].groupby(["fecha","svc"], as_index=False)["cantidad"].sum().rename(columns={"cantidad":"ship_crowd"})
    out = tot.merge(crw, on=["fecha","svc"], how="left").fillna({"ship_crowd":0.0})
    out["share_crowd_obj"] = np.where(out["ship_total"]>0, (out["ship_crowd"]/out["ship_total"]).clip(0,1), 0.0)
    return out[["fecha","svc","share_crowd_obj","ship_total","ship_crowd"]]

# Scheduler de descansos (SDD: 6x7; Spot: 5x7 ó 6x7 según necesidad)
def schedule_mlp_rest(df_day: pd.DataFrame) -> pd.DataFrame:
    out = df_day.copy()
    out["week_key"] = out["fecha"].apply(lambda d: f"{_iso_year_week(d)[0]}-{_iso_year_week(d)[1]:02d}")
    out["sdd_trabaja"]  = 1
    out["spot_trabaja"] = 1

    def proc(g):
        n = len(g)
        need_days = int((g["routes_after_rentals"]>0).sum())
        work_sdd  = min(6, need_days)
        rest_sdd  = max(n - work_sdd, 0)

        work_spot = 5
        if need_days >= 6: work_spot = 6
        elif need_days < 5: work_spot = need_days
        rest_spot = max(n - work_spot, 0)

        gs = g.sort_values(["routes_after_rentals","fecha"], ascending=[True,True])
        if rest_sdd>0:
            idx = gs.head(rest_sdd).index
            g.loc[idx,"sdd_trabaja"] = 0
        gs2 = g.sort_values(["routes_after_rentals","fecha"], ascending=[True,True])
        if rest_spot>0:
            idx2 = gs2.head(rest_spot).index
            g.loc[idx2,"spot_trabaja"] = 0
        return g

    out = out.groupby(["svc","week_key"], group_keys=False).apply(proc)
    return out.drop(columns=["week_key"])

# ================== Motor principal ==================
def compute_plan(spr_mode: str) -> pd.DataFrame:
    # 1) carga
    fcst       = load_fcst()
    spr_real   = load_spr_real()
    capacity   = load_capacity()
    srm        = load_srm()
    rentals    = load_rentals()
    crowd_caps = load_crowd()

    # 2) spr escenarios
    spr_tbl = compute_spr_scenarios(fcst, spr_real, capacity)
    spr_col = {"promedio":"spr_promedio","peak":"spr_peak","plan":"spr_plan"}[spr_mode]

    # 3) shipments DC/SP y share crowd
    dcsp   = compute_dc_sp_shipments(capacity)
    share  = compute_crowd_share(capacity)

    # 4) base por día (svc,fecha)
    df = (fcst
          .merge(dcsp,  on=["fecha","svc"], how="left")
          .merge(share, on=["fecha","svc"], how="left")
          .merge(srm,    on="svc", how="left")
          .merge(rentals,on="svc", how="left")
          .merge(crowd_caps, on="svc", how="left")
          .merge(spr_tbl[["fecha","svc",spr_col]], on=["fecha","svc"], how="left")
          )

    # completar nulos
    for c in ["ship_dc","ship_sp","share_crowd_obj","sdd_routes_max","spot_routes_max",
              "rentals_routes_max","base_wd","base_sa","base_su","e1_wd","e1_sa","e1_su"]:
        if c in df.columns:
            df[c] = _to_num(df[c]).fillna(0.0)
        else:
            df[c] = 0.0

    df = df.rename(columns={spr_col:"spr_objetivo"})
    df["spr_objetivo"] = _to_num(df["spr_objetivo"])

    # 5) Remanente: FCST - DC - SP (nunca negativo)
    df["ship_fcst_neto"] = (df["shipments"] - df["ship_dc"] - df["ship_sp"]).clip(lower=0.0)
    df["alerta_spr_missing"] = ((df["ship_fcst_neto"]>0) & (df["spr_objetivo"].isna() | (df["spr_objetivo"]<=0)))

    # 6) Rutas totales requeridas (sobre remanente)
    df["routes_need_total"] = np.where(
        (df["ship_fcst_neto"]>0) & (df["spr_objetivo"]>0),
        np.ceil(df["ship_fcst_neto"]/df["spr_objetivo"]).astype(int),
        0
    )

    # 7) Rentals primero (1 unidad = 1 ruta)
    df["routes_rentals_alloc"] = np.minimum(df["routes_need_total"], df["rentals_routes_max"]).astype(int)
    df["routes_after_rentals"] = (df["routes_need_total"] - df["routes_rentals_alloc"]).clip(lower=0).astype(int)

    # 8) Crowd target (sobre remanente tras rentals)
    #     Capacidad por día según DOW
    def _crowd_caps_day(row):
        dow = _weekday(row["fecha"])
        if dow <=4:
            base, e1 = row["base_wd"], row["e1_wd"]
        elif dow==5:
            base, e1 = row["base_sa"], row["e1_sa"]
        else:
            base, e1 = row["base_su"], row["e1_su"]
        return pd.Series({"crowd_base_day":int(base), "crowd_e1_day":int(e1)})

    tmp = df.apply(_crowd_caps_day, axis=1)
    df = pd.concat([df, tmp], axis=1)

    df["routes_crowd_target"] = np.ceil(df["routes_after_rentals"] * df["share_crowd_obj"]).astype(int)
    df["routes_crowd_base"]   = np.minimum(df["routes_crowd_target"], df["crowd_base_day"]).astype(int)
    df["routes_crowd_e1"]     = np.minimum(
        (df["routes_crowd_target"] - df["routes_crowd_base"]).clip(lower=0),
        df["crowd_e1_day"]
    ).astype(int)
    df["routes_crowd_alloc"]  = df["routes_crowd_base"] + df["routes_crowd_e1"]

    # 9) Restante para MLPs
    df["routes_after_crowd"] = (df["routes_after_rentals"] - df["routes_crowd_alloc"]).clip(lower=0).astype(int)

    # 10) Asignar descansos MLP por semana (usa 'routes_after_rentals' como señal)
    rest_base = df[["fecha","svc","routes_after_rentals"]].copy()
    sched = schedule_mlp_rest(rest_base)
    df = df.merge(sched[["fecha","svc","sdd_trabaja","spot_trabaja"]], on=["fecha","svc"], how="left").fillna({"sdd_trabaja":1,"spot_trabaja":1})

    df["routes_mlp_cap_day"] = (df["sdd_routes_max"]*df["sdd_trabaja"] + df["spot_routes_max"]*df["spot_trabaja"]).astype(int)
    df["routes_mlp_alloc"]   = np.minimum(df["routes_after_crowd"], df["routes_mlp_cap_day"]).astype(int)

    # 11) Si aún hay déficit, usar Crowd EXTRA (E1 restante) sin pasarse de la nube
    df["routes_after_mlp"]  = (df["routes_after_crowd"] - df["routes_mlp_alloc"]).clip(lower=0).astype(int)
    e1_left = (df["crowd_e1_day"] - df["routes_crowd_e1"]).clip(lower=0).astype(int)
    df["routes_crowd_extra"] = np.minimum(df["routes_after_mlp"], e1_left).astype(int)

    # 12) Totales y riesgos
    df["routes_total_alloc"] = (df["routes_rentals_alloc"] + df["routes_crowd_alloc"] +
                                df["routes_mlp_alloc"] + df["routes_crowd_extra"]).astype(int)

    df["shipments_plan"] = np.where(
        df["spr_objetivo"]>0,
        df["routes_total_alloc"] * df["spr_objetivo"],
        0.0
    )
    df["routes_deficit"] = (df["routes_need_total"] - df["routes_total_alloc"]).clip(lower=0).astype(int)
    df["alerta_deficit"] = df["shipments_plan"] + 1e-6 < (df["ship_fcst_neto"])

    df["spr_logrado"] = np.where(df["routes_total_alloc"]>0,
                                 df["ship_fcst_neto"] / df["routes_total_alloc"], np.nan)
    df["share_crowd_real"] = np.where(df["routes_need_total"]>0,
                                      (df["routes_crowd_alloc"]+df["routes_crowd_extra"]) / df["routes_need_total"], 0.0)
    df["risk_flag"] = df["alerta_deficit"] | df["alerta_spr_missing"]

    cols = [
        "fecha","svc",
        "shipments","ship_dc","ship_sp","ship_fcst_neto",
        "spr_objetivo","routes_need_total",
        "rentals_routes_max","routes_rentals_alloc",
        "share_crowd_obj","crowd_base_day","crowd_e1_day",
        "routes_crowd_target","routes_crowd_base","routes_crowd_e1","routes_crowd_extra",
        "sdd_routes_max","spot_routes_max","sdd_trabaja","spot_trabaja",
        "routes_mlp_cap_day","routes_mlp_alloc",
        "routes_total_alloc","routes_deficit","shipments_plan",
        "spr_logrado","share_crowd_real",
        "alerta_spr_missing","alerta_deficit","risk_flag"
    ]
    return df[cols].sort_values(["fecha","svc"]).reset_index(drop=True)

# ================== UI ==================

with st.sidebar:
    st.header("📂 Proyecto")
    st.write(f"**Sheet:** `{SHEET_ID or '—'}`")
    st.subheader("🔐 Credenciales")
    svc_email = get_service_account_email()
    if svc_email:
        st.caption("Comparte el Sheet con:")
        st.code(svc_email, language="text")
    else:
        st.warning("No se detectó Service Account.")

st.title("Mel-IA — Plan táctico (diario por SVC)")

spr_mode = st.radio("SPR objetivo", ["promedio","peak","plan"], index=0, horizontal=True)

# Carga + cálculo
try:
    with st.expander("Cargando datos..."):
        st.write("1/6 FCST…")
        _ = load_fcst()
        st.write("2/6 SPR (real)…")
        _ = load_spr_real()
        st.write("3/6 Capacity…")
        _ = load_capacity()
        st.write("4/6 SRM…")
        _ = load_srm()
        st.write("5/6 Rentals…")
        _ = load_rentals()
        st.write("6/6 Crowd…")
        _ = load_crowd()
        st.success("Datos listos ✅")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Plano
try:
    plan = compute_plan(spr_mode)
except Exception as e:
    st.error(f"Error al calcular el plan: {e}")
    st.stop()

# Filtro SVC seguro (sin Series ambiguo)
svc_list = sorted(plan["svc"].dropna().astype(str).unique().tolist())
sel_svcs = st.multiselect("Filtrar SVC", svc_list, default=svc_list)
if sel_svcs:
    plan = plan[plan["svc"].isin(sel_svcs)]

# Tabla principal (una fila por svc-fecha)
st.subheader("Tabla principal — (svc, fecha) × Delivery model")
st.dataframe(plan, use_container_width=True, hide_index=True)

# Resumen de riesgos
st.subheader("Riesgos por fecha")
resumen = (plan.groupby("fecha", as_index=False)
           .agg(svcs_con_deficit=("alerta_deficit","sum"),
                rutas_deficit=("routes_deficit","sum"),
                svcs_sin_spr=("alerta_spr_missing","sum")))
st.dataframe(resumen, use_container_width=True, hide_index=True)




