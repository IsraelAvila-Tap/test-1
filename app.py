# ----------------------------- #
# Mel-IA – Plan táctico (SVC/día)
# ----------------------------- #

import os, json, yaml, math
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# ====== Credenciales desde Secrets (2 formatos posibles) ======
if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
elif "gcp_service_account" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps(dict(st.secrets["gcp_service_account"]))

if "PROJECT_KEY" in st.secrets:
    os.environ["PROJECT_KEY"] = st.secrets["PROJECT_KEY"]

# ====== Imports locales ======
from utils_gsheets import read_ws, _client, get_service_account_email

# ====== Config Streamlit ======
st.set_page_config(page_title="Mel-IA — Plan táctico", layout="wide")


# ====== Helpers robustos ======

def SSTRIP(s: pd.Series) -> pd.Series:
    """strip seguro para Series (convierte a str y aplica .str.strip())."""
    return s.astype(str).str.strip()

def _to_num(s: pd.Series) -> pd.Series:
    """limpia comas/% y convierte a numérico"""
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip(),
        errors="coerce"
    )

def _weekday(d) -> int:
    return pd.Timestamp(d).weekday()  # 0=Lun ... 6=Dom

def _iso_yr_week(d):
    iso = pd.Timestamp(d).isocalendar()
    return int(iso.year), int(iso.week)

def _safe_mean(vals):
    vals = [float(v) for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan

def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    rn = {c: str(c).strip().lower() for c in df.columns}
    return df.rename(columns=rn)

def _norm_date_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True).dt.date
    return df

import re

def _find_header_row_by_svc_df(raw: pd.DataFrame, search_rows: int = 12) -> int | None:
    """Devuelve el índice (0-based) de la fila que contiene 'svc'/'svcs'. None si no aparece."""
    for i in range(min(search_rows, len(raw))):
        vals = (
            raw.iloc[i]
               .astype(str)
               .str.replace(r"\s+", " ", regex=True)
               .str.strip()
               .str.lower()
        )
        if any(v in ("svc", "svcs", "svc ") for v in vals):
            return i
    return None

def _apply_header_from_row(raw: pd.DataFrame, hdr_idx: int) -> pd.DataFrame:
    hdr = (
        raw.iloc[hdr_idx]
           .astype(str)
           .str.replace(r"[\r\n]+", " ", regex=True)
           .str.replace(r"\s+", " ", regex=True)
           .str.strip()
           .str.lower()
           .tolist()
    )
    hdr = [h if h else f"col_{i+1}" for i, h in enumerate(hdr)]
    body = raw.iloc[hdr_idx + 1 :].copy()
    body.columns = hdr
    body = body.dropna(how="all", axis=1).reset_index(drop=True)
    return body

def _guess_svc_col(df: pd.DataFrame) -> str | None:
    """Adivina la columna SVC por patrón (p.ej. SPB1, SMX9, SGD1, SMT1)."""
    svc_regex = re.compile(r"^[A-Z]{3}\d$")  # 3 letras + 1 dígito (e.g., SPB1)
    best_col, best_ratio = None, 0.0
    for c in df.columns:
        s = df[c].astype(str).str.strip().str.upper()
        valid = s[(s != "") & (s != "NAN")]
        if valid.empty:
            continue
        ratio = valid.map(lambda x: bool(svc_regex.match(x))).mean()
        if ratio > best_ratio:
            best_ratio, best_col = ratio, c
    return best_col if best_ratio >= 0.5 else None


# ====== Carga de config.yaml ======

@st.cache_resource
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()
proj_key = list(cfg["projects"].keys())[0]
proj = cfg["projects"][proj_key]
SHEET_ID = proj["sheet_id"]


# ====== Loaders ======

def load_fcst() -> pd.DataFrame:
    df = _lower_cols(read_ws(SHEET_ID, "FCST"))
    # columnas candidatas
    c_svc   = "svc" if "svc" in df.columns else ("svcs" if "svcs" in df.columns else None)
    c_fecha = "fecha"
    c_ship  = "shipments"
    need = [c_svc, c_fecha, c_ship]
    if any(c is None or c not in df.columns for c in need):
        raise ValueError(f"FCST: columnas requeridas no detectadas. Encabezados: {list(df.columns)}")
    df["svc"] = SSTRIP(df[c_svc]).str.upper()
    df = _norm_date_col(df, c_fecha).rename(columns={c_fecha:"fecha"})
    df["shipments"] = _to_num(df[c_ship]).fillna(0.0)
    return df[["fecha","svc","shipments"]]


def load_spr_sheet() -> pd.DataFrame:
    df = _lower_cols(read_ws(SHEET_ID, "SPR"))

    # Mapeo tolerante de columnas
    map_cand = {
        "delivery_model": ["delivery_model", "delivery model", "dm"],
        "fecha":          ["fecha", "date"],
        "svc":            ["svc", "svcs"],
        "veh":            ["shp_lg_vehicle", "veh", "vehicle", "tipo de vehículo", "tipo dm"],
        "spr":            ["spr", "shipments per route"],
    }
    picked = {}
    for k, opts in map_cand.items():
        for o in opts:
            if o in df.columns:
                picked[k] = o
                break

    need = ["fecha", "svc", "spr"]
    if any(k not in picked for k in need):
        # no hay hoja compatible -> devolvemos vacía
        return pd.DataFrame(columns=["fecha","svc","dm","veh","spr","dow","iso_year","iso_week"])

    # Normalización básica
    df["svc"] = SSTRIP(df[picked.get("svc","svc")]).str.upper()
    df["dm"]  = SSTRIP(df[picked.get("delivery_model","")]).str.upper() if "delivery_model" in picked else "GEN"
    df["veh"] = SSTRIP(df[picked.get("veh","")]).str.upper() if "veh" in picked else "GEN"

    # Fechas y SPR
    df = _norm_date_col(df, picked["fecha"]).rename(columns={picked["fecha"]: "fecha"})
    df["spr"] = _to_num(df[picked["spr"]]).astype(float)

    # 🔒 Filtra filas sin fecha válida para evitar NaTType/isocalendar
    df = df.dropna(subset=["fecha"]).copy()

    # DOW y año/semana ISO de forma segura
    dt = pd.to_datetime(df["fecha"], errors="coerce")
    iso = dt.dt.isocalendar()
    df["dow"] = dt.dt.weekday
    df["iso_year"] = iso["year"].astype("Int64")
    df["iso_week"] = iso["week"].astype("Int64")

    # Descarta cualquier fila que aún no tenga año/semana/dow calculados (muy raro)
    df = df.dropna(subset=["dow","iso_year","iso_week"]).copy()
    df[["dow","iso_year","iso_week"]] = df[["dow","iso_year","iso_week"]].astype(int)

    return df[["fecha","svc","dm","veh","spr","dow","iso_year","iso_week"]]


def load_capacity() -> pd.DataFrame:
    df = _lower_cols(read_ws(SHEET_ID, "Capacity"))
    # normalizamos nombres (delivery model, tipo, svc, tipo dm, fecha, cantidad)
    # permite encabezados con espacios/variantes
    ren = {}
    for c in df.columns:
        cc = c.strip().lower()
        if cc in ("delivery model","delivery_model","dm"): ren[c] = "dm"
        elif cc in ("tipo","type"):                         ren[c] = "tipo"
        elif cc in ("svc","svcs"):                          ren[c] = "svc"
        elif cc in ("tipo dm","shp_lg_vehicle","vehicle","veh","tipo de vehículo"): ren[c] = "veh"
        elif cc in ("fecha","date"):                        ren[c] = "fecha"
        elif cc in ("cantidad","qty","cantidad "):          ren[c] = "cantidad"
    df = df.rename(columns=ren)

    need = {"dm","tipo","svc","fecha","cantidad"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Capacity: faltan columnas {miss}. Encabezados: {list(df.columns)}")

    df["dm"]   = SSTRIP(df["dm"]).str.upper()
    df["tipo"] = SSTRIP(df["tipo"]).str.lower()
    df["svc"]  = SSTRIP(df["svc"]).str.upper()
    df["veh"]  = SSTRIP(df["veh"]).str.upper() if "veh" in df.columns else "GEN"
    df         = _norm_date_col(df, "fecha")
    df["cantidad"] = _to_num(df["cantidad"]).fillna(0.0)

    return df[["fecha","svc","dm","veh","tipo","cantidad"]]


# --- Helpers SRM (detectar encabezado y aplicarlo) ---
def _find_header_row_by_svc_df(raw: pd.DataFrame, search_rows: int = 12) -> int:
    """Devuelve el índice (0-based) de la fila que contiene 'svc'/'svcs'.
    Si no aparece en las primeras 'search_rows' filas, usa fila 5 (índice 4)."""
    for i in range(min(search_rows, len(raw))):
        vals = (
            raw.iloc[i]
               .astype(str)
               .str.replace(r"\s+", " ", regex=True)
               .str.strip()
               .str.lower()
        )
        if any(v in ("svc", "svcs", "svc ") for v in vals):
            return i
    return 4  # fallback: encabezado en fila 5

def _apply_header_from_row(raw: pd.DataFrame, hdr_idx: int) -> pd.DataFrame:
    """Usa la fila hdr_idx como header y devuelve el body con columnas normalizadas."""
    hdr = (
        raw.iloc[hdr_idx]
           .astype(str)
           .str.replace(r"[\r\n]+", " ", regex=True)
           .str.replace(r"\s+", " ", regex=True)
           .str.strip()
           .str.lower()
           .tolist()
    )
    hdr = [h if h else f"col_{i+1}" for i, h in enumerate(hdr)]
    body = raw.iloc[hdr_idx + 1 :].copy()
    body.columns = hdr
    body = body.dropna(how="all", axis=1).reset_index(drop=True)
    return body

# --- SRM loader robusto (reemplaza tu load_srm actual por este) ---
def load_srm() -> pd.DataFrame:
    # 1) Leemos tal cual (sin header_row)
    raw = read_ws(SHEET_ID, "SRM")

    # 2) Detectamos la fila de encabezados por presencia de 'svc'
    hdr_idx = _find_header_row_by_svc_df(raw)
    df = _apply_header_from_row(raw, hdr_idx)

    # 3) Normaliza nombres
    df = df.rename(columns={c: (str(c).strip().lower()) for c in df.columns})

    # 4) Localiza columna SVC
    svc_col = None
    for c in ("svc", "svcs", "svc "):
        if c in df.columns:
            svc_col = c
            break
    if not svc_col:
        raise ValueError("SRM: no se encontró columna 'SVC' en el encabezado detectado.")

    df["svc"] = df[svc_col].astype(str).str.strip().str.upper()

    # 5) Detecta columnas SDD y SPOT (muy flexible; soporta cientos de columnas)
    cols = list(df.columns)
    sdd_cols  = [c for c in cols if "sdd"  in c]
    spot_cols = [c for c in cols if "spot" in c]

    if not sdd_cols and not spot_cols:
        raise ValueError(f"SRM: no se hallaron columnas con 'sdd' o 'spot'. Encabezados: {cols[:40]}")

    # 6) Suma segura (coerción numérica) por fila
    def _sum_cols(clist):
        if not clist:
            return pd.Series(0, index=df.index)
        tmp = df[clist].apply(_to_num, axis=0).fillna(0)
        return tmp.sum(axis=1)

    df["_sdd"]  = _sum_cols(sdd_cols)
    df["_spot"] = _sum_cols(spot_cols)

    # 7) Agrega por SVC y devuelve capacidades diarias máximas en rutas
    out = (df.groupby("svc", as_index=False)[["_sdd", "_spot"]]
             .sum()
             .rename(columns={"_sdd": "sdd_routes_max", "_spot": "spot_routes_max"}))

    out[["sdd_routes_max", "spot_routes_max"]] = (
        out[["sdd_routes_max", "spot_routes_max"]].round(0).astype(int)
    )
    return out



def load_rentals_by_vehicle() -> pd.DataFrame:
    df = _lower_cols(read_ws(SHEET_ID, "Rentals"))
    # buscamos: svcs/svc, "unidades disponibles", "tipo de vehículo" (opcional)
    svc_col = "svc" if "svc" in df.columns else ("svcs" if "svcs" in df.columns else None)
    qty_col = None
    for c in df.columns:
        if "unidades" in c and "dispon" in c:
            qty_col = c
            break
    veh_col = None
    for c in df.columns:
        if "tipo de vehículo" in c or c in ("veh","vehicle","tipo de vehiculo"):
            veh_col = c
            break

    if not svc_col or not qty_col:
        # devolvemos vacía si no hay columnas mínimas
        return pd.DataFrame(columns=["svc","veh","units"])

    df["svc"] = SSTRIP(df[svc_col]).str.upper()
    df["veh"] = SSTRIP(df[veh_col]).str.upper() if veh_col else "GEN"
    df["units"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0).astype(int)
    out = df.groupby(["svc","veh"], as_index=False)["units"].sum()
    return out


def load_crowd() -> pd.DataFrame:
    """Devuelve: svc, base_wd, base_sa, base_su, e1_wd, e1_sa, e1_su (enteros)."""
    raw = read_ws(SHEET_ID, "Crowd").dropna(how="all", axis=1)

    # 1) Intento: detectar fila de encabezado con 'svc'
    hdr_idx = _find_header_row_by_svc_df(raw)
    if hdr_idx is not None:
        df = _apply_header_from_row(raw, hdr_idx)
    else:
        # si no hay 'svc' visible en las primeras filas, usamos tal cual
        df = raw.copy()

    # Normaliza nombres de columnas
    cols = [str(c).strip().lower() if str(c).strip() else f"col_{i+1}" for i, c in enumerate(df.columns)]
    df.columns = cols

    # 2) Localizar/crear columna SVC
    svc_col = None
    for cand in ("svc", "svcs", "svc "):
        if cand in df.columns:
            svc_col = cand
            break
    if svc_col is None:
        guess = _guess_svc_col(df)
        if guess:
            df = df.rename(columns={guess: "svc"})
            svc_col = "svc"
    if svc_col is None:
        raise ValueError("Crowd: falta columna 'svc' y no se pudo inferir por patrón. "
                         f"Encabezados vistos: {list(df.columns)}")

    # 3) Layout A: encabezado en 2 filas (grupos 'base' y 'e1' + subetiquetas wd/sa/su en la primera fila de datos)
    if ("base" in df.columns) and ("e1" in df.columns):
        # la primera fila bajo el header contiene 'entre / sab / dom'
        labrow = df.iloc[0].astype(str).str.strip().str.lower()

        def _take3(start_idx: int) -> list[str]:
            return cols[start_idx:start_idx+3]

        b_idx = cols.index("base")
        e_idx = cols.index("e1")
        base_cols = _take3(b_idx)
        e1_cols   = _take3(e_idx)

        def _reorder(group_cols: list[str]) -> tuple[str, str, str]:
            labs = [str(labrow.get(c, "")).lower() for c in group_cols]
            mapping = {}
            for c, lab in zip(group_cols, labs):
                if ("entre" in lab) or ("sem" in lab) or (lab in ("wd", "entre semana")):
                    mapping["wd"] = c
                elif "sab" in lab:
                    mapping["sa"] = c
                elif "dom" in lab:
                    mapping["su"] = c
            # completa por orden si falta
            order = ["wd", "sa", "su"]
            out = [mapping.get(k) for k in order]
            rest = [c for c in group_cols if c not in out]
            for i in range(3):
                if out[i] is None and rest:
                    out[i] = rest.pop(0)
            return out[0], out[1], out[2]

        b_wd, b_sa, b_su = _reorder(base_cols)
        e_wd, e_sa, e_su = _reorder(e1_cols)

        # quita fila de etiquetas
        df = df.iloc[1:].reset_index(drop=True)

        for c in [b_wd, b_sa, b_su, e_wd, e_sa, e_su]:
            df[c] = _to_num(df[c]).fillna(0)

        out = pd.DataFrame({
            "svc": df[svc_col].astype(str).str.strip().str.upper(),
            "base_wd": df[b_wd].astype(int),
            "base_sa": df[b_sa].astype(int),
            "base_su": df[b_su].astype(int),
            "e1_wd": df[e_wd].astype(int),
            "e1_sa": df[e_sa].astype(int),
            "e1_su": df[e_su].astype(int),
        })
        return out.groupby("svc", as_index=False).sum()

    # 4) Layout B: detallado (columnas ya separadas)
    def _pick(name_opts: list[str]) -> str | None:
        for n in df.columns:
            n2 = str(n).lower()
            if any(opt in n2 for opt in name_opts):
                return n
        return None

    c_base_wd = _pick(["base entre", "entre sem"])
    c_base_sa = _pick(["base sab"])
    c_base_su = _pick(["base dom"])
    c_e1_wd   = _pick(["holgura entre", "e1 entre"])
    c_e1_sa   = _pick(["holgura sab", "e1 sab"])
    c_e1_su   = _pick(["holgura dom", "e1 dom"])

    need_cols = [c_base_wd, c_base_sa, c_base_su, c_e1_wd, c_e1_sa, c_e1_su]
    if any(c is None for c in need_cols):
        raise ValueError(
            f"Crowd: no se reconoció layout. Encabezados: {list(df.columns)}\n"
            "Soportado:\n"
            "  • Header en 2 filas (base/e1 + entre/sab/dom)\n"
            "  • Detallado (Base entre semana/sábado/domingo + Holgura entre/sábado/domingo)"
        )

    for c in need_cols:
        df[c] = _to_num(df[c]).fillna(0)

    out = df[[svc_col, c_base_wd, c_base_sa, c_base_su, c_e1_wd, c_e1_sa, c_e1_su]].copy()
    out.columns = ["svc", "base_wd", "base_sa", "base_su", "e1_wd", "e1_sa", "e1_su"]
    out["svc"] = out["svc"].astype(str).str.strip().str.upper()
    return out.groupby("svc", as_index=False).sum()


# --- compatibilidad con código previo ---
def load_crowd_caps() -> pd.DataFrame:
    """Alias compatible: devuelve las mismas columnas que load_crowd()."""
    return load_crowd()


# ====== SPR (promedio/peak/plan) por SVC-fecha ======

def compute_spr_obj(fcst: pd.DataFrame, spr_real: pd.DataFrame, capacity: pd.DataFrame, mode: str) -> pd.DataFrame:
    target = fcst[["fecha","svc"]].drop_duplicates().copy()
    target["dow"] = target["fecha"].apply(_weekday)
    target["iso_year"] = target["fecha"].apply(lambda d: int(pd.Timestamp(d).isocalendar().year))

    # promedio: últimas 4 semanas mismo DOW
    exec_map = spr_real.set_index(["fecha","svc"])["spr"] if not spr_real.empty else pd.Series(dtype=float)

    def avg_last4(row):
        if spr_real.empty: return np.nan
        d, s = row["fecha"], row["svc"]
        vals = []
        for k in (7,14,21,28):
            dk = d - timedelta(days=k)
            v = exec_map.get((dk,s), np.nan)
            if pd.notna(v): vals.append(float(v))
        if not vals:
            mask = (spr_real["svc"].eq(s) &
                    spr_real["fecha"].between(d - timedelta(days=28), d - timedelta(days=1)))
            vals = list(spr_real.loc[mask, "spr"])
        return _safe_mean(vals)

    def avg_peak(row):
        if spr_real.empty: return np.nan
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
        # plan: Capacity tipo==SPR (por fecha si existe)
        cap = capacity.copy()
        m = cap["tipo"].eq("spr")
        spr_plan = cap.loc[m, ["fecha","svc","cantidad"]].rename(columns={"cantidad":"spr_plan"})
        if spr_plan.empty:
            # por SVC promedio
            spr_plan = (cap.loc[m].groupby("svc", as_index=False)["cantidad"]
                        .mean().rename(columns={"cantidad":"spr_plan"}))
            target = target.merge(spr_plan, on="svc", how="left")
        else:
            target = target.merge(spr_plan, on=["fecha","svc"], how="left")
        target = target.rename(columns={"spr_plan":"spr_objetivo"})

    return target[["fecha","svc","spr_objetivo"]]


# ====== Share crowd objetivo (desde Capacity – Shipments) ======

def compute_crowd_share(capacity: pd.DataFrame) -> pd.DataFrame:
    cap = capacity.copy()
    cap["tipo"] = SSTRIP(cap["tipo"]).str.lower()
    cap["dm"]   = SSTRIP(cap["dm"]).str.upper()
    m_ship = cap["tipo"].eq("shipments")
    ship = cap.loc[m_ship, ["fecha","svc","dm","cantidad"]].copy()

    tot = ship.groupby(["fecha","svc"], as_index=False)["cantidad"].sum().rename(columns={"cantidad":"ship_total_plan"})
    crw = (ship.loc[ship["dm"].str.contains("CROWD|CRO|CRWD")]
           .groupby(["fecha","svc"], as_index=False)["cantidad"].sum()
           .rename(columns={"cantidad":"ship_crowd_plan"}))

    out = tot.merge(crw, on=["fecha","svc"], how="left").fillna({"ship_crowd_plan":0.0})
    out["share_crowd_obj"] = np.where(out["ship_total_plan"]>0,
                                      (out["ship_crowd_plan"]/out["ship_total_plan"]).clip(0,1), 0.0)
    return out[["fecha","svc","share_crowd_obj"]]


# ====== Scheduler de descansos (6x7 / 5x7) ======

def schedule_mlp_rest(df_day: pd.DataFrame) -> pd.DataFrame:
    """
    df_day: FECHA,SVC,routes_mlp_need,sdd_routes_max,spot_routes_max
    devuelve flags sdd_trabaja / spot_trabaja por día, distribuyendo descansos
    """
    out = df_day.copy()
    out["week_key"] = out["fecha"].apply(lambda d: f"{_iso_yr_week(d)[0]}-{_iso_yr_week(d)[1]:02d}")
    out["sdd_trabaja"]  = 1
    out["spot_trabaja"] = 1

    def proc(g):
        n = len(g)
        need_days = int((g["routes_mlp_need"]>0).sum())
        work_sdd  = min(6, max(need_days, 0))
        work_spot = 5 if need_days <= 5 else 6 if need_days >= 6 else 5

        rest_sdd  = max(n - work_sdd, 0)
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


# ====== Motor principal ======

def compute_plan(spr_mode: str, svc_filter: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 1) Cargas
    fcst       = load_fcst()
    spr_real   = load_spr_sheet()
    capacity   = load_capacity()
    srm        = load_srm()
    rentals    = load_rentals_by_vehicle()
    crowd_caps = load_crowd_caps()

    if svc_filter:
        fcst = fcst[fcst["svc"].isin(svc_filter)]

    # 2) SPR objetivo por SVC-fecha
    spr_tbl = compute_spr_obj(fcst, spr_real, capacity, spr_mode)

    # 3) share crowd
    share_tbl = compute_crowd_share(capacity)

    # 4) Base (una fila por SVC-fecha)
    base = (fcst.merge(spr_tbl, on=["fecha","svc"], how="left")
                 .merge(share_tbl, on=["fecha","svc"], how="left")
            )
    base["share_crowd_obj"] = base["share_crowd_obj"].fillna(0.0)
    base["spr_objetivo"]    = _to_num(base["spr_objetivo"]).fillna(0.0)

    # 5) Shipments DC/SP desde Capacity (Shipments por DM)
    cap_ship = capacity[capacity["tipo"].eq("shipments")].copy()
    cap_ship["dm"] = cap_ship["dm"].astype(str).str.upper()

    def dm_is_dc(s):  return any(x in s for x in ("DC","DELIVERY CELLS","DELIVERY_CELLS","CELLS"))
    def dm_is_sp(s):  return any(x in s for x in ("SP","SERVICE PARTNERS","SERVICE_PARTNERS"))
    cap_ship["is_dc"] = cap_ship["dm"].apply(dm_is_dc)
    cap_ship["is_sp"] = cap_ship["dm"].apply(dm_is_sp)

    dc = (cap_ship[cap_ship["is_dc"]]
          .groupby(["fecha","svc"], as_index=False)["cantidad"].sum()
          .rename(columns={"cantidad":"ship_dc"}))
    sp = (cap_ship[cap_ship["is_sp"]]
          .groupby(["fecha","svc"], as_index=False)["cantidad"].sum()
          .rename(columns={"cantidad":"ship_sp"}))

    base = (base
            .merge(dc, on=["fecha","svc"], how="left")
            .merge(sp, on=["fecha","svc"], how="left")
           ).fillna({"ship_dc":0.0,"ship_sp":0.0})

    # 6) Rentals: pasar unidades a shipments usando SPR (fallback: spr_objetivo)
    rent = rentals.copy()
    if rent.empty:
        base["ship_rentals"] = 0.0
    else:
        # para cada SVC-fecha tomamos SPR fallback (el elegido) – si luego agregas SPR por veh/DM,
        # acá se puede refinar.
        rent_total = rent.groupby("svc", as_index=False)["units"].sum().rename(columns={"units":"units_total"})
        base = base.merge(rent_total, on="svc", how="left").fillna({"units_total":0})
        base["ship_rentals"] = base["units_total"] * base["spr_objetivo"]

    # 7) Remanente tras DC/SP/Rentals
    base["ship_fcst_neto"] = (base["shipments"] - base["ship_dc"] - base["ship_sp"] - base["ship_rentals"]).clip(lower=0.0)

    # 8) Crowd: objetivo por % del plan y capacidad por DOW (base/E1)
    #    Primero mapeamos capacidad por DOW al SVC-fecha
    def map_crowd_cap(row):
        s, d = row["svc"], row["fecha"]
        dow = _weekday(d)
        r = crowd_caps[crowd_caps["svc"].eq(s)]
        if r.empty:
            return pd.Series({"crowd_base_routes":0, "crowd_e1_routes":0})
        r = r.iloc[0]
        if dow <= 4:
            base_r, e1_r = r["base_wd"], r["e1_wd"]
        elif dow == 5:
            base_r, e1_r = r["base_sa"], r["e1_sa"]
        else:
            base_r, e1_r = r["base_su"], r["e1_su"]
        return pd.Series({"crowd_base_routes":int(base_r), "crowd_e1_routes":int(e1_r)})

    caps = base.apply(map_crowd_cap, axis=1)
    base = pd.concat([base, caps], axis=1)

    # shipments crowd target y rutas
    base["ship_crowd_target"] = (base["ship_fcst_neto"] * base["share_crowd_obj"]).clip(lower=0.0)
    spr_crowd = base["spr_objetivo"].replace(0, np.nan)
    base["routes_crowd_target"] = np.ceil(base["ship_crowd_target"] / spr_crowd).fillna(0).astype(int)

    base["routes_crowd_base"] = np.minimum(base["routes_crowd_target"], base["crowd_base_routes"]).astype(int)
    base["routes_crowd_e1"]   = np.minimum(
        (base["routes_crowd_target"] - base["routes_crowd_base"]).clip(lower=0),
        base["crowd_e1_routes"]
    ).astype(int)

    base["routes_crowd_alloc"] = base["routes_crowd_base"] + base["routes_crowd_e1"]
    base["ship_crowd_alloc"]   = base["routes_crowd_alloc"] * base["spr_objetivo"]

    # Remanente tras Crowd
    base["ship_after_crowd"] = (base["ship_fcst_neto"] - base["ship_crowd_alloc"]).clip(lower=0.0)

    # 9) MLP (SDD 6×7, Spot 5×7): rutas disponibles desde SRM con scheduler
    base = base.merge(srm, on="svc", how="left").fillna({"sdd_routes_max":0.0,"spot_routes_max":0.0})

    # rutas que aún se necesitan (en rutas) después de Crowd
    base["routes_need_after_crowd"] = np.ceil(base["ship_after_crowd"] / spr_crowd).fillna(0).astype(int)

    # plan de descansos por semana con base en necesidad
    sched_input = base[["fecha","svc","routes_need_after_crowd","sdd_routes_max","spot_routes_max"]].rename(
        columns={"routes_need_after_crowd":"routes_mlp_need"}
    )
    sched = schedule_mlp_rest(sched_input)
    base = base.merge(sched[["fecha","svc","sdd_trabaja","spot_trabaja"]], on=["fecha","svc"], how="left").fillna(1)

    base["routes_sdd_avail"]  = (base["sdd_routes_max"]  * base["sdd_trabaja"]).astype(int)
    base["routes_spot_avail"] = (base["spot_routes_max"] * base["spot_trabaja"]).astype(int)

    base["routes_sdd_alloc"] = np.minimum(base["routes_need_after_crowd"], base["routes_sdd_avail"]).astype(int)
    ship_left = (base["routes_need_after_crowd"] - base["routes_sdd_alloc"]).clip(lower=0)
    base["routes_spot_alloc"] = np.minimum(ship_left, base["routes_spot_avail"]).astype(int)

    base["ship_mlp_sdd"]  = base["routes_sdd_alloc"]  * base["spr_objetivo"]
    base["ship_mlp_spot"] = base["routes_spot_alloc"] * base["spr_objetivo"]

    # Remanente tras MLP
    base["ship_after_mlp"] = (base["ship_after_crowd"] - base["ship_mlp_sdd"] - base["ship_mlp_spot"]).clip(lower=0.0)

    # 10) Crowd extra (si aún falta) hasta tope (base+E1)
    base["routes_crowd_max"] = (base["crowd_base_routes"] + base["crowd_e1_routes"]).astype(int)
    base["routes_crowd_extra_cap"] = (base["routes_crowd_max"] - base["routes_crowd_alloc"]).clip(lower=0).astype(int)
    need_extra_routes = np.ceil(base["ship_after_mlp"] / spr_crowd).fillna(0).astype(int)
    base["routes_crowd_extra"] = np.minimum(need_extra_routes, base["routes_crowd_extra_cap"]).astype(int)

    base["ship_crowd_extra"] = base["routes_crowd_extra"] * base["spr_objetivo"]

    # Shipments logrados y déficit
    base["ship_total_alloc"] = (base["ship_dc"] + base["ship_sp"] + base["ship_rentals"] +
                                base["ship_crowd_alloc"] + base["ship_mlp_sdd"] + base["ship_mlp_spot"] +
                                base["ship_crowd_extra"])
    base["ship_deficit"] = (base["shipments"] - base["ship_total_alloc"]).clip(lower=0.0)

    # ===== Tabla principal (una fila por SVC-fecha) con columnas por DM =====
    main_cols = [
        "fecha","svc","shipments","spr_objetivo",
        "ship_dc","ship_sp","ship_rentals",
        "ship_crowd_alloc","ship_mlp_sdd","ship_mlp_spot","ship_crowd_extra",
        "ship_total_alloc","ship_deficit",
        "routes_crowd_base","routes_crowd_e1","routes_sdd_alloc","routes_spot_alloc","routes_crowd_extra"
    ]
    main = base[main_cols].sort_values(["fecha","svc"])

    # Resumen riesgo por fecha
    resumen = (main.assign(risk_flag = main["ship_deficit"] > 1e-6)
                    .groupby("fecha", as_index=False)
                    .agg(
                        svcs_con_deficit=("risk_flag","sum"),
                        rutas_deficit=("ship_deficit", lambda s: int(np.ceil(s.sum() / (main["spr_objetivo"].replace(0,np.nan).mean() or 1)) ) ),
                    ))

    return main, resumen


# ====== UI ======

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

# Filtro SVC dinámico
try:
    fcst_preview = load_fcst()
    svc_list = sorted(fcst_preview["svc"].unique().tolist())
except Exception:
    svc_list = []

sel_svcs = st.multiselect("Filtrar SVC", options=svc_list, default=svc_list[:4])

# Ejecución
try:
    with st.expander("Cargando datos…", expanded=True):
        st.write("1/6 FCST…")
        _ = load_fcst()
        st.write("2/6 SPR (real)…")
        _ = load_spr_sheet()
        st.write("3/6 Capacity…")
        _ = load_capacity()
        st.write("4/6 SRM…")
        _ = load_srm()
        st.write("5/6 Rentals…")
        _ = load_rentals_by_vehicle()
        st.write("6/6 Crowd…")
        _ = load_crowd_caps()
        st.success("Datos listos ✅")

    main, resumen = compute_plan(spr_mode, svc_filter=sel_svcs)

    st.subheader("Tabla principal — (svc, fecha) × Delivery model")
    st.dataframe(main, use_container_width=True, hide_index=True)

    st.subheader("Riesgos por fecha")
    st.dataframe(resumen, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Error: {e}")




