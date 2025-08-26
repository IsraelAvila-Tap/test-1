import os, json
import yaml                       # <-- IMPORTANTE
import pandas as pd               # súbelo antes de usar pd en _to_num
import numpy as np
import streamlit as st
from math import ceil
from datetime import timedelta, datetime, date


# Lee credenciales desde Secrets y las expone como variable de entorno
if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
    # Caso 1: secreto como string JSON en una sola línea (con \n en la private_key)
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
elif "gcp_service_account" in st.secrets:
    # Caso 2: secreto como bloque [gcp_service_account] (TOML)
    # Lo convertimos a JSON válido para utils_gsheets
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps(dict(st.secrets["gcp_service_account"]))

# (opcional) si guardaste PROJECT_KEY en Secrets
if "PROJECT_KEY" in st.secrets:
    os.environ["PROJECT_KEY"] = st.secrets["PROJECT_KEY"]


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip(),
        errors="coerce"
    )


import numpy as np
from math import ceil
from datetime import timedelta, datetime, date
import streamlit as st

from utils_gsheets import read_ws, _client, get_service_account_email

# ------------- CONFIG -------------
st.set_page_config(page_title="Mel-IA — Plan táctico", layout="wide")

@st.cache_resource
def load_config():
    with open("config.yaml","r",encoding="utf-8") as f:
        return yaml.safe_load(f)
cfg = load_config()
proj_key = list(cfg["projects"].keys())[0]
proj = cfg["projects"][proj_key]
SHEET_ID = proj["sheet_id"]

# ------------- HELPERS -------------
def _norm_cols(df: pd.DataFrame):
    rn = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=rn)
    return df

def _weekday(d: date) -> int:
    return pd.Timestamp(d).weekday()  # 0=Lunes ... 6=Domingo

def _iso_yr_week(d: date):
    iso = pd.Timestamp(d).isocalendar()
    return int(iso.year), int(iso.week)

def _safe_mean(vals):
    vals = [float(v) for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan

def _read_sheet(tab: str) -> pd.DataFrame:
    df = read_ws(SHEET_ID, tab)
    return _norm_cols(df)

# ------------- READ TABS -------------
def load_fcst() -> pd.DataFrame:
    df = _read_sheet("FCST")
    # columnas esperadas: svc, fecha, shipments
    need = {"svc","fecha","shipments"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"FCST: faltan columnas {miss}")
    df["shipments"] = pd.to_numeric(df["shipments"], errors="coerce").fillna(0).astype(float)
    return df[["svc","fecha","shipments"]]

def load_spr_real() -> pd.DataFrame:
    df = _read_sheet("SPR")
    need = {"fecha","svc","spr"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"SPR: faltan columnas {miss}")

    # 🔧 clave: forzar SPR a numérico
    df["spr"] = _to_num(df["spr"])
    df = df.dropna(subset=["spr"])  # quita filas sin valor numérico

    day = (df.groupby(["fecha","svc"], as_index=False)["spr"]
             .mean()
             .rename(columns={"spr":"spr_exec"}))

    day["dow"] = day["fecha"].apply(_weekday)
    iso = day["fecha"].apply(lambda d: pd.Timestamp(d).isocalendar())
    day["iso_year"] = [int(x.year) for x in iso]
    day["iso_week"] = [int(x.week) for x in iso]
    return day


def load_capacity() -> pd.DataFrame:
    df = _read_sheet("Capacity")
    # columnas: delivery model, tipo (Routes/Shipments/SPR), svc, tipo dm, fecha, cantidad
    need = {"delivery model","tipo","svc","fecha","cantidad"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Capacity: faltan columnas {miss}")
    df["cantidad"] = _to_num(df["cantidad"]).fillna(0).astype(float)
    return df

def load_srm() -> pd.DataFrame:
    df = _read_sheet("SRM")
    # buscamos SVC y columnas que contengan "Total SDD" y "Total SPOT"
    svc_col = [c for c in df.columns if c in ("svc","svcs","svc ")]
    if not svc_col:
        raise ValueError("SRM: no se encontró columna SVC.")
    svc_col = svc_col[0]
    sdd_cols = [c for c in df.columns if "total" in c and "sdd" in c]
    spot_cols = [c for c in df.columns if "total" in c and "spot" in c]
    if not sdd_cols or not spot_cols:
        raise ValueError("SRM: no se hallaron columnas 'Total SDD ...' y/o 'Total SPOT ...'")
    out = df[[svc_col]+sdd_cols+spot_cols].copy()
    out = out.rename(columns={svc_col:"svc"})
    for c in sdd_cols+spot_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    out["sdd_routes_max"]  = out[sdd_cols].sum(axis=1)
    out["spot_routes_max"] = out[spot_cols].sum(axis=1)
    out = (out.groupby("svc", as_index=False)[["sdd_routes_max","spot_routes_max"]].sum())
    return out

def load_rentals() -> pd.DataFrame:
    df = _read_sheet("Rentals")
    # columnas: svc o svcs, unidades disponibles
    svc_col = "svc" if "svc" in df.columns else ("svcs" if "svcs" in df.columns else None)
    if not svc_col or "unidades disponibles" not in df.columns:
        raise ValueError("Rentals: se esperan columnas 'SVC/SVCs' y 'Unidades disponibles'.")
    df["unidades disponibles"] = pd.to_numeric(df["unidades disponibles"], errors="coerce").fillna(0).astype(int)
    out = df.groupby(svc_col, as_index=False)["unidades disponibles"].sum()
    out = out.rename(columns={svc_col:"svc","unidades disponibles":"rentals_routes_max"})
    return out

def load_crowd_caps() -> pd.DataFrame:
    df = _read_sheet("Crowd")
    # columnas parecidas a: 'svc', 'base entre sem', 'base sabado', 'base domingo',
    # 'holgura entre sem', 'holgura sabado', 'holgura domingo'
    if "svc" not in df.columns:
        raise ValueError("Crowd: falta columna 'SVC'.")
    def _pick(name_opts):
        for n in df.columns:
            for opt in name_opts:
                if opt in n:
                    return n
        return None
    c_base_wd = _pick(["base entre"])
    c_base_sa = _pick(["base sab"])
    c_base_su = _pick(["base domingo","base dom"])
    c_e1_wd   = _pick(["holgura entre","e1 entre"])
    c_e1_sa   = _pick(["holgura sab","e1 sab"])
    c_e1_su   = _pick(["holgura domingo","e1 dom"])

    need_cols = [c_base_wd,c_base_sa,c_base_su,c_e1_wd,c_e1_sa,c_e1_su]
    if any(c is None for c in need_cols):
        raise ValueError("Crowd: no se encontraron columnas base/e1 esperadas (entre semana/sábado/domingo).")

    for c in need_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    out = df[["svc", c_base_wd,c_base_sa,c_base_su,c_e1_wd,c_e1_sa,c_e1_su]].copy()
    out = out.rename(columns={
        c_base_wd:"base_wd", c_base_sa:"base_sa", c_base_su:"base_su",
        c_e1_wd:"e1_wd", c_e1_sa:"e1_sa", c_e1_su:"e1_su",
    })
    return out

# ------------- SPR SCENARIOS -------------
def compute_spr_scenarios(fcst: pd.DataFrame, spr_real: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    target = fcst[["fecha","svc"]].drop_duplicates().copy()
    target["dow"] = target["fecha"].apply(_weekday)
    target["iso_year"] = target["fecha"].apply(lambda d: int(pd.Timestamp(d).isocalendar().year))

    # SPR_promedio: últimas 4 semanas, MISMO DOW
    spr_exec_map = spr_real.set_index(["fecha","svc"])["spr_exec"]

    def avg_last4(row):
        d, s = row["fecha"], row["svc"]
        vals = []
        for k in [7,14,21,28]:
            dk = d - timedelta(days=k)
            v = spr_exec_map.get((dk,s), np.nan)
            if pd.notna(v):
                vals.append(float(v))
        if not vals:
            # fallback: últimos 28 días cualquier DOW
            mask = (spr_real["svc"]==s) & (spr_real["fecha"].between(d - timedelta(days=28), d - timedelta(days=1)))
            vals = list(spr_real.loc[mask,"spr_exec"])
        return _safe_mean(vals)

    target["spr_promedio"] = target.apply(avg_last4, axis=1)

    # SPR_peak: semanas 20-22 del mismo año, MISMO DOW (fallback 19..23)
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

    target["spr_peak"] = target.apply(avg_peak, axis=1)

    # SPR_plan desde Capacity (Tipo == 'SPR')
    cap = capacity.copy()
    m_spr = cap["tipo"].str.strip().str.lower().eq("spr")
    spr_plan = cap.loc[m_spr, ["svc","fecha","cantidad"]].rename(columns={"cantidad":"spr_plan"})
    if spr_plan.empty:
        # si no hay por fecha, intenta por SVC (promedio)
        spr_by_svc = cap.loc[m_spr].groupby("svc", as_index=False)["cantidad"].mean().rename(columns={"cantidad":"spr_plan"})
        spr_plan = target[["fecha","svc"]].merge(spr_by_svc, on="svc", how="left")
    target = target.merge(spr_plan, on=["fecha","svc"], how="left")

    return target[["fecha","svc","spr_promedio","spr_peak","spr_plan"]]

# ------------- SHARE CROWD (desde Capacity → Shipments) -------------
def compute_crowd_share(capacity: pd.DataFrame) -> pd.DataFrame:
    cap = capacity.copy()
    cap["tipo"] = cap["tipo"].str.strip().str.lower()
    cap["delivery model"] = cap["delivery model"].str.strip().str.lower()

    m_ship = cap["tipo"].eq("shipments")
    ship = cap.loc[m_ship, ["fecha","svc","delivery model","cantidad"]].copy()

    tot = ship.groupby(["fecha","svc"], as_index=False)["cantidad"].sum().rename(columns={"cantidad":"ship_total"})
    crw = ship.loc[ship["delivery model"].eq("crowd")].groupby(["fecha","svc"], as_index=False)["cantidad"].sum().rename(columns={"cantidad":"ship_crowd"})

    out = tot.merge(crw, on=["fecha","svc"], how="left").fillna({"ship_crowd":0.0})
    out["share_crowd_obj"] = np.where(out["ship_total"]>0, (out["ship_crowd"]/out["ship_total"]).clip(0,1), 0.0)
    return out[["fecha","svc","share_crowd_obj","ship_total","ship_crowd"]]

# ------------- CROWD CAP POR DÍA -------------
def map_crowd_capacity_by_date(target_days: pd.DataFrame, crowd_caps: pd.DataFrame) -> pd.DataFrame:
    # target_days: FECHA×SVC
    def cap_for(row):
        s, d = row["svc"], row["fecha"]
        dow = _weekday(d)  # 0-4 wd, 5 sab, 6 dom
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

# ------------- SCHEDULER MLP DESCANSOS -------------
def schedule_mlp_rest(df_day: pd.DataFrame) -> pd.DataFrame:
    # df_day: FECHA,SVC,routes_mlp_need,sdd_routes_max,spot_routes_max
    out = df_day.copy()
    out["week_key"] = out["fecha"].apply(lambda d: f"{_iso_yr_week(d)[0]}-{_iso_yr_week(d)[1]:02d}")
    out["sdd_trabaja"]  = 1
    out["spot_trabaja"] = 1

    def proc(g):
        n = len(g)
        need_days = int((g["routes_mlp_need"]>0).sum())
        # SDD: 6x7 pero no más que días con necesidad
        work_sdd = min(6, need_days)
        rest_sdd = max(n - work_sdd, 0)
        # Spot: 5x7; si hay déficit en ≥6 días, subir a 6x7; si hay <5 días con necesidad, trabajar solo esos
        work_spot = 5
        if need_days >= 6:
            work_spot = 6
        elif need_days < 5:
            work_spot = need_days
        rest_spot = max(n - work_spot, 0)

        # asigna descansos a días con menor necesidad
        g_sorted = g.sort_values(["routes_mlp_need","fecha"], ascending=[True,True])
        if rest_sdd > 0:
            idx = g_sorted.head(rest_sdd).index
            g.loc[idx,"sdd_trabaja"] = 0
        # Recalcula orden para Spot (levemente desalineado si hay empates)
        g_sorted2 = g.sort_values(["routes_mlp_need","fecha"], ascending=[True,True])
        if rest_spot > 0:
            idx2 = g_sorted2.head(rest_spot).index
            g.loc[idx2,"spot_trabaja"] = 0
        return g

    out = out.groupby(["svc","week_key"], group_keys=False).apply(proc)
    return out.drop(columns=["week_key"])

# ------------- MOTOR PRINCIPAL -------------
def compute_plan(spr_mode: str):
    # Carga
    fcst       = load_fcst()
    spr_real   = load_spr_real()
    capacity   = load_capacity()
    srm        = load_srm()
    rentals    = load_rentals()
    crowd_caps = load_crowd_caps()

    # SPRs
    spr_tbl = compute_spr_scenarios(fcst, spr_real, capacity)
    spr_col = {"promedio":"spr_promedio","peak":"spr_peak","plan":"spr_plan"}[spr_mode]

    # Share crowd objetivo desde Capacity (Shipments)
    share_tbl = compute_crowd_share(capacity)

    # CROWD caps por día
    target_days = fcst[["fecha","svc"]].drop_duplicates()
    crowd_daily = map_crowd_capacity_by_date(target_days, crowd_caps)

    # Merge base
    df = (fcst
          .merge(share_tbl, on=["fecha","svc"], how="left")
          .merge(crowd_daily, on=["fecha","svc"], how="left")
          .merge(srm, on="svc", how="left")
          .merge(rentals, on="svc", how="left")
          .merge(spr_tbl[["fecha","svc",spr_col]], on=["fecha","svc"], how="left")
         )

    # Limpieza/nulos
    for c in ["share_crowd_obj","crowd_base_routes","crowd_e1_routes","sdd_routes_max","spot_routes_max","rentals_routes_max"]:
        df[c] = pd.to_numeric(df.get(c,0), errors="coerce").fillna(0)
    df["spr_objetivo"] = pd.to_numeric(df[spr_col], errors="coerce")
    df.drop(columns=[spr_col], inplace=True)

    # Remanente (por ahora no se descuenta DC/SP explícito)
    df["q_rem"] = df["shipments"].clip(lower=0)

    # Rutas requeridas
    df["routes_need_total"] = np.where(
        (df["q_rem"]>0) & (df["spr_objetivo"]>0),
        np.ceil(df["q_rem"]/df["spr_objetivo"]).astype(int),
        0
    )
    df["alerta_spr_missing"] = ((df["q_rem"]>0) & (df["spr_objetivo"].isna() | (df["spr_objetivo"]<=0)))

    # Target Crowd
    df["routes_crowd_target"] = np.ceil(df["routes_need_total"] * df["share_crowd_obj"]).astype(int)
    # Asignación Crowd base y E1 (high cost)
    df["routes_crowd_base"] = np.minimum(df["routes_crowd_target"], df["crowd_base_routes"]).astype(int)
    df["routes_crowd_e1"] = np.minimum(
        (df["routes_crowd_target"] - df["routes_crowd_base"]).clip(lower=0),
        df["crowd_e1_routes"]
    ).astype(int)
    df["routes_crowd_alloc"] = df["routes_crowd_base"] + df["routes_crowd_e1"]
    df["alerta_crowd_high"] = df["routes_crowd_e1"] > 0

    # Rentals directo (rutas fijas)
    df["routes_after_crowd"] = (df["routes_need_total"] - df["routes_crowd_alloc"]).clip(lower=0).astype(int)
    df["routes_rentals_alloc"] = np.minimum(df["routes_after_crowd"], df["rentals_routes_max"]).astype(int)

    # Necesidad para MLP
    df["routes_mlp_need"] = (df["routes_after_crowd"] - df["routes_rentals_alloc"]).clip(lower=0).astype(int)

    # Descansos MLP por semana y SVC
    rest_base = df[["fecha","svc","routes_mlp_need","sdd_routes_max","spot_routes_max"]]
    rest_sched = schedule_mlp_rest(rest_base)
    df = df.merge(rest_sched[["fecha","svc","sdd_trabaja","spot_trabaja"]], on=["fecha","svc"], how="left")
    df["routes_mlp_cap_day"] = (df["sdd_routes_max"]*df["sdd_trabaja"] + df["spot_routes_max"]*df["spot_trabaja"]).fillna(0).astype(int)
    df["routes_mlp_alloc"] = np.minimum(df["routes_mlp_need"], df["routes_mlp_cap_day"]).astype(int)

    # Déficit + Shipments logrados
    df["routes_deficit"] = (df["routes_mlp_need"] - df["routes_mlp_alloc"]).clip(lower=0).astype(int)
    df["routes_total_alloc"] = (df["routes_crowd_alloc"] + df["routes_rentals_alloc"] + df["routes_mlp_alloc"]).astype(int)
    df["shipments_plan"] = np.where(
        df["spr_objetivo"]>0,
        df["routes_total_alloc"] * df["spr_objetivo"],
        0.0
    )
    df["alerta_deficit"] = df["shipments_plan"] + 1e-6 < df["shipments"]

    # Métricas
    df["spr_logrado"] = np.where(
        df["routes_total_alloc"]>0,
        df["q_rem"] / df["routes_total_alloc"],
        np.nan
    )
    df["share_crowd_real"] = np.where(
        df["routes_need_total"]>0,
        df["routes_crowd_alloc"] / df["routes_need_total"],
        0.0
    )
    df["risk_flag"] = df["alerta_deficit"] | df["alerta_spr_missing"]

    cols = [
        "fecha","svc","shipments","spr_objetivo",
        "q_rem","routes_need_total",
        "share_crowd_obj","routes_crowd_target","routes_crowd_base","routes_crowd_e1","routes_crowd_alloc","alerta_crowd_high",
        "rentals_routes_max","routes_rentals_alloc",
        "sdd_routes_max","spot_routes_max","sdd_trabaja","spot_trabaja","routes_mlp_need","routes_mlp_cap_day","routes_mlp_alloc",
        "routes_deficit","routes_total_alloc","shipments_plan","spr_logrado","share_crowd_real",
        "alerta_spr_missing","alerta_deficit","risk_flag"
    ]
    return df[cols].sort_values(["fecha","svc"])

# ------------- UI -------------
with st.sidebar:
    st.header("⚙️ Proyecto")
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

try:
    plan = compute_plan(spr_mode)
    st.dataframe(plan, use_container_width=True, hide_index=True)

    st.subheader("Riesgos por fecha")
    resumen = (plan.groupby("fecha")
               .agg(
                   svcs_con_deficit=("alerta_deficit","sum"),
                   rutas_deficit=("routes_deficit","sum"),
                   svcs_sin_spr=("alerta_spr_missing","sum"),
               ).reset_index())
    st.dataframe(resumen, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Error: {e}")



