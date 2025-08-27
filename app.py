# ---------------- Credenciales desde Secrets ----------------
import os, json, streamlit as st
# Caso 1: secreto como string JSON (GOOGLE_SERVICE_ACCOUNT_JSON)
if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
# Caso 2: secreto como bloque [gcp_service_account]
elif "gcp_service_account" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps(dict(st.secrets["gcp_service_account"]))
# Opcional: PROJECT_KEY
if "PROJECT_KEY" in st.secrets:
    os.environ["PROJECT_KEY"] = st.secrets["PROJECT_KEY"]

# ---------------- Imports ----------------
import yaml, pandas as pd, numpy as np
from math import ceil
from datetime import timedelta, date, datetime
from utils_gsheets import read_ws, _client, get_service_account_email

st.set_page_config(page_title="Mel-IA — Plan táctico", layout="wide")

# ---------------- Helpers ----------------
@st.cache_resource
def load_config():
    with open("config.yaml","r",encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()
proj_key = list(cfg["projects"].keys())[0]
proj = cfg["projects"][proj_key]
SHEET_ID = proj["sheet_id"]

def _weekday(d: date) -> int:
    return pd.Timestamp(d).weekday()  # 0=lun ... 6=dom

def _safe_mean(vals):
    vals = [float(v) for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",","",regex=False).str.replace("%","",regex=False).str.strip(),
        errors="coerce"
    )

def _read_sheet(tab: str) -> pd.DataFrame:
    df = read_ws(SHEET_ID, tab)
    if df is None: 
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    # normaliza fechas si vienen como texto d/m/Y
    for c in df.columns:
        if c.lower() in ("fecha","date"):
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True).dt.date
    return df

# ---------------- Loaders base ----------------
@st.cache_data(ttl=300)
def load_fcst() -> pd.DataFrame:
    df = _read_sheet("FCST")
    need = {"SVC","Fecha","Shipments"}
    if miss := (need - set(df.columns)):
        raise ValueError(f"FCST: faltan columnas {miss}")
    out = df.rename(columns={"SVC":"svc","Fecha":"fecha","Shipments":"shipments"})
    out["svc"] = out["svc"].astype(str).str.strip().str.upper()
    out["shipments"] = _to_num(out["shipments"]).fillna(0.0)
    return out[["fecha","svc","shipments"]]

@st.cache_data(ttl=300)
def load_capacity() -> pd.DataFrame:
    df = _read_sheet("Capacity")
    need = {"Delivery model","Tipo","SVC","Tipo DM","Fecha","Cantidad"}
    if miss := (need - set(df.columns)):
        raise ValueError(f"Capacity: faltan columnas {miss}")
    df = df.rename(columns={
        "Delivery model":"dm","Tipo":"tipo","SVC":"svc","Tipo DM":"veh","Fecha":"fecha","Cantidad":"cantidad"
    })
    df["dm"]  = df["dm"].astype(str).strip().str.upper()
    df["veh"] = df["veh"].astype(str).strip().str.upper()
    df["svc"] = df["svc"].astype(str).strip().str.upper()
    df["cantidad"] = _to_num(df["cantidad"]).fillna(0.0)
    return df

@st.cache_data(ttl=300)
def compute_crowd_share(capacity: pd.DataFrame) -> pd.DataFrame:
    cap = capacity.copy()
    cap["tipo"] = cap["tipo"].str.strip().str.lower()
    cap["dm"]   = cap["dm"].astype(str).str.upper()
    m_ship = cap["tipo"].eq("shipments")
    ship = cap.loc[m_ship, ["fecha","svc","dm","cantidad"]].copy()
    tot = ship.groupby(["fecha","svc"], as_index=False)["cantidad"].sum().rename(columns={"cantidad":"ship_total"})
    crw = (ship.loc[ ship["dm"].str.contains("CROWD", na=False) ]
           .groupby(["fecha","svc"], as_index=False)["cantidad"].sum()
           .rename(columns={"cantidad":"ship_crowd"}))
    out = tot.merge(crw, on=["fecha","svc"], how="left").fillna({"ship_crowd":0.0})
    out["share_crowd_obj"] = np.where(out["ship_total"]>0, (out["ship_crowd"]/out["ship_total"]).clip(0,1), 0.0)
    return out

# ---------------- SPR ejecutado (por DM/veh) y SPR plan ----------------
@st.cache_data(ttl=300)
def load_spr_real_dm() -> pd.DataFrame:
    df = _read_sheet("SPR")
    need = {"DELIVERY_MODEL","FECHA","SVC","SHP_LG_VEHICLE","SPR"}
    if miss := (need - set(df.columns)):
        raise ValueError(f"SPR: faltan columnas {miss}")
    df = df.rename(columns={
        "DELIVERY_MODEL":"dm","FECHA":"fecha","SVC":"svc","SHP_LG_VEHICLE":"veh","SPR":"spr"
    })
    df["dm"]  = df["dm"].astype(str).str.strip().str.upper()
    df["veh"] = df["veh"].astype(str).str.strip().str.upper()
    df["svc"] = df["svc"].astype(str).str.strip().str.upper()
    df["spr"] = _to_num(df["spr"])
    df = df.dropna(subset=["spr"])
    df["dow"] = df["fecha"].apply(_weekday)
    iso = df["fecha"].apply(lambda d: pd.Timestamp(d).isocalendar())
    df["iso_year"] = [int(x.year) for x in iso]
    df["iso_week"] = [int(x.week) for x in iso]
    return df[["fecha","svc","dm","veh","spr","dow","iso_year","iso_week"]]

@st.cache_data(ttl=300)
def load_spr_plan_dm() -> pd.DataFrame:
    cap = load_capacity()
    m = cap["tipo"].str.strip().str.lower().eq("spr")
    df = cap.loc[m, ["fecha","svc","dm","veh","cantidad"]].copy()
    df["dm"]  = df["dm"].astype(str).str.strip().str.upper()
    df["veh"] = df["veh"].astype(str).str.strip().str.upper()
    df["spr_plan"] = _to_num(df["cantidad"])
    return df.drop(columns=["cantidad"])

def spr_scenario_lookup(spr_mode: str, spr_real_dm: pd.DataFrame, spr_plan_dm: pd.DataFrame,
                        keys: pd.DataFrame) -> pd.DataFrame:
    t = keys.copy()
    t["dow"] = t["fecha"].apply(_weekday)
    t["iso_year"] = t["fecha"].apply(lambda d: int(pd.Timestamp(d).isocalendar().year))

    if spr_mode in ("promedio","peak"):
        r = spr_real_dm
        def avg_last4(row):
            d,s,dm,v = row["fecha"],row["svc"],row["dm"],row["veh"]
            vals=[]
            for k in (7,14,21,28):
                dk=d - timedelta(days=k)
                m=(r["fecha"].eq(dk)&r["svc"].eq(s)&r["dm"].eq(dm)&r["veh"].eq(v))
                if m.any(): vals.append(float(r.loc[m,"spr"].mean()))
            if not vals:
                m=(r["svc"].eq(s)&r["dm"].eq(dm)&r["veh"].eq(v)&
                   r["fecha"].between(d-timedelta(days=28), d - timedelta(days=1)))
                vals=list(r.loc[m,"spr"])
            return _safe_mean(vals)
        def avg_peak(row):
            s,dm,v,yr,dow=row["svc"],row["dm"],row["veh"],row["iso_year"],row["dow"]
            m=(r["svc"].eq(s)&r["dm"].eq(dm)&r["veh"].eq(v)&
               r["iso_year"].eq(yr)&r["iso_week"].isin([20,21,22])&r["dow"].eq(dow))
            vals=list(r.loc[m,"spr"])
            if not vals:
                m=(r["svc"].eq(s)&r["dm"].eq(dm)&r["veh"].eq(v)&
                   r["iso_year"].eq(yr)&r["iso_week"].between(19,23)&r["dow"].eq(dow))
                vals=list(r.loc[m,"spr"])
            return _safe_mean(vals)

        if spr_mode=="promedio":
            t["spr_obj_dmveh"]=t.apply(avg_last4,axis=1)
        else:
            t["spr_obj_dmveh"]=t.apply(avg_peak,axis=1)
    else:
        t = t.merge(spr_plan_dm, on=["fecha","svc","dm","veh"], how="left")
        t = t.rename(columns={"spr_plan":"spr_obj_dmveh"})
    return t[["fecha","svc","dm","veh","spr_obj_dmveh"]]

# ---------------- Rentals (unidades por vehículo) ----------------
@st.cache_data(ttl=300)
def load_rentals_by_vehicle() -> pd.DataFrame:
    df = _read_sheet("Rentals")
    if df.empty:
        return pd.DataFrame(columns=["svc","veh","units"])
    svc_col = next((c for c in df.columns if "svc" in c.lower().replace(" ","")), None)
    qty_col = next((c for c in df.columns if "unidades" in c.lower()), None)
    veh_col = next((c for c in df.columns if "tipo de veh" in c.lower()), None)
    if not svc_col or not qty_col:
        raise ValueError("Rentals: se esperan columnas SVC/SVCS y 'Unidades ...'")
    df["svc"] = df[svc_col].astype(str).str.strip().str.upper()
    df["veh"] = (df[veh_col].astype(str).str.strip().str.upper() if veh_col else "GEN")
    df["units"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0).astype(int)
    out = df.groupby(["svc","veh"], as_index=False)["units"].sum()
    return out  # svc, veh, units

# ---------------- Crowd (cap base/E1 por día) ----------------
@st.cache_data(ttl=300)
def load_crowd_caps() -> pd.DataFrame:
    # robust loader (agrupado, detallado, compacto)
    df_raw = _read_sheet("Crowd")
    if df_raw.empty:
        return pd.DataFrame(columns=["svc","base_wd","base_sa","base_su","e1_wd","e1_sa","e1_su"])
    df = df_raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # detect svc
    svc_col = None
    for c in df.columns:
        if "svc" in c.replace(" ",""):
            svc_col = c; break
    if svc_col is None:
        raise ValueError("Crowd: falta columna 'SVC'.")

    df["svc"] = df[svc_col].astype(str).str.strip().str.upper()

    cols = list(df.columns)
    # Agrupado (tu caso): base, col_*, col_* y e1, col_*, col_*
    if ("base" in cols) and ("e1" in cols):
        ib, ie = cols.index("base"), cols.index("e1")
        base_group = [c for c in cols[ib:ib+3] if c in df.columns]
        e1_group   = [c for c in cols[ie:ie+3] if c in df.columns]
        if len(base_group)==3 and len(e1_group)==3:
            out = df[["svc"] + base_group + e1_group].copy()
            out.columns = ["svc","base_wd","base_sa","base_su","e1_wd","e1_sa","e1_su"]
            for c in ["base_wd","base_sa","base_su","e1_wd","e1_sa","e1_su"]:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
            st.caption(f"Crowd (agrupado): base={base_group}, e1={e1_group}.")
            return out

    # fallback: vacío
    raise ValueError(f"Crowd: no se reconoció el layout. Encabezados: {list(df_raw.columns)}")

def map_crowd_capacity_by_date(target_days: pd.DataFrame, crowd_caps: pd.DataFrame) -> pd.DataFrame:
    def cap_for(row):
        s, d = row["svc"], row["fecha"]
        dow = _weekday(d)
        r = crowd_caps.loc[crowd_caps["svc"]==s]
        if r.empty:
            return pd.Series({"crowd_base_routes":0,"crowd_e1_routes":0})
        r = r.iloc[0]
        if   dow<=4: base,e1 = r["base_wd"], r["e1_wd"]
        elif dow==5: base,e1 = r["base_sa"], r["e1_sa"]
        else:        base,e1 = r["base_su"], r["e1_su"]
        return pd.Series({"crowd_base_routes":int(base), "crowd_e1_routes":int(e1)})
    tmp = target_days.apply(cap_for, axis=1)
    return pd.concat([target_days.reset_index(drop=True), tmp], axis=1)

# ---------------- SRM (cap MLP SDD/SPOT por SVC) ----------------
@st.cache_data(ttl=300)
def load_srm_caps() -> pd.DataFrame:
    df = _read_sheet("SRM")
    if df.empty:
        return pd.DataFrame(columns=["svc","sdd_routes_max","spot_routes_max"])
    # Detectar SVC y sumar columnas que contengan "TOTAL SDD" y "TOTAL SPOT"
    svc_col = next((c for c in df.columns if c.strip().lower() in ("svc","svcs","svc ")), None)
    if not svc_col: 
        raise ValueError("SRM: no se encontró columna SVC.")
    sdd_cols  = [c for c in df.columns if ("sdd"  in c.lower()) and ("total" in c.lower())]
    spot_cols = [c for c in df.columns if ("spot" in c.lower()) and ("total" in c.lower())]
    if not sdd_cols or not spot_cols:
        # si tus totales están en un rango fijo (encabezado a partir de fila 5), puedes mapearlos aquí
        pass
    out = df[[svc_col] + sdd_cols + spot_cols].copy()
    out = out.rename(columns={svc_col:"svc"})
    for c in sdd_cols+spot_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    out["sdd_routes_max"]  = out[sdd_cols].sum(axis=1)
    out["spot_routes_max"] = out[spot_cols].sum(axis=1)
    out = out.groupby("svc", as_index=False)[["sdd_routes_max","spot_routes_max"]].sum()
    out["svc"] = out["svc"].astype(str).str.strip().str.upper()
    return out

def schedule_mlp_rest(df_day: pd.DataFrame) -> pd.DataFrame:
    out = df_day.copy()
    out["week_key"] = out["fecha"].apply(lambda d: f"{pd.Timestamp(d).isocalendar().year}-{pd.Timestamp(d).isocalendar().week:02d}")
    out["sdd_trabaja"]  = 1
    out["spot_trabaja"] = 1
    def proc(g):
        n = len(g)
        need_days = int((g["ship_rem_after_crowd"]>0).sum())
        work_sdd  = min(6, need_days)  # 6x7
        work_spot = 5 if need_days>=5 else need_days  # 5x7, pero puede ser <5
        rest_sdd  = max(n - work_sdd, 0)
        rest_spot = max(n - work_spot, 0)
        g_sorted = g.sort_values(["ship_rem_after_crowd","fecha"], ascending=[True,True])
        if rest_sdd>0:
            g.loc[g_sorted.head(rest_sdd).index,"sdd_trabaja"]=0
        g_sorted2 = g.sort_values(["ship_rem_after_crowd","fecha"], ascending=[True,True])
        if rest_spot>0:
            g.loc[g_sorted2.head(rest_spot).index,"spot_trabaja"]=0
        return g
    out = out.groupby(["svc","week_key"], group_keys=False).apply(proc)
    return out.drop(columns=["week_key"])

# ---------------- Planificador principal (en SHIPMENTS) ----------------
def compute_plan(spr_mode: str, filter_svcs: list[str] | None = None) -> pd.DataFrame:
    fcst        = load_fcst()
    capacity    = load_capacity()
    share_tbl   = compute_crowd_share(capacity)
    spr_real_dm = load_spr_real_dm()
    spr_plan_dm = load_spr_plan_dm()
    rentals_inv = load_rentals_by_vehicle()
    crowd_caps  = load_crowd_caps()
    srm_caps    = load_srm_caps()

    if filter_svcs:
        fcst = fcst[fcst["svc"].isin(filter_svcs)]

    # --- DC y SP (desde Capacity: tipo=Shipments; heurística por nombre del DM) ---
    ship_dm = (capacity[capacity["tipo"].str.lower().eq("shipments")]
               .pivot_table(index=["fecha","svc"], columns="dm", values="cantidad", aggfunc="sum").fillna(0.0))
    ship_dm.columns = [str(c).upper() for c in ship_dm.columns]
    ship_dm = ship_dm.reset_index()
    def sum_cols(df, patt):
        cols = [c for c in df.columns if isinstance(c,str) and patt in c]
        return df[cols].sum(axis=1) if cols else 0.0
    ship_dm["ship_dc"] = sum_cols(ship_dm, "DC")
    ship_dm["ship_sp"] = sum_cols(ship_dm, "SP ") + (ship_dm["SP"] if "SP" in ship_dm.columns else 0.0)
    ship_dcsp = ship_dm[["fecha","svc","ship_dc","ship_sp"]].copy()

    base = (fcst.merge(ship_dcsp, on=["fecha","svc"], how="left")
                 .fillna({"ship_dc":0.0,"ship_sp":0.0}))
    base["ship_rem"] = (base["shipments"] - base["ship_dc"] - base["ship_sp"]).clip(lower=0)

    # --- Rentals en SHIPMENTS: unidades × SPR(dm='RENT', veh) ---
    # Keys para rentals (usamos todas las combinaciones svc-fecha con inventario/veh)
    if not rentals_inv.empty:
        rentals_keys = (base[["fecha","svc"]]
                        .merge(rentals_inv[["svc","veh"]].drop_duplicates(), on="svc", how="left")
                        .dropna(subset=["veh"]))
        rentals_keys["dm"] = "RENT"
        spr_rentals = spr_scenario_lookup(spr_mode, spr_real_dm, spr_plan_dm, rentals_keys)
        rent = (rentals_keys.merge(spr_rentals, on=["fecha","svc","dm","veh"], how="left")
                          .merge(rentals_inv, on=["svc","veh"], how="left"))
        rent["spr_obj_dmveh"] = pd.to_numeric(rent["spr_obj_dmveh"], errors="coerce")
        # Shipments que pueden servir los rentals ese día = unidades * SPR del vehículo
        rent["ship_rent_veh"] = (rent["units"].fillna(0) * rent["spr_obj_dmveh"].fillna(0)).astype(float)
        rentals_day = rent.groupby(["fecha","svc"], as_index=False)["ship_rent_veh"].sum()
    else:
        rentals_day = pd.DataFrame(columns=["fecha","svc","ship_rent_veh"])

    base = base.merge(rentals_day, on=["fecha","svc"], how="left").fillna({"ship_rent_veh":0.0})
    base["ship_rem_after_rent"] = (base["ship_rem"] - base["ship_rent_veh"]).clip(lower=0)

    # --- Crowd: objetivo por share y cap base/E1 (en rutas) → convertir a shipments con SPR crowd ---
    target_days = base[["fecha","svc"]].drop_duplicates()
    crowd_daily = map_crowd_capacity_by_date(target_days, crowd_caps)
    # SPR crowd (usamos DM='CROWD' y agregamos por svc-fecha si hay múltiples vehículos)
    keys_crowd = crowd_daily.copy(); keys_crowd["dm"]="CROWD"; keys_crowd["veh"]="*"
    # Truco: pedir por DM= 'CROWD' y veh wildcard no existe → haremos fallback global por SVC si NaN
    spr_glob_real = (spr_real_dm.groupby(["fecha","svc"], as_index=False)["spr"].mean()
                     .rename(columns={"spr":"spr_glob_exec"}))
    # share crowd
    share_tbl = share_tbl.rename(columns={"share_crowd_obj":"share_crowd"})
    df = (base.merge(share_tbl[["fecha","svc","share_crowd"]], on=["fecha","svc"], how="left")
              .merge(crowd_daily, on=["fecha","svc"], how="left")
              .merge(spr_glob_real, on=["fecha","svc"], how="left")
         ).fillna({"share_crowd":0.0,
                   "crowd_base_routes":0,"crowd_e1_routes":0})

    # SPR crowd (preferente por DM/CROWD si existe plan; si no, usa spr_glob_exec o el plan global)
    # Para simplificar, usa el SPR global por SVC-fecha (promedio ejecutado); si tienes plan por CROWD, puedes integrarlo igual que rentals.
    df["spr_crowd"] = df["spr_glob_exec"]

    # Crowd objetivo en shipments
    df["ship_crowd_target"] = (df["ship_rem_after_rent"] * df["share_crowd"]).clip(lower=0)

    # Capacidad crowd por día (rutas) → shipments
    df["ship_crowd_base_cap"] = df["crowd_base_routes"].astype(float) * df["spr_crowd"].fillna(0)
    df["ship_crowd_e1_cap"]   = df["crowd_e1_routes"].astype(float)   * df["spr_crowd"].fillna(0)

    df["ship_crowd_base_alloc"] = np.minimum(df["ship_crowd_target"], df["ship_crowd_base_cap"])
    rem_after_crowd_base = (df["ship_rem_after_rent"] - df["ship_crowd_base_alloc"]).clip(lower=0)
    add_needed = (df["ship_crowd_target"] - df["ship_crowd_base_alloc"]).clip(lower=0)
    df["ship_crowd_e1_alloc"]   = np.minimum(add_needed, df["ship_crowd_e1_cap"])
    df["ship_crowd_alloc"]      = df["ship_crowd_base_alloc"] + df["ship_crowd_e1_alloc"]

    df["ship_rem_after_crowd"]  = (df["ship_rem_after_rent"] - df["ship_crowd_alloc"]).clip(lower=0)

    # --- MLP SDD (6x7) y Spot (5x7) en SHIPMENTS: cap rutas × SPR MLP ---
    df = df.merge(srm_caps, on="svc", how="left").fillna({"sdd_routes_max":0,"spot_routes_max":0})

    # SPR para MLP: si tienes por DM/vehículo, úsalo; si no, cae al SPR global por SVC-fecha
    # Keys para SDD y SPOT
    mlp_keys = df[["fecha","svc"]].drop_duplicates()
    mlp_keys_sdd  = mlp_keys.copy(); mlp_keys_sdd["dm"]="SDD";  mlp_keys_sdd["veh"]="*"
    mlp_keys_spot = mlp_keys.copy(); mlp_keys_spot["dm"]="SPOT"; mlp_keys_spot["veh"]="*"
    spr_sdd = spr_scenario_lookup(spr_mode, spr_real_dm, spr_plan_dm, mlp_keys_sdd).rename(columns={"spr_obj_dmveh":"spr_sdd"})
    spr_spot= spr_scenario_lookup(spr_mode, spr_real_dm, spr_plan_dm, mlp_keys_spot).rename(columns={"spr_obj_dmveh":"spr_spot"})
    df = (df.merge(spr_sdd, on=["fecha","svc"], how="left")
            .merge(spr_spot, on=["fecha","svc"], how="left"))
    # fallbacks
    df["spr_sdd"]  = df["spr_sdd"].fillna(df["spr_glob_exec"])
    df["spr_spot"] = df["spr_spot"].fillna(df["spr_glob_exec"])

    # Programar descansos por semana con base en remanente
    sched = schedule_mlp_rest(df[["fecha","svc","ship_rem_after_crowd"]])
    df = df.merge(sched[["fecha","svc","sdd_trabaja","spot_trabaja"]], on=["fecha","svc"], how="left").fillna(0)

    df["routes_sdd_day"]  = (df["sdd_routes_max"]  * df["sdd_trabaja"]).astype(float)
    df["routes_spot_day"] = (df["spot_routes_max"] * df["spot_trabaja"]).astype(float)

    df["ship_sdd_cap"]  = df["routes_sdd_day"]  * df["spr_sdd"].fillna(0)
    df["ship_spot_cap"] = df["routes_spot_day"] * df["spr_spot"].fillna(0)

    df["ship_sdd_alloc"]  = np.minimum(df["ship_rem_after_crowd"], df["ship_sdd_cap"])
    rem_after_sdd = (df["ship_rem_after_crowd"] - df["ship_sdd_alloc"]).clip(lower=0)
    df["ship_spot_alloc"] = np.minimum(rem_after_sdd, df["ship_spot_cap"])
    df["ship_rem_after_mlp"] = (rem_after_sdd - df["ship_spot_alloc"]).clip(lower=0)

    # --- Crowd extra si aún falta (sin sobrepasar nube E1 total) ---
    # Capacidad extra disponible = E1_cap - lo usado en E1
    df["ship_crowd_e1_left"] = (df["ship_crowd_e1_cap"] - df["ship_crowd_e1_alloc"]).clip(lower=0)
    df["ship_crowd_extra"]   = np.minimum(df["ship_rem_after_mlp"], df["ship_crowd_e1_left"])
    df["ship_final_deficit"] = (df["ship_rem_after_mlp"] - df["ship_crowd_extra"]).clip(lower=0)

    # --- Resumen columnas útiles ---
    cols = [
        "fecha","svc","shipments","ship_dc","ship_sp",
        "ship_rent_veh",
        "ship_crowd_base_alloc","ship_crowd_e1_alloc","ship_crowd_extra",
        "ship_sdd_alloc","ship_spot_alloc",
        "ship_final_deficit",
        # referencias y caps
        "share_crowd","crowd_base_routes","crowd_e1_routes",
        "sdd_routes_max","spot_routes_max","sdd_trabaja","spot_trabaja",
        "spr_crowd","spr_sdd","spr_spot"
    ]
    out = df[cols].sort_values(["fecha","svc"])

    # flags
    out["risk_flag"] = out["ship_final_deficit"] > 0
    return out

# ---------------- UI ----------------
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

# Filtro SVC
try:
    fcst_all = load_fcst()
    svc_list = sorted(fcst_all["svc"].unique().tolist())
except Exception:
    svc_list = []

sel_svcs = st.multiselect("Filtrar SVC", options=svc_list, default=svc_list[:4])

try:
    plan = compute_plan(spr_mode, filter_svcs=sel_svcs if sel_svcs else None)

    st.subheader("Tabla principal — (svc, fecha) × Delivery model")
    # Mostrar con nombres más claros
    show = plan.rename(columns={
        "shipments":"FCST",
        "ship_dc":"DC",
        "ship_sp":"SP",
        "ship_rent_veh":"Rentals (ship)",
        "ship_crowd_base_alloc":"Crowd base (ship)",
        "ship_crowd_e1_alloc":"Crowd E1 (ship)",
        "ship_crowd_extra":"Crowd extra (ship)",
        "ship_sdd_alloc":"MLP SDD (ship)",
        "ship_spot_alloc":"MLP Spot (ship)",
        "ship_final_deficit":"Déficit (ship)"
    })
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.subheader("Riesgos por fecha")
    resumen = (plan.groupby("fecha")
               .agg(
                   svcs_con_deficit=("risk_flag","sum"),
                   ship_deficit=("ship_final_deficit","sum"),
               ).reset_index())
    st.dataframe(resumen, use_container_width=True, hide_index=True)

    with st.expander("Datos verificados (DC, SP, Share crowd, caps y SPR usados)"):
        ver = plan[[
            "fecha","svc","share_crowd","crowd_base_routes","crowd_e1_routes",
            "sdd_routes_max","spot_routes_max","sdd_trabaja","spot_trabaja",
            "spr_crowd","spr_sdd","spr_spot"
        ]].sort_values(["fecha","svc"])
        st.dataframe(ver, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Error: {e}")




