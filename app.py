# =========================
# Mel-IA — Plan táctico APP
# =========================
import os, json, yaml
import pandas as pd
import numpy as np
import streamlit as st
from math import ceil
from datetime import timedelta, datetime, date

# --- Secrets (Streamlit Cloud) ---
# Acepta GOOGLE_SERVICE_ACCOUNT_JSON como string o bloque [gcp_service_account]
if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
elif "gcp_service_account" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps(dict(st.secrets["gcp_service_account"]))
if "PROJECT_KEY" in st.secrets:
    os.environ["PROJECT_KEY"] = st.secrets["PROJECT_KEY"]

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

# --- Limpieza numérica/entera seguras ---
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip(),
        errors="coerce"
    )

def _to_float0(s: pd.Series) -> pd.Series:
    x = _to_num(s)
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

def _to_int0(s: pd.Series) -> pd.Series:
    x = _to_num(s)
    return x.replace([np.inf, -np.inf], np.nan).fillna(0).round().astype(int)

def _to_intsafe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).round().astype(int)

# ------------- READ TABS -------------
@st.cache_data(ttl=300)
def load_fcst() -> pd.DataFrame:
    df = _read_sheet("FCST")
    need = {"svc","fecha","shipments"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"FCST: faltan columnas {miss}")
    df["shipments"] = _to_float0(df["shipments"])
    df["svc"] = df["svc"].astype(str).str.strip().str.upper()
    return df[["svc","fecha","shipments"]]

@st.cache_data(ttl=300)
def load_spr_real() -> pd.DataFrame:
    df = _read_sheet("SPR")
    need = {"fecha","svc","spr"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"SPR: faltan columnas {miss}")

    df["spr"] = _to_float0(df["spr"])
    df = df.dropna(subset=["spr"])
    day = (df.groupby(["fecha","svc"], as_index=False)["spr"]
             .mean()
             .rename(columns={"spr":"spr_exec"}))

    day["dow"] = day["fecha"].apply(_weekday)
    iso = day["fecha"].apply(lambda d: pd.Timestamp(d).isocalendar())
    day["iso_year"] = [int(x.year) for x in iso]
    day["iso_week"] = [int(x.week) for x in iso]
    day["svc"] = day["svc"].astype(str).str.strip().str.upper()
    return day

@st.cache_data(ttl=300)
def load_capacity() -> pd.DataFrame:
    df = _read_sheet("Capacity")
    need = {"delivery model","tipo","svc","fecha","cantidad"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Capacity: faltan columnas {miss}")
    df["cantidad"] = _to_float0(df["cantidad"])
    df["delivery model"] = df["delivery model"].astype(str).str.strip().str.lower()
    df["tipo"] = df["tipo"].astype(str).str.strip().str.lower()
    df["svc"] = df["svc"].astype(str).str.strip().str.upper()
    return df

@st.cache_data(ttl=300)
def load_srm() -> pd.DataFrame:
    # Lectura cruda para detectar header real (por si hay filas superiores con totales)
    gc = _client()
    sh = gc.open_by_key(SHEET_ID)
    ws = sh.worksheet("SRM")
    values = ws.get_all_values()
    if not values:
        raise ValueError("SRM: hoja vacía.")

    # Detectar la fila de encabezado
    header_idx = None
    best_score, best_i = -1, None
    for i, row in enumerate(values[:100]):
        row_lower = [c.strip().lower() for c in row]
        nonempty = sum(1 for c in row_lower if c)
        alphas   = sum(1 for c in row_lower if any(ch.isalpha() for ch in c))
        has_svc  = any("svc"  in c for c in row_lower)
        has_sdd  = any("sdd"  in c for c in row_lower)
        has_spot = any("spot" in c for c in row_lower)
        has_tot  = any("total" in c for c in row_lower)
        score = (2 if has_svc else 0) + (1 if has_sdd else 0) + (1 if has_spot else 0) \
                + (1 if has_tot else 0) + (1 if nonempty >= 4 else 0) + (1 if alphas >= 3 else 0)
        if score > best_score:
            best_score, best_i = score, i
        if has_svc and (has_sdd or has_spot) and nonempty >= 4:
            header_idx = i
            break
    if header_idx is None:
        header_idx = best_i if best_i is not None else 0

    # Saneamos headers
    headers_raw = values[header_idx]
    seen, headers = {}, []
    for j, h in enumerate(headers_raw):
        base = (h or "").replace("\n", " ").strip()
        if not base:
            base = f"col_{j+1}"
        name = base
        k = 1
        while name in seen:
            k += 1
            name = f"{base}_{k}"
        seen[name] = True
        headers.append(name)

    rows = values[header_idx + 1 :]
    df = pd.DataFrame(rows, columns=headers)
    df.columns = [c.strip().lower() for c in df.columns]

    # Detectar columna SVC
    svc_cols = [c for c in df.columns if "svc" in c.replace(" ", "")]
    if not svc_cols:
        import re
        pat = re.compile(r"^[A-Za-z]{2,4}\d{1,2}$")
        best, best_c = -1, None
        for c in df.columns:
            hits = (df[c].astype(str).str.strip().str.upper().str.match(pat, na=False)).sum()
            if hits > best:
                best, best_c = hits, c
        if best >= 3:
            svc_cols = [best_c]
    if not svc_cols:
        raise ValueError(f"SRM: no se encontró columna SVC. Encabezados detectados: {list(df.columns)}")
    svc_col = svc_cols[0]

    # Columnas SDD / SPOT
    sdd_cols  = [c for c in df.columns if ("sdd"  in c)]
    spot_cols = [c for c in df.columns if ("spot" in c)]
    sdd_total  = [c for c in sdd_cols  if "total" in c] or sdd_cols
    spot_total = [c for c in spot_cols if "total" in c] or spot_cols
    if not sdd_total and not spot_total:
        raise ValueError(f"SRM: no se hallaron columnas con 'sdd' o 'spot'. Encabezados: {list(df.columns)}")

    out = df[[svc_col] + list(set(sdd_total + spot_total))].copy()
    out = out.rename(columns={svc_col: "svc"})

    for c in out.columns:
        if c != "svc":
            out[c] = _to_float0(out[c])

    out["sdd_routes_max"]  = out[[c for c in out.columns if c != "svc" and "sdd"  in c]].sum(axis=1)
    out["spot_routes_max"] = out[[c for c in out.columns if c != "svc" and "spot" in c]].sum(axis=1)

    out = (out.groupby("svc", as_index=False)[["sdd_routes_max","spot_routes_max"]].sum())
    out["svc"] = out["svc"].astype(str).str.strip().str.upper()
    st.caption(f"SRM: header en fila {header_idx+1}. SVC='{svc_col}'. SDD cols={len(sdd_total)} · SPOT cols={len(spot_total)}")
    return out

@st.cache_data(ttl=300)
def load_rentals() -> pd.DataFrame:
    df = _read_sheet("Rentals")
    if df.empty:
        raise ValueError("Rentals: hoja vacía.")

    svc_candidates = [c for c in df.columns if "svc" in c.replace(" ", "")]
    if not svc_candidates:
        raise ValueError(f"Rentals: no se encontró columna SVC. Encabezados: {list(df.columns)}")
    svc_col = svc_candidates[0]

    qty_prefer = [
        "unidades disponibles", "unidades", "cantidad", "capacidad",
        "units", "available", "disp", "available units"
    ]
    qty_col = None
    for name in qty_prefer:
        for c in df.columns:
            if name in c:
                qty_col = c; break
        if qty_col: break
    if qty_col is None:
        counts = []
        for c in df.columns:
            if c == svc_col: continue
            nums = _to_num(df[c])
            counts.append((nums.notna().sum(), c))
        counts.sort(reverse=True)
        if counts and counts[0][0] > 0:
            qty_col = counts[0][1]
    if qty_col is None:
        raise ValueError(f"Rentals: no se encontró columna de cantidad. Encabezados: {list(df.columns)}")

    df[qty_col] = _to_int0(df[qty_col])
    out = (df.groupby(svc_col, as_index=False)[qty_col].sum()
             .rename(columns={svc_col: "svc", qty_col: "rentals_routes_max"}))
    out["svc"] = out["svc"].astype(str).str.strip().str.upper()
    st.caption(f"Rentals: usando SVC='{svc_col}' · cantidad='{qty_col}'")
    return out

@st.cache_data(ttl=300)
def load_crowd_caps() -> pd.DataFrame:
    df = _read_sheet("Crowd")
    if df.empty:
        raise ValueError("Crowd: hoja vacía.")

    # SVC por nombre/patrón
    cand_svc = [c for c in df.columns if any(k in c.replace(" ", "")
                 for k in ["svc","svcs","facility","facilidad","centro","centrooperativo","estacion","station","lc","logisticcenter"])]
    if not cand_svc:
        import re
        pat = re.compile(r"^[A-Z]{3,4}\d{1,2}$")
        best_hits, best_col = -1, None
        for c in df.columns:
            vals = df[c].astype(str).str.strip().str.upper()
            hits = (vals.str.match(pat, na=False)).sum()
            if hits > best_hits:
                best_hits, best_col = hits, c
        if best_hits >= 3:
            cand_svc = [best_col]
    if not cand_svc:
        raise ValueError(f"Crowd: falta columna SVC. Encabezados detectados: {list(df.columns)}")
    svc_col = cand_svc[0]

    # Intentar formato detallado (6 columnas)
    cols = list(df.columns)
    def _pick(tags_main, tags_day):
        for c in cols:
            cc = c.lower()
            if any(t in cc for t in tags_main) and any(t in cc for t in tags_day):
                return c
        return None

    base_tags = ["base","normal"]
    e1_tags   = ["e1","holgura","escala","escalada","alto","high"]
    wd_tags   = ["entre", "sem", "weekday", "wd", "laboral"]
    sa_tags   = ["sab", "sábado", "sat"]
    su_tags   = ["dom", "domingo", "sun"]

    c_base_wd = _pick(base_tags, wd_tags)
    c_base_sa = _pick(base_tags, sa_tags)
    c_base_su = _pick(base_tags, su_tags)
    c_e1_wd   = _pick(e1_tags,   wd_tags)
    c_e1_sa   = _pick(e1_tags,   sa_tags)
    c_e1_su   = _pick(e1_tags,   su_tags)

    if all([c_base_wd, c_base_sa, c_base_su, c_e1_wd, c_e1_sa, c_e1_su]):
        for c in [c_base_wd, c_base_sa, c_base_su, c_e1_wd, c_e1_sa, c_e1_su]:
            df[c] = _to_int0(df[c])
        out = df[[svc_col, c_base_wd, c_base_sa, c_base_su, c_e1_wd, c_e1_sa, c_e1_su]].copy()
        out = out.rename(columns={
            svc_col: "svc",
            c_base_wd: "base_wd", c_base_sa: "base_sa", c_base_su: "base_su",
            c_e1_wd:   "e1_wd",   c_e1_sa:   "e1_sa",   c_e1_su:   "e1_su",
        })
        out["svc"] = out["svc"].astype(str).str.strip().str.upper()
        st.caption(
            f"Crowd (detallado): SVC='{svc_col}'. "
            f"Base→ wd='{c_base_wd}', sa='{c_base_sa}', dom='{c_base_su}'. "
            f"E1→ wd='{c_e1_wd}', sa='{c_e1_sa}', dom='{c_e1_su}'."
        )
        return out

    # Formato compacto (base/e1 únicos)
    def _find_single(tag_list):
        for c in cols:
            cc = c.lower()
            if any(t in cc for t in tag_list) and not any(d in cc for d in (wd_tags + sa_tags + su_tags)):
                return c
        return None
    c_base = _find_single(base_tags)
    c_e1   = _find_single(e1_tags)
    if c_base is None:
        raise ValueError(
            "Crowd: no se encontraron columnas esperadas. Para formato compacto, crea columnas 'base' y (opcional) 'e1'. "
            f"Encabezados actuales: {list(df.columns)}"
        )
    df[c_base] = _to_int0(df[c_base])
    if c_e1 and c_e1 in df.columns:
        df[c_e1] = _to_int0(df[c_e1])
    else:
        df["__e1_tmp__"] = 0; c_e1 = "__e1_tmp__"

    out = df[[svc_col, c_base, c_e1]].copy().rename(columns={svc_col:"svc", c_base:"base", c_e1:"e1"})
    out["base_wd"] = out["base"]; out["base_sa"] = out["base"]; out["base_su"] = out["base"]
    out["e1_wd"]   = out["e1"];   out["e1_sa"]   = out["e1"];   out["e1_su"]   = out["e1"]
    out = out[["svc","base_wd","base_sa","base_su","e1_wd","e1_sa","e1_su"]]
    out["svc"] = out["svc"].astype(str).str.strip().str.upper()
    st.caption(f"Crowd (compacto): SVC='{svc_col}'. base='{c_base}' → wd/sa/su; e1='{c_e1}' → wd/sa/su.")
    return out

# ------------- SPR SCENARIOS -------------
def compute_spr_scenarios(fcst: pd.DataFrame, spr_real: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    target = fcst[["fecha","svc"]].drop_duplicates().copy()
    target["dow"] = target["fecha"].apply(_weekday)
    target["iso_year"] = target["fecha"].apply(lambda d: int(pd.Timestamp(d).isocalendar().year))

    spr_exec_map = spr_real.set_index(["fecha","svc"])["spr_exec"]

    def avg_last4(row):
        d, s = row["fecha"], row["svc"]
        vals = []
        for k in [7,14,21,28]:
            dk = d - timedelta(days=k)
            v = spr_exec_map.get((dk,s), np.nan)
            if pd.notna(v): vals.append(float(v))
        if not vals:
            mask = (spr_real["svc"]==s) & (spr_real["fecha"].between(d - timedelta(days=28), d - timedelta(days=1)))
            vals = list(spr_real.loc[mask,"spr_exec"])
        return _safe_mean(vals)
    target["spr_promedio"] = target.apply(avg_last4, axis=1)

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

    cap = capacity.copy()
    m_spr = cap["tipo"].str.strip().str.lower().eq("spr")
    spr_plan = cap.loc[m_spr, ["svc","fecha","cantidad"]].rename(columns={"cantidad":"spr_plan"})
    if spr_plan.empty:
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
            return pd.Series({"crowd_base_routes": 0.0, "crowd_e1_routes": 0.0})
        r = r.iloc[0]
        if dow <= 4:
            base, e1 = r.get("base_wd", 0), r.get("e1_wd", 0)
        elif dow == 5:
            base, e1 = r.get("base_sa", 0), r.get("e1_sa", 0)
        else:
            base, e1 = r.get("base_su", 0), r.get("e1_su", 0)

        # saneo numérico -> 0 si NaN/inf
        base = pd.to_numeric(base, errors="coerce")
        e1   = pd.to_numeric(e1,   errors="coerce")
        base = 0.0 if not np.isfinite(base) else float(base)
        e1   = 0.0 if not np.isfinite(e1)   else float(e1)

        return pd.Series({"crowd_base_routes": base, "crowd_e1_routes": e1})

    tmp = target_days.apply(cap_for, axis=1)
    out = pd.concat([target_days.reset_index(drop=True), tmp], axis=1)

    # cast final seguro (por si algo quedó float)
    out["crowd_base_routes"] = _to_intsafe(out["crowd_base_routes"])
    out["crowd_e1_routes"]   = _to_intsafe(out["crowd_e1_routes"])
    return out

    return pd.concat([target_days.reset_index(drop=True), tmp], axis=1)

# ------------- SCHEDULER MLP DESCANSOS -------------
def schedule_mlp_rest(df_day: pd.DataFrame) -> pd.DataFrame:
    out = df_day.copy()
    out["week_key"] = out["fecha"].apply(lambda d: f"{_iso_yr_week(d)[0]}-{_iso_yr_week(d)[1]:02d}")
    out["sdd_trabaja"]  = 1
    out["spot_trabaja"] = 1

    def proc(g):
        n = len(g)
        need_days = int((g["routes_mlp_need"]>0).sum())
        work_sdd = min(6, need_days)
        rest_sdd = max(n - work_sdd, 0)
        work_spot = 5
        if need_days >= 6: work_spot = 6
        elif need_days < 5: work_spot = need_days
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

# ------------- MOTOR PRINCIPAL -------------

def compute_plan(spr_mode: str, sel_svcs=None):

    with st.status("Cargando datos...", expanded=True) as status:
        st.write("1/6 FCST…");       fcst       = load_fcst()
        st.write("2/6 SPR (real)…"); spr_real   = load_spr_real()
        st.write("3/6 Capacity…");   capacity   = load_capacity()
        st.write("4/6 SRM…");        srm        = load_srm()
        st.write("5/6 Rentals…");    rentals    = load_rentals()
        st.write("6/6 Crowd…");      crowd_caps = load_crowd_caps()
        status.update(label="Datos listos ✅", state="complete")

        # ---- FILTRO DE SVC (aplicar temprano para que todo el cálculo use solo esos) ----
    if sel_svcs:
        sel_svcs = set([str(s).strip().upper() for s in sel_svcs])
        fcst       = fcst[fcst["svc"].isin(sel_svcs)]
        spr_real   = spr_real[spr_real["svc"].isin(sel_svcs)]
        capacity   = capacity[capacity["svc"].isin(sel_svcs)]
        srm        = srm[srm["svc"].isin(sel_svcs)]
        rentals    = rentals[rentals["svc"].isin(sel_svcs)]
        crowd_caps = crowd_caps[crowd_caps["svc"].isin(sel_svcs)]


    # SPRs
    spr_tbl = compute_spr_scenarios(fcst, spr_real, capacity)
    spr_col = {"promedio":"spr_promedio","peak":"spr_peak","plan":"spr_plan"}[spr_mode]

    # Share crowd
    share_tbl   = compute_crowd_share(capacity)
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

    # Limpieza/nulos (mantén todo en float por ahora)
    for c in ["share_crowd_obj","crowd_base_routes","crowd_e1_routes",
              "sdd_routes_max","spot_routes_max","rentals_routes_max"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0)

    df["spr_objetivo"] = pd.to_numeric(df[spr_col], errors="coerce")
    df.drop(columns=[spr_col], inplace=True)

    # Demanda remanente
    df["q_rem"] = df["shipments"].clip(lower=0)

    # Rutas requeridas (deja en float; casteamos al final)
    df["routes_need_total"] = np.where(
        (df["q_rem"]>0) & (df["spr_objetivo"]>0),
        np.ceil(df["q_rem"]/df["spr_objetivo"]),
        0.0
    )
    df["alerta_spr_missing"] = ((df["q_rem"]>0) & (df["spr_objetivo"].isna() | (df["spr_objetivo"]<=0)))

    # Crowd
    df["routes_crowd_target"] = np.ceil(df["routes_need_total"] * df["share_crowd_obj"])
    df["routes_crowd_base"]   = np.minimum(df["routes_crowd_target"], df["crowd_base_routes"])
    df["routes_crowd_e1"]     = np.minimum(
        (df["routes_crowd_target"] - df["routes_crowd_base"]).clip(lower=0),
        df["crowd_e1_routes"]
    )
    df["routes_crowd_alloc"]  = df["routes_crowd_base"] + df["routes_crowd_e1"]
    df["alerta_crowd_high"]   = df["routes_crowd_e1"] > 0

    # Rentals y MLP (todo en float)
    df["routes_after_crowd"]    = (df["routes_need_total"] - df["routes_crowd_alloc"]).clip(lower=0)
    df["routes_rentals_alloc"]  = np.minimum(df["routes_after_crowd"], df["rentals_routes_max"])
    df["routes_mlp_need"]       = (df["routes_after_crowd"] - df["routes_rentals_alloc"]).clip(lower=0)

    # Descansos MLP por semana y SVC
    rest_base   = df[["fecha","svc","routes_mlp_need","sdd_routes_max","spot_routes_max"]].copy()
    rest_sched  = schedule_mlp_rest(rest_base)
    df = df.merge(rest_sched[["fecha","svc","sdd_trabaja","spot_trabaja"]], on=["fecha","svc"], how="left")

    # Capacidad MLP diaria (sin casts)
    df["routes_mlp_cap_day"] = (df["sdd_routes_max"]*df["sdd_trabaja"] + df["spot_routes_max"]*df["spot_trabaja"]).fillna(0.0)
    df["routes_mlp_alloc"]   = np.minimum(df["routes_mlp_need"], df["routes_mlp_cap_day"])

    # Déficit + shipments logrados
    df["routes_deficit"]      = (df["routes_mlp_need"] - df["routes_mlp_alloc"]).clip(lower=0)
    df["routes_total_alloc"]  = (df["routes_crowd_alloc"] + df["routes_rentals_alloc"] + df["routes_mlp_alloc"])
    df["shipments_plan"]      = np.where(df["spr_objetivo"]>0, df["routes_total_alloc"] * df["spr_objetivo"], 0.0)
    df["alerta_deficit"]      = df["shipments_plan"] + 1e-6 < df["shipments"]

    # Métricas
    df["spr_logrado"] = np.where(df["routes_total_alloc"]>0, df["q_rem"] / df["routes_total_alloc"], np.nan)
    df["share_crowd_real"] = np.where(df["routes_need_total"]>0, df["routes_crowd_alloc"] / df["routes_need_total"], 0.0)
    df["risk_flag"] = df["alerta_deficit"] | df["alerta_spr_missing"]

    # --- SANIDAD FINAL: convertir a enteros de forma SEGURA ---
    int_cols = [
        "routes_need_total","routes_crowd_target","routes_crowd_base","routes_crowd_e1",
        "routes_crowd_alloc","routes_after_crowd","routes_rentals_alloc","routes_mlp_need",
        "sdd_trabaja","spot_trabaja","routes_mlp_cap_day","routes_mlp_alloc",
        "routes_deficit","routes_total_alloc"
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = _to_intsafe(df[c])

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

    # ⬇⬇⬇ NUEVO: selector de SVC
    svcs_all = sorted(load_fcst()["svc"].unique().tolist())
    default_svcs = ["SPB1", "SMX9", "SGD1", "SMT1"]
    sel_svcs = st.multiselect(
        "Filtrar SVC",
        options=svcs_all,
        default=[s for s in default_svcs if s in svcs_all],
        help="El cálculo y las tablas solo incluirán estos SVC."
    )



st.title("Mel-IA — Plan táctico (diario por SVC)")
spr_mode = st.radio("SPR objetivo", ["promedio","peak","plan"], index=0, horizontal=True)

try:
    plan = compute_plan(spr_mode, sel_svcs)
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




