# ====== app.py  (Mel-IA Ops — Copiloto Flota / Jarvis) ======

import os, re, json, glob
import numpy as np
import pandas as pd
import streamlit as st
import gspread
from google.oauth2 import service_account
from datetime import datetime

# ========= Config de página =========
st.set_page_config(page_title="Copiloto Flota — Mel-IA Ops (Jarvis)", layout="wide")

# ========= IDs / opciones base =========
SHEET_ID = "1UBjU3-ftGCow3EzTD0NB6UaYwMUYUARbn9QjD7SlxtY"
TABS = dict(EJ="Ejecución", SRM="SRM", FCST="FCST",
            OUT_RES="Plan_14_resumen", OUT_DET="Plan_14_detalle")

RUTAS_SRM_IS_DAILY = True
FACTOR_SRM_SEM_A_DIA = 1/6
CROWD_VT = "Car"

# ========= OpenAI opcional =========
_HAS_OPENAI = False
try:
    from openai import OpenAI
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
        _client = OpenAI()
        _HAS_OPENAI = True
    else:
        _client = None
except Exception:
    _client = None
    _HAS_OPENAI = False

# Determinismo primero; si no entiende, LLM
_PARSER_LOCAL_FIRST = True

# ========= Estado =========
if "params" not in st.session_state:
    st.session_state["params"] = {
        "escalar_si_excede_fcst": True,     # si MLP+Rentals exceden FCST, escalar
        "factor_escalado_mlp": 1.0,         # multiplicadores “rápidos”
        "factor_escalado_rentals": 1.0,
    }

# overrides: valores fijos de rutas por modelo (GLOBAL o por SVC)
if "overrides" not in st.session_state:
    st.session_state["overrides"] = {"mlp": {}, "rentals": {}}

# ========= Google Sheets client =========
@st.cache_resource
def get_client():
    creds_dict = st.secrets.get("gcp_service_account", None)
    if not creds_dict:
        st.stop()  # corta la app con mensaje claro
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"]
    )
    return gspread.authorize(creds)

def _unique_headers(headers):
    out, seen = [], set()
    for i, h in enumerate(headers):
        h = (h or "").strip()
        if not h: h = f"col_{i+1}"
        h = re.sub(r"\s+", " ", h)
        base, k, name = h, 1, h
        while name in seen:
            k += 1; name = f"{base}__{k}"
        out.append(name); seen.add(name)
    return out

def read_ws(sheet, title, expected=None, any_keys=None, header_probe=10):
    ws = sheet.worksheet(title)
    vals = ws.get_all_values()
    if not vals: return pd.DataFrame()
    hdr_i = 0
    if expected:
        for i,row in enumerate(vals[:header_probe]):
            r = [c.strip().lower() for c in row]
            if all(k.lower() in r for k in expected): hdr_i = i; break
        else:
            if any_keys:
                for i,row in enumerate(vals[:header_probe]):
                    r = [c.strip().lower() for c in row]
                    if any(k.lower() in r for k in any_keys): hdr_i = i; break
    headers = _unique_headers(vals[hdr_i])
    df = pd.DataFrame(vals[hdr_i+1:], columns=headers)
    df = df[~(df.astype(str).apply(lambda x: "".join(x).strip(), axis=1)=="")]
    return df

def write_ws(sheet, title, df):
    try:
        ws = sheet.worksheet(title)
        sheet.del_worksheet(ws)
    except Exception:
        pass
    rows = len(df) + 10
    cols = max(10, len(df.columns)+2)
    ws = sheet.add_worksheet(title=title, rows=str(rows), cols=str(cols))
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    ws.update([out.columns.tolist()] + out.astype(object).values.tolist())

def clean_num(s):
    return (pd.Series(s).astype(str)
            .str.replace(r"[^\d\-\.\,]", "", regex=True)
            .str.replace(",", "", regex=False)
            .replace({"": "0"}).astype(float))

def winsor(s, low=0.05, high=0.95, min_n=10):
    s = pd.Series(s).dropna()
    if len(s) < min_n: return s
    ql,qh = s.quantile([low,high])
    return s.clip(ql,qh)

# ========= Carga & normalización =========
@st.cache_data(ttl=300)
def load_all():
    client = get_client()
    sheet  = client.open_by_key(SHEET_ID)

    df_ej  = read_ws(sheet, TABS["EJ"],
                     expected=["DELIVERY_MODEL","HOMOLOGACION_VEHICULO","Shps Dispatched","TOTAL_RUTAS"],
                     any_keys=["SVC","Fecha","Año de DATE","DELIVERY_MODEL"])
    df_srm = read_ws(sheet, TABS["SRM"], expected=["MLP","Region","SVC"], any_keys=["MLP","SVC"])
    df_fc  = read_ws(sheet, TABS["FCST"], expected=["SVC"], any_keys=["SVC"])

    # --- normalización ejecución
    if "DELIVERY_MODEL" not in df_ej.columns:
        for alt in ["DELIVERY_MODEL 1","Delivery Model","DELIVERY MODEL"]:
            if alt in df_ej.columns: df_ej["DELIVERY_MODEL"]=df_ej[alt]; break
    if "SVC" not in df_ej.columns:
        for alt in ["Estación","Estacion"]:
            if alt in df_ej.columns: df_ej["SVC"]=df_ej[alt]; break
    for c in ["Shps Dispatched","TOTAL_RUTAS"]:
        if c in df_ej.columns: df_ej[c]=pd.to_numeric(df_ej[c], errors="coerce")

    _fecha = None
    for cand in ["DATE","Fecha","FECHA","Date"]:
        if cand in df_ej.columns:
            _fecha = pd.to_datetime(df_ej[cand], errors="coerce"); break
    if _fecha is None and {"Año de DATE","Mes de DATE","Día de DATE"}.issubset(df_ej.columns):
        mes_map = {"enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
                   "julio":7,"agosto":8,"septiembre":9,"octubre":10,"noviembre":11,"diciembre":12}
        month_series = df_ej["Mes de DATE"].map(mes_map).fillna(pd.to_numeric(df_ej["Mes de DATE"], errors="coerce"))
        _fecha = pd.to_datetime(dict(
            year=pd.to_numeric(df_ej["Año de DATE"], errors="coerce"),
            month=pd.to_numeric(month_series, errors="coerce"),
            day=pd.to_numeric(df_ej["Día de DATE"], errors="coerce")
        ), errors="coerce")

    df_ej["_fecha"] = _fecha
    df_ej = df_ej[df_ej["_fecha"].notna()].copy()
    df_ej["_dow"]   = df_ej["_fecha"].dt.weekday

    # --- FCST tidy
    date_cols = [c for c in df_fc.columns if re.match(r"\d{1,2}\-\w{3}", str(c))]
    for c in date_cols: df_fc[c]=clean_num(df_fc[c])
    YEAR = int(df_ej["_fecha"].dt.year.max()) if not df_ej.empty else datetime.now().year

    df_fcst = df_fc.melt(
        id_vars=["SVC"], value_vars=date_cols,
        var_name="Fecha_txt", value_name="Shipments_FCST"
    ).dropna(subset=["SVC"])
    df_fcst["Fecha"] = pd.to_datetime(df_fcst["Fecha_txt"] + f"-{YEAR}", format="%d-%b-%Y", errors="coerce")
    df_fcst = df_fcst[["SVC","Fecha","Shipments_FCST"]]
    fechas = sorted(df_fcst["Fecha"].dropna().unique())
    cal = pd.DataFrame({"Fecha": fechas})
    cal["_dow"] = cal["Fecha"].dt.weekday

    # --- SPR por DOW
    df_ej["SPR_raw"] = df_ej["Shps Dispatched"] / df_ej["TOTAL_RUTAS"]
    df_ej["SPR_w"]   = df_ej["SPR_raw"]
    _w = df_ej.groupby(["DELIVERY_MODEL","HOMOLOGACION_VEHICULO","_dow"])["SPR_raw"].transform(
        lambda s: winsor(s,0.05,0.95,10)
    )
    df_ej.loc[_w.notna(),"SPR_w"] = _w

    spr_dow = (df_ej.groupby(["DELIVERY_MODEL","HOMOLOGACION_VEHICULO","_dow"])["SPR_w"]
               .mean().reset_index().rename(columns={"SPR_w":"SPR_dow"}))
    spr_dow_vt = (spr_dow.groupby(["HOMOLOGACION_VEHICULO","_dow"])["SPR_dow"]
                  .mean().reset_index().rename(columns={"SPR_dow":"SPR_dow_vt"}))
    spr_dow_global = (spr_dow.groupby(["_dow"])["SPR_dow"]
                      .mean().reset_index().rename(columns={"SPR_dow":"SPR_dow_global"}))
    spr_global = float(df_ej["SPR_w"].mean()) if not df_ej.empty else 0.0

    # --- MLP capacity (SRM)
    veh_cols = [c for c in ["Extra Large Van","Large Van","Small Van","Car"] if c in df_srm.columns]
    df_srm_long = df_srm.melt(id_vars=["MLP","Region","SVC"], value_vars=veh_cols,
                              var_name="HOMOLOGACION_VEHICULO", value_name="Rutas_MLP").fillna(0)
    df_srm_long["Rutas_MLP"] = pd.to_numeric(df_srm_long["Rutas_MLP"], errors="coerce").fillna(0)
    if not RUTAS_SRM_IS_DAILY:
        df_srm_long["Rutas_MLP"] *= FACTOR_SRM_SEM_A_DIA
    mlp_capacity = df_srm_long.groupby(["SVC","HOMOLOGACION_VEHICULO"], as_index=False)\
                              .agg(Rutas_MLP=("Rutas_MLP","sum"))

    # Rentals capacity (promedio)
    is_rent = df_ej["DELIVERY_MODEL"].astype(str).str.upper().str.contains("RENT")
    df_rent = df_ej[is_rent].copy()
    grp = ["SVC","HOMOLOGACION_VEHICULO"] if "SVC" in df_rent.columns else ["HOMOLOGACION_VEHICULO"]
    rent_rutas = df_rent.groupby(grp, dropna=False)["TOTAL_RUTAS"].mean().reset_index()\
                        .rename(columns={"TOTAL_RUTAS":"Rutas_Rentals_avg"})
    if "SVC" not in rent_rutas.columns:
        svcs = mlp_capacity["SVC"].dropna().unique().tolist()
        rent_rutas = rent_rutas.assign(_k=1).merge(pd.DataFrame({"SVC":svcs,"_k":[1]*len(svcs)}), on="_k").drop(columns="_k")
    rentals_capacity = rent_rutas[["SVC","HOMOLOGACION_VEHICULO","Rutas_Rentals_avg"]].copy()

    return (sheet, df_ej, df_srm, df_fcst, cal, spr_dow, spr_dow_vt,
            spr_dow_global, spr_global, mlp_capacity, rentals_capacity)

(sheet, df_exec, df_srm, df_fcst, cal, spr_dow, spr_dow_vt,
 spr_dow_global, spr_global, mlp_capacity, rentals_capacity) = load_all()

# ========= Motor principal (aplica overrides) =========
def run(params: dict):
    scale_if_exceed = bool(params.get("escalar_si_excede_fcst", True))
    f_mlp   = float(params.get("factor_escalado_mlp", 1.0))
    f_rent  = float(params.get("factor_escalado_rentals", 1.0))

    # --- MLP (capacidad * f_mlp) + SPR
    spr_mlp = spr_dow[spr_dow["DELIVERY_MODEL"]=="MLP"][["HOMOLOGACION_VEHICULO","_dow","SPR_dow"]]
    mlp = mlp_capacity.assign(_k=1).merge(cal.assign(_k=1), on="_k").drop(columns="_k")
    mlp = (mlp.merge(spr_mlp, on=["HOMOLOGACION_VEHICULO","_dow"], how="left")
               .merge(spr_dow_vt, on=["HOMOLOGACION_VEHICULO","_dow"], how="left")
               .merge(spr_dow_global, on=["_dow"], how="left"))
    mlp["SPR_final"] = mlp["SPR_dow"].fillna(mlp["SPR_dow_vt"]).fillna(mlp["SPR_dow_global"]).fillna(spr_global)
    mlp["Rutas_MLP_int"] = np.round(mlp["Rutas_MLP"] * f_mlp).clip(lower=0).astype(int)

    # --- Rentals (promedio * f_rent) + SPR
    rent = rentals_capacity.assign(_k=1).merge(cal.assign(_k=1), on="_k").drop(columns="_k")
    rent = (rent.merge(spr_dow_vt, on=["HOMOLOGACION_VEHICULO","_dow"], how="left")
               .merge(spr_dow_global, on=["_dow"], how="left"))
    rent["SPR_final"] = rent["SPR_dow_vt"].fillna(rent["SPR_dow_global"]).fillna(spr_global)
    rent["Rutas_Rentals_int"] = np.round(rent["Rutas_Rentals_avg"] * f_rent).clip(lower=0).astype(int)

    # ===== Overrides (GLOBAL y por SVC) =====
    ov = st.session_state["overrides"]

    # MLP
    if "GLOBAL" in ov["mlp"]:
        mlp["Rutas_MLP_int"] = int(ov["mlp"]["GLOBAL"])
    if ov["mlp"]:
        # por SVC
        for svc_k, rutas_k in ov["mlp"].items():
            if svc_k == "GLOBAL": continue
            mask = mlp["SVC"].astype(str).str.upper().eq(str(svc_k).upper())
            mlp.loc[mask, "Rutas_MLP_int"] = int(rutas_k)
    mlp["Shipments_MLP"] = mlp["Rutas_MLP_int"] * mlp["SPR_final"]

    # Rentals
    if "GLOBAL" in ov["rentals"]:
        rent["Rutas_Rentals_int"] = int(ov["rentals"]["GLOBAL"])
    if ov["rentals"]:
        for svc_k, rutas_k in ov["rentals"].items():
            if svc_k == "GLOBAL": continue
            mask = rent["SVC"].astype(str).str.upper().eq(str(svc_k).upper())
            rent.loc[mask, "Rutas_Rentals_int"] = int(rutas_k)
    rent["Shipments_Rentals"] = rent["Rutas_Rentals_int"] * rent["SPR_final"]

    # --- Crowd SPR
    crowd = cal.merge(
        spr_dow_vt[spr_dow_vt["HOMOLOGACION_VEHICULO"]==CROWD_VT][["_dow","SPR_dow_vt"]],
        on="_dow", how="left"
    ).merge(spr_dow_global, on="_dow", how="left")
    crowd["SPR_Crowd"] = crowd["SPR_dow_vt"].fillna(crowd["SPR_dow_global"]).fillna(spr_global)

    fcst = df_fcst.copy()

    # --- Detalle base
    det_mlp = mlp.rename(columns={"Rutas_MLP_int":"Rutas","SPR_final":"SPR","Shipments_MLP":"Shipments"}) \
                 .assign(Modelo="MLP")[["SVC","Fecha","HOMOLOGACION_VEHICULO","Modelo","Rutas","SPR","Shipments"]]
    det_rent= rent.rename(columns={"Rutas_Rentals_int":"Rutas","SPR_final":"SPR","Shipments_Rentals":"Shipments"}) \
                 .assign(Modelo="Rentals")[["SVC","Fecha","HOMOLOGACION_VEHICULO","Modelo","Rutas","SPR","Shipments"]]
    det_crowd = fcst[["SVC","Fecha"]].drop_duplicates().merge(crowd[["Fecha","SPR_Crowd"]], on="Fecha", how="left")
    det_crowd["HOMOLOGACION_VEHICULO"] = f"Crowd ({CROWD_VT})"
    det_crowd["Modelo"] = "Crowd"
    det_crowd["Rutas"] = 0
    det_crowd["SPR"] = det_crowd["SPR_Crowd"]
    det_crowd["Shipments"] = 0.0
    det_crowd = det_crowd[["SVC","Fecha","HOMOLOGACION_VEHICULO","Modelo","Rutas","SPR","Shipments"]]

    det = pd.concat([det_mlp, det_rent, det_crowd], ignore_index=True)

    # --- Ajuste vs FCST (si excede escala; si no, Crowd cierra gap)
    def ajustar(gr):
        svc, fecha = gr.name
        fc = float(fcst[(fcst.SVC==svc)&(fcst.Fecha==fecha)]["Shipments_FCST"].sum())
        m = gr["Modelo"].eq("MLP"); r = gr["Modelo"].eq("Rentals"); c = gr["Modelo"].eq("Crowd")

            ov = st.session_state.get("overrides", {"mlp":{},"rentals":{}})
        svc_key = str(svc).upper() if svc is not None else ""
        has_mlp_ov = ("GLOBAL" in ov["mlp"]) or (svc_key in ov["mlp"])
        has_rent_ov = ("GLOBAL" in ov["rentals"]) or (svc_key in ov["rentals"])
        base = gr.loc[m|r,"Shipments"].sum()
        fixed = 0.0
        if has_mlp_ov:
            fixed += gr.loc[m, "Shipments"].sum()
        if has_rent_ov:
            fixed += gr.loc[r, "Shipments"].sum()
        free = base - fixed
        if scale_if_exceed and base > fc and base > 0:
            f = fc/base
            for mask in [m,r]:
                gr.loc[mask,"Rutas"] = np.round(gr.loc[mask,"Rutas"]*f).clip(lower=0).astype(int)
                gr.loc[mask,"Shipments"] = gr.loc[mask,"Rutas"]*gr.loc[mask,"SPR"]
            if fixed >= fc:
                gr.loc[(m & ~has_mlp_ov) | (r & ~has_rent_ov), ["Rutas","Shipments"]] = (0,0.0)
                gr.loc[c,["Rutas","Shipments"]] = (0,0.0)
                return gr
            remain = fc - fixed
            f = remain/free if free>0 else 0.0
            if not has_mlp_ov:
                gr.loc[m,"Rutas"] = np.round(gr.loc[m,"Rutas"]*f).clip(lower=0).astype(int)
                gr.loc[m,"Shipments"] = gr.loc[m,"Rutas"]*gr.loc[m,"SPR"]
            if not has_rent_ov:
                gr.loc[r,"Rutas"] = np.round(gr.loc[r,"Rutas"]*f).clip(lower=0).astype(int)
                gr.loc[r,"Shipments"] = gr.loc[r,"Rutas"]*gr.loc[r,"SPR"]
            gr.loc[c,["Rutas","Shipments"]] = (0,0.0)
            return gr
        base = gr.loc[m|r,"Shipments"].sum()


                
        gap = max(fc-base, 0.0)
        if gap>0:
            spr_c = float(gr.loc[c,"SPR"].iloc[0]) if gr.loc[c,"SPR"].notna().any() else 0.0
            if spr_c>0:
                rr = int(np.round(gap/spr_c))
                gr.loc[c,"Rutas"] = rr
                gr.loc[c,"Shipments"] = rr*spr_c
        return gr

    det = det.groupby(["SVC","Fecha"], group_keys=False).apply(ajustar)

    # --- Resumen
    g = det.groupby(["SVC","Fecha","Modelo"], as_index=False).agg(Rutas=("Rutas","sum"),
                                                                  Shipments=("Shipments","sum"))
    mlp_s = g[g["Modelo"]=="MLP"][["SVC","Fecha","Rutas","Shipments"]].rename(columns={"Rutas":"Rutas_MLP","Shipments":"Shipments_MLP"})
    ren_s = g[g["Modelo"]=="Rentals"][["SVC","Fecha","Rutas","Shipments"]].rename(columns={"Rutas":"Rutas_Rentals","Shipments":"Shipments_Rentals"})
    crw_s = g[g["Modelo"]=="Crowd"][["SVC","Fecha","Rutas","Shipments"]].rename(columns={"Rutas":"Rutas_Crowd","Shipments":"Shipments_Crowd"})
    res = (mlp_s.merge(ren_s,on=["SVC","Fecha"],how="outer")
               .merge(crw_s,on=["SVC","Fecha"],how="outer")).fillna(0)
    res = res.merge(df_fcst, on=["SVC","Fecha"], how="left")
    res["Shipments_Totales"] = res["Shipments_MLP"]+res["Shipments_Rentals"]+res["Shipments_Crowd"]
    res["Dif_vs_FCST"] = res["Shipments_Totales"] - res["Shipments_FCST"]
    return res, det

# Primer cálculo
if "plan_res" not in st.session_state or "plan_det" not in st.session_state:
    st.session_state.plan_res, st.session_state.plan_det = run(st.session_state["params"])

# ========= Parser de instrucciones (Jarvis) =========
def parse_actions(instruccion: str) -> list[dict]:
    acciones: list[dict] = []
    txt = (instruccion or "").strip()
    if not txt:
        return acciones
    low = txt.lower()

    def _local_parse() -> list[dict]:
        L: list[dict] = []

        # set rutas
        m = re.search(r"\b(pon|ajusta|set|fija)\s+(mlp|rentals)\s*(?:a\s*)?(\d+)\b", low)
        if m:
            L.append({"tipo":"set_routes","modelo":m.group(2),"rutas":int(m.group(3))})

        m = re.search(r"\b(pon|ajusta|set|fija)\s+(mlp|rentals)\s*(?:a\s*)?(\d+)\s+en\s+([a-z0-9]+)\b", low)
        if m:
            L.append({"tipo":"set_routes","modelo":m.group(2),"rutas":int(m.group(3)),"svc":m.group(4).upper()})

        m = re.search(r"\b(mlp|rentals)\s*=\s*(\d+)\b", low) or re.search(r"\b(mlp|rentals)\s+(\d+)\b", low)
        if m:
            L.append({"tipo":"set_routes","modelo":m.group(1),"rutas":int(m.group(2))})

        m = re.search(r"(mlp|rentals).{0,8}?(\d+).{0,16}?\ben\s+([a-z0-9]+)\b", low)
        if m:
            L.append({"tipo":"set_routes","modelo":m.group(1),"rutas":int(m.group(2)),"svc":m.group(3).upper()})

        # quitar override
        m = re.search(r"\bquita.*override.*\b(mlp|rentals)\b(?:.*\ben\s+([a-z0-9]+)\b)?", low)
        if m:
            L.append({"tipo":"clear_override","modelo":m.group(1),"svc": (m.group(2).upper() if m.group(2) else None)})

        # flags
        m = re.search(r"\b(apaga|desactiva|prende|activa)\s+(escalar_si_excede_fcst)\b", low)
        if m:
            val = m.group(1) in ("prende","activa")
            L.append({"tipo":"set_param","nombre":"escalar_si_excede_fcst","valor":val})

        # factores
        m = re.search(r"\bfactor_escalado_mlp\s*=\s*([0-9]+(?:[.,][0-9]+)?)\b", low)
        if m:
            L.append({"tipo":"set_param","nombre":"factor_escalado_mlp","valor":float(m.group(1).replace(',','.'))})
        m = re.search(r"\bfactor_escalado_rentals\s*=\s*([0-9]+(?:[.,][0-9]+)?)\b", low)
        if m:
            L.append({"tipo":"set_param","nombre":"factor_escalado_rentals","valor":float(m.group(1).replace(',','.'))})

        # crowd needed
        if re.search(r"\bcrowd\b|\bneeded\b|\bnecesit", low):
            msvc   = re.search(r"\bsvc\s*([a-z0-9]+)\b", low)
            mfecha = re.search(r"(20\d{2}-\d{2}-\d{2})", low)
            L.append({"tipo":"crowd_need",
                      "svc":   (msvc.group(1).upper() if msvc else None) if msvc else None,
                      "fecha": (mfecha.group(1) if mfecha else None)})

        # recalc
        if re.search(r"\brecalc|recalcula|recalcular\b", low):
            L.append({"tipo":"recalc"})
        return L

    def _llm_parse() -> list[dict]:
        if not _HAS_OPENAI:
            return []
        try:
            system_msg = (
                "Devuelve SOLO JSON con key 'acciones' (lista). "
                "Cada item: {tipo:'set_routes|clear_override|set_param|recalc|crowd_need', ...}."
            )
            resp = _client.responses.create(
                model="gpt-4o-mini",
                input=[{"role":"system","content":system_msg},
                       {"role":"user","content":f"Instrucción: {txt}"}],
            )
            m = re.search(r"\{[\s\S]*\}", (resp.output_text or ""))
            if not m: return []
            data = json.loads(m.group(0))
            arr = data.get("acciones", [])
            return arr if isinstance(arr, list) else []
        except Exception:
            return []

    raw = _local_parse() if _PARSER_LOCAL_FIRST else _llm_parse()
    if not raw:
        raw = _llm_parse() if _PARSER_LOCAL_FIRST else _local_parse()

    out: list[dict] = []
    for a in raw:
        if not isinstance(a, dict): 
            continue
        t = a.get("tipo")
        if t in {"set_routes","clear_override","set_param","recalc","crowd_need"}:
            out.append(a)
    return out

def apply_actions(acciones: list[dict]) -> str:
    if not acciones:
        return "(Ignorado) No se reconoció ninguna acción."

    msg = []
    need_recalc = False

    for a in acciones:
        if not isinstance(a, dict) or "tipo" not in a:
            continue
        t = a.get("tipo")

        if t == "set_routes":
            modelo = str(a.get("modelo","")).lower()
            rutas  = int(a.get("rutas",0))
            svc    = a.get("svc")
            if modelo not in ("mlp","rentals"):
                msg.append(f"(Ignorado) Modelo inválido: {modelo}"); 
                continue
            key = "mlp" if modelo=="mlp" else "rentals"
            if svc:
                st.session_state["overrides"][key][str(svc).upper()] = rutas
                msg.append(f"Override {modelo}={rutas} en {str(svc).upper()}")
            else:
                st.session_state["overrides"][key]["GLOBAL"] = rutas
                msg.append(f"Override GLOBAL {modelo}={rutas}")
            need_recalc = True

        elif t == "clear_override":
            modelo = str(a.get("modelo","")).lower()
            svc    = a.get("svc")
            key = "mlp" if modelo=="mlp" else "rentals"
            if svc:
                st.session_state["overrides"][key].pop(str(svc).upper(), None)
                msg.append(f"Quité override {modelo} en {str(svc).upper()}")
            else:
                st.session_state["overrides"][key].pop("GLOBAL", None)
                msg.append(f"Quité override GLOBAL de {modelo}")
            need_recalc = True

        elif t == "set_param":
            nombre = a.get("nombre"); valor = a.get("valor")
            if nombre in st.session_state["params"]:
                st.session_state["params"][nombre] = valor
                msg.append(f"Parámetro '{nombre}' → {valor}")
                need_recalc = True
            else:
                msg.append(f"(Ignorado) Parámetro desconocido: {nombre}")

        elif t == "recalc":
            need_recalc = True
            msg.append("Recalculo solicitado.")

        elif t == "crowd_need":
            svc   = a.get("svc")
            fecha = a.get("fecha")
            res = st.session_state.plan_res.copy()
            if fecha: res = res[res["Fecha"]==pd.to_datetime(fecha)]
            if svc:   res = res[res["SVC"].astype(str).str.upper()==str(svc).upper()]
            if res.empty:
                msg.append("No encontré filas para calcular Crowd needed (revisa SVC/fecha).")
            else:
                gap  = (res["Shipments_FCST"] - (res["Shipments_MLP"]+res["Shipments_Rentals"])).clip(lower=0)
                sprc = st.session_state.plan_det[st.session_state.plan_det["Modelo"]=="Crowd"]["SPR"].mean()
                if pd.isna(sprc) or sprc<=0: 
                    sprc = spr_global if 'spr_global' in globals() else 1.0
                rutas = int(np.round(gap.sum()/max(sprc,1e-6)))
                msg.append(f"Rutas Crowd necesarias totales: {rutas}.")
        else:
            msg.append(f"(Ignorado) Acción no soportada: {t}")

    if need_recalc:
        st.session_state.plan_res, st.session_state.plan_det = run(st.session_state["params"])
        msg.append("Recalculé el plan.")
    return " | ".join(msg)

# ========= UI: encabezado =========
def _pick_logo():
    prefer = [
        "20250813_1028_Camión Futurista Amarillo_remix_01k2j400zxfp0te4vpq1kq1wnv.png",
        "mel-ia-ops.png", "mel_ia_ops.png", "Mel-IA Ops.png", "Mel-IA Ops.jpg", "mel-ia-ops.jpg",
    ]
    for p in prefer:
        if os.path.exists(p):
            return p
    pngs = glob.glob("*.png") + glob.glob("assets/*.png")
    if pngs:
        pngs.sort(key=lambda p: os.path.getsize(p), reverse=True)
        return pngs[0]
    return None

logo_path = _pick_logo()
col_logo, col_title = st.columns([1,4])
with col_logo:
    if logo_path:
        st.image(logo_path, use_container_width=True)
with col_title:
    st.markdown("## **Mel-IA Ops — Copiloto de Planeación de Flota**")

# ========= Sidebar: Jarvis (CAMBIOS) =========
with st.sidebar.expander("🧠 Jarvis — Instrucciones en español", expanded=True):
    st.caption("CAMBIA el plan. Ej.: `rentals=100`, `pon mlp 80 en SPB1`, `apaga escalar_si_excede_fcst`, "
               "`crowd needed`, `quita override de rentals en SPB2`.")
    instruccion = st.text_input("Instrucción", placeholder="Ej: pon rentals 50")
    st.markdown(f"OpenAI: {'🟢 conectado' if _HAS_OPENAI else '⚪ opcional (no configurado)'}")
    if st.checkbox("Ver params", value=False):
        st.json(st.session_state["params"])
    if st.checkbox("Ver overrides", value=False):
        st.json(st.session_state["overrides"])

    if st.button("Ejecutar instrucción"):
        acciones = parse_actions(instruccion)
        st.info(f"Acciones detectadas: {acciones}")
        msg = apply_actions(acciones)
        st.success(msg)
        st.rerun()

with st.sidebar:
    st.header("Parámetros rápidos")
    scale_if_exceed = st.checkbox("Escalar MLP/Rentals si exceden el FCST",
                                  value=st.session_state["params"]["escalar_si_excede_fcst"])
    st.session_state["params"]["escalar_si_excede_fcst"] = bool(scale_if_exceed)
    st.caption(f"Sheet ID: {SHEET_ID}")

# ========= UI Principal (tablas) =========
st.subheader("Resumen (incluye Shipments_FCST)")
cols_pref = [
    "SVC","Fecha","Rutas_MLP","Rutas_Rentals","Rutas_Crowd",
    "Shipments_MLP","Shipments_Rentals","Shipments_Crowd",
    "Shipments_Totales","Shipments_FCST","Dif_vs_FCST"
]
res_view = st.session_state.plan_res.copy()
res_view = res_view[[c for c in cols_pref if c in res_view.columns]]

col1, col2 = st.columns([2,1])
with col1:
    st.dataframe(
        res_view.sort_values(["SVC","Fecha"]).reset_index(drop=True),
        use_container_width=True
    )
with col2:
    st.metric("Max |Dif_vs_FCST|", f"{st.session_state.plan_res['Dif_vs_FCST'].abs().max():,.0f}")
    if st.button("🔁 Recalcular"):
        st.session_state.plan_res, st.session_state.plan_det = run(st.session_state["params"])
        st.rerun()

st.subheader("Detalle")
st.dataframe(
    st.session_state.plan_det.sort_values(["SVC","Fecha","Modelo","HOMOLOGACION_VEHICULO"]).reset_index(drop=True),
    use_container_width=True, height=380
)

# ========= Escribir a Google Sheet =========
st.subheader("📝 Escribir al Google Sheet")
if st.button("Escribir 'Plan_14_resumen' y 'Plan_14_detalle'"):
    try:
        write_ws(get_client().open_by_key(SHEET_ID), TABS["OUT_RES"],
                 st.session_state.plan_res.sort_values(["SVC","Fecha"]))
        write_ws(get_client().open_by_key(SHEET_ID), TABS["OUT_DET"],
                 st.session_state.plan_det.sort_values(["SVC","Fecha","Modelo","HOMOLOGACION_VEHICULO"]))
        st.success("¡Listo! Se actualizaron las pestañas.")
    except Exception as e:
        st.error(f"No se pudo escribir: {e}")

# ========= Chat inferior (CONSULTAS) =========
st.subheader("💬 Chat de consultas (no cambia el plan)")
if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)

q = st.chat_input("Consulta: ‘resumen svc SPB1’, ‘para fecha 2025-08-10’, ‘totales por modelo’, ‘¿mayor Dif_vs_FCST?’")

def _nl_answer(question: str) -> str:
    txt = question or ""
    plan_res = st.session_state.plan_res
    plan_det = st.session_state.plan_det
    try:
        low = txt.lower()

        # SPR promedio por SVC: "spr spb1"
        m = re.search(r"\bspr\b.*\b(spb[0-9]+)\b", low)
        if m:
            svc = m.group(1).upper()
            sprs = (plan_det.query("Modelo!='Crowd' and SVC==@svc")["SPR"])
            if sprs.empty:
                return f"No tengo SPR para {svc}."
            return f"SPR promedio en {svc}: **{sprs.mean():.2f}** (p50={sprs.median():.2f})."

        # Resumen por SVC
        m = re.search(r"\bresumen\b.*\bsvc\b.*\b(spb[0-9]+)\b", low)
        if m:
            svc = m.group(1).upper()
            df = (plan_res[plan_res["SVC"]==svc]
                  .sort_values("Fecha")[["Fecha","Rutas_MLP","Rutas_Rentals",
                                         "Shipments_Totales","Shipments_FCST","Dif_vs_FCST"]]
                  .tail(10))
            if df.empty: return f"Sin filas para {svc}."
            st.dataframe(df, use_container_width=True)
            return f"Te mostré las últimas 10 fechas de **{svc}**."

        # Para una fecha exacta
        m = re.search(r"(20\d{2}-\d{2}-\d{2})", low)
        if "para fecha" in low or "el día" in low or m:
            if m:
                fecha = pd.to_datetime(m.group(1))
                df = plan_res[plan_res["Fecha"]==fecha][["SVC","Rutas_MLP","Rutas_Rentals",
                                                          "Shipments_Totales","Shipments_FCST","Dif_vs_FCST"]]
                if not df.empty:
                    st.dataframe(df.sort_values("SVC"), use_container_width=True)
                    return f"Mostré el resumen para **{fecha.date()}**."
                return "No encontré esa fecha en el plan."

        # Mayor diferencia absoluta
        if "max" in low or "mayor dif" in low:
            row = plan_res.loc[plan_res["Dif_vs_FCST"].abs().idxmax()]
            return (f"El máximo |Dif_vs_FCST| es **{abs(row['Dif_vs_FCST']):,.0f}** en "
                    f"**{row['SVC']}** el **{pd.to_datetime(row['Fecha']).date()}**.")

        # Fallback con LLM (contexto liviano)
        if _HAS_OPENAI:
            def _safe_preview(df, n=6):
                d = df.head(n).copy()
                for c in d.columns:
                    if str(d[c].dtype).startswith("datetime") or "datetime64" in str(d[c].dtype):
                        d[c] = pd.to_datetime(d[c], errors="coerce").dt.strftime("%Y-%m-%d")
                return d.astype(object).where(pd.notnull(d), None).to_dict(orient="records")

            schema = {
                "plan_res_columns": list(plan_res.columns),
                "plan_det_columns": list(plan_det.columns),
                "ejemplo_plan_res": _safe_preview(plan_res, 6),
                "ejemplo_plan_det": _safe_preview(plan_det, 6),
            }

            resp = _client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {"role":"system","content":"Eres analista de planeación. Responde conciso y pide SVC/fecha si falta."},
                    {"role":"user","content": f"Pregunta: {question}\nContexto:\n{json.dumps(schema, default=str)[:8000]}"},
                ],
            )
            return resp.output_text
        else:
            return "Este chat es para CONSULTAS. Para CAMBIOS usa Jarvis en el panel izquierdo."

    except Exception as e:
        return f"Ocurrió un error respondiendo: {e}"

if q:
    st.session_state.chat.append(("user", q))
    ans = _nl_answer(q)
    st.session_state.chat.append(("assistant", ans))
    st.rerun()
