import os, re, json, glob
import numpy as np
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# ========= Configuración de página (primero) =========
st.set_page_config(page_title="Copiloto Flota — Mel-IA Ops", layout="wide")

# ========= Config general =========
SHEET_ID  = "1UBjU3-ftGCow3EzTD0NB6UaYwMUYUARbn9QjD7SlxtY"  # tu Google Sheet

TABS = dict(
    EJ="Ejecución",
    SRM="SRM",
    FCST="FCST",
    OUT_RES="Plan_14_resumen",
    OUT_DET="Plan_14_detalle"
)

RUTAS_SRM_IS_DAILY = True
FACTOR_SRM_SEM_A_DIA = 1/6
CROWD_VT = "Car"

# ========= Cliente OpenAI (opcional) =========
try:
    from openai import OpenAI
    _client = OpenAI()
    _HAS_OPENAI = True
except Exception:
    _client = None
    _HAS_OPENAI = False
    st.sidebar.warning("No se detectó OpenAI o la API key. El agente y chat funcionan en modo manual.")

# ========= Estado de parámetros controlados por agente =========
if "params" not in st.session_state:
    st.session_state["params"] = {
        "escalar_si_excede_fcst": True,   # bool
        "factor_escalado_mlp": 0.85,      # float [0..1]
        "factor_escalado_rentals": 0.75,  # float [0..1]
    }

# ========= Auth Google Sheets con Secrets (sin archivo) =========
@st.cache_resource
def get_client():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    # Toma credenciales de Streamlit Secrets
    if "gcp_service_account" in st.secrets:
        info = dict(st.secrets["gcp_service_account"])
        if "private_key" in info and "\\n" in info["private_key"]:
            info["private_key"] = info["private_key"].replace("\\n", "\n")
        creds = Credentials.from_service_account_info(info, scopes=scope)
        return gspread.authorize(creds)

    st.error(
        "No se encontraron credenciales. En Streamlit Cloud configura **Settings → Secrets** con la sección [gcp_service_account]."
    )
    st.stop()

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
    except:
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

# ========= Carga de datos =========
@st.cache_data(ttl=300)
def load_all():
    client = get_client()
    sheet  = client.open_by_key(SHEET_ID)
    df_ej  = read_ws(sheet, TABS["EJ"],
                     expected=["delivery_model","homologacion_vehiculo","shps dispatched","total_rutas"],
                     any_keys=["svc","fecha","año de date","delivery_model"])
    df_srm = read_ws(sheet, TABS["SRM"], expected=["mlp","region","svc"], any_keys=["mlp","svc"])
    df_fc  = read_ws(sheet, TABS["FCST"], expected=["svc"], any_keys=["svc"])
    return sheet, df_ej, df_srm, df_fc

sheet, df_exec, df_srm, df_fcst_raw = load_all()

# ========= Normalización de ejecución =========
if "DELIVERY_MODEL" not in df_exec.columns:
    for alt in ["DELIVERY_MODEL 1","Delivery Model","DELIVERY MODEL"]:
        if alt in df_exec.columns: df_exec["DELIVERY_MODEL"]=df_exec[alt]; break
if "SVC" not in df_exec.columns:
    for alt in ["Estación","Estacion"]:
        if alt in df_exec.columns: df_exec["SVC"]=df_exec[alt]; break
for c in ["Shps Dispatched","TOTAL_RUTAS"]:
    if c in df_exec.columns: df_exec[c]=pd.to_numeric(df_exec[c], errors="coerce")

_fecha = None
for cand in ["DATE","Fecha","FECHA","Date"]:
    if cand in df_exec.columns:
        _fecha = pd.to_datetime(df_exec[cand], errors="coerce"); break
if _fecha is None and {"Año de DATE","Mes de DATE","Día de DATE"}.issubset(df_exec.columns):
    mes_map = {"enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
               "julio":7,"agosto":8,"septiembre":9,"octubre":10,"noviembre":11,"diciembre":12}
    month_series = df_exec["Mes de DATE"].map(mes_map).fillna(pd.to_numeric(df_exec["Mes de DATE"], errors="coerce"))
    _fecha = pd.to_datetime(dict(
        year=pd.to_numeric(df_exec["Año de DATE"], errors="coerce"),
        month=pd.to_numeric(month_series, errors="coerce"),
        day=pd.to_numeric(df_exec["Día de DATE"], errors="coerce")
    ), errors="coerce")

df_exec["_fecha"] = _fecha
df_exec = df_exec[df_exec["_fecha"].notna()].copy()
df_exec["_dow"]   = df_exec["_fecha"].dt.weekday

# ========= FCST tidy =========
date_cols = [c for c in df_fcst_raw.columns if re.match(r"\d{1,2}\-\w{3}", str(c))]
for c in date_cols: df_fcst_raw[c]=clean_num(df_fcst_raw[c])
YEAR = int(df_exec["_fecha"].dt.year.max()) if not df_exec.empty else datetime.now().year

df_fcst = df_fcst_raw.melt(
    id_vars=["SVC"], value_vars=date_cols,
    var_name="Fecha_txt", value_name="Shipments_FCST"
).dropna(subset=["SVC"])
df_fcst["Fecha"] = pd.to_datetime(df_fcst["Fecha_txt"] + f"-{YEAR}", format="%d-%b-%Y", errors="coerce")
df_fcst = df_fcst[["SVC","Fecha","Shipments_FCST"]]
fechas = sorted(df_fcst["Fecha"].dropna().unique())
cal = pd.DataFrame({"Fecha": fechas})
cal["_dow"] = cal["Fecha"].dt.weekday

# ========= SPR por DOW =========
df_exec["SPR_raw"] = df_exec["Shps Dispatched"] / df_exec["TOTAL_RUTAS"]
df_exec["SPR_w"]   = df_exec["SPR_raw"]
_w = df_exec.groupby(["DELIVERY_MODEL","HOMOLOGACION_VEHICULO","_dow"])["SPR_raw"].transform(
    lambda s: winsor(s,0.05,0.95,10)
)
df_exec.loc[_w.notna(),"SPR_w"] = _w

spr_dow = (df_exec.groupby(["DELIVERY_MODEL","HOMOLOGACION_VEHICULO","_dow"])["SPR_w"]
           .mean().reset_index().rename(columns={"SPR_w":"SPR_dow"}))
spr_dow_vt = (spr_dow.groupby(["HOMOLOGACION_VEHICULO","_dow"])["SPR_dow"]
              .mean().reset_index().rename(columns={"SPR_dow":"SPR_dow_vt"}))
spr_dow_global = (spr_dow.groupby(["_dow"])["SPR_dow"]
                  .mean().reset_index().rename(columns={"SPR_dow":"SPR_dow_global"}))
spr_global = float(df_exec["SPR_w"].mean()) if not df_exec.empty else 0.0

# ========= Capacidad MLP (SRM) =========
veh_cols = [c for c in ["Extra Large Van","Large Van","Small Van","Car"] if c in df_srm.columns]
df_srm_long = df_srm.melt(id_vars=["MLP","Region","SVC"], value_vars=veh_cols,
                          var_name="HOMOLOGACION_VEHICULO", value_name="Rutas_MLP").fillna(0)
df_srm_long["Rutas_MLP"] = pd.to_numeric(df_srm_long["Rutas_MLP"], errors="coerce").fillna(0)
if not RUTAS_SRM_IS_DAILY:
    df_srm_long["Rutas_MLP"] *= FACTOR_SRM_SEM_A_DIA
mlp_capacity = df_srm_long.groupby(["SVC","HOMOLOGACION_VEHICULO"], as_index=False).agg(Rutas_MLP=("Rutas_MLP","sum"))

# ========= Capacidad Rentals (promedio de rutas) =========
is_rent = df_exec["DELIVERY_MODEL"].astype(str).str.upper().str.contains("RENT")
df_rent = df_exec[is_rent].copy()
grp = ["SVC","HOMOLOGACION_VEHICULO"] if "SVC" in df_rent.columns else ["HOMOLOGACION_VEHICULO"]
rent_rutas = df_rent.groupby(grp, dropna=False)["TOTAL_RUTAS"].mean().reset_index().rename(columns={"TOTAL_RUTAS":"Rutas_Rentals_avg"})
if "SVC" not in rent_rutas.columns:
    svcs = mlp_capacity["SVC"].dropna().unique().tolist()
    rent_rutas = rent_rutas.assign(_k=1).merge(pd.DataFrame({"SVC":svcs,"_k":[1]*len(svcs)}), on="_k").drop(columns="_k")
rentals_capacity = rent_rutas[["SVC","HOMOLOGACION_VEHICULO","Rutas_Rentals_avg"]].copy()

# ========= Motor principal que USA params =========
def run(params: dict):
    scale_if_exceed = bool(params.get("escalar_si_excede_fcst", True))
    f_mlp   = float(params.get("factor_escalado_mlp", 1.0))
    f_rent  = float(params.get("factor_escalado_rentals", 1.0))

    # --- MLP ---
    spr_mlp = spr_dow[spr_dow["DELIVERY_MODEL"]=="MLP"][["HOMOLOGACION_VEHICULO","_dow","SPR_dow"]]
    mlp = mlp_capacity.assign(_k=1).merge(cal.assign(_k=1), on="_k").drop(columns="_k")
    mlp = (mlp.merge(spr_mlp, on=["HOMOLOGACION_VEHICULO","_dow"], how="left")
               .merge(spr_dow_vt, on=["HOMOLOGACION_VEHICULO","_dow"], how="left")
               .merge(spr_dow_global, on=["_dow"], how="left"))
    mlp["SPR_final"] = mlp["SPR_dow"].fillna(mlp["SPR_dow_vt"]).fillna(mlp["SPR_dow_global"]).fillna(spr_global)
    mlp["Rutas_MLP_int"] = np.round(mlp["Rutas_MLP"] * f_mlp).clip(lower=0).astype(int)
    mlp["Shipments_MLP"] = mlp["Rutas_MLP_int"] * mlp["SPR_final"]

    # --- Rentals ---
    rent = rentals_capacity.assign(_k=1).merge(cal.assign(_k=1), on="_k").drop(columns="_k")
    rent = (rent.merge(spr_dow_vt, on=["HOMOLOGACION_VEHICULO","_dow"], how="left")
               .merge(spr_dow_global, on=["_dow"], how="left"))
    rent["SPR_final"] = rent["SPR_dow_vt"].fillna(rent["SPR_dow_global"]).fillna(spr_global)
    rent["Rutas_Rentals_int"] = np.round(rent["Rutas_Rentals_avg"] * f_rent).clip(lower=0).astype(int)
    rent["Shipments_Rentals"] = rent["Rutas_Rentals_int"] * rent["SPR_final"]

    # --- Crowd ---
    crowd = cal.merge(
        spr_dow_vt[spr_dow_vt["HOMOLOGACION_VEHICULO"]==CROWD_VT][["_dow","SPR_dow_vt"]],
        on="_dow", how="left"
    ).merge(spr_dow_global, on="_dow", how="left")
    crowd["SPR_Crowd"] = crowd["SPR_dow_vt"].fillna(crowd["SPR_dow_global"]).fillna(spr_global)

    fcst = df_fcst.copy()

    # --- Detalle base ---
    det_mlp = (
        mlp.rename(columns={"Rutas_MLP_int":"Rutas","SPR_final":"SPR","Shipments_MLP":"Shipments"})
           .assign(Modelo="MLP")[["SVC","Fecha","HOMOLOGACION_VEHICULO","Modelo","Rutas","SPR","Shipments"]]
    )
    det_rent = (
        rent.rename(columns={"Rutas_Rentals_int":"Rutas","SPR_final":"SPR","Shipments_Rentals":"Shipments"})
            .assign(Modelo="Rentals")[["SVC","Fecha","HOMOLOGACION_VEHICULO","Modelo","Rutas","SPR","Shipments"]]
    )
    det_crowd = fcst[["SVC","Fecha"]].drop_duplicates().merge(crowd[["Fecha","SPR_Crowd"]], on="Fecha", how="left")
    det_crowd["HOMOLOGACION_VEHICULO"] = f"Crowd ({CROWD_VT})"
    det_crowd["Modelo"] = "Crowd"
    det_crowd["Rutas"] = 0
    det_crowd["SPR"] = det_crowd["SPR_Crowd"]
    det_crowd["Shipments"] = 0.0
    det_crowd = det_crowd[["SVC","Fecha","HOMOLOGACION_VEHICULO","Modelo","Rutas","SPR","Shipments"]]

    det = pd.concat([det_mlp, det_rent, det_crowd], ignore_index=True)

    # --- Ajuste vs FCST ---
    def ajustar(gr):
        svc, fecha = gr.name
        fc = float(fcst[(fcst.SVC==svc)&(fcst.Fecha==fecha)]["Shipments_FCST"].sum())
        m = gr["Modelo"].eq("MLP"); r = gr["Modelo"].eq("Rentals"); c = gr["Modelo"].eq("Crowd")
        base = gr.loc[m|r,"Shipments"].sum()
        if scale_if_exceed and base > fc and base > 0:
            f = fc/base
            for mask in [m,r]:
                gr.loc[mask,"Rutas"] = np.round(gr.loc[mask,"Rutas"]*f).clip(lower=0).astype(int)
                gr.loc[mask,"Shipments"] = gr.loc[mask,"Rutas"]*gr.loc[mask,"SPR"]
            gr.loc[c,["Rutas","Shipments"]] = (0,0.0)
            return gr
        gap = max(fc-base, 0.0)
        if gap>0:
            spr_c = float(gr.loc[c,"SPR"].iloc[0]) if gr.loc[c,"SPR"].notna().any() else 0.0
            if spr_c>0:
                rr = int(np.round(gap/spr_c))
                gr.loc[c,"Rutas"] = rr
                gr.loc[c,"Shipments"] = rr*spr_c
        return gr

    det = det.groupby(["SVC","Fecha"], group_keys=False).apply(ajustar)

    # --- Resumen ---
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

# === Resultados cacheados en estado (primer cálculo) ===
if "plan_res" not in st.session_state or "plan_det" not in st.session_state:
    st.session_state.plan_res, st.session_state.plan_det = run(st.session_state["params"])

# ========= Utilidades del agente =========
_NUMERIC_TRUE = {"true","t","1","yes","y","si","sí","on"}
_NUMERIC_FALSE = {"false","f","0","no","n","off"}

def _coerce_value(val_str: str):
    s = str(val_str).strip()
    if s.lower() in _NUMERIC_TRUE: return True
    if s.lower() in _NUMERIC_FALSE: return False
    if re.fullmatch(r"[+-]?\d+", s): return int(s)
    if re.fullmatch(r"[+-]?\d+(\.\d+)?", s): return float(s)
    return s

def _apply_changes(cambios):
    resultados = []
    for c in cambios:
        nombre = c.get("nombre"); valor = c.get("valor")
        if nombre not in st.session_state["params"]:
            resultados.append({"ok": False, "msg": f"Parámetro desconocido: '{nombre}'"}); continue
        if nombre in ("factor_escalado_mlp","factor_escalado_rentals"):
            try:
                v = float(valor)
                if not (0.0 <= v <= 1.0):
                    resultados.append({"ok": False, "msg": f"'{nombre}' debe estar en [0,1]."}); continue
                valor = v
            except:
                resultados.append({"ok": False, "msg": f"'{nombre}' debe ser numérico."}); continue
        if nombre == "escalar_si_excede_fcst":
            if isinstance(valor, str): valor = _coerce_value(valor)
            if not isinstance(valor, bool):
                resultados.append({"ok": False, "msg": f"'{nombre}' debe ser booleano."}); continue
        st.session_state["params"][nombre] = valor
        resultados.append({"ok": True, "msg": f"{nombre} → {valor}"})
    return resultados

def _extract_json(texto: str):
    m = re.search(r"\{[\s\S]*\}", texto)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except:
        return None

# ========= Encabezado con imagen =========
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
col_logo, col_title = st.columns([1,4])  # compatible con versiones antiguas de Streamlit
with col_logo:
    if logo_path:
        st.image(logo_path, use_container_width=True)
with col_title:
    st.markdown("## **Mel-IA Ops — Copiloto de Planeación de Flota**")

# ========= Sidebar: Agente de ajustes =========
with st.sidebar.expander("🛠️ Ajustes por lenguaje natural", expanded=True):
    st.caption("Pídeme cambios y que recalcule. Ejemplos:")
    st.code("factor_escalado_mlp=0.9 y recalcula\napaga escalar_si_excede_fcst\nbaja rentals 10%", language="text")
    instruccion = st.text_input("Instrucción", placeholder="Ej: sube mlp a 0.9 y recalcula")
    if st.checkbox("Ver params", value=False):
        st.json(st.session_state["params"])

    if st.button("Ejecutar instrucción"):
        cambios = []; hacer_recalculo = True
        if _HAS_OPENAI and instruccion.strip():
            try:
                system_msg = ("Devuelve SOLO JSON con keys: cambios (lista de {nombre, valor}) y recalcular (bool). "
                              f"Parámetros válidos: {list(st.session_state['params'].keys())}. "
                              "Si dice 'apaga/desactiva' → false; 'prende/activa' → true.")
                resp = _client.responses.create(
                    model="gpt-4o-mini",
                    input=[{"role":"system","content":system_msg},
                           {"role":"user","content":f"Instrucción: {instruccion}"}],
                )
                data = _extract_json(resp.output_text)
                if data:
                    cambios = data.get("cambios", [])
                    hacer_recalculo = bool(data.get("recalcular", True))
            except Exception as e:
                st.warning(f"No se pudo usar OpenAI: {e}")

        # --------- Fallback robusto (sin IA) ----------
        if not cambios and instruccion.strip():
            txt = instruccion.strip().lower()
            alias = {"mlp": "factor_escalado_mlp", "rentals": "factor_escalado_rentals"}

            # nombre=valor
            for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9\.,]+)", txt):
                nombre, valor_str = m.group(1), m.group(2).replace(",", ".")
                nombre = alias.get(nombre, nombre)
                if nombre in st.session_state["params"]:
                    cambios.append({"nombre": nombre, "valor": float(valor_str)})

            # “sube/baja/ajusta X a N”
            m = re.search(r"(sube|incrementa|baja|reduce|ajusta)\s+(mlp|rentals|factor_escalado_mlp|factor_escalado_rentals)\s+a\s*([0-9\.,]+)", txt)
            if m:
                verbo, nombre, valor = m.groups()
                nombre = alias.get(nombre, nombre)
                try:
                    v = float(valor.replace(",", "."))
                    cambios.append({"nombre": nombre, "valor": v})
                except:
                    pass

            # “sube/baja X %”
            m = re.search(r"(sube|incrementa|baja|reduce)\s+(mlp|rentals|factor_escalado_mlp|factor_escalado_rentals)\s+([0-9]{1,3})\s*%", txt)
            if m:
                verbo, nombre, p = m.groups()
                nombre = alias.get(nombre, nombre)
                base = float(st.session_state["params"].get(nombre, 1.0))
                factor = (1 + int(p)/100.0) if verbo in ("sube","incrementa") else (1 - int(p)/100.0)
                cambios.append({"nombre": nombre, "valor": max(0.0, min(1.0, base * factor))})

            # “sube/baja X” sin número -> ±0.05
            m = re.search(r"(sube|incrementa|baja|reduce)\s+(mlp|rentals|factor_escalado_mlp|factor_escalado_rentals)\b", txt)
            if m and not cambios:
                verbo, nombre = m.groups()
                nombre = alias.get(nombre, nombre)
                base = float(st.session_state["params"].get(nombre, 1.0))
                delta = 0.05 if verbo in ("sube","incrementa") else -0.05
                cambios.append({"nombre": nombre, "valor": max(0.0, min(1.0, base + delta))})

            # on/off de la bandera
            for m in re.finditer(r"(apaga|desactiva|prende|activa)\s+(escalar_si_excede_fcst)", txt):
                accion, nombre = m.groups()
                cambios.append({"nombre": nombre, "valor": accion in ("prende","activa")})

            # Recalcular si se pide explícitamente (por defecto sí)
            if re.search(r"\bno\s+recalc", txt):
                hacer_recalculo = False
            elif re.search(r"\brecalc", txt):
                hacer_recalculo = True

        if cambios:
            resultados = _apply_changes(cambios)
            st.write("Resultados:", resultados)
            if hacer_recalculo:
                st.session_state.plan_res, st.session_state.plan_det = run(st.session_state["params"])
                st.success("Plan recalculado con parámetros actualizados.")
                st.rerun()
        else:
            st.info("No se detectaron cambios.")

# ========= Sidebar: Parámetros rápidos =========
with st.sidebar:
    st.header("Parámetros rápidos")
    scale_if_exceed = st.checkbox("Escalar MLP/Rentals si exceden el FCST",
                                  value=st.session_state["params"]["escalar_si_excede_fcst"])
    st.session_state["params"]["escalar_si_excede_fcst"] = bool(scale_if_exceed)
    st.caption(f"Sheet ID: {SHEET_ID}")

# ========= UI Principal =========
st.subheader("Resumen (incluye Shipments_FCST)")

cols_pref = [
    "SVC","Fecha",
    "Rutas_MLP","Rutas_Rentals","Rutas_Crowd",
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

# ========= Agente: parseo y ejecución de instrucciones desde el chat =========

def _agent_parse_instruction(txt: str) -> dict:
    """
    Devuelve un dict con:
      {
        "acciones": [
            {"tipo":"update_params","cambios":[{"nombre":"factor_escalado_mlp","valor":0.9}]},
            {"tipo":"recalcular"},
            {"tipo":"escribir_sheet"},
            {"tipo":"mostrar_resumen_svc","svc":"SPB1","n":10},
            {"tipo":"mostrar_fecha","fecha":"2025-08-10"},
        ]
      }
    """
    txt = (txt or "").strip()
    out = {"acciones": []}
    if not txt:
        return out

    # --- Con IA (si disponible) ---
    if _HAS_OPENAI:
        try:
            system = (
                "Eres un agente de planeación. Devuelve SOLO JSON con la forma:\n"
                "{'acciones':[...]} donde cada acción es de tipos: "
                "update_params(cambios:[{nombre,valor}]), recalcular(), escribir_sheet(), "
                "mostrar_resumen_svc(svc, n opcional), mostrar_fecha(fecha 'YYYY-MM-DD'). "
                f"Parámetros válidos: {list(st.session_state['params'].keys())}.\n"
                "Si el usuario pide 'sube/baja X %', calcula el nuevo valor en [0,1]. "
                "Si pide 'apaga/prende escalar...', mapea a booleano."
            )
            resp = _client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {"role":"system","content":system},
                    {"role":"user","content":f"Instrucción: {txt}"},
                ],
            )
            data = _extract_json(resp.output_text)
            if isinstance(data, dict) and "acciones" in data:
                return data
        except Exception as e:
            st.info(f"Agente (IA) no disponible, usando reglas locales. Detalle: {e}")

    # --- Fallback sin IA (reglas locales) ---
    acciones = []
    low = txt.lower()

    # update_params (reutiliza tu parser del sidebar)
    cambios = []
    alias = {"mlp":"factor_escalado_mlp","rentals":"factor_escalado_rentals"}
    for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9\.,]+)", low):
        nombre, valor_str = m.group(1), m.group(2).replace(",", ".")
        nombre = alias.get(nombre, nombre)
        if nombre in st.session_state["params"]:
            cambios.append({"nombre": nombre, "valor": float(valor_str)})

    m = re.search(r"(sube|incrementa|baja|reduce|ajusta)\s+(mlp|rentals|factor_escalado_mlp|factor_escalado_rentals)\s+a\s*([0-9\.,]+)", low)
    if m:
        _, nombre, valor = m.groups()
        nombre = alias.get(nombre, nombre)
        try:
            v = float(valor.replace(",", "."))
            cambios.append({"nombre": nombre, "valor": v})
        except:
            pass

    m = re.search(r"(sube|incrementa|baja|reduce)\s+(mlp|rentals|factor_escalado_mlp|factor_escalado_rentals)\s+([0-9]{1,3})\s*%", low)
    if m:
        verbo, nombre, p = m.groups()
        nombre = alias.get(nombre, nombre)
        base = float(st.session_state["params"].get(nombre, 1.0))
        factor = (1 + int(p)/100.0) if verbo in ("sube","incrementa") else (1 - int(p)/100.0)
        cambios.append({"nombre": nombre, "valor": max(0.0, min(1.0, base * factor))})

    m = re.search(r"(sube|incrementa|baja|reduce)\s+(mlp|rentals|factor_escalado_mlp|factor_escalado_rentals)\b", low)
    if m and not cambios:
        _, nombre = m.groups()
        nombre = alias.get(nombre, nombre)
        base = float(st.session_state["params"].get(nombre, 1.0))
        delta = 0.05 if m.group(1) in ("sube","incrementa") else -0.05
        cambios.append({"nombre": nombre, "valor": max(0.0, min(1.0, base + delta))})

    for m in re.finditer(r"(apaga|desactiva|prende|activa)\s+(escalar_si_excede_fcst)", low):
        accion, nombre = m.groups()
        cambios.append({"nombre": nombre, "valor": accion in ("prende","activa")})

    if cambios:
        acciones.append({"tipo":"update_params","cambios":cambios})

    # recalcular explícito
    if re.search(r"\b(recalcula|recalcular|recalc)\b", low):
        acciones.append({"tipo":"recalcular"})

    # escribir al sheet
    if re.search(r"\b(escribe|escribir)\b.*\b(sheet|hoja|google)\b", low):
        acciones.append({"tipo":"escribir_sheet"})

    # mostrar_resumen_svc
    m = re.search(r"\bresumen\b.*\bsvc\b\s*([A-Z0-9]+)", low)
    if m:
        acciones.append({"tipo":"mostrar_resumen_svc","svc":m.group(1).upper(),"n":10})

    # mostrar_fecha
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", low)
    if m:
        acciones.append({"tipo":"mostrar_fecha","fecha":m.group(1)})

    out["acciones"] = acciones
    return out


def _agent_execute(parsed: dict) -> str:
    """
    Ejecuta las acciones y devuelve un mensaje para el chat.
    Puede mostrar dataframes/metricas en pantalla (side-effects).
    """
    acciones = parsed.get("acciones", [])
    mensajes = []

    for a in acciones:
        t = a.get("tipo")

        if t == "update_params":
            res = _apply_changes(a.get("cambios", []))
            st.write("Resultados:", res)
            mensajes.append("Actualicé parámetros.")

        elif t == "recalcular":
            st.session_state.plan_res, st.session_state.plan_det = run(st.session_state["params"])
            st.success("Plan recalculado.")
            mensajes.append("Recalculé el plan.")

        elif t == "escribir_sheet":
            try:
                cli = get_client()
                write_ws(cli.open_by_key(SHEET_ID), TABS["OUT_RES"],
                         st.session_state.plan_res.sort_values(["SVC","Fecha"]))
                write_ws(cli.open_by_key(SHEET_ID), TABS["OUT_DET"],
                         st.session_state.plan_det.sort_values(["SVC","Fecha","Modelo","HOMOLOGACION_VEHICULO"]))
                st.success("¡Listo! Se actualizaron las pestañas en Google Sheet.")
                mensajes.append("Escribí 'Plan_14_resumen' y 'Plan_14_detalle'.")
            except Exception as e:
                st.error(f"No se pudo escribir: {e}")
                mensajes.append(f"No pude escribir: {e}")

        elif t == "mostrar_resumen_svc":
            svc = a.get("svc")
            n = int(a.get("n", 10))
            pr = st.session_state.plan_res
            if svc and "SVC" in pr.columns:
                df = (pr[pr["SVC"]==svc]
                      .sort_values("Fecha")[["Fecha","Rutas_MLP","Rutas_Rentals",
                                             "Shipments_Totales","Shipments_FCST","Dif_vs_FCST"]]
                      .tail(n))
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                    mensajes.append(f"Mostré resumen de **{svc}** (últimos {n}).")
                else:
                    mensajes.append(f"No encontré filas para {svc}.")
            else:
                mensajes.append("Pásame el SVC para el resumen.")

        elif t == "mostrar_fecha":
            fecha = a.get("fecha")
            if fecha:
                try:
                    f = pd.to_datetime(fecha)
                    df = st.session_state.plan_res[st.session_state.plan_res["Fecha"]==f][
                        ["SVC","Rutas_MLP","Rutas_Rentals","Shipments_Totales","Shipments_FCST","Dif_vs_FCST"]
                    ]
                    if not df.empty:
                        st.dataframe(df.sort_values("SVC"), use_container_width=True)
                        mensajes.append(f"Mostré el resumen para **{f.date()}**.")
                    else:
                        mensajes.append("No encontré esa fecha en el plan.")
                except Exception:
                    mensajes.append("Fecha inválida; usa YYYY-MM-DD.")

    if not mensajes:
        return "Listo. Si quieres, dime: 'sube mlp a 0.9 y recalcula', 'escribe al sheet', 'resumen svc SPB1', 'para fecha 2025-08-10', etc."
    return " ".join(mensajes)



# ========= Chat con Copiloto (Q&A sobre los datos) =========
st.subheader("💬 Chat con Copiloto (pregúntame de tus datos)")
if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)

q = st.chat_input("Ej: 'sube mlp a 0.9 y recalcula' · 'resumen svc SPB1' · 'para fecha 2025-08-10' · o pregúntame algo de los datos")

def _nl_answer(question: str) -> str:
    # ... (deja tu función _nl_answer tal como la tienes ahora) ...
    # (no la repito para ahorrar espacio)
    # Importante: esta se usa solo si el agente no detecta acciones relevantes
    return "No entendí la pregunta."

if q:
    st.session_state.chat.append(("user", q))

    # 1) Intentar como instrucción/acción
    parsed = _agent_parse_instruction(q)
    if parsed.get("acciones"):
        msg = _agent_execute(parsed)
        st.session_state.chat.append(("assistant", msg))
        st.rerun()

    # 2) Si no hubo acciones -> Q&A sobre los datos
    ans = _nl_answer(q)
    st.session_state.chat.append(("assistant", ans))
    st.rerun()
