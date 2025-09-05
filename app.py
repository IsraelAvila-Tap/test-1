# app.py
# =============================================================================
# Mel-IA — Plan táctico (Copiloto de tu flota en Mercado Libre)
# Tabs: FCST, SPR, Capacity, Rentals, Crowd, SRM (MLP caps).
# Encabezado autodetectado (busca SVC/SVCs/LC/Facility) + encabezado de 2 filas (Crowd).
# Robusto ante vacíos/alias y pestañas ausentes.
# =============================================================================
import os, json, re, unicodedata, textwrap, traceback
from datetime import date
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 0) Credenciales y configuración
# -----------------------------------------------------------------------------
if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
elif "gcp_service_account" in st.secrets:
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps(dict(st.secrets["gcp_service_account"]))

SERVICE_EMAIL = "planificacion@planificacion.iam.gserviceaccount.com"

DEFAULT_SVCS = ["SGD1", "SMT1", "SMX9", "SPB1"]
DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1UBjU3-ftGCow3EzTD0NB6UaYwMUYUARbn9QjD7SlxtY/edit?gid=148917403#gid=148917403"


# ---------------------------------------------------------
# Branding (logo): búsqueda robusta y carga
# ---------------------------------------------------------
from PIL import Image
import unicodedata

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    return s.lower()

def find_logo_file(preferred: list[str] | None = None) -> str | None:
    """
    1) Prueba rutas preferidas (tal cual);
    2) Busca recursivamente un .png/.jpg cuyo nombre contenga 'camion' y 'amarillo' o 'mel-ia'.
    """
    import os
    preferred = preferred or []
    # 1) preferred exactas
    for p in preferred:
        if p and os.path.exists(p):
            return p
    # 2) secretos opcionales
    psec = (st.secrets.get("LOGO_PATH") if isinstance(st.secrets, dict) else None) or os.environ.get("LOGO_PATH")
    if psec and os.path.exists(psec):
        return psec
    # 3) búsqueda recursiva
    hits = []
    for root, _, files in os.walk("."):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                fn = _norm(f)
                # tokens que identifican tu archivo (soporta acentos/variantes)
                if (("camion" in fn and "amarillo" in fn) or ("mel-ia" in fn) or ("meli" in fn)):
                    hits.append(os.path.join(root, f))
    return hits[0] if hits else None

# Intenta con el nombre que vimos en tu repo + búsqueda flexible
LOGO_PATH = find_logo_file([
    # ↙️ si cambia, no pasa nada: el buscador recursivo lo encuentra igual
    "20250813_1028_Camión Futurista Amarillo_remix_01k2j40zxfp0te4vpq1kq1wnv.png",
    "20250813_1028_Camion Futurista Amarillo_remix_01k2j40zxfp0te4vpq1kq1wnv.png",
])

LOGO_IMAGE = None
if LOGO_PATH:
    try:
        LOGO_IMAGE = Image.open(LOGO_PATH)
    except Exception as _e:
        LOGO_IMAGE = None  # fallback sin romper la app

from PIL import Image, ImageChops

def autocrop_whitespace(img: Image.Image) -> Image.Image:
    try:
        # Si tiene alpha, recorta contra transparente
        if img.mode in ("RGBA", "LA"):
            bg = Image.new(img.mode, img.size, (255, 255, 255, 0))
            diff = ImageChops.difference(img, bg)
            bbox = diff.getbbox()
            return img.crop(bbox) if bbox else img
        # Si no, recorta contra el color del pixel (0,0) (normalmente blanco)
        base = img.convert("RGB")
        bg = Image.new("RGB", base.size, base.getpixel((0, 0)))
        diff = ImageChops.difference(base, bg)
        bbox = diff.getbbox()
        return img.crop(bbox) if bbox else img
    except Exception:
        return img

# Quita fondo blanco (o casi blanco) -> lo vuelve transparente
def white_to_alpha(img: Image.Image, threshold: int = 245) -> Image.Image:
    """Convierte a RGBA y pone alpha=0 donde R,G,B >= threshold (blancos/near-white)."""
    try:
        import numpy as np
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        arr = np.array(img)
        r, g, b, a = arr[...,0], arr[...,1], arr[...,2], arr[...,3]
        mask = (r >= threshold) & (g >= threshold) & (b >= threshold)
        arr[mask, 3] = 0  # vuelve transparentes los blancos
        return Image.fromarray(arr, mode="RGBA")
    except Exception:
        # Fallback sin romper la app
        return img


# Después de abrir el logo:
if LOGO_IMAGE is not None:
    LOGO_IMAGE = autocrop_whitespace(LOGO_IMAGE)
    # ← limpia el fondo blanco; sube/baja el umbral si hace falta
    LOGO_IMAGE = white_to_alpha(LOGO_IMAGE, threshold=245)



def sanitize_sheet_id(text: str | None) -> str | None:
    if not text: return None
    text = text.strip()
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", text)
    if m: return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9-_]{20,}", text): return text
    return text

SHEET_ID = sanitize_sheet_id(
    st.secrets.get("SHEET_ID")
    or os.environ.get("SHEET_ID")
    or os.environ.get("PROJECT_SHEET_ID")
    or st.secrets.get("PROJECT_SHEET_ID")
    or DEFAULT_SHEET_URL
)

SHEET_TABS = {
    "fcst":     "FCST",
    "spr":      "SPR",
    "capacity": "Capacity",
    "rentals":  "Rentals",
    "crowd":    "Crowd",
    "srm":      "SRM",     # <--- NUEVO (capacidad MLP)
}

# ---------------------------------------------------------
# Header fijo (fixed), compacto y con control de tamaños
# ---------------------------------------------------------
def render_fixed_header(
    logo_img=None,
    title="Mel-IA (Copiloto de tu flota en Mercado Libre)",
    subtitle="Copiloto de planificación táctica de flota • Rentals • Crowd • MLP",
    header_h=104,     # altura total de la barra
    logo_h=800,       # altura del logo dentro de la barra
):
    import io, base64

    def _img_to_b64(img):
        buf = io.BytesIO()
        try:
            img.save(buf, format="PNG")
        except Exception:
            img = img.convert("RGBA")
            img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    if logo_img is not None:
        try:
            b64 = _img_to_b64(logo_img)
            logo_html = (
                f'<img src="data:image/png;base64,{b64}" alt="logo" '
                f'style="height:{int(logo_h)}px; width:auto; display:block;" />'
            )
        except Exception:
            logo_html = f'<div style="font-size:{int(logo_h)}px; line-height:{int(logo_h)}px;">🚛</div>'
    else:
        logo_html = f'<div style="font-size:{int(logo_h)}px; line-height:{int(logo_h)}px;">🚛</div>'

    st.markdown(
        f"""
        <style>
          :root {{ --meli-header-h: {int(header_h)}px; }}

          /* Barra fija arriba de todo */
          .meli-fixed {{
            position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
            background: #FFFFFF;
            height: var(--meli-header-h);
            border-bottom: 1px solid rgba(0,0,0,.06);
            box-shadow: 0 1px 6px rgba(0,0,0,.04);
          }}
          .meli-wrap {{
            max-width: 1200px; margin: 0 auto;
            height: 100%;
            display: grid; grid-template-columns: auto 1fr;
            align-items: center; gap: 14px; padding: 6px 12px;
          }}
          .meli-title {{
            margin: 0; font-size: 32px; line-height: 1.15; font-weight: 800;
          }}
          .meli-sub {{ margin: 2px 0 0 0; color: #6b7280; font-size: 13px; }}
          /* Quita el header nativo y compensa el espacio superior del main */
          [data-testid="stHeader"] {{ display: none; }}
          .block-container {{ padding-top: calc(var(--meli-header-h) + 12px) !important; }}
        </style>

        <div class="meli-fixed">
          <div class="meli-wrap">
            <div class="meli-logo">{logo_html}</div>
            <div>
              <h1 class="meli-title">{title}</h1>
              <div class="meli-sub">{subtitle}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )





# -----------------------------------------------------------------------------
# 1) Normalización / coerción
# -----------------------------------------------------------------------------
def _canon_name(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    return re.sub(r"[ \-_/\.]", "", s).lower()

def find_and_rename(df: pd.DataFrame, candidates: List[str], new_name: str,
                    required: bool = True, source_label: str = "") -> Optional[str]:
    cmap = {_canon_name(c): c for c in df.columns}
    for cand in candidates:
        key = _canon_name(cand)
        if key in cmap:
            real = cmap[key]
            if real != new_name: df.rename(columns={real: new_name}, inplace=True)
            return new_name
    if required:
        raise ValueError(f"{source_label}: falta columna equivalente a {candidates}. Encabezados: {list(df.columns)}")
    return None

def ensure_columns(df: pd.DataFrame, defaults: Dict[str, object]) -> pd.DataFrame:
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v
    return df

def _clean_svc_values(series_like) -> list[str]:
    """
    Devuelve una lista de SVCs válidos, eliminando None/nan/'' y
    cadenas 'none'/'nan' y cosas raras.
    """
    import pandas as pd, re
    if series_like is None:
        return []
    s = pd.Series(series_like)
    # quita nulos reales
    s = s[~s.isna()]
    # a str para normalizar
    s = s.astype(str).str.strip()
    # quita valores vacíos o 'none'/'nan'
    s = s[~s.str.lower().isin(["", "none", "nan", "(none)"])]
    # patrón simple de SVC (alfa-num, guion/guion_bajo)
    s = s[s.str.match(r"^[A-Za-z0-9_\-]{2,}$", na=False)]
    return sorted(s.unique().tolist())


def _as_str_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def _find_units_like_column(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    def canon(x: str) -> str:
        return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", str(x).lower()).encode("ascii","ignore").decode("ascii"))
    cands = [canon(c) for c in df.columns]
    real  = list(df.columns)
    targets = [r"^unidades", r"^units?$", r"^cantidad$", r"^qty$", r"^count$"]
    for i, can in enumerate(cands):
        for pat in targets:
            if re.search(pat, can):
                return real[i]
    return None

_NUM_SEP_RE = re.compile(r"[ ,\u00A0]")

def _maybe_to_numeric(series_like):
    import pandas as pd
    if isinstance(series_like, pd.DataFrame):
        for sub in series_like.columns:
            series_like[sub] = _maybe_to_numeric(series_like[sub])
        return series_like
    s = series_like
    if getattr(s, "dtype", None) is not None and s.dtype.kind in "iufc":
        return s
    sample = s.dropna().astype(str).head(60) if hasattr(s, "dropna") else []
    if len(sample) == 0:
        try:    return pd.to_numeric(s, errors="coerce")
        except Exception: return s
    looks_numeric = 0
    for v in sample:
        v2 = _NUM_SEP_RE.sub("", v).replace("%","").replace("−","-")
        if re.fullmatch(r"-?\d+(\.\d+)?", v2): looks_numeric += 1
    if looks_numeric / max(1, len(sample)) >= 0.75:
        s2 = s.astype(str).str.replace("%","", regex=False)
        s2 = s2.str.replace("−","-", regex=False)
        s2 = s2.apply(lambda x: _NUM_SEP_RE.sub("", x) if x is not None else x)
        return pd.to_numeric(s2, errors="coerce")
    return s

def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = _maybe_to_numeric(df[c])
    return df

def coerce_date_column(df: pd.DataFrame, candidates: List[str], new_name: str,
                       source_label: str, required: bool = False) -> Optional[str]:
    col = find_and_rename(df, candidates, new_name, required=required, source_label=source_label)
    if col:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True, infer_datetime_format=True).dt.date
        if df[col].notna().sum() == 0:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=False, infer_datetime_format=True).dt.date
    return col

def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: List[str], how="left", suffixes=("_x","_y")):
    if right is None or right.empty:
        return left.copy()
    return left.merge(right, how=how, on=on, suffixes=suffixes)

def show_exception(e: Exception, title: str):
    with st.expander(f"⚠️ {title}", expanded=False):
        st.code("".join(traceback.format_exception(None, e, e.__traceback__)))

# -----------------------------------------------------------------------------
# 2) Header único + 2 filas + AUTODETECCIÓN
# -----------------------------------------------------------------------------
def _make_unique_headers(headers):
    clean = [(h or "").strip() for h in headers]
    out, seen = [], {}
    for i, h in enumerate(clean, start=1):
        base = h if h else f"col_{i}"
        if base in seen:
            seen[base] += 1
            out.append(f"{base}__{seen[base]}")
        else:
            seen[base] = 1
            out.append(base)
    return out

def _combine_two_header_rows(r1: List[str], r2: List[str]) -> List[str]:
    n = max(len(r1), len(r2))
    out = []
    for i in range(n):
        a = (r1[i] if i < len(r1) else "").strip()
        b = (r2[i] if i < len(r2) else "").strip()
        name = (a + " " + b).strip() if (a or b) else ""
        out.append(name)
    return out

def _looks_group_header(row_lower: List[str]) -> bool:
    words = ("base", "e1", "spot", "back", "up", "sdd")
    hits = sum(1 for c in row_lower if any(w in c for w in words))
    return hits >= max(2, len(row_lower)//6)

# -----------------------------------------------------------------------------
# 3) Carga desde Google Sheets con header autodetectado
# -----------------------------------------------------------------------------
def _get_gspread_client():
    import gspread
    from google.oauth2.service_account import Credentials
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw: raise RuntimeError("Faltan credenciales en GOOGLE_SERVICE_ACCOUNT_JSON.")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(json.loads(raw), scopes=scopes)
    return gspread.authorize(creds)

@st.cache_data(show_spinner=False, ttl=300)
def read_sheet(sheet_id: str, tab_name: str) -> pd.DataFrame:
    import gspread
    gc = _get_gspread_client()
    sh = gc.open_by_key(sanitize_sheet_id(sheet_id))
    try:
        ws = sh.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        return pd.DataFrame()

    values = ws.get_all_values()
    if not values: return pd.DataFrame()

    def _is_header_row(row: list[str]) -> bool:
        row_can = [_canon_name(c) for c in row]
        header_keys = {"svc", "svcs", "logisticcenterid", "facility", "lc"}
        return any(c in header_keys for c in row_can)

    header_idx = None
    limit = min(50, len(values))
    for i in range(limit):
        if _is_header_row(values[i]):
            header_idx = i
            break
    if header_idx is None:
        for i in range(limit):
            nonempty = sum(1 for x in values[i] if x.strip())
            if nonempty >= 2:
                header_idx = i
                break
    if header_idx is None:
        header_idx = 0

    r1 = values[header_idx]
    r1_lower = [c.strip().lower() for c in r1]

    combine = False
    if header_idx + 1 < len(values):
        r2 = values[header_idx + 1]
        r2_nonempty = sum(1 for x in r2 if x.strip())
        if _looks_group_header(r1_lower) and r2_nonempty >= max(2, len(r2)//4):
            combine = True

    if combine:
        header = _combine_two_header_rows(values[header_idx], values[header_idx+1])
        data_rows = values[header_idx+2:]
    else:
        header = values[header_idx]
        data_rows = values[header_idx+1:]

    header = _make_unique_headers(header)
    df = pd.DataFrame(data_rows, columns=header)
    return coerce_numeric_df(df)

def quick_healthcheck(sheet_id: str) -> Dict[str, str]:
    out = {"sheet_id": sanitize_sheet_id(sheet_id) or "", "ok": "false", "note": ""}
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(sanitize_sheet_id(sheet_id))
        titles = [w.title for w in sh.worksheets()]
        out["ok"] = "true"
        out["note"] = f"Pestañas: {', '.join(titles[:12])}" + ("…" if len(titles) > 12 else "")
    except Exception as e:
        out["note"] = f"{e}"
    return out

# -----------------------------------------------------------------------------
# 4) Loaders
# -----------------------------------------------------------------------------
def _finalize(df: pd.DataFrame, wanted: List[str]) -> pd.DataFrame:
    df = ensure_columns(df, {"FECHA": pd.NaT, "SVC": None})
    cols = [c for c in wanted if c in df.columns]
    return df[cols].copy() if cols else pd.DataFrame(columns=wanted)

def load_fcst() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["fcst"])
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","FCST"])
    coerce_date_column(df, ["FECHA","Fecha","DATE","OP_DT"], "FECHA", "FCST", required=False)
    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","FACILITY","LC"], "SVC", False, "FCST")
    find_and_rename(df, ["Shipments","SHIPMENTS","FCST","Forecast","PLAN"], "FCST", False, "FCST")
    df = ensure_columns(df, {"FCST":0})
    hoy = date.today()
    if "FECHA" in df.columns:
        df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce").dt.date
        df = df[df["FECHA"].notna()]
        df = df[df["FECHA"] <= hoy] if (df["FECHA"] <= hoy).any() else df
        if not df.empty:
            idx = df.groupby("SVC")["FECHA"].idxmax()
            df = df.loc[idx]
    return _finalize(df, ["FECHA","SVC","FCST"])

def load_spr() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["spr"])
    if df.empty: return pd.DataFrame(columns=["FECHA","SVC","SPR_OBJ","SPR_PEAK","SPR_PROM"])
    coerce_date_column(df, ["FECHA","Fecha","DATE","OP_DT"], "FECHA", "SPR", required=False)
    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","FACILITY","LC"], "SVC", False, "SPR")
    find_and_rename(df, ["SPR","Spr"], "SPR_VAL", False, "SPR")
    df = ensure_columns(df, {"SPR_VAL": np.nan})
    g = df.groupby("SVC")["SPR_VAL"]
    out = g.agg(SPR_PROM="mean", SPR_PEAK=lambda x: np.nanpercentile(x.dropna(), 95) if x.notna().any() else np.nan).reset_index()
    out["SPR_OBJ"] = out["SPR_PROM"]
    out["FECHA"] = date.today()
    return _finalize(out, ["FECHA","SVC","SPR_OBJ","SPR_PEAK","SPR_PROM"])

# ---- Rentals (homologación + SPR histórico ponderado) ----
def _norm_txt(x: str) -> str:
    x = (x or "").strip().lower()
    rep = {"eléctrica":"electric","eléctrico":"electric","electrica":"electric","electrico":"electric"}
    for a,b in rep.items(): x = x.replace(a,b)
    x = re.sub(r"\brental(s)?\b", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def homologar_vehicle_type(raw: str) -> str:
    t = _norm_txt(str(raw))
    if re.search(r"\blarge\s+van\s+electric\b", t): return "Large Van Electric"
    if re.search(r"\blarge\s+van\b", t):           return "Large Van"
    if re.search(r"\bsmall\s+van\s+electric\b", t):return "Small Van Electric"
    if re.search(r"\bsmall\s+van\b", t):           return "Small Van"
    if re.search(r"\blarge.*electric\b", t):       return "Large Van Electric"
    if re.search(r"\bsmall.*electric\b", t):       return "Small Van Electric"
    if "large" in t:                               return "Large Van"
    if "small" in t:                               return "Small Van"
    return "Large Van"

def load_spr_hist_from_sheet() -> pd.DataFrame:
    spr = read_sheet(SHEET_ID, SHEET_TABS["spr"])
    if spr.empty:
        return pd.DataFrame(columns=["SVC", "VEHICULO_TIPO_HOM", "SPR_HIST"])
    find_and_rename(spr, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", False, "SPR")
    find_and_rename(spr, ["SHP_LG_VEHICLE_TYPE","Vehicle type","Tipo de vehículo","Tipo de vehiculo"], "SHP_LG_VEHICLE_TYPE", False, "SPR")
    find_and_rename(spr, ["SPR","spr","Ships per route"], "SPR", False, "SPR")
    spr = ensure_columns(spr, {"SVC": None, "SHP_LG_VEHICLE_TYPE":"", "SPR": np.nan})
    spr["SPR"] = pd.to_numeric(spr["SPR"], errors="coerce")

    spr = _as_str_cols(spr, ["SVC", "SHP_LG_VEHICLE_TYPE"])
    spr["VEHICULO_TIPO_HOM"] = spr["SHP_LG_VEHICLE_TYPE"].map(homologar_vehicle_type)

    grp_local = spr.groupby(["SVC","VEHICULO_TIPO_HOM"], dropna=False)["SPR"].median().reset_index().rename(columns={"SPR":"SPR_HIST"})
    grp_glob  = spr.groupby("VEHICULO_TIPO_HOM")["SPR"].median().rename("SPR_GLOBAL_TIPO").reset_index()

    out = grp_local.merge(grp_glob, on="VEHICULO_TIPO_HOM", how="right")
    out["SVC"] = out["SVC"].fillna("__GLOBAL__")
    out["SPR_HIST"] = out["SPR_HIST"].fillna(out["SPR_GLOBAL_TIPO"])
    out = out.drop(columns=["SPR_GLOBAL_TIPO"])
    out = _as_str_cols(out, ["SVC","VEHICULO_TIPO_HOM"])
    return out

def load_rentals_caps_from_sheet() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["rentals"])
    if df.empty:
        return pd.DataFrame(columns=["SVC","RUTAS_RENTALS","SPR_RENTALS"])

    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", False, "Rentals")
    find_and_rename(df, ["Tipo de vehiculo","Tipo de vehículo","Vehicle type","Tipo"], "TIPO_VEHICULO", False, "Rentals")
    df = ensure_columns(df, {"SVC": None, "TIPO_VEHICULO":"", })

    # Columna UNIDADES (exactos + fuzzy)
    units_col = None
    for keys in [["Unidades","Unidades dispon","Unidades disponibles"],
                 ["Units","Cantidad","Qty","QTY","COUNT"]]:
        for k in keys:
            col = find_and_rename(df, [k], "UNIDADES", required=False, source_label="Rentals")
            if col: units_col = "UNIDADES"; break
        if units_col: break
    if not units_col:
        guessed = _find_units_like_column(df)
        if guessed:
            if guessed != "UNIDADES":
                df.rename(columns={guessed:"UNIDADES"}, inplace=True)
            units_col = "UNIDADES"
    if not units_col:
        return pd.DataFrame(columns=["SVC","RUTAS_RENTALS","SPR_RENTALS"])

    df["UNIDADES"] = pd.to_numeric(df["UNIDADES"], errors="coerce").fillna(0)
    df = _as_str_cols(df, ["SVC","TIPO_VEHICULO"])
    df["VEHICULO_TIPO_HOM"] = df["TIPO_VEHICULO"].map(homologar_vehicle_type)

    by_type = df.groupby(["SVC","VEHICULO_TIPO_HOM"], dropna=False)["UNIDADES"].sum().reset_index()
    by_type = _as_str_cols(by_type, ["SVC","VEHICULO_TIPO_HOM"])

    spr_hist = load_spr_hist_from_sheet()
    spr_loc  = _as_str_cols(spr_hist[spr_hist["SVC"] != "__GLOBAL__"], ["SVC","VEHICULO_TIPO_HOM"])
    spr_glob = _as_str_cols(spr_hist[spr_hist["SVC"] == "__GLOBAL__"].drop(columns=["SVC"]).rename(columns={"SPR_HIST":"SPR_GLOBAL_TIPO"}), ["VEHICULO_TIPO_HOM"])

    by_type = by_type.merge(spr_loc,  on=["SVC","VEHICULO_TIPO_HOM"], how="left") \
                     .merge(spr_glob, on=["VEHICULO_TIPO_HOM"],      how="left")
    by_type["SPR_HIST"] = by_type["SPR_HIST"].fillna(by_type["SPR_GLOBAL_TIPO"])
    by_type.drop(columns=["SPR_GLOBAL_TIPO"], inplace=True)

    rentals_sum = by_type.groupby("SVC", dropna=False)["UNIDADES"].sum().rename("RUTAS_RENTALS").reset_index()

    by_type["PESO"] = by_type["UNIDADES"]
    by_type["POND"] = by_type["UNIDADES"] * by_type["SPR_HIST"]
    spr_r = by_type.groupby("SVC", dropna=False)[["POND","PESO"]].sum().reset_index()
    spr_r["SPR_RENTALS"] = (spr_r["POND"] / spr_r["PESO"]).replace([np.inf,-np.inf], np.nan)

    out = rentals_sum.merge(spr_r[["SVC","SPR_RENTALS"]], on="SVC", how="left")
    out = _as_str_cols(out, ["SVC"])

    # Rellena con fallback simple por SVC si existiera
    rents_fb = load_rentals_fallback()
    if not rents_fb.empty:
        rents_fb = _as_str_cols(rents_fb, ["SVC"])
        out = out.merge(rents_fb[["SVC","RUTAS_RENTALS"]].rename(columns={"RUTAS_RENTALS":"RUTAS_RENTALS_FB"}),
                        on="SVC", how="outer")
        out["RUTAS_RENTALS"] = out["RUTAS_RENTALS"].fillna(out["RUTAS_RENTALS_FB"]).fillna(0)
        out.drop(columns=["RUTAS_RENTALS_FB"], inplace=True)

    return out[["SVC","RUTAS_RENTALS","SPR_RENTALS"]]

def load_rentals_fallback() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["rentals"])
    target_cols = ["FECHA","SVC","RUTAS_RENTALS"]
    if df.empty:
        return pd.DataFrame(columns=target_cols)

    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","FACILITY","LC"], "SVC", required=False, source_label="Rentals")
    if "SVC" not in df.columns:
        out = pd.DataFrame(columns=target_cols)
        out["FECHA"] = date.today()
        return out

    find_and_rename(df, ["Unidades disponibles","Unidades dispon","Units","Cantidad","Qty","QTY","COUNT"], "CANT", required=False, source_label="Rentals")
    df = ensure_columns(df, {"CANT":0})
    df["CANT"] = pd.to_numeric(df["CANT"], errors="coerce").fillna(0)

    out = df.groupby("SVC", as_index=False)["CANT"].sum().rename(columns={"CANT":"RUTAS_RENTALS"})
    out["FECHA"] = date.today()
    return _finalize(out, target_cols)

from datetime import date as _date

def load_crowd_caps_for(op_date: _date) -> pd.DataFrame:
    """
    Lee 'Crowd' y entrega, por SVC:
      - BASE_ENTRE_SEM, BASE_SAB, BASE_DOM
      - E1_ENTRE_SEM,  E1_SAB,  E1_DOM  (Holgura - Base, acotado a >=0)
      - RUTAS_CROWD_CAP  (Base del día op_date)
      - CROWD_E1_CAP     (Holgura del día - Base del día, >=0)
      - FECHA = op_date
    """
    wanted = [
        "SVC",
        "BASE_ENTRE_SEM","BASE_SAB","BASE_DOM",
        "E1_ENTRE_SEM","E1_SAB","E1_DOM",
        "RUTAS_CROWD_CAP","CROWD_E1_CAP","FECHA"
    ]

    df = read_sheet(SHEET_ID, SHEET_TABS["crowd"])
    if df.empty:
        return pd.DataFrame(columns=wanted)

    # --- SVC robusto ---
    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","FACILITY","LC"], "SVC", required=False, source_label="Crowd")
    if "SVC" not in df.columns:
        cmap = {_canon_name(c): c for c in df.columns}
        for key, real in cmap.items():
            if key.startswith("svc") or key in {"svcs","logisticcenterid","facility","lc"}:
                if real != "SVC":
                    df.rename(columns={real: "SVC"}, inplace=True)
                break
    if "SVC" not in df.columns:
        return pd.DataFrame(columns=wanted)
    _as_str_cols(df, ["SVC"])

    # --- ayudante para renombrado tolerante a headers fusionados ---
    def rename_fuzzy(variants: list[str], new_name: str):
        # 1) intento exacto con nuestro find_and_rename
        found = find_and_rename(df, variants, new_name, required=False, source_label="Crowd")
        if found:
            return
        # 2) intento "fuzzy": coincide si el canon contiene/termina con el objetivo
        cmap = {_canon_name(c): c for c in df.columns}
        targets = [_canon_name(v) for v in variants]
        for can, real in cmap.items():
            if any(can.endswith(t) or t in can for t in targets):
                if real != new_name:
                    df.rename(columns={real: new_name}, inplace=True)
                return

    # --- claves (incluyen variantes con el prefijo del grupo "Base"/"E1") ---
    base_sem_keys = ["Base entre semana","Base entre sem","Base semana","Base entre sem.", "Base Base entre semana"]
    base_sab_keys = ["Base sabado","Base sábado"]
    base_dom_keys = ["Base domingo"]

    holg_sem_keys = ["Holgura entre semana","Holgura entre sem","Holgura semana", "E1 Holgura entre semana"]
    holg_sab_keys = ["Holgura sabado","Holgura sábado"]
    holg_dom_keys = ["Holgura domingo"]

    rename_fuzzy(base_sem_keys, "BASE_SEM")
    rename_fuzzy(base_sab_keys, "BASE_SAB")
    rename_fuzzy(base_dom_keys, "BASE_DOM")
    rename_fuzzy(holg_sem_keys, "HOLG_SEM")
    rename_fuzzy(holg_sab_keys, "HOLG_SAB")
    rename_fuzzy(holg_dom_keys, "HOLG_DOM")

    # --- asegurar numérico e inexistentes en 0 ---
    for c in ["BASE_SEM","BASE_SAB","BASE_DOM","HOLG_SEM","HOLG_SAB","HOLG_DOM"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # --- E1 = Holgura - Base (no negativo) ---
    df["E1_SEM"] = (df["HOLG_SEM"] - df["BASE_SEM"]).clip(lower=0).astype(int)
    df["E1_SAB"] = (df["HOLG_SAB"] - df["BASE_SAB"]).clip(lower=0).astype(int)
    df["E1_DOM"] = (df["HOLG_DOM"] - df["BASE_DOM"]).clip(lower=0).astype(int)

    # --- Día de operación ---
    wd = op_date.weekday()  # 0..6 => Lun..Dom
    if wd <= 4:
        base_sel = df["BASE_SEM"]
        e1_sel   = df["E1_SEM"]
    elif wd == 5:
        base_sel = df["BASE_SAB"]
        e1_sel   = df["E1_SAB"]
    else:
        base_sel = df["BASE_DOM"]
        e1_sel   = df["E1_DOM"]

    out = pd.DataFrame({
        "SVC": df["SVC"].astype(str).values,
        "BASE_ENTRE_SEM": df["BASE_SEM"].astype(int).values,
        "BASE_SAB":       df["BASE_SAB"].astype(int).values,
        "BASE_DOM":       df["BASE_DOM"].astype(int).values,
        "E1_ENTRE_SEM":   df["E1_SEM"].astype(int).values,
        "E1_SAB":         df["E1_SAB"].astype(int).values,
        "E1_DOM":         df["E1_DOM"].astype(int).values,
        "RUTAS_CROWD_CAP": pd.to_numeric(base_sel, errors="coerce").fillna(0).astype(int).values,
        "CROWD_E1_CAP":    pd.to_numeric(e1_sel,   errors="coerce").fillna(0).astype(int).values,
    })
    out["FECHA"] = op_date
    return out[wanted]



def load_crowd_pct_from_capacity() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["capacity"])
    if df.empty:
        return pd.DataFrame(columns=["SVC", "CROWD_PCT"])

    find_and_rename(df, ["Delivery model","Deliverymodel","Model","DM"], "DELIVERY_MODEL", False, "Capacity")
    find_and_rename(df, ["Tipo","Type","Category"], "TIPO", False, "Capacity")
    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","FACILITY","LC"], "SVC", False, "Capacity")
    find_and_rename(df, ["Tipo DM","TipoDM","DM Type"], "TIPO_DM", False, "Capacity")
    coerce_date_column(df, ["Fecha","FECHA","Date","OP_DT"], "FECHA", "Capacity", required=False)
    find_and_rename(df, ["Cantidad","Qty","Quantity","COUNT","QTY"], "CANT", False, "Capacity")

    df = ensure_columns(df, {"DELIVERY_MODEL":"", "TIPO":"", "TIPO_DM":"", "SVC":None, "FECHA": pd.NaT, "CANT":0})
    df["CANT"] = pd.to_numeric(df["CANT"], errors="coerce").fillna(0)
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce").dt.date
    _as_str_cols(df, ["SVC", "DELIVERY_MODEL", "TIPO", "TIPO_DM"])  # <- asegura str

    # toma último día por SVC si hay fechas
    if df["FECHA"].notna().any():
        last_by_svc = df.groupby("SVC")["FECHA"].transform("max")
        df = df[df["FECHA"] == last_by_svc]

    # normaliza textos
    dm     = df["DELIVERY_MODEL"].fillna("").astype(str)
    tipo   = df["TIPO"].fillna("").astype(str)
    tipodm = df["TIPO_DM"].fillna("").astype(str)

    is_shipments = tipo.str.contains("ship", case=False, regex=False)
    is_crowd = (
        dm.str.contains("crowd", case=False, regex=False) |
        tipodm.str.contains("crowd", case=False, regex=False) |
        tipo.str.contains("crowd", case=False, regex=False)
    )
    is_crowd = is_shipments & is_crowd

    tot = df[is_shipments].groupby("SVC", dropna=False)["CANT"].sum().rename("SHIP_TOT")
    crd = df[is_crowd].groupby("SVC", dropna=False)["CANT"].sum().rename("SHIP_CROWD")

    out = pd.concat([tot, crd], axis=1).reset_index()
    _as_str_cols(out, ["SVC"])
    out["SHIP_TOT"]   = pd.to_numeric(out["SHIP_TOT"], errors="coerce").fillna(0)
    out["SHIP_CROWD"] = pd.to_numeric(out["SHIP_CROWD"], errors="coerce").fillna(0)
    out["CROWD_PCT"]  = (out["SHIP_CROWD"] / out["SHIP_TOT"]).replace([np.inf,-np.inf], 0).fillna(0)
    return out[["SVC", "CROWD_PCT"]]


def load_plan_spr_from_capacity(op_date: date) -> pd.DataFrame:
    """
    SPR 'plan' por SVC y Delivery Model a partir de la pestaña Capacity.
    Regla: filtra Tipo = SPR, Delivery model/Tipo DM = Crowd (fuzzy),
    toma el registro del día op_date; si no existe, usa el último <= op_date.
    Devuelve columnas: SVC, SPR_CROWD_PLAN
    """
    df = read_sheet(SHEET_ID, SHEET_TABS["capacity"])
    if df.empty:
        return pd.DataFrame(columns=["SVC", "SPR_CROWD_PLAN"])

    # Normalización de columnas
    find_and_rename(df, ["Delivery model","Deliverymodel","Model","DM"], "DELIVERY_MODEL", False, "Capacity")
    find_and_rename(df, ["Tipo","Type","Category"], "TIPO", False, "Capacity")
    find_and_rename(df, ["Tipo DM","TipoDM","DM Type"], "TIPO_DM", False, "Capacity")
    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","FACILITY","LC"], "SVC", False, "Capacity")
    find_and_rename(df, ["Fecha","FECHA","Date","OP_DT"], "FECHA", False, "Capacity")
    find_and_rename(df, ["Cantidad","Qty","Quantity","COUNT","QTY"], "CANT", False, "Capacity")

    df = ensure_columns(df, {"DELIVERY_MODEL":"", "TIPO":"", "TIPO_DM":"", "SVC":None, "FECHA": pd.NaT, "CANT":np.nan})
    _as_str_cols(df, ["SVC","DELIVERY_MODEL","TIPO","TIPO_DM"])
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce").dt.date
    df["CANT"]  = pd.to_numeric(df["CANT"], errors="coerce")

    # Filtros: Tipo = SPR y Crowd por cualquier columna relacionada
    tip  = df["TIPO"].str.lower()
    dm   = df["DELIVERY_MODEL"].str.lower()
    tdm  = df["TIPO_DM"].str.lower()
    is_spr   = tip.str.contains("spr", regex=False)
    is_crowd = dm.str.contains("crowd", regex=False) | tdm.str.contains("crowd", regex=False) | tip.str.contains("crowd", regex=False)

    sub = df[is_spr & is_crowd].copy()
    if sub.empty:
        return pd.DataFrame(columns=["SVC","SPR_CROWD_PLAN"])

    # Para cada SVC, escoger el registro del día; si no hay, el último <= op_date; si tampoco, el más reciente
    def pick_for_svc(g):
        g = g.sort_values("FECHA")
        exact = g[g["FECHA"] == op_date]
        if not exact.empty:
            return exact.tail(1)
        leq = g[g["FECHA"].notna() & (g["FECHA"] <= op_date)]
        if not leq.empty:
            return leq.tail(1)
        return g.tail(1)

    picked = sub.groupby("SVC", group_keys=False).apply(pick_for_svc)
    out = picked[["SVC","CANT"]].rename(columns={"CANT":"SPR_CROWD_PLAN"}).reset_index(drop=True)
    out["SPR_CROWD_PLAN"] = pd.to_numeric(out["SPR_CROWD_PLAN"], errors="coerce")
    return _as_str_cols(out, ["SVC"])



def load_spr_crowd() -> pd.DataFrame:
    spr = read_sheet(SHEET_ID, SHEET_TABS["spr"])
    if spr.empty:
        return pd.DataFrame(columns=["SVC","SPR_CROWD"])
    find_and_rename(spr, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", False, "SPR")
    find_and_rename(spr, ["SPR","spr","Ships per route"], "SPR", False, "SPR")
    find_and_rename(spr, ["Delivery model","Deliverymodel","Model","DM"], "DELIVERY_MODEL", False, "SPR")
    find_and_rename(spr, ["Tipo","Type","Category"], "TIPO", False, "SPR")
    find_and_rename(spr, ["SHP_LG_VEHICLE_TYPE","Vehicle type","Tipo de vehículo","Tipo de vehiculo"], "VEH_TYPE", False, "SPR")

    spr = ensure_columns(spr, {"SVC":None, "SPR":np.nan, "DELIVERY_MODEL":"", "TIPO":"", "VEH_TYPE":""})
    spr["SPR"] = pd.to_numeric(spr["SPR"], errors="coerce")
    spr = _as_str_cols(spr, ["SVC","DELIVERY_MODEL","TIPO","VEH_TYPE"])

    is_crowd = spr["DELIVERY_MODEL"].str.lower().str.contains("crowd", regex=False) \
               | spr["TIPO"].str.lower().str.contains("crowd", regex=False) \
               | spr["VEH_TYPE"].str.lower().str.contains("crowd", regex=False)

    spr_crowd = spr[is_crowd].copy()
    if spr_crowd.empty:
        return pd.DataFrame(columns=["SVC","SPR_CROWD"])

    grp_local = spr_crowd.groupby("SVC")["SPR"].median().rename("SPR_CROWD").reset_index()
    global_val = spr_crowd["SPR"].median()
    grp_local["SPR_CROWD"] = grp_local["SPR_CROWD"].fillna(global_val)
    return _as_str_cols(grp_local, ["SVC"])

# ---- Capacity caps (MLP/Crowd caps + Shipments DC/SP) ----
def load_capacity_caps() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, SHEET_TABS["capacity"])
    wanted = ["FECHA","SVC","RUTAS_MLP_SDD","RUTAS_MLP_SPOT","RUTAS_RENTALS","RUTAS_CROWD_CAP","SHIPMENTS_DC","SHIPMENTS_SP"]
    if df.empty:
        return pd.DataFrame(columns=wanted)

    find_and_rename(df, ["Delivery model","Deliverymodel","Model","DM"], "DELIVERY_MODEL", False, "Capacity")
    find_and_rename(df, ["Tipo","Type","Category"], "TIPO", False, "Capacity")
    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","FACILITY","LC"], "SVC", False, "Capacity")
    find_and_rename(df, ["Tipo DM","TipoDM","DM Type"], "TIPO_DM", False, "Capacity")
    coerce_date_column(df, ["Fecha","FECHA","Date","OP_DT"], "FECHA", "Capacity", required=False)
    find_and_rename(df, ["Cantidad","Qty","Quantity","COUNT","QTY"], "CANT", False, "Capacity")
    df = ensure_columns(df, {"DELIVERY_MODEL":"", "TIPO":"", "TIPO_DM":"", "SVC":None, "FECHA": pd.NaT, "CANT":0})
    df["CANT"] = pd.to_numeric(df["CANT"], errors="coerce").fillna(0)

    hoy = date.today()
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce").dt.date
    if df["FECHA"].notna().any():
        sub = df[df["FECHA"].notna()]
        sub = sub[sub["FECHA"] <= hoy]
        target = sub["FECHA"].max() if not sub.empty else df["FECHA"].max()
        df = df[df["FECHA"] == target]
    else:
        df["FECHA"] = hoy

    dm_norm     = df["DELIVERY_MODEL"].fillna("").astype(str).str.lower()
    tipo_norm   = df["TIPO"].fillna("").astype(str).str.lower()
    tipodm_norm = df["TIPO_DM"].fillna("").astype(str).str.lower()

    is_rentals = dm_norm.str.contains("rent",  regex=False)
    is_crowd_routes = dm_norm.str.contains("crowd", regex=False) & (
                        tipo_norm.str.contains("route", regex=False) |
                        tipodm_norm.str.contains("route", case=False, regex=False)
                      )
    is_mlp_spot = dm_norm.str.contains("mlp", regex=False) & tipodm_norm.str.contains("spot", regex=False)
    is_mlp_sdd  = dm_norm.str.contains("mlp", regex=False) & (~is_mlp_spot) & (
                        tipodm_norm.str.contains("mlp", regex=False) |
                        tipodm_norm.str.contains("sdd", regex=False) |
                        (tipodm_norm == "")
                  )

    is_shipments = tipo_norm.str.contains("ship", regex=False)
    is_dc = (
        (dm_norm.str.contains("delivery", regex=False) & dm_norm.str.contains("cell", regex=False)) |
        tipodm_norm.str.contains("delivery cell", regex=False) |
        tipodm_norm.str.contains(r"^(dc|cell)$", case=False, regex=True)
    )
    is_sp = (
        dm_norm.str.contains(r"^(s\.?p\.?|sp)$", case=False, regex=True) |
        (dm_norm.str.contains("service", regex=False) & dm_norm.str.contains("partner", regex=False)) |
        tipodm_norm.str.contains(r"\bsp\b|service partner", case=False, regex=True)
    )
    is_dc_ship = is_shipments & is_dc
    is_sp_ship = is_shipments & is_sp

    g = df.groupby(["FECHA","SVC"])["CANT"]
    agg = pd.DataFrame({
        "RUTAS_MLP_SDD":   g.apply(lambda s: s[is_mlp_sdd.loc[s.index]].sum()),
        "RUTAS_MLP_SPOT":  g.apply(lambda s: s[is_mlp_spot.loc[s.index]].sum()),
        "RUTAS_RENTALS":   g.apply(lambda s: s[is_rentals.loc[s.index]].sum()),
        "RUTAS_CROWD_CAP": g.apply(lambda s: s[is_crowd_routes.loc[s.index]].sum()),
        "SHIPMENTS_DC":    g.apply(lambda s: s[is_dc_ship.loc[s.index]].sum()),
        "SHIPMENTS_SP":    g.apply(lambda s: s[is_sp_ship.loc[s.index]].sum()),
    }).reset_index()

    return _finalize(agg, wanted)

# ---- NUEVO: MLP caps desde SRM (SDD/SPOT/BACKLOG) — SIN “TOTAL” y con desglose por tipo ----

# ---- MLP caps desde SRM (robusto: SDD/SPOT/Back Up; ignora "Total") ----
def load_mlp_caps_from_srm() -> pd.DataFrame:
    """
    Lee la pestaña SRM y arma capacidades MLP por SVC:
      SDD:  MLP_SDD_LV, MLP_SDD_SV, MLP_SDD_CAR, MLP_SDD_CAP
      SPOT: MLP_SPOT_LV, MLP_SPOT_SV, MLP_SPOT_CAR, MLP_SPOT_CAP
      BACK: MLP_BACK_CAP  (Back/Backup/BU/Backlog)
    Ignora columnas 'Total ...' para no contar doble.
    También arrastra la columna textual 'MLP' (proveedor) como referencia.
    """
    df = read_sheet(SHEET_ID, SHEET_TABS["srm"])
    out_cols = [
        "SVC",
        "MLP",  # referencia textual (proveedor)
        "MLP_SDD_LV","MLP_SDD_SV","MLP_SDD_CAR","MLP_SDD_CAP",
        "MLP_SPOT_LV","MLP_SPOT_SV","MLP_SPOT_CAR","MLP_SPOT_CAP",
        "MLP_BACK_CAP",
    ]
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    # --- SVC (con fallback fuzzy: acepta "SVC SCV1", "SVC (ID)", etc.) ---
    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", required=False, source_label="SRM")
    if "SVC" not in df.columns:
        cmap = {_canon_name(c): c for c in df.columns}
        for key, real in cmap.items():
            if key.startswith("svc"):
                if real != "SVC":
                    df.rename(columns={real: "SVC"}, inplace=True)
                break
    if "SVC" not in df.columns:
        return pd.DataFrame(columns=out_cols)
    df = _as_str_cols(df, ["SVC"])

    # --- Proveedor MLP (solo referencia textual) ---
    find_and_rename(df, ["MLP","Proveedor","Carrier","Proveedor MLP","Partner"], "MLP", required=False, source_label="SRM")
    if "MLP" not in df.columns:
        df["MLP"] = ""
    else:
        df["MLP"] = df["MLP"].astype(str).str.strip()

    # ---------- Canon de columnas para matching ----------
    def canon_col(name: str) -> str:
        c = _canon_name(name)           # quita espacios/acentos
        c = c.replace("h&b", "hb")
        c = re.sub(r"w\d+", "", c)      # quita W36, W37...
        c = re.sub(r"\d+$", "", c)      # quita número suelto final (“ … 3”)
        return c

    canon = {c: canon_col(c) for c in df.columns}

    def is_not_svc(cc: str) -> bool:
        return cc not in ("svc", "svcs", "logisticcenterid", "facility", "lc", "mlp")

    def has(cc: str, token: str) -> bool:
        return token in cc

    def has_any(cc: str, tokens: list[str]) -> bool:
        return any(t in cc for t in tokens)

    def pick_cols(type_tokens: list[str], family_tokens: list[str], exclude_tokens: list[str]) -> list[str]:
        sel = []
        for col, cc in canon.items():
            if not is_not_svc(cc):
                continue
            if all(has(cc, ft) for ft in family_tokens) and has_any(cc, type_tokens) and not has_any(cc, exclude_tokens):
                sel.append(col)
        return sel

    def pick_cols_any(family_tokens: list[str], include_any: list[str], exclude_tokens: list[str]) -> list[str]:
        sel = []
        for col, cc in canon.items():
            if not is_not_svc(cc):
                continue
            if all(has(cc, ft) for ft in family_tokens) and has_any(cc, include_any) and not has_any(cc, exclude_tokens):
                sel.append(col)
        return sel

    # Tokens (amplios) para detectar tipo/familia
    LV  = ["largevan","largev","large","lv","xlarge","xlv","heavybulky","hb"]
    SV  = ["smallvan","small","sv"]
    CAR = ["car","auto"]

    EXC_TOTAL = ["total"]
    EXC_BACK  = ["bu","backup","back","backlog"]
    BACK_ANY  = ["bu","backup","back","backlog"]

    # Columnas por familia/tipo
    sdd_lv_cols   = pick_cols(LV,  ["sdd"], EXC_TOTAL + [])
    sdd_sv_cols   = pick_cols(SV,  ["sdd"], EXC_TOTAL + [])
    sdd_car_cols  = pick_cols(CAR, ["sdd"], EXC_TOTAL + [])

    spot_lv_cols  = pick_cols(LV,  ["spot"], EXC_TOTAL + EXC_BACK)
    spot_sv_cols  = pick_cols(SV,  ["spot"], EXC_TOTAL + EXC_BACK)
    spot_car_cols = pick_cols(CAR, ["spot"], EXC_TOTAL + EXC_BACK)

    back_cols     = pick_cols_any(["spot"], BACK_ANY, EXC_TOTAL)

    # Asegura numérico en todas las columnas sumables
    for c in set(sdd_lv_cols + sdd_sv_cols + sdd_car_cols +
                 spot_lv_cols + spot_sv_cols + spot_car_cols + back_cols):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    grp = df.groupby("SVC", dropna=False)

    def sum_cols(cols: list[str]) -> pd.Series:
        if not cols:
            return grp.size().mul(0)
        # suma por columna dentro de cada grupo y luego por fila
        return grp[cols].sum().sum(axis=1)

    # Agrega referencia MLP (concatena nombres únicos por SVC)
    mlp_ref = grp["MLP"].apply(
        lambda s: ", ".join(pd.Series(s).astype(str).str.strip().replace("", np.nan).dropna().unique()[:5])
    ).rename("MLP").reset_index()

    out = pd.DataFrame({"SVC": grp.size().index}).reset_index(drop=True)

    out["MLP_SDD_LV"]   = sum_cols(sdd_lv_cols).values
    out["MLP_SDD_SV"]   = sum_cols(sdd_sv_cols).values
    out["MLP_SDD_CAR"]  = sum_cols(sdd_car_cols).values
    out["MLP_SDD_CAP"]  = (out["MLP_SDD_LV"] + out["MLP_SDD_SV"] + out["MLP_SDD_CAR"]).astype(int)

    out["MLP_SPOT_LV"]  = sum_cols(spot_lv_cols).values
    out["MLP_SPOT_SV"]  = sum_cols(spot_sv_cols).values
    out["MLP_SPOT_CAR"] = sum_cols(spot_car_cols).values
    out["MLP_SPOT_CAP"] = (out["MLP_SPOT_LV"] + out["MLP_SPOT_SV"] + out["MLP_SPOT_CAR"]).astype(int)

    out["MLP_BACK_CAP"] = sum_cols(back_cols).astype(int).values

    # Tipado final
    for c in ["MLP_SDD_LV","MLP_SDD_SV","MLP_SDD_CAR",
              "MLP_SPOT_LV","MLP_SPOT_SV","MLP_SPOT_CAR",
              "MLP_SDD_CAP","MLP_SPOT_CAP","MLP_BACK_CAP"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).round(0).astype(int)

    # Adjunta MLP (texto)
    out = out.merge(mlp_ref, on="SVC", how="left")
    out["MLP"] = out["MLP"].fillna("")

    return out[out_cols]


# ---- NUEVO: SPR de MLP ----
def load_spr_mlp() -> pd.DataFrame:
    spr = read_sheet(SHEET_ID, SHEET_TABS["spr"])
    if spr.empty:
        return pd.DataFrame(columns=["SVC","SPR_MLP"])
    find_and_rename(spr, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", False, "SPR")
    find_and_rename(spr, ["SPR","spr","Ships per route"], "SPR", False, "SPR")
    find_and_rename(spr, ["Delivery model","Deliverymodel","Model","DM"], "DELIVERY_MODEL", False, "SPR")
    find_and_rename(spr, ["Tipo","Type","Category"], "TIPO", False, "SPR")
    find_and_rename(spr, ["SHP_LG_VEHICLE_TYPE","Vehicle type","Tipo de vehículo","Tipo de vehiculo"], "VEH_TYPE", False, "SPR")

    spr = ensure_columns(spr, {"SVC":None, "SPR":np.nan, "DELIVERY_MODEL":"", "TIPO":"", "VEH_TYPE":""})
    spr["SPR"] = pd.to_numeric(spr["SPR"], errors="coerce")
    spr = _as_str_cols(spr, ["SVC","DELIVERY_MODEL","TIPO","VEH_TYPE"])

    is_mlp = (
        spr["DELIVERY_MODEL"].str.lower().str.contains("mlp|sdd|spot|back", regex=True)
        | spr["TIPO"].str.lower().str.contains("mlp|sdd|spot|back", regex=True)
        | spr["VEH_TYPE"].str.lower().str.contains("mlp|sdd|spot|back", regex=True)
    )
    mlp_rows = spr[is_mlp]
    if mlp_rows.empty:
        return pd.DataFrame(columns=["SVC","SPR_MLP"])
    grp_local = mlp_rows.groupby("SVC")["SPR"].median().rename("SPR_MLP").reset_index()
    global_val = mlp_rows["SPR"].median()
    grp_local["SPR_MLP"] = grp_local["SPR_MLP"].fillna(global_val)
    return grp_local

# --- SPR por Delivery Model (4 semanas previas, mismo día de la semana) ---
def load_spr_dm_recent_weekday_stats(op_date: date) -> pd.DataFrame:
    """
    Devuelve SPR por SVC y Delivery Model (RENTALS/CROWD/MLP) con:
      - SPR_PROM4: mediana de los últimos 4 mismos días de la semana previos a op_date
      - SPR_P95_4: p95 del mismo set (≈max si son 4 puntos)
      - SPR_TODAY: valor del día op_date (si existe) para 'plan'
    """
    spr = read_sheet(SHEET_ID, SHEET_TABS["spr"])
    if spr.empty:
        return pd.DataFrame(columns=["SVC","DM","SPR_PROM4","SPR_P95_4","SPR_TODAY"])

    # Normaliza columnas
    find_and_rename(spr, ["DELIVERY_MODEL","Delivery model","Model","DM"], "DELIVERY_MODEL", False, "SPR")
    find_and_rename(spr, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", False, "SPR")
    find_and_rename(spr, ["SPR","spr","Ships per route"], "SPR", False, "SPR")
    coerce_date_column(spr, ["FECHA","Fecha","DATE","OP_DT"], "FECHA", "SPR", required=False)

    spr = ensure_columns(spr, {"DELIVERY_MODEL":"", "SVC":None, "FECHA": pd.NaT, "SPR": np.nan})
    spr["SPR"] = pd.to_numeric(spr["SPR"], errors="coerce")
    spr = _as_str_cols(spr, ["SVC","DELIVERY_MODEL"])
    spr["FECHA_TS"] = pd.to_datetime(spr["FECHA"], errors="coerce")
    spr["WD"] = spr["FECHA_TS"].dt.weekday

    # Bucket de Delivery Model → {RENTALS, CROWD, MLP}
    def dm_bucket(x: str) -> str:
        x = (x or "").lower()
        if "crowd" in x: return "CROWD"
        if "rent"  in x: return "RENTALS"
        if ("mlp" in x) or ("sdd" in x) or ("spot" in x) or ("back" in x): return "MLP"
        return "__OTHER__"

    spr["DM"] = spr["DELIVERY_MODEL"].map(dm_bucket)
    spr = spr[spr["DM"] != "__OTHER__"]

    # Ventana: 4 semanas previas, mismo weekday
    op_ts  = pd.Timestamp(op_date)
    start  = op_ts - pd.Timedelta(days=28)
    wd     = op_ts.weekday()
    mask4w = spr["FECHA_TS"].notna() & (spr["FECHA_TS"] < op_ts) & (spr["FECHA_TS"] >= start) & (spr["WD"] == wd)
    recent = spr[mask4w].copy()

    g = recent.groupby(["SVC","DM"])["SPR"]
    stats = pd.DataFrame({
        "SPR_PROM4": g.median(),
        "SPR_P95_4": g.apply(lambda x: np.nanpercentile(x.dropna(), 95) if x.notna().any() else np.nan)
    }).reset_index()

    # Valor del propio día (para 'plan')
    today = spr[spr["FECHA_TS"] == op_ts].copy()
    today_v = today.groupby(["SVC","DM"])["SPR"].median().rename("SPR_TODAY").reset_index()

    out = stats.merge(today_v, on=["SVC","DM"], how="outer")
    return out


# ---- SPR por Delivery Model (RENTALS/CROWD/MLP) según modo y día de análisis ----
def _norm_dm_label(s: str) -> str:
    s = (s or "").strip().lower()
    if re.search(r"crowd", s): return "CROWD"
    if re.search(r"rent",  s): return "RENTALS"
    # MLP / SDD / SPOT / Backlog en la misma canasta
    if re.search(r"mlp|sdd|spot|back", s): return "MLP"
    # DC / SP no participan en SPR de rutas (solo shipments)
    return ""

def _weekday_tail(df: pd.DataFrame, limit_date: date, same_weekday_only: bool = True) -> pd.DataFrame:
    """Filtra histórico <= limit_date (mismo weekday si aplica)."""
    if "FECHA" not in df.columns:
        return df.iloc[0:0]
    df2 = df.copy()
    df2["FECHA"] = pd.to_datetime(df2["FECHA"], errors="coerce").dt.date
    df2 = df2[df2["FECHA"].notna()]
    df2 = df2[df2["FECHA"] <= limit_date]
    if same_weekday_only:
        w = limit_date.weekday()
        df2 = df2[df2["FECHA"].apply(lambda d: d.weekday()) == w]
    return df2

def _agg_prom_last4_same_weekday(s: pd.Series) -> float:
    # Mediana de los últimos 4 puntos (si hay menos, con lo que haya)
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if vals.empty: 
        return np.nan
    if len(vals) > 4:
        vals = vals.iloc[-4:]
    return float(vals.median())

def _agg_peak_p95_same_weekday(s: pd.Series) -> float:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if vals.empty: 
        return np.nan
    return float(np.percentile(vals, 95))

@st.cache_data(show_spinner=False, ttl=300)
def load_spr_dm_by_mode(op_date: date, mode: str) -> pd.DataFrame:
    """
    Devuelve, por SVC:
      - SPR_RENTALS_SEL
      - SPR_CROWD_SEL
      - SPR_MLP_SEL
    Cálculo:
      - promedio: mediana de los **últimos 4** del mismo weekday (<= op_date)
      - peak:     p95 del mismo weekday (<= op_date)
      - plan:     último valor del mismo weekday (<= op_date)
    Fuente: pestaña SPR (col: DELIVERY_MODEL, FECHA, SVC, SPR)
    """
    df = read_sheet(SHEET_ID, SHEET_TABS["spr"])
    out_cols = ["SVC","SPR_RENTALS_SEL","SPR_CROWD_SEL","SPR_MLP_SEL"]
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    # Normalización mínima
    find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", False, "SPR")
    find_and_rename(df, ["SPR","spr","Ships per route"], "SPR", False, "SPR")
    find_and_rename(df, ["Delivery model","Deliverymodel","Model","DM"], "DELIVERY_MODEL", False, "SPR")
    coerce_date_column(df, ["FECHA","Fecha","DATE","OP_DT"], "FECHA", "SPR", required=False)

    if "SVC" not in df.columns or "SPR" not in df.columns or "DELIVERY_MODEL" not in df.columns:
        return pd.DataFrame(columns=out_cols)

    df = ensure_columns(df, {"SVC":None, "SPR":np.nan, "DELIVERY_MODEL":"", "FECHA": pd.NaT})
    df["SPR"] = pd.to_numeric(df["SPR"], errors="coerce")
    _as_str_cols(df, ["SVC","DELIVERY_MODEL"])
    df["DM"] = df["DELIVERY_MODEL"].map(_norm_dm_label)

    # Solo DM relevantes
    df = df[df["DM"].isin(["RENTALS","CROWD","MLP"])].copy()
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    # Filtrado temporal (<= día y mismo weekday)
    dfw = _weekday_tail(df, op_date, same_weekday_only=True).sort_values(["SVC","DM","FECHA"])
    if dfw.empty:
        return pd.DataFrame(columns=out_cols)

    # Agregadores por modo
    if mode == "promedio":
        agg_fun = _agg_prom_last4_same_weekday
    elif mode == "peak":
        agg_fun = _agg_peak_p95_same_weekday
    else:  # plan: último valor del mismo weekday
        def agg_fun(s):
            vals = pd.to_numeric(s, errors="coerce").dropna()
            return float(vals.iloc[-1]) if len(vals) else np.nan

    # Aplica agregación por SVC x DM
    stats = (
        dfw.groupby(["SVC","DM"], dropna=False)["SPR"]
           .apply(agg_fun)
           .reset_index()
           .rename(columns={"SPR":"SPR_SEL"})
    )

    # Pivot a columnas por DM
    piv = stats.pivot(index="SVC", columns="DM", values="SPR_SEL").reset_index().rename_axis(None, axis=1)
    # Renombra a columnas objetivo
    piv = piv.rename(columns={
        "RENTALS": "SPR_RENTALS_SEL",
        "CROWD":   "SPR_CROWD_SEL",
        "MLP":     "SPR_MLP_SEL",
    })

    # Asegura columnas
    for c in ["SPR_RENTALS_SEL","SPR_CROWD_SEL","SPR_MLP_SEL"]:
        if c not in piv.columns: piv[c] = np.nan

    return piv[["SVC","SPR_RENTALS_SEL","SPR_CROWD_SEL","SPR_MLP_SEL"]]


def load_spr_dm_stats_from_sheet() -> pd.DataFrame:
    """
    Lee la pestaña SPR y calcula, por SVC y por Delivery Model (RENTALS, CROWD, MLP),
    el SPR 'prom' (mediana) y 'peak' (p95).
    Devuelve columnas:
      SVC,
      SPR_RENTALS_PROM, SPR_RENTALS_PEAK,
      SPR_CROWD_PROM,   SPR_CROWD_PEAK,
      SPR_MLP_PROM,     SPR_MLP_PEAK
    """
    spr = read_sheet(SHEET_ID, SHEET_TABS["spr"])
    if spr.empty:
        return pd.DataFrame(columns=[
            "SVC",
            "SPR_RENTALS_PROM","SPR_RENTALS_PEAK",
            "SPR_CROWD_PROM","SPR_CROWD_PEAK",
            "SPR_MLP_PROM","SPR_MLP_PEAK",
        ])

    find_and_rename(spr, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", False, "SPR")
    find_and_rename(spr, ["SPR","spr","Ships per route"], "SPR", False, "SPR")
    find_and_rename(spr, ["Delivery model","Deliverymodel","Model","DM"], "DELIVERY_MODEL", False, "SPR")
    find_and_rename(spr, ["Tipo","Type","Category"], "TIPO", False, "SPR")
    find_and_rename(spr, ["SHP_LG_VEHICLE_TYPE","Vehicle type","Tipo de vehículo","Tipo de vehiculo"], "VEH_TYPE", False, "SPR")

    spr = ensure_columns(spr, {"SVC":None, "SPR":np.nan, "DELIVERY_MODEL":"", "TIPO":"", "VEH_TYPE":""})
    spr["SPR"] = pd.to_numeric(spr["SPR"], errors="coerce")
    spr = _as_str_cols(spr, ["SVC","DELIVERY_MODEL","TIPO","VEH_TYPE"])

    dm  = spr["DELIVERY_MODEL"].str.lower()
    tip = spr["TIPO"].str.lower()
    veh = spr["VEH_TYPE"].str.lower()

    is_crowd = dm.str.contains("crowd", regex=False) | tip.str.contains("crowd", regex=False) | veh.str.contains("crowd", regex=False)
    is_mlp   = dm.str.contains("mlp|sdd|spot|back", regex=True) | tip.str.contains("mlp|sdd|spot|back", regex=True) | veh.str.contains("mlp|sdd|spot|back", regex=True)
    is_rent  = dm.str.contains("rent", regex=False) | tip.str.contains("rent", regex=False) | veh.str.contains("rent", regex=False)

    def agg_stats(df):
        if df.empty:
            return pd.DataFrame(columns=["SVC","PROM","PEAK"])
        g = df.groupby("SVC")["SPR"]
        return pd.DataFrame({
            "SVC": g.median().index,
            "PROM": g.median().values,
            "PEAK": g.apply(lambda x: np.nanpercentile(x.dropna(), 95) if x.notna().any() else np.nan).values
        })

    a = agg_stats(spr[is_rent]).rename(columns={"PROM":"SPR_RENTALS_PROM","PEAK":"SPR_RENTALS_PEAK"})
    b = agg_stats(spr[is_crowd]).rename(columns={"PROM":"SPR_CROWD_PROM","PEAK":"SPR_CROWD_PEAK"})
    c = agg_stats(spr[is_mlp]).rename(columns={"PROM":"SPR_MLP_PROM","PEAK":"SPR_MLP_PEAK"})

    out = pd.DataFrame({"SVC": pd.concat([a["SVC"], b["SVC"], c["SVC"]], axis=0).drop_duplicates()})
    out = out.merge(a, on="SVC", how="left").merge(b, on="SVC", how="left").merge(c, on="SVC", how="left")
    return out



def _bucket_dm(dm: str) -> str:
    dm = (dm or "").strip().lower()
    if "crowd" in dm:                  return "CROWD"
    if "rent" in dm:                   return "RENTALS"
    if "mlp" in dm or "sdd" in dm or "spot" in dm or "back" in dm:
        return "MLP"
    if "dc" in dm or "delivery cell" in dm or dm in ("dc","cell"):
        return "DC"
    return dm.upper() if dm else "OTHER"

def load_spr_dm_real_peak_for(op_date: date,
                              real_weeks: int = 4,
                              peak_weeks: int = 12) -> pd.DataFrame:
    """
    SPR por Delivery Model usando la pestaña SPR:
      - REAL4W_* : mediana de los últimos 4 (por defecto) del mismo día de semana (<= op_date)
      - PEAK_*   : p95 de los últimos 12 del mismo día de semana (<= op_date)
    Devuelve columnas por SVC:
      SPR_RENTALS_REAL4W, SPR_CROWD_REAL4W, SPR_MLP_REAL4W,
      SPR_RENTALS_PEAK,   SPR_CROWD_PEAK,   SPR_MLP_PEAK
    """
    spr = read_sheet(SHEET_ID, SHEET_TABS["spr"])
    want_cols = ["SVC","DELIVERY_MODEL","FECHA","SPR"]
    if spr.empty:
        return pd.DataFrame(columns=["SVC",
            "SPR_RENTALS_REAL4W","SPR_CROWD_REAL4W","SPR_MLP_REAL4W",
            "SPR_RENTALS_PEAK","SPR_CROWD_PEAK","SPR_MLP_PEAK"])

    find_and_rename(spr, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", False, "SPR")
    find_and_rename(spr, ["Delivery model","Deliverymodel","Model","DM"], "DELIVERY_MODEL", False, "SPR")
    find_and_rename(spr, ["SPR","spr","Ships per route"], "SPR", False, "SPR")
    coerce_date_column(spr, ["FECHA","Fecha","DATE","OP_DT"], "FECHA", "SPR", required=False)

    spr = ensure_columns(spr, {"SVC": None, "DELIVERY_MODEL": "", "FECHA": pd.NaT, "SPR": np.nan})
    spr["SPR"] = pd.to_numeric(spr["SPR"], errors="coerce")
    spr["FECHA"] = pd.to_datetime(spr["FECHA"], errors="coerce").dt.date
    spr = spr[spr["FECHA"].notna()]
    _as_str_cols(spr, ["SVC","DELIVERY_MODEL"])

    # mismo día de semana y <= op_date
    wd = op_date.weekday()
    spr = spr[(spr["FECHA"] <= op_date) & (spr["FECHA"].apply(lambda d: d.weekday()) == wd)]

    if spr.empty:
        return pd.DataFrame(columns=["SVC",
            "SPR_RENTALS_REAL4W","SPR_CROWD_REAL4W","SPR_MLP_REAL4W",
            "SPR_RENTALS_PEAK","SPR_CROWD_PEAK","SPR_MLP_PEAK"])

    spr["DM_BUCKET"] = spr["DELIVERY_MODEL"].map(_bucket_dm)

    spr = spr.sort_values("FECHA", ascending=False)
    spr["rank"] = spr.groupby(["SVC","DM_BUCKET"]).cumcount()

    last_real = spr[spr["rank"] < real_weeks]
    last_peak = spr[spr["rank"] < peak_weeks]

    def p95(x):
        x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
        return np.nan if x.empty else float(np.nanpercentile(x, 95))

    real4 = last_real.groupby(["SVC","DM_BUCKET"])["SPR"].median().unstack("DM_BUCKET")
    peaks = last_peak.groupby(["SVC","DM_BUCKET"])["SPR"].apply(p95).unstack("DM_BUCKET")

    out = pd.DataFrame(index=sorted(spr["SVC"].astype(str).unique()))
    out = out.join(real4.add_prefix("REAL4W_"), how="left").join(peaks.add_prefix("PEAK_"), how="left")
    out = out.reset_index().rename(columns={"index":"SVC"})

    # renombres estándar
    ren = {
        "REAL4W_RENTALS":"SPR_RENTALS_REAL4W",
        "REAL4W_CROWD":"SPR_CROWD_REAL4W",
        "REAL4W_MLP":"SPR_MLP_REAL4W",
        "PEAK_RENTALS":"SPR_RENTALS_PEAK",
        "PEAK_CROWD":"SPR_CROWD_PEAK",
        "PEAK_MLP":"SPR_MLP_PEAK",
    }
    for a,b in ren.items():
        if a in out.columns: out.rename(columns={a:b}, inplace=True)
        else: out[b] = np.nan
    return out[["SVC",
                "SPR_RENTALS_REAL4W","SPR_CROWD_REAL4W","SPR_MLP_REAL4W",
                "SPR_RENTALS_PEAK","SPR_CROWD_PEAK","SPR_MLP_PEAK"]]


# -----------------------------------------------------------------------------
# 5) Cálculo del plan
# -----------------------------------------------------------------------------

def apply_output_adjustments(resumen: pd.DataFrame) -> pd.DataFrame:
    # Quita algún legado que pueda colarse
    resumen = resumen.drop(columns=["Demanda esperada", "DEMANDA_ESPERADA"], errors="ignore")

    orden = [
        "SVC","FECHA",
        "FCST","SHIPMENTS_DC","SHIPMENTS_SP","FCST (sin DC & sin SP)","DEMANDA_AJUSTADA",
        "RUTAS_RENTALS","SPR_RENTALS","SHIP_RENTALS",
        "CROWD_PCT","SPR_CROWD","SHIP_OBJ_CROWD","RUTAS_CROWD_OBJ",
        "RUTAS_CROWD_CAP","CROWD_E1_CAP",
        "RUTAS_CROWD_BASE","RUTAS_CROWD_ESCALADO",
        "SHIP_CROWD","SHIP_RESTANTES_PRE_MLP",
        "SPR_USADO","SPR_PROM","SPR_PEAK","SPR_OBJ","SPR_MLP",
        "MLP_SDD_LV","MLP_SDD_SV","MLP_SDD_CAR","MLP_SDD_CAP",
        "MLP_SPOT_LV","MLP_SPOT_SV","MLP_SPOT_CAR","MLP_SPOT_CAP",
        "MLP_BACK_CAP",
        "RUTAS_MLP_NEEDED",
        "RUTAS_MLP_SDD_USADAS","RUTAS_MLP_SPOT_USADAS","RUTAS_MLP_BACKLOG_USADAS",
        "RUTAS_SPR_BASE","RUTAS_RESTANTES","RUTAS_FALTANTES",
        "CAP_TOTAL","CAP_VS_FCST","CAP_DIFF_ABS","RIESGO",
    ]

    for c in orden:
        if c not in resumen.columns:
            resumen[c] = 0

    df = resumen.copy()

    # --- Formato solicitado ---
    # 1) CAP_VS_FCST como porcentaje "100%" / "96%"
    df["CAP_VS_FCST"] = (
        pd.to_numeric(df["CAP_VS_FCST"], errors="coerce").fillna(0).clip(lower=0) * 100
    ).round(0).astype(int).astype(str) + "%"

    # 2) CAP_DIFF_ABS negativa cuando haya RIESGO
    diff_signed = (
        pd.to_numeric(df["CAP_TOTAL"], errors="coerce").fillna(0)
        - pd.to_numeric(df["FCST"], errors="coerce").fillna(0)
    ).round(0)
    df["CAP_DIFF_ABS"] = np.where(
        df.get("RIESGO","") == "RIESGO",
        diff_signed,      # negativo si falta capacidad
        diff_signed.abs() # positivo si estamos OK
    )

    # 3) Formato con comas y sin decimales para todas las demás numéricas (excepto %)
    for c in df.columns:
        if c not in ["SVC","FECHA","CAP_VS_FCST","RIESGO"]:
            if np.issubdtype(df[c].dtype, np.number):
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).round(0).astype(int)
                df[c] = df[c].map(lambda x: f"{x:,}")

    return df[orden]


def compute_plan(spr_mode: str, sel_svcs: Optional[List[str]] = None) -> pd.DataFrame:
    hoy = date.today()

    # ---- carga de datos base ----
    fcst      = load_fcst()
    spr       = load_spr()                        # SPR_PROM / SPR_PEAK (agregado general)
    caps      = load_capacity_caps()              # DC/SP, caps MLP legacy (no crowd aquí)
    crowdday  = load_crowd_caps_for(hoy)          # base crowd del día + E1 del día
    rents     = load_rentals_caps_from_sheet()    # RUTAS_RENTALS + SPR_RENTALS ponderado por mix
    rents_fb  = load_rentals_fallback()           # fallback de rutas rentals (unidades)
    crowd_pct = load_crowd_pct_from_capacity()    # % crowd objetivo
    spr_crowd = load_spr_crowd()                  # SPR_CROWD (histórico, por si hace falta fallback)
    mlp_caps  = load_mlp_caps_from_srm()          # caps SDD/SPOT/BACK desde SRM
    spr_mlp   = load_spr_mlp()                    # SPR_MLP (histórico, por si hace falta fallback)

    # Stats SPR por Delivery Model desde pestaña SPR (debe incluir *_PROM4 / *_P95_4).
    spr_stats_dm = load_spr_dm_stats_from_sheet()

    # --- helper: SPR plan por DM (Capacity, día exacto) ---
    def get_plan_spr_from_capacity(op_date: date) -> pd.DataFrame:
        df = read_sheet(SHEET_ID, SHEET_TABS["capacity"])
        if df.empty:
            return pd.DataFrame(columns=["SVC","SPR_PLAN_RENTALS","SPR_PLAN_CROWD","SPR_PLAN_MLP"])
        # Normalización mínima
        find_and_rename(df, ["Delivery model","Deliverymodel","Model","DM"], "DELIVERY_MODEL", False, "Capacity")
        find_and_rename(df, ["Tipo","Type","Category"], "TIPO", False, "Capacity")
        find_and_rename(df, ["SVC","SVCs","LOGISTIC_CENTER_ID","FACILITY","LC"], "SVC", False, "Capacity")
        coerce_date_column(df, ["Fecha","FECHA","Date","OP_DT"], "FECHA", "Capacity", required=False)
        find_and_rename(df, ["Cantidad","Qty","Quantity","COUNT","QTY"], "CANT", False, "Capacity")
        df = ensure_columns(df, {"DELIVERY_MODEL":"", "TIPO":"", "SVC":None, "FECHA": pd.NaT, "CANT":0})
        df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce").dt.date
        df["CANT"]  = pd.to_numeric(df["CANT"], errors="coerce").fillna(0)
        _as_str_cols(df, ["SVC","DELIVERY_MODEL","TIPO"])

        # Sólo filas Tipo=SPR del día exacto
        is_spr = df["TIPO"].str.lower().str.contains("spr", regex=False)
        df = df[is_spr & (df["FECHA"] == op_date)]
        if df.empty:
            return pd.DataFrame(columns=["SVC","SPR_PLAN_RENTALS","SPR_PLAN_CROWD","SPR_PLAN_MLP"])

        dm = df["DELIVERY_MODEL"].str.lower()
        m_rent = dm.str.contains("rent", regex=False)
        m_crwd = dm.str.contains("crowd", regex=False)
        m_mlp  = dm.str.contains("mlp", regex=False)

        out_r = df[m_rent].groupby("SVC", dropna=False)["CANT"].median().rename("SPR_PLAN_RENTALS")
        out_c = df[m_crwd].groupby("SVC", dropna=False)["CANT"].median().rename("SPR_PLAN_CROWD")
        out_m = df[m_mlp].groupby("SVC", dropna=False)["CANT"].median().rename("SPR_PLAN_MLP")

        out = pd.concat([out_r, out_c, out_m], axis=1).reset_index()
        return _as_str_cols(out, ["SVC"])

    plan_spr_cap = get_plan_spr_from_capacity(hoy)

    # --- normaliza SVCs de todos ---
    for d in (fcst, spr, caps, crowdday, rents, rents_fb, crowd_pct, spr_crowd, mlp_caps, spr_mlp, spr_stats_dm, plan_spr_cap):
        if not d.empty and "SVC" in d.columns:
            _as_str_cols(d, ["SVC"])

    # ----------------- BASE de SVCs -----------------
    bases = []
    for d in [fcst, spr, caps, crowdday, rents, rents_fb, crowd_pct, mlp_caps, spr_stats_dm, plan_spr_cap]:
        if "SVC" in d.columns and not d.empty:
            bases.append(d[["SVC"]])
    base = pd.concat(bases, axis=0).drop_duplicates() if bases else pd.DataFrame(columns=["SVC"])
    base = _as_str_cols(base, ["SVC"])

    out = base.copy()
    out["FECHA"] = hoy

    # ----------------- MERGES PRINCIPALES -----------------
    if not fcst.empty:
        out = safe_merge(out, fcst[["SVC","FCST"]], ["SVC"])
    if not spr.empty:
        out = safe_merge(out, spr[["SVC","SPR_OBJ","SPR_PEAK","SPR_PROM"]], ["SVC"])
    out = ensure_columns(out, {"SPR_OBJ":np.nan, "SPR_PEAK":np.nan, "SPR_PROM":np.nan})

    # SPR genérico de base (fallback para todo)
    spr_mode_col = {"promedio":"SPR_PROM", "peak":"SPR_PEAK", "plan":"SPR_OBJ"}.get(spr_mode, "SPR_PROM")
    spr_usado = out[spr_mode_col].where(out[spr_mode_col].notna(), out["SPR_OBJ"]).fillna(20)
    out["SPR_USADO"] = pd.to_numeric(spr_usado, errors="coerce").fillna(20).clip(lower=1)

    # Capacity DC/SP
    if not caps.empty:
        out = safe_merge(out, caps[["SVC","RUTAS_MLP_SDD","RUTAS_MLP_SPOT","RUTAS_RENTALS","SHIPMENTS_DC","SHIPMENTS_SP"]], ["SVC"])
    else:
        out = ensure_columns(out, {"RUTAS_MLP_SDD":0,"RUTAS_MLP_SPOT":0,"RUTAS_RENTALS":0,"SHIPMENTS_DC":0,"SHIPMENTS_SP":0})

    # Rentals (rutas y spr ponderado por mix de vehículos)
    out = out.drop(columns=["RUTAS_RENTALS"], errors="ignore")
    if not rents.empty:
        out = safe_merge(out, rents[["SVC","RUTAS_RENTALS","SPR_RENTALS"]], ["SVC"])
    elif not rents_fb.empty:
        out = safe_merge(out, rents_fb[["SVC","RUTAS_RENTALS"]], ["SVC"])
        out["SPR_RENTALS"] = np.nan
    else:
        out["RUTAS_RENTALS"] = 0
        out["SPR_RENTALS"]   = np.nan
    out["RUTAS_RENTALS"] = pd.to_numeric(out.get("RUTAS_RENTALS", 0), errors="coerce").fillna(0).astype(int)

    # CROWD (cap del día + E1 del día)
    out = out.drop(columns=["RUTAS_CROWD_CAP"], errors="ignore")
    if not crowdday.empty:
        out = safe_merge(out, crowdday[["SVC","RUTAS_CROWD_CAP","CROWD_E1_CAP"]], ["SVC"])
    out = ensure_columns(out, {"RUTAS_CROWD_CAP":0, "CROWD_E1_CAP":0})
    out["RUTAS_CROWD_CAP"] = pd.to_numeric(out["RUTAS_CROWD_CAP"], errors="coerce").fillna(0).astype(int)
    out["CROWD_E1_CAP"]    = pd.to_numeric(out["CROWD_E1_CAP"], errors="coerce").fillna(0).astype(int)

    # % objetivo Crowd
    if not crowd_pct.empty:
        out = safe_merge(out, crowd_pct, ["SVC"])
    out["CROWD_PCT"] = pd.to_numeric(out.get("CROWD_PCT", 0), errors="coerce").fillna(0).clip(0,1)

    # SPR de Crowd histórico como posible fallback
    if not spr_crowd.empty:
        out = safe_merge(out, spr_crowd, ["SVC"])

    # ----------------- DEMANDA & SPR BASE -----------------
    out = ensure_columns(out, {"FCST":0, "SHIPMENTS_DC":0, "SHIPMENTS_SP":0})
    out["FCST (sin DC & sin SP)"] = (
        pd.to_numeric(out["FCST"], errors="coerce").fillna(0)
        - pd.to_numeric(out["SHIPMENTS_DC"], errors="coerce").fillna(0)
        - pd.to_numeric(out["SHIPMENTS_SP"], errors="coerce").fillna(0)
    ).clip(lower=0)
    out["DEMANDA_AJUSTADA"] = out["FCST (sin DC & sin SP)"]
    out["RUTAS_SPR_BASE"]   = np.ceil(out["DEMANDA_AJUSTADA"] / out["SPR_USADO"]).astype(int)

    # ----------------- SPR por Delivery Model (según modo) -----------------
    # Garantiza existencia de columnas de stats
    for c in [
        "SPR_RENTALS_PROM4","SPR_RENTALS_P95_4","SPR_RENTALS_PROM","SPR_RENTALS_PEAK",
        "SPR_CROWD_PROM4","SPR_CROWD_P95_4","SPR_CROWD_PROM","SPR_CROWD_PEAK",
        "SPR_MLP_PROM4","SPR_MLP_P95_4","SPR_MLP_PROM","SPR_MLP_PEAK",
    ]:
        if spr_stats_dm.empty or c not in spr_stats_dm.columns:
            # se crearán como NaN y luego habrá fallback
            pass
    if not spr_stats_dm.empty:
        out = safe_merge(out, spr_stats_dm, ["SVC"])

    # Merge plan SPR de Capacity (día) si existe
    if not plan_spr_cap.empty:
        out = safe_merge(out, plan_spr_cap, ["SVC"])

    def pick_dm(prom4, p95_4, prom, peak, plan_cap, plan_sheet):
        """
        Selecciona el SPR final para un DM con fallbacks.
        - promedio: PROM4 -> PROM -> SPR_USADO
        - peak:     P95_4 -> PEAK -> PROM -> SPR_USADO
        - plan:     plan_cap -> plan_sheet -> PROM4 -> PROM -> SPR_USADO
        """
        if spr_mode == "promedio":
            s = out.get(prom4)
            if s is None:
                s = out.get(prom)
            return pd.to_numeric(s, errors="coerce")
        elif spr_mode == "peak":
            s = out.get(p95_4)
            if s is None:
                s = out.get(peak)
            if s is None:
                s = out.get(prom)
            return pd.to_numeric(s, errors="coerce")
        else:  # plan
            s = out.get(plan_cap)  # desde Capacity del día
            if s is None or pd.isna(s).all():
                s = out.get(plan_sheet)  # valor “plan” que ya tengas en sheet por DM (si aplica)
            if s is None or pd.isna(s).all():
                s = out.get(prom4)
            if s is None or pd.isna(s).all():
                s = out.get(prom)
            return pd.to_numeric(s, errors="coerce")

    # Rentals
    out["SPR_RENTALS"] = pick_dm(
        "SPR_RENTALS_PROM4","SPR_RENTALS_P95_4","SPR_RENTALS_PROM","SPR_RENTALS_PEAK",
        "SPR_PLAN_RENTALS","SPR_RENTALS"  # plan_sheet = el ponderado por mix si no hubiera en Capacity
    ).fillna(out["SPR_USADO"]).clip(lower=1)

    # Crowd
    out["SPR_CROWD"] = pick_dm(
        "SPR_CROWD_PROM4","SPR_CROWD_P95_4","SPR_CROWD_PROM","SPR_CROWD_PEAK",
        "SPR_PLAN_CROWD","SPR_CROWD"  # si no hay en Capacity, usa el del sheet SPR
    ).fillna(out["SPR_USADO"]).clip(lower=1)

    # MLP
    out["SPR_MLP"] = pick_dm(
        "SPR_MLP_PROM4","SPR_MLP_P95_4","SPR_MLP_PROM","SPR_MLP_PEAK",
        "SPR_PLAN_MLP","SPR_MLP"  # fallback al del sheet SPR si no hay en Capacity
    ).fillna(out["SPR_USADO"]).clip(lower=1)

    # ----------------- CROWD ASIGNACIÓN -----------------
    out["SHIP_OBJ_CROWD"] = pd.to_numeric(out["FCST"], errors="coerce").fillna(0) * out["CROWD_PCT"]
    out["RUTAS_CROWD_OBJ"] = np.ceil(
        pd.to_numeric(out["SHIP_OBJ_CROWD"], errors="coerce").fillna(0) / out["SPR_CROWD"].replace(0, np.nan)
    ).replace([np.inf,-np.inf], np.nan).fillna(0).astype(int)

    leftover_vs_spr = np.maximum(out["RUTAS_SPR_BASE"] - out["RUTAS_RENTALS"], 0)
    out["RUTAS_CROWD_BASE"] = np.minimum.reduce([
        leftover_vs_spr,
        out["RUTAS_CROWD_CAP"],
        out["RUTAS_CROWD_OBJ"]
    ]).astype(int)

    exceso_obj       = (out["RUTAS_CROWD_OBJ"] - out["RUTAS_CROWD_BASE"]).clip(lower=0)
    rem_despues_base = (leftover_vs_spr - out["RUTAS_CROWD_BASE"]).clip(lower=0)
    out["RUTAS_CROWD_ESCALADO"] = np.minimum.reduce([exceso_obj, out["CROWD_E1_CAP"], rem_despues_base]).astype(int)

    # Shipments por Rentals y Crowd
    out["SHIP_RENTALS"] = pd.to_numeric(out["RUTAS_RENTALS"], errors="coerce").fillna(0) * out["SPR_RENTALS"]
    out["SHIP_CROWD"]   = (out["RUTAS_CROWD_BASE"] + out["RUTAS_CROWD_ESCALADO"]) * out["SPR_CROWD"]

    # Restantes para MLP
    base_otros = pd.to_numeric(out["SHIPMENTS_DC"], errors="coerce").fillna(0) + pd.to_numeric(out["SHIPMENTS_SP"], errors="coerce").fillna(0)
    out["SHIP_RESTANTES_PRE_MLP"] = (
        pd.to_numeric(out["FCST"], errors="coerce").fillna(0) - base_otros - out["SHIP_RENTALS"] - out["SHIP_CROWD"]
    ).clip(lower=0)

    # ----------------- MLP (caps SRM + SPR_MLP) -----------------
    if not mlp_caps.empty:
        out = safe_merge(out, mlp_caps, ["SVC"])
    else:
        out = ensure_columns(out, {
            "MLP_SDD_LV":0,"MLP_SDD_SV":0,"MLP_SDD_CAR":0,"MLP_SDD_CAP":0,
            "MLP_SPOT_LV":0,"MLP_SPOT_SV":0,"MLP_SPOT_CAR":0,"MLP_SPOT_CAP":0,
            "MLP_BACK_CAP":0
        })

    need     = np.ceil(pd.to_numeric(out.get("SHIP_RESTANTES_PRE_MLP", 0), errors="coerce").fillna(0) / out["SPR_MLP"].replace(0,np.nan)) \
                    .replace([np.inf,-np.inf], np.nan).fillna(0).astype(int)
    sdd_cap  = pd.to_numeric(out["MLP_SDD_CAP"],  errors="coerce").fillna(0)
    spot_cap = pd.to_numeric(out["MLP_SPOT_CAP"], errors="coerce").fillna(0)
    back_cap = pd.to_numeric(out["MLP_BACK_CAP"], errors="coerce").fillna(0)

    use_sdd  = np.minimum(need, sdd_cap)
    need2    = (need - use_sdd).clip(lower=0)
    use_spot = np.minimum(need2, spot_cap)
    need3    = (need2 - use_spot).clip(lower=0)
    use_back = np.minimum(need3, back_cap)

    out["RUTAS_MLP_SDD_USADAS"]     = use_sdd.astype(int)
    out["RUTAS_MLP_SPOT_USADAS"]    = use_spot.astype(int)
    out["RUTAS_MLP_BACKLOG_USADAS"] = use_back.astype(int)

    out["RUTAS_RESTANTES"] = (need3 - use_back).clip(lower=0).astype(int)
    out["RUTAS_POST_MLP"]  = out["RUTAS_RESTANTES"]
    out["RUTAS_FALTANTES"] = out["RUTAS_RESTANTES"]

    # KPIs finales
    cap_mlp = (out["RUTAS_MLP_SDD_USADAS"] + out["RUTAS_MLP_SPOT_USADAS"] + out["RUTAS_MLP_BACKLOG_USADAS"]) * out["SPR_MLP"]
    out["CAP_TOTAL"]    = base_otros.fillna(0) + out["SHIP_RENTALS"].fillna(0) + out["SHIP_CROWD"].fillna(0) + cap_mlp.fillna(0)
    out["CAP_VS_FCST"]  = (out["CAP_TOTAL"] / out["FCST"].replace(0, np.nan)).fillna(0).round(2)
    out["CAP_DIFF_ABS"] = (pd.to_numeric(out["FCST"], errors="coerce").fillna(0) - out["CAP_TOTAL"]).abs().round(2)
    out["RIESGO"]       = np.where(out["CAP_TOTAL"] + 1e-9 >= pd.to_numeric(out["FCST"], errors="coerce").fillna(0), "OK", "RIESGO")

    # Filtro SVC (si aplica)
    if sel_svcs:
        sel_svcs = _clean_svc_values(sel_svcs)
        if sel_svcs:
            out = out[out["SVC"].isin(sel_svcs)]

    # Orden y salida
    out = apply_output_adjustments(out).fillna(0).sort_values("SVC").reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# 5.1) Detalle por vehículo – SVC – día (respetando estructura)
# -----------------------------------------------------------------------------

def _to_int(x):
    if pd.isna(x): return 0
    if isinstance(x, (int, np.integer)): return int(x)
    s = str(x).replace(",", "").replace("%", "").strip()
    try:
        return int(float(s))
    except Exception:
        return 0

def _largest_remainder_allocation(total: int, weights: pd.Series) -> pd.Series:
    """Asigna enteros que sumen 'total' según 'weights'."""
    w = pd.to_numeric(weights, errors="coerce").fillna(0)
    if total <= 0 or w.sum() <= 0:
        return pd.Series(0, index=w.index)
    p = w / w.sum()
    raw = p * total
    flo = np.floor(raw).astype(int)
    rem = raw - flo
    short = total - int(flo.sum())
    if short > 0:
        order = np.argsort(-rem.values)
        for i in order[:short]:
            flo.iloc[i] += 1
    return flo

def _read_vehicle_mix_from_spr(op_date: date, dm_bucket: str, weeks: int = 2) -> pd.DataFrame:
    """
    Lee la pestaña SPR y arma el mix de vehículos por SVC para el DM indicado
    usando la **suma de 'Rutas'** de las últimas 'weeks' semanas (mismo weekday).
    Devuelve columnas: SVC, VEHICULO_TIPO_HOM, PESO_RUTAS
    """
    spr = read_sheet(SHEET_ID, SHEET_TABS["spr"])
    out_cols = ["SVC", "VEHICULO_TIPO_HOM", "PESO_RUTAS"]
    if spr.empty:
        return pd.DataFrame(columns=out_cols)

    # Normalización mínima de columnas
    find_and_rename(spr, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", False, "SPR")
    find_and_rename(spr, ["SHP_LG_VEHICLE_TYPE","Vehicle type","Tipo de vehículo","Tipo de vehiculo"], "VEH_TYPE", False, "SPR")
    # La columna 'Rutas' la usaremos como peso para el mix
    find_and_rename(spr, ["Rutas","RUTAS","Routes"], "RUTAS", False, "SPR")
    find_and_rename(spr, ["Delivery model","Deliverymodel","Model","DM"], "DELIVERY_MODEL", False, "SPR")
    coerce_date_column(spr, ["FECHA","Fecha","DATE","OP_DT"], "FECHA", "SPR", required=False)

    if "RUTAS" not in spr.columns:
        return pd.DataFrame(columns=out_cols)

    # Tipos básicos
    spr["RUTAS"] = pd.to_numeric(spr["RUTAS"], errors="coerce").fillna(0)
    _as_str_cols(spr, ["SVC","VEH_TYPE","DELIVERY_MODEL"])

    # --- Bucket de DM (CROWD / RENTALS / MLP / DC / SP) ---
    def dm_bucket_map(dm: str) -> str:
        dm = (dm or "").strip().lower()
        if "crowd" in dm: return "CROWD"
        if "rent"  in dm: return "RENTALS"
        if ("mlp" in dm) or ("sdd" in dm) or ("spot" in dm) or ("back" in dm): return "MLP"
        if "dc" in dm or "delivery cell" in dm or dm in ("dc","cell"): return "DC"
        if dm in ("sp","s.p.","service partner") or ("service" in dm and "partner" in dm): return "SP"
        return "__OTHER__"

    spr["DM_BKT"] = spr["DELIVERY_MODEL"].map(dm_bucket_map)

    # --- Homologación de tipo de vehículo para el mix ---
    # Unifica variantes de Car (Car 5h/8h/SH...), normaliza vans y preserva Bike.
    def veh_hom(v: str) -> str:
        t = (v or "").strip().lower()
        if "bike" in t:                 return "Bike"
        if "car"  in t:                 return "Car"
        # Para vans usamos la homologación existente y quitamos 'Electric' al mix
        h = homologar_vehicle_type(v)
        h_l = (h or "").lower()
        if "small van" in h_l:          return "Small Van"
        if "large van" in h_l:          return "Large Van"
        return "Others"

    spr["VEHICULO_TIPO_HOM"] = spr["VEH_TYPE"].map(veh_hom)

    # --- Ventana temporal: últimas N semanas, mismo weekday que op_date ---
    op_ts = pd.Timestamp(op_date)
    start = op_ts - pd.Timedelta(days=7 * weeks)
    wd    = op_ts.weekday()

    mask = (
        spr["FECHA"].notna()
        & (pd.to_datetime(spr["FECHA"]) <= op_ts)
        & (pd.to_datetime(spr["FECHA"]) >= start)
        & (pd.to_datetime(spr["FECHA"]).dt.weekday == wd)
        & (spr["DM_BKT"] == dm_bucket)
    )

    recent = spr[mask].copy()
    if recent.empty:
        return pd.DataFrame(columns=out_cols)

    # --- Mix por SVC a partir de la suma de Rutas homologadas ---
    g = (
        recent.groupby(["SVC","VEHICULO_TIPO_HOM"], dropna=False)["RUTAS"]
              .sum()
              .rename("PESO_RUTAS")
              .reset_index()
    )

    g = _as_str_cols(g, ["SVC","VEHICULO_TIPO_HOM"])
    return g[out_cols]



def _spr_por_vehiculo_hist() -> pd.DataFrame:
    """SPR histórico por SVC y tipo homologado (local con fallback global)."""
    return load_spr_hist_from_sheet()

def _alloc_mlp_by_type(used_total: int, caps_lv: int, caps_sv: int, caps_car: int) -> dict:
    """Prioridad LV → SV → Car, acotado por caps."""
    rem = max(0, int(used_total))
    lv = min(rem, int(caps_lv));  rem -= lv
    sv = min(rem, int(caps_sv));  rem -= sv
    car = min(rem, int(caps_car)); rem -= car
    return {"Large Van": lv, "Small Van": sv, "Car": car}

def _veh_mix_hom(v: str) -> str:
    t = (v or "").strip().lower()
    if "bike" in t:  return "Bike"
    if "car"  in t:  return "Car"
    h = homologar_vehicle_type(v)
    hl = (h or "").lower()
    if "small van" in hl: return "Small Van"
    if "large van" in hl: return "Large Van"
    return "Others"

def _dm_bucket_for_mix(dm: str) -> str:
    dm = (dm or "").strip().lower()
    if "crowd" in dm: return "CROWD"
    if "rent"  in dm: return "RENTALS"
    if ("mlp" in dm) or ("sdd" in dm) or ("spot" in dm) or ("back" in dm): return "MLP"
    if "dc" in dm or "delivery cell" in dm or dm in ("dc","cell"): return "DC"
    if dm in ("sp","s.p.","service partner") or ("service" in dm and "partner" in dm): return "SP"
    return "__OTHER__"

@st.cache_data(show_spinner=False, ttl=300)
def load_spr_by_dm_vehicle(op_date: date, mode: str) -> pd.DataFrame:
    """
    SPR seleccionado por SVC × (RENTALS/CROWD/MLP) × Vehículo (LV/SV/Car/Bike/Others),
    usando el mismo criterio que la cabecera:
      - promedio: mediana de los últimos 4 del mismo weekday (<= op_date)
      - peak:     p95 de los últimos 12 del mismo weekday (<= op_date)
      - plan:     último valor del mismo weekday (<= op_date)
    """
    spr = read_sheet(SHEET_ID, SHEET_TABS["spr"])
    want = ["SVC","DM","VEH","SPR_SEL"]
    if spr.empty:
        return pd.DataFrame(columns=want)

    # Normalización de columnas
    find_and_rename(spr, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", False, "SPR")
    find_and_rename(spr, ["Delivery model","Deliverymodel","Model","DM"], "DELIVERY_MODEL", False, "SPR")
    find_and_rename(spr, ["SHP_LG_VEHICLE_TYPE","Vehicle type","Tipo de vehículo","Tipo de vehiculo"], "VEH_TYPE", False, "SPR")
    find_and_rename(spr, ["SPR","spr","Ships per route"], "SPR", False, "SPR")
    coerce_date_column(spr, ["FECHA","Fecha","DATE","OP_DT"], "FECHA", "SPR", required=False)

    spr = ensure_columns(spr, {"SVC":None,"DELIVERY_MODEL":"","VEH_TYPE":"","SPR":np.nan,"FECHA":pd.NaT})
    spr["SPR"] = pd.to_numeric(spr["SPR"], errors="coerce")
    _as_str_cols(spr, ["SVC","DELIVERY_MODEL","VEH_TYPE"])
    spr["DM_BKT"] = spr["DELIVERY_MODEL"].map(_dm_bucket_for_mix)
    spr["VEH_HOM"] = spr["VEH_TYPE"].map(_veh_mix_hom)

    # Sólo DM relevantes
    spr = spr[spr["DM_BKT"].isin(["RENTALS","CROWD","MLP"])].copy()
    spr["FECHA"] = pd.to_datetime(spr["FECHA"], errors="coerce")
    if spr["FECHA"].isna().all():
        return pd.DataFrame(columns=want)

    # Misma day-of-week y <= op_date
    op_ts = pd.Timestamp(op_date)
    wd    = op_ts.weekday()
    spr = spr[(spr["FECHA"].dt.weekday == wd) & (spr["FECHA"] <= op_ts)].copy()
    if spr.empty:
        return pd.DataFrame(columns=want)

    # Agregadores por modo
    def agg_prom(g):
        vals = pd.to_numeric(g["SPR"], errors="coerce").dropna()
        if len(vals) > 4: vals = vals.iloc[-4:]
        return float(vals.median()) if len(vals) else np.nan

    def agg_peak(g):
        vals = pd.to_numeric(g["SPR"], errors="coerce").dropna()
        if len(vals) > 12: vals = vals.iloc[-12:]
        return float(np.nanpercentile(vals, 95)) if len(vals) else np.nan

    def agg_plan(g):
        vals = pd.to_numeric(g["SPR"], errors="coerce").dropna()
        return float(vals.iloc[-1]) if len(vals) else np.nan

    if mode == "promedio":
        sel = spr.groupby(["SVC","DM_BKT","VEH_HOM"], dropna=False).apply(agg_prom).reset_index(name="SPR_SEL")
    elif mode == "peak":
        sel = spr.groupby(["SVC","DM_BKT","VEH_HOM"], dropna=False).apply(agg_peak).reset_index(name="SPR_SEL")
    else:  # plan
        sel = spr.groupby(["SVC","DM_BKT","VEH_HOM"], dropna=False).apply(agg_plan).reset_index(name="SPR_SEL")

    sel.rename(columns={"DM_BKT":"DM","VEH_HOM":"VEH"}, inplace=True)
    return sel[want]




def expand_to_vehicle_level(plan: pd.DataFrame, spr_mode: str) -> pd.DataFrame:
    """
    Expande la tabla 'plan' a nivel vehículo–SVC–día.
    Columnas: DELIVERY_MOD, FECHA, SVC, SHP_LG_VEHICLE_TYPE, SPR, Rutas, Shipments
    """
    op_date = date.today()

    # SPR histórico por vehículo (local + global) para fallbacks finos
    sprveh = _spr_por_vehiculo_hist()
    sprveh_glob = sprveh[sprveh["SVC"] == "__GLOBAL__"][["VEHICULO_TIPO_HOM","SPR_HIST"]].rename(
        columns={"SPR_HIST":"SPR_GLOBAL"}
    )
    sprveh_loc  = sprveh[sprveh["SVC"] != "__GLOBAL__"][["SVC","VEHICULO_TIPO_HOM","SPR_HIST"]]

    # SPR seleccionado por SVC × DM × VEH (según 'spr_mode')
    spr_dmveh = load_spr_by_dm_vehicle(op_date, spr_mode)

    # Mix de 2 semanas SOLO para DM sin apertura (Crowd, DC, SP)
    mix_crowd = _read_vehicle_mix_from_spr(op_date, "CROWD", weeks=2)
    mix_dc    = _read_vehicle_mix_from_spr(op_date, "DC",    weeks=2)
    mix_sp    = _read_vehicle_mix_from_spr(op_date, "SP",    weeks=2)

    rows = []

    def spr_for(svc: str, dm: str, veh: str, fallback: float) -> float:
        """Prioridad: SPR_dmveh(SVC,DM,veh) → SPR_hist_local(SVC,veh) → SPR_hist_global(veh) → fallback."""
        # 1) por DM × veh
        if not spr_dmveh.empty:
            hit = spr_dmveh[(spr_dmveh["SVC"] == svc) & (spr_dmveh["DM"] == dm) & (spr_dmveh["VEH"] == veh)]
            if not hit.empty:
                v = float(pd.to_numeric(hit["SPR_SEL"].iloc[0], errors="coerce"))
                if np.isfinite(v) and v > 0:
                    return v
        # 2) local por SVC × veh
        loc = sprveh_loc[(sprveh_loc["SVC"] == svc) & (sprveh_loc["VEHICULO_TIPO_HOM"] == veh)]
        if not loc.empty:
            v = float(pd.to_numeric(loc["SPR_HIST"].iloc[0], errors="coerce"))
            if np.isfinite(v) and v > 0:
                return v
        # 3) global por veh
        glob = sprveh_glob[sprveh_glob["VEHICULO_TIPO_HOM"] == veh]
        if not glob.empty:
            v = float(pd.to_numeric(glob["SPR_GLOBAL"].iloc[0], errors="coerce"))
            if np.isfinite(v) and v > 0:
                return v
        # 4) fallback final
        return float(fallback)

    for _, r in plan.iterrows():
        svc = str(r.get("SVC", ""))
        fch = r.get("FECHA", op_date)

        # ---------- RENTALS (apertura nativa; SPR por vehículo y DM='RENTALS') ----------
        rutas_r = _to_int(r.get("RUTAS_RENTALS", 0))
        if rutas_r > 0:
            rentals_raw = read_sheet(SHEET_ID, SHEET_TABS["rentals"])
            find_and_rename(rentals_raw, ["SVC","SVCs","LOGISTIC_CENTER_ID","LC","Facility"], "SVC", False, "Rentals")
            find_and_rename(rentals_raw, ["Tipo de vehiculo","Tipo de vehículo","Vehicle type","Tipo"], "TIPO_VEHICULO", False, "Rentals")
            # columna unidades (tolerante)
            units_col = None
            for k in ["Unidades disponibles","Unidades dispon","Units","Cantidad","Qty","QTY","COUNT"]:
                if find_and_rename(rentals_raw, [k], "UNIDADES", required=False, source_label="Rentals"):
                    units_col = "UNIDADES"; break
            if not units_col:
                gcol = _find_units_like_column(rentals_raw)
                if gcol:
                    rentals_raw.rename(columns={gcol:"UNIDADES"}, inplace=True)
                    units_col = "UNIDADES"

            mix_tbl = pd.DataFrame(columns=["VEHICULO_TIPO_HOM","PESO"])
            if units_col:
                rentals_raw["UNIDADES"] = pd.to_numeric(rentals_raw["UNIDADES"], errors="coerce").fillna(0)
                rentals_raw = rentals_raw[rentals_raw["SVC"].astype(str) == svc].copy()
                rentals_raw["VEHICULO_TIPO_HOM"] = rentals_raw["TIPO_VEHICULO"].map(homologar_vehicle_type)
                mix_tbl = rentals_raw.groupby("VEHICULO_TIPO_HOM")["UNIDADES"].sum().rename("PESO").reset_index()

            if mix_tbl.empty:
                mix_glob = _read_vehicle_mix_from_spr(op_date, "RENTALS", weeks=2)
                if not mix_glob.empty:
                    mix_tbl = mix_glob.groupby("VEHICULO_TIPO_HOM")["PESO_RUTAS"].sum().rename("PESO").reset_index()
                else:
                    mix_tbl = pd.DataFrame({"VEHICULO_TIPO_HOM":["Large Van","Small Van","Car"], "PESO":[3,2,1]})

            weights = mix_tbl.set_index("VEHICULO_TIPO_HOM")["PESO"]
            alloc = _largest_remainder_allocation(rutas_r, weights)
            diff = rutas_r - int(alloc.sum())
            if diff != 0 and not alloc.empty:
                top = weights.sort_values(ascending=False).index[0]
                alloc.loc[top] = int(alloc.loc[top]) + diff

            for veh, rutas_v in alloc.items():
                if rutas_v <= 0: continue
                spr_v = spr_for(
                    svc, "RENTALS", veh,
                    fallback=float(pd.to_numeric(r.get("SPR_RENTALS", np.nan), errors="coerce") or 20)
                )
                ships = int(round(int(rutas_v) * spr_v))
                rows.append(["Rentals", fch, svc, veh, float(spr_v), int(rutas_v), ships])

        # ---------- CROWD (mix 2 semanas; SPR por DM='CROWD' y vehículo) ----------
        rutas_crowd = _to_int(r.get("RUTAS_CROWD_BASE", 0)) + _to_int(r.get("RUTAS_CROWD_ESCALADO", 0))
        if rutas_crowd > 0:
            mix_svc = mix_crowd[mix_crowd["SVC"] == svc]
            if mix_svc.empty:
                mix_svc = mix_crowd.groupby("VEHICULO_TIPO_HOM")["PESO_RUTAS"].sum().rename("PESO_RUTAS").reset_index()
                mix_svc["SVC"] = svc
            alloc = _largest_remainder_allocation(rutas_crowd, mix_svc.set_index("VEHICULO_TIPO_HOM")["PESO_RUTAS"])
            for veh, rutas_v in alloc.items():
                if rutas_v <= 0: continue
                spr_v = spr_for(
                    svc, "CROWD", veh,
                    fallback=float(pd.to_numeric(r.get("SPR_CROWD", np.nan), errors="coerce") or 20)
                )
                ships = int(round(int(rutas_v) * spr_v))
                rows.append(["CROWD", fch, svc, veh, float(spr_v), int(rutas_v), ships])

        # ---------- MLP (apertura por caps; SPR por DM='MLP' y vehículo) ----------
        # SDD
        used_sdd  = _to_int(r.get("RUTAS_MLP_SDD_USADAS", 0))
        caps_lv   = _to_int(r.get("MLP_SDD_LV", 0))
        caps_sv   = _to_int(r.get("MLP_SDD_SV", 0))
        caps_car  = _to_int(r.get("MLP_SDD_CAR", 0))
        if used_sdd > 0:
            alloc = _alloc_mlp_by_type(used_sdd, caps_lv, caps_sv, caps_car)
            for veh, rutas_v in alloc.items():
                if rutas_v <= 0: continue
                spr_v = spr_for(
                    svc, "MLP", veh,
                    fallback=float(pd.to_numeric(r.get("SPR_MLP", np.nan), errors="coerce") or 25)
                )
                ships = int(round(int(rutas_v) * spr_v))
                rows.append(["MLP SDD", fch, svc, veh, float(spr_v), int(rutas_v), ships])

        # SPOT
        used_spot = _to_int(r.get("RUTAS_MLP_SPOT_USADAS", 0))
        caps_lv   = _to_int(r.get("MLP_SPOT_LV", 0))
        caps_sv   = _to_int(r.get("MLP_SPOT_SV", 0))
        caps_car  = _to_int(r.get("MLP_SPOT_CAR", 0))
        if used_spot > 0:
            alloc = _alloc_mlp_by_type(used_spot, caps_lv, caps_sv, caps_car)
            for veh, rutas_v in alloc.items():
                if rutas_v <= 0: continue
                spr_v = spr_for(
                    svc, "MLP", veh,
                    fallback=float(pd.to_numeric(r.get("SPR_MLP", np.nan), errors="coerce") or 25)
                )
                ships = int(round(int(rutas_v) * spr_v))
                rows.append(["MLP SPOT", fch, svc, veh, float(spr_v), int(rutas_v), ships])

        # BACKLOG (prioridad LV→SV→Car; sin caps por tipo)
        used_back = _to_int(r.get("RUTAS_MLP_BACKLOG_USADAS", 0))
        if used_back > 0:
            alloc_back = _alloc_mlp_by_type(used_back, 10**9, 10**9, 10**9)
            for veh, rutas_v in alloc_back.items():
                if rutas_v <= 0: continue
                spr_v = spr_for(
                    svc, "MLP", veh,
                    fallback=float(pd.to_numeric(r.get("SPR_MLP", np.nan), errors="coerce") or
                                   pd.to_numeric(r.get("SPR_USADO", np.nan), errors="coerce") or 25)
                )
                ships = int(round(int(rutas_v) * spr_v))
                rows.append(["MLP BACKLOG", fch, svc, veh, float(spr_v), int(rutas_v), ships])

        # ---------- DC / SP (mix 2 semanas; rutas = shipments/SPR_v, usando SPR_USADO como fallback) ----------
        ship_dc = _to_int(r.get("SHIPMENTS_DC", 0))
        if ship_dc > 0:
            mix_svc = mix_dc[mix_dc["SVC"] == svc]
            if mix_svc.empty:
                mix_svc = mix_dc.groupby("VEHICULO_TIPO_HOM")["PESO_RUTAS"].sum().rename("PESO_RUTAS").reset_index()
                mix_svc["SVC"] = svc
            alloc_ship = _largest_remainder_allocation(ship_dc, mix_svc.set_index("VEHICULO_TIPO_HOM")["PESO_RUTAS"])
            for veh, ships_v in alloc_ship.items():
                if ships_v <= 0: continue
                spr_v = spr_for(svc, "DC", veh, fallback=float(pd.to_numeric(r.get("SPR_USADO", np.nan), errors="coerce") or 25))
                rutas_v = int(round(ships_v / max(1.0, spr_v)))
                rows.append(["DC", fch, svc, veh, float(spr_v), int(rutas_v), int(ships_v)])

        ship_sp = _to_int(r.get("SHIPMENTS_SP", 0))
        if ship_sp > 0:
            mix_svc = mix_sp[mix_sp["SVC"] == svc]
            if mix_svc.empty:
                mix_svc = mix_sp.groupby("VEHICULO_TIPO_HOM")["PESO_RUTAS"].sum().rename("PESO_RUTAS").reset_index()
                mix_svc["SVC"] = svc
            alloc_ship = _largest_remainder_allocation(ship_sp, mix_svc.set_index("VEHICULO_TIPO_HOM")["PESO_RUTAS"])
            for veh, ships_v in alloc_ship.items():
                if ships_v <= 0: continue
                spr_v = spr_for(svc, "SP", veh, fallback=float(pd.to_numeric(r.get("SPR_USADO", np.nan), errors="coerce") or 25))
                rutas_v = int(round(ships_v / max(1.0, spr_v)))
                rows.append(["SP", fch, svc, veh, float(spr_v), int(rutas_v), int(ships_v)])

    detalle = pd.DataFrame(rows, columns=[
        "DELIVERY_MOD","FECHA","SVC","SHP_LG_VEHICLE_TYPE","SPR","Rutas","Shipments"
    ])

    # Orden visual
    dm_order = pd.CategoricalDtype(["Rentals","CROWD","MLP SDD","MLP SPOT","MLP BACKLOG","DC","SP"], ordered=True)
    if "DELIVERY_MOD" in detalle.columns and not detalle.empty:
        try:
            detalle["DELIVERY_MOD"] = detalle["DELIVERY_MOD"].astype(dm_order)
            detalle = detalle.sort_values(["SVC","DELIVERY_MOD","SHP_LG_VEHICLE_TYPE"]).reset_index(drop=True)
        except Exception:
            detalle = detalle.sort_values(["SVC","DELIVERY_MOD","SHP_LG_VEHICLE_TYPE"]).reset_index(drop=True)

    detalle["Rutas"] = detalle["Rutas"].astype(int)
    detalle["Shipments"] = detalle["Shipments"].astype(int)
    return detalle

def _to_num_series(s):
    """Convierte serie con comas/% a numérico (NaN→0)."""
    return pd.to_numeric(
        pd.Series(s).astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False),
        errors="coerce"
    ).fillna(0)

def reconcile_plan_with_detail(plan_df: pd.DataFrame, detalle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Toma la tabla de arriba (plan_df) y el detalle (detalle_df),
    y reemplaza en 'arriba' los SHIP_* por la suma del detalle.
    Además muestra SPR resultantes = shipments / rutas.
    No toca las rutas (las rutas son la “verdad” arriba).
    """
    if plan_df is None or plan_df.empty or detalle_df is None or detalle_df.empty:
        return plan_df

    df = plan_df.copy()

    # --- extraemos totales de shipments del detalle por DM/SVC ---
    det = detalle_df.copy()
    det["Shipments"] = pd.to_numeric(det["Shipments"], errors="coerce").fillna(0)
    det["Rutas"]     = pd.to_numeric(det["Rutas"], errors="coerce").fillna(0)
    dm = det["DELIVERY_MOD"].astype(str)

    ship_rent = det[dm.eq("Rentals")].groupby("SVC")["Shipments"].sum().rename("SHIP_RENTALS_DET")
    ship_crow = det[dm.eq("CROWD")].groupby("SVC")["Shipments"].sum().rename("SHIP_CROWD_DET")
    ship_mlp  = det[dm.isin(["MLP SDD","MLP SPOT","MLP BACKLOG"])].groupby("SVC")["Shipments"].sum().rename("SHIP_MLP_DET")
    # (Opcional) DC/SP
    ship_dc   = det[dm.eq("DC")].groupby("SVC")["Shipments"].sum().rename("SHIP_DC_DET")
    ship_sp   = det[dm.eq("SP")].groupby("SVC")["Shipments"].sum().rename("SHIP_SP_DET")

    # Mergear a la tabla de arriba
    for ser in [ship_rent, ship_crow, ship_mlp, ship_dc, ship_sp]:
        if not ser.empty:
            df = df.merge(ser.reset_index(), on="SVC", how="left")

    # --- columnas numéricas que usaremos como denominadores ---
    rutas_r = _to_num_series(df.get("RUTAS_RENTALS", 0))
    rutas_c = _to_num_series(df.get("RUTAS_CROWD_BASE", 0)) + _to_num_series(df.get("RUTAS_CROWD_ESCALADO", 0))
    rutas_m = (
        _to_num_series(df.get("RUTAS_MLP_SDD_USADAS", 0)) +
        _to_num_series(df.get("RUTAS_MLP_SPOT_USADAS", 0)) +
        _to_num_series(df.get("RUTAS_MLP_BACKLOG_USADAS", 0))
    )

    # backups de SPR objetivo (solo referencia)
    if "SPR_RENTALS" in df.columns: df["SPR_RENTALS_OBJ"] = df["SPR_RENTALS"]
    if "SPR_CROWD"   in df.columns: df["SPR_CROWD_OBJ"]   = df["SPR_CROWD"]
    if "SPR_MLP"     in df.columns: df["SPR_MLP_OBJ"]     = df["SPR_MLP"]

    # --- Reemplazar shipments “arriba” por los del detalle ---
    df["SHIP_RENTALS"] = _to_num_series(df.get("SHIP_RENTALS", 0))
    df["SHIP_CROWD"]   = _to_num_series(df.get("SHIP_CROWD", 0))
    # DC/SP “arriba” pueden no existir; solo informativo si los quieres usar en KPIs
    base_dc = _to_num_series(df.get("SHIPMENTS_DC", 0))
    base_sp = _to_num_series(df.get("SHIPMENTS_SP", 0))

    if "SHIP_RENTALS_DET" in df.columns:
        df["SHIP_RENTALS"] = _to_num_series(df["SHIP_RENTALS_DET"])
    if "SHIP_CROWD_DET" in df.columns:
        df["SHIP_CROWD"] = _to_num_series(df["SHIP_CROWD_DET"])

    ship_mlp_det = _to_num_series(df.get("SHIP_MLP_DET", 0))  # total MLP desde detalle

    # --- SPR resultantes = shipments / rutas (evita división por 0) ---
    def safe_div(num, den):
        den = den.replace(0, np.nan)
        out = (num / den).fillna(0)
        return out

    df["SPR_RENTALS"] = safe_div(_to_num_series(df["SHIP_RENTALS"]), rutas_r)
    df["SPR_CROWD"]   = safe_div(_to_num_series(df["SHIP_CROWD"]), rutas_c)
    if rutas_m.sum() > 0:
        df["SPR_MLP"] = safe_div(ship_mlp_det, rutas_m)

    # --- Recalcular KPIs de capacidad con shipments reconciliados ---
    fcst = _to_num_series(df.get("FCST", 0))
    cap_total = base_dc + base_sp + _to_num_series(df["SHIP_RENTALS"]) + _to_num_series(df["SHIP_CROWD"]) + ship_mlp_det
    df["CAP_TOTAL"]   = cap_total
    df["CAP_VS_FCST"] = (cap_total / fcst.replace(0, np.nan)).fillna(0).round(2)
    df["CAP_DIFF_ABS"] = (fcst - cap_total).abs().round(2)
    df["RIESGO"] = np.where(cap_total + 1e-9 >= fcst, "OK", "RIESGO")

    # Limpiar columnas auxiliares
    df.drop(columns=[c for c in ["SHIP_RENTALS_DET","SHIP_CROWD_DET","SHIP_MLP_DET","SHIP_DC_DET","SHIP_SP_DET"] if c in df.columns],
            inplace=True, errors="ignore")

    # Devolvemos con el mismo formato que la tabla 1 (llama a tu formateador)
    return apply_output_adjustments(df)




# -----------------------------------------------------------------------------
# 6) UI
# -----------------------------------------------------------------------------

# ---------------------------------------------------------
# Estilos de marca MELI para Streamlit
# ---------------------------------------------------------
def inject_brand_css(
    yellow="#FFE600",   # Amarillo MELI
    blue="#2D3277",     # Azul MELI
    soft="#F6F7FB",     # Fondo suave de tarjetas
    text="#111827"      # Texto principal
):
    st.markdown(
        f"""
        <style>
        :root {{
          --meli-yellow: {yellow};
          --meli-blue: {blue};
          --meli-soft: {soft};
          --meli-text: {text};
        }}

        /* Header transparente (solo deja tu título) */
        [data-testid="stHeader"] {{ background: transparent; }}

        /* Botón primario */
        .stButton>button {{
          background: var(--meli-blue) !important;
          color: #fff !important;
          border: 0 !important;
          border-radius: 12px !important;
          padding: 0.55rem 1.0rem !important;
          box-shadow: 0 2px 10px rgba(0,0,0,.06);
        }}
        .stButton>button:hover {{ filter: brightness(1.06); }}

        /* Radio / checkbox con color de acento */
        input[type="radio"], input[type="checkbox"] {{ accent-color: var(--meli-blue); }}

        /* Chips del multiselect */
        div[data-baseweb="tag"] {{
          background: var(--meli-yellow) !important;
          color: #111827 !important;
          border-radius: 10px !important;
          border: 0 !important;
        }}

        /* Métricas (st.metric) como tarjetas */
        div[data-testid="stMetric"] {{
          background: var(--meli-soft);
          border: 1px solid rgba(0,0,0,.06);
          border-radius: 14px;
          padding: 10px 14px;
        }}
        div[data-testid="stMetric"] label p {{
          color: var(--meli-blue) !important;
          font-weight: 700 !important;
        }}

        /* Expanders más limpios */
        details>summary {{
          background: var(--meli-soft);
          border-radius: 10px;
          padding: 6px 10px;
        }}

        /* DataFrame: cabecera azul, filas zebra suaves */
        .stDataFrame table thead tr th {{
          background: var(--meli-blue) !important;
          color: #fff !important;
        }}
        .stDataFrame tbody tr:nth-child(odd) {{
          background: rgba(0,0,0,.015);
        }}

        /* Links del título */
        h1, h2, h3 {{ color: var(--meli-text); }}
        a {{ color: var(--meli-blue); }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Opcional: helper para “vista bonita” estática con pandas Styler
def render_pretty_table(df, caption=None):
    if df is None or df.empty:
        st.info("Sin datos para mostrar.")
        return
    # Si tus números vienen con comas/%, los dejamos como están (solo pintamos riesgo)
    cols = [c for c in df.columns if c.upper() == "RIESGO"]
    def color_riesgo(row):
        return [("background-color: #FFE1E1; color:#B00020;" if str(v).upper()=="RIESGO" else "") for v in row]
    sty = (df.style
        .apply(color_riesgo, axis=1)
        .hide(axis="index"))
    if caption:
        st.caption(caption)
    st.table(sty)

# ---------------------------------------------------------------------
# 6) UI
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Mel-IA (Copiloto de tu flota en Mercado Libre)",
    layout="wide",
    page_icon=LOGO_IMAGE if LOGO_IMAGE is not None else "🚛"
)
inject_brand_css()  # ← ACTIVA los estilos de marca

# --- Tamaño del logo: default + persistencia ---
DEFAULT_LOGO_WIDTH = 200
try:
    DEFAULT_LOGO_WIDTH = int(
        st.secrets.get("LOGO_WIDTH", os.environ.get("LOGO_WIDTH", DEFAULT_LOGO_WIDTH))
    )
except Exception:
    pass

if "logo_width" not in st.session_state:
    st.session_state["logo_width"] = DEFAULT_LOGO_WIDTH

LOGO_WIDTH = int(st.session_state["logo_width"])


with st.sidebar.expander("🎨 Apariencia", expanded=False):
    _def_h = int(st.session_state.get("header_h", 60))
    _def_l = int(st.session_state.get("logo_h", 200))

    header_h = st.slider("Altura barra (px)", 64, 1200, _def_h, step=2)
    # ↑ también amplié el máximo por si quieres una barra muy alta

    logo_h   = st.slider("Altura logo (px)",  24, 1200, min(_def_l, 400), step=2)
    #            ^^^^^^^^^^^^^^^^^^ sube el máximo (ej. 1200)

    st.session_state["header_h"] = int(header_h)
    st.session_state["logo_h"]   = int(logo_h)

# La barra debe ser al menos tan alta como el logo
final_header_h = max(int(st.session_state["header_h"]), int(st.session_state["logo_h"]) -5)

render_fixed_header(
    logo_img=LOGO_IMAGE,
    header_h=final_header_h,
    logo_h=int(st.session_state["logo_h"]),
)



def render_fixed_header(
    logo_img=None,
    title="Mel-IA — Plan táctico (diario por SVC)",
    subtitle="Copiloto de planificación táctica de flota • Rentals • Crowd • MLP",
    header_h=104,
    logo_h=500,
):
    import io, base64
    def _img_to_b64(img):
        buf = io.BytesIO()
        try:
            img.save(buf, format="PNG")
        except Exception:
            img = img.convert("RGBA")
            img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    if logo_img is not None:
        try:
            b64 = _img_to_b64(logo_img)
            # estilo inline con !important por si hay reglas globales
            logo_html = (
                f'<img src="data:image/png;base64,{b64}" alt="logo" '
                f'style="height:{int(logo_h)}px !important; '
                f'width:auto !important; max-height:none !important; '
                f'max-width:none !important; display:block;" />'
            )
        except Exception:
            logo_html = f'<div style="font-size:{int(logo_h)}px; line-height:{int(logo_h)}px;">🚛</div>'
    else:
        logo_html = f'<div style="font-size:{int(logo_h)}px; line-height:{int(logo_h)}px;">🚛</div>'

    st.markdown(
        f"""
        <style>
          :root {{
            --meli-header-h: {int(header_h)}px;
            --meli-logo-h: {int(logo_h)}px;
          }}

          .meli-fixed {{
            position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
            background: #fff; height: var(--meli-header-h);
            border-bottom: 1px solid rgba(0,0,0,.06);
            box-shadow: 0 1px 6px rgba(0,0,0,.04);
            overflow: visible; /* por si el navegador quisiera recortar */
          }}
          .meli-wrap {{
            max-width: 1200px; margin: 0 auto;
            height: 100%;
            display: grid; grid-template-columns: auto 1fr;
            align-items: center; gap: 14px; padding: 6px 12px;
          }}

          /* Mata cualquier límite global de imágenes dentro del header */
          .meli-logo img {{
            height: var(--meli-logo-h) !important;
            width: auto !important;
            max-height: none !important;
            max-width: none !important;
            display: block;
            image-rendering: crisp-edges;
          }}

          .meli-title {{ margin: 0; font-size: 32px; line-height: 1.15; font-weight: 800; }}
          .meli-sub {{ margin: 2px 0 0 0; color: #6b7280; font-size: 13px; }}

          /* Quita header nativo y compensa offset del main */
          [data-testid="stHeader"] {{ display: none; }}
          .block-container {{ padding-top: calc(var(--meli-header-h) + 12px) !important; }}
        </style>

        <div class="meli-fixed">
          <div class="meli-wrap">
            <div class="meli-logo">{logo_html}</div>
            <div>
              <h1 class="meli-title">{title}</h1>
              <div class="meli-sub">{subtitle}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )







# --- Sidebar: configuración del proyecto / credenciales / healthcheck ---
st.sidebar.markdown("## 🗂️ Proyecto")
raw_input = st.sidebar.text_input(
    "SHEET_ID (puede ser URL o ID)",
    value=SHEET_ID or "",
    placeholder="pega aquí la URL o el ID del Sheet",
)
new_sheet_id = sanitize_sheet_id(raw_input)
if new_sheet_id != SHEET_ID:
    SHEET_ID = new_sheet_id
    st.cache_data.clear()
    st.session_state["sheet_id"] = SHEET_ID

if SHEET_ID:
    st.sidebar.markdown(f"**Sheet (ID):** `{SHEET_ID}`")
else:
    st.sidebar.error("Configura/Pega `SHEET_ID`.")

st.sidebar.markdown("## 🔐 Credenciales")
st.sidebar.code(f"Comparte el Sheet con:\n{SERVICE_EMAIL}")

with st.sidebar.expander("Estado de conexión", expanded=False):
    try:
        if SHEET_ID:
            hc = quick_healthcheck(SHEET_ID)
            ok = hc.get("ok") == "true"
            st.write("OK ✅" if ok else "Fallo ❌")
            st.caption(hc.get("note", ""))
        else:
            st.info("Proporciona SHEET_ID para checar acceso.")
    except Exception as e:
        st.error("No se pudo validar acceso.")
        st.caption(str(e))


spr_mode = st.radio(
    "SPR objetivo",
    options=["promedio", "peak", "plan"],
    horizontal=True,
    index=0,
)
spr_mode = spr_mode.strip().lower()  # normaliza

run_btn = False
auto_run = False
sel_svcs: List[str] = []

# --- Carga de opciones (SVCs disponibles) ---
with st.expander("▶️ Cargando datos...", expanded=True):
    try:
        if not SHEET_ID:
            st.warning("Falta `SHEET_ID`. Pégalo en la barra lateral.")
            svc_list = []
        else:
            fcst_svcs    = load_fcst()[["SVC"]]
            caps         = load_capacity_caps()
            cap_svcs     = caps[["SVC"]] if "SVC" in caps.columns else caps.to_frame(name="SVC")
            crowd_svcs   = load_crowd_caps_for(date.today())[["SVC"]]
            rents_svcs   = load_rentals_caps_from_sheet()[["SVC"]]
            rent_fb_svcs = load_rentals_fallback()[["SVC"]]
            mlp_svcs     = load_mlp_caps_from_srm()[["SVC"]]

            base_svcs = pd.concat(
                [fcst_svcs, cap_svcs, crowd_svcs, rents_svcs, rent_fb_svcs, mlp_svcs],
                axis=0
            ).drop_duplicates()
            base_svcs = _as_str_cols(base_svcs, ["SVC"])
            svc_list = sorted(base_svcs["SVC"].dropna().astype(str).unique().tolist())

        default_sel = [s for s in DEFAULT_SVCS if s in svc_list] or svc_list[:4]
        sel_svcs = st.multiselect("Filtrar SVC", options=svc_list, default=default_sel, placeholder="Selecciona SVCs")
        st.write(" ")
        run_btn = st.button("Calcular plan", type="primary")
    except Exception as e:
        st.error("No se pudieron preparar los filtros.")
        show_exception(e, "Detalles (filtros)")

# --- Helper para sumar columnas numéricas formateadas (para métricas) ---
def _sum_numeric_col(df, col):
    s = (
        df[col].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    return int(pd.to_numeric(s, errors="coerce").fillna(0).sum())

# --- Auto-run (primer render) ---
if 'auto_run_once' not in st.session_state:
    st.session_state['auto_run_once'] = True
    auto_run = True
else:
    auto_run = False

# --- Orquestación: plan base → detalle → reconciliar → mostrar ---
try:
    # ✅ Clave: además de run_btn/auto_run, muestra el último plan guardado en session_state
    if run_btn or auto_run or ("plan_df" in st.session_state and st.session_state["plan_df"] is not None):
        if not SHEET_ID:
            st.warning("Proporciona `SHEET_ID` para calcular.")
        else:
            # Si se presionó Calcular (o es el primer render), recalculamos y PERSISTIMOS
            if run_btn or auto_run:
                # 1) Calcula plan base (objetivo)
                plan_base = compute_plan(spr_mode, sel_svcs or DEFAULT_SVCS)

                if plan_base.empty:
                    st.warning("No hay datos para mostrar con los filtros seleccionados.")
                    st.session_state["plan_df"] = None
                    st.session_state["detalles_df"] = None
                else:
                    # 2) Detalle por vehículo (usa rutas “arriba” y SPR por DM/vehículo)
                    try:
                        detalles = expand_to_vehicle_level(plan_base, spr_mode)
                    except Exception as e:
                        st.error("No se pudo construir el detalle por vehículo.")
                        show_exception(e, "Detalle vehículo (traceback)")
                        detalles = pd.DataFrame(columns=["DELIVERY_MOD","FECHA","SVC","SHP_LG_VEHICLE_TYPE","SPR","Rutas","Shipments"])

                    # 3) Reconciliación: shipments de abajo → arriba; SPR resultante arriba
                    plan = reconcile_plan_with_detail(plan_base, detalles)

                    # ✅ NUEVO: persistir para que el chat y los reruns lo usen
                    st.session_state["plan_df"] = plan.copy()
                    st.session_state["detalles_df"] = detalles.copy()

            # ---- Mostrar siempre lo último disponible (aunque este run no haya recalculado)
            plan = st.session_state.get("plan_df")
            detalles = st.session_state.get("detalles_df")

            if plan is None or plan.empty:
                st.warning("No hay datos para mostrar. Calcula el plan.")
            else:
                # 4) KPIs header basados en el plan reconciliado
                svcs_uniques = plan["SVC"].nunique()

                dem_aj_total = _sum_numeric_col(plan, "DEMANDA_AJUSTADA")
                rutas_falt_total = _sum_numeric_col(plan, "RUTAS_FALTANTES")

                ship_dc   = _sum_numeric_col(plan, "SHIPMENTS_DC")
                ship_sp   = _sum_numeric_col(plan, "SHIPMENTS_SP")
                ship_rent = _sum_numeric_col(plan, "SHIP_RENTALS")
                ship_crwd = _sum_numeric_col(plan, "SHIP_CROWD")
                cap_total = _sum_numeric_col(plan, "CAP_TOTAL")

                # Shipments MLP = CAP_TOTAL - (DC + SP + Rentals + Crowd)
                ship_mlp_total = max(cap_total - ship_dc - ship_sp - ship_rent - ship_crwd, 0)

                # Shipments de capacidad (solo Rentals + Crowd + MLP)
                ship_capacidad_total = ship_rent + ship_crwd + ship_mlp_total

                # Rutas de capacidad = Rentals + Crowd (base+esc) + MLP (SDD/SPOT/BACKLOG)
                rutas_cap_total = (
                    _sum_numeric_col(plan, "RUTAS_RENTALS")
                    + _sum_numeric_col(plan, "RUTAS_CROWD_BASE")
                    + _sum_numeric_col(plan, "RUTAS_CROWD_ESCALADO")
                    + _sum_numeric_col(plan, "RUTAS_MLP_SDD_USADAS")
                    + _sum_numeric_col(plan, "RUTAS_MLP_SPOT_USADAS")
                    + _sum_numeric_col(plan, "RUTAS_MLP_BACKLOG_USADAS")
                )

                spr_cap_result = ship_capacidad_total / max(rutas_cap_total, 1)
                spr_cap_label  = f"{spr_cap_result:,.1f}"

                # 5) Mostrar KPIs (sin "Rutas SPR base")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("SVCs", svcs_uniques)
                c2.metric("Demanda ajustada", f"{dem_aj_total:,}")
                c3.metric("Rutas faltantes", f"{rutas_falt_total:,}")
                c4.metric("SPR (resultante capacidad)", spr_cap_label)

                # 6) Mostrar tabla “arriba” (reconciliada)
                st.dataframe(plan, use_container_width=True, hide_index=True)

                # 7) Mostrar detalle
                st.markdown("### Detalle por vehículo – SVC – día")
                st.dataframe(detalles, use_container_width=True, hide_index=True)

except Exception as e:
    st.error("Ocurrió un error durante el cálculo.")
    show_exception(e, "Traceback completo")

# =========================
# 🤖 Chat Mel-IA (Bloque B)
# =========================
import os, textwrap
import streamlit as st

# --- IMPORT & CLIENTE OPENAI ROBUSTO ---
try:
    from openai import OpenAI
    _openai_ok = True
except Exception:
    _openai_ok = False
    OpenAI = None  # evita NameError

client = None
if _openai_ok:
    api_key = (
        st.secrets.get("openai", {}).get("api_key") or
        st.secrets.get("OPENAI_API_KEY") or
        os.environ.get("OPENAI_API_KEY")
    )
    try:
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key  # el SDK la toma del env
        client = OpenAI()
    except Exception as e:
        client = None
        st.info(f"El chat no está disponible (OpenAI client): {e}")
else:
    st.info("El chat no está disponible porque falta el paquete 'openai'.")

# --- ASEGURA PERSISTENCIA DE RESULTADOS DEL PLAN ---
# (Si en otra parte ya guardas estos dfs en session_state, puedes omitir este bloque)
plan = st.session_state.get("plan_df")   # DataFrame de la tabla de arriba
detalles = st.session_state.get("detalles_df")  # DataFrame de la tabla de abajo

# --- UI DEL CHAT ---
st.markdown("## 🤖 Pregunta a Mel-IA sobre tus datos")
st.caption("Puedes preguntarle a Mel-IA sobre los datos del plan, riesgos, spr, mlps, etc.")

# Si no hay plan persistido, avisa (evita consultas vacías)
if not st.session_state.get("plan_df") is not None:
    st.info("Primero calcula el plan (botón 'Calcular plan') para que el chat tenga contexto.")


# Historial persistente del chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Aísla el input en un formulario para evitar reruns al teclear/Enter
with st.form("qa_form", clear_on_submit=False):
    user_q = st.text_input(
        "Escribe tu pregunta (ej. “¿cuántas rutas rentals en SGD1?”)",
        key="user_q",
        placeholder="Pregunta sobre tus datos…"
    )
    ask = st.form_submit_button("Preguntar")

if ask:
    if not client:
        st.error("El chat no está disponible (cliente OpenAI no inicializado o paquete faltante).")
    elif not user_q.strip():
        st.warning("Escribe una pregunta.")
    else:
        # Construye contexto desde los dataframes persistidos
        context_parts = []
        def _df_to_context(title, df, n=150):
            try:
                return f"{title}:\n" + df.head(n).to_csv(index=False)
            except Exception:
                return f"{title}: (no disponible)\n"

        if plan is not None and hasattr(plan, "empty") and not plan.empty:
            context_parts.append(_df_to_context("Tabla plan (arriba)", plan, 200))
        else:
            context_parts.append("Tabla plan (arriba): (no disponible)\n")

        if detalles is not None and hasattr(detalles, "empty") and not detalles.empty:
            context_parts.append(_df_to_context("Tabla detalle (abajo)", detalles, 250))
        else:
            context_parts.append("Tabla detalle (abajo): (no disponible)\n")

        system_msg = (
            "Eres un analista de planeación táctica. Responde en español, "
            "claro y conciso. Si haces cálculos, muéstralos. "
            "Usa solo el contexto provisto; si algo no está en los datos, dilo."
        )
        context_block = "\n".join(context_parts)

        # OJO: nada de backslashes dentro de { } en f-strings
        user_prompt = textwrap.dedent(
            f"""CONTEXTO:
{context_block}

PREGUNTA:
{user_q}
"""
        )

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",   # usa el modelo que tengas habilitado
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=700,
            )
            answer = resp.choices[0].message.content.strip()
            st.session_state.chat_history.append((user_q, answer))
        except Exception as e:
            st.error(f"No se pudo obtener respuesta: {e}")

# --- Mostrar historial (no dispara recalculos del plan) ---
if st.session_state.chat_history:
    st.write("---")
    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")
