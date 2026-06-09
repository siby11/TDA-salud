import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import haversine_distances
from ripser import ripser as _ripser_fn

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TDA Salud CDMX",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


COLORS = {
    "Muy bajo":  "#2ecc71",
    "Bajo":      "#a8e6a3",
    "Medio":     "#f9e07a",
    "Alto":      "#f0854f",
    "Muy alto":  "#c0392b",
    "Privado":   "#4e8df5",
    "Público":   "#e74c3c",
    "No gubernamental": "#9b59b6",
}

COLORS_IDS = {
    "Muy bajo":  "#c0392b",
    "Bajo":      "#f0854f",
    "Medio":     "#f9e07a",
    "Alto":      "#a8e6a3",
    "Muy alto":  "#2ecc71",
}

# Mapa clave AGEB 
MUN_MAP = {
    "002": "Azcapotzalco",        "003": "Coyoacán",
    "004": "Cuajimalpa de Morelos","005": "Gustavo A. Madero",
    "006": "Iztacalco",           "007": "Iztapalapa",
    "008": "La Magdalena Contreras","009": "Milpa Alta",
    "010": "Álvaro Obregón",      "011": "Tláhuac",
    "012": "Tlalpan",             "013": "Xochimilco",
    "014": "Benito Juárez",       "015": "Cuauhtémoc",
    "016": "Miguel Hidalgo",      "017": "Venustiano Carranza",
}
MUN_INV = {v: k for k, v in MUN_MAP.items()}

# ── Helpers TDA ────────────────────────────────────────────────────────────────
def _dist_km(pts_deg: np.ndarray) -> np.ndarray:
    """Matriz de distancias haversine en km."""
    return haversine_distances(np.radians(pts_deg)) * 6371.0

def _dist_cross_km(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Matriz de distancias cruzadas haversine en km (a vs b)."""
    return haversine_distances(np.radians(a_deg), np.radians(b_deg)) * 6371.0

def _circles(lats, lons, r_km: float, n: int = 32):
    """Un solo trace con todos los círculos separados por None."""
    R, angles = 6371.0, np.linspace(0, 2 * np.pi, n + 1)
    all_lat, all_lon = [], []
    for lat, lon in zip(lats, lons):
        c = np.cos(np.radians(lat))
        all_lat += (lat + np.degrees(r_km / R * np.cos(angles))).tolist() + [None]
        all_lon += (lon + np.degrees(r_km / R / c * np.sin(angles))).tolist() + [None]
    return all_lat, all_lon

def _edges(lats, lons, D: np.ndarray, eps: float, max_edges: int = 3000):
    """Aristas del complejo VR: pares (i,j) con d(i,j) ≤ eps."""
    el, eo = [], []
    n, count = len(lats), 0
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= eps:
                el += [lats[i], lats[j], None]
                eo += [lons[i], lons[j], None]
                count += 1
                if count >= max_edges:
                    return el, eo
    return el, eo

def _barcode_fig(dgm: np.ndarray, title: str, color: str, max_eps: float):
    """Diagrama de barras de persistencia (barcode) en Plotly."""
    if len(dgm) == 0:
        return go.Figure().update_layout(title=title, height=300)
    births = dgm[:, 0]
    deaths = np.where(np.isinf(dgm[:, 1]), max_eps * 1.05, dgm[:, 1])
    pers = deaths - births
    order = np.argsort(pers)        
    x_seg, y_seg = [], []
    for rank, idx in enumerate(order):
        x_seg += [float(births[idx]), float(deaths[idx]), None]
        y_seg += [rank, rank, None]
    fig = go.Figure(go.Scatter(
        x=x_seg, y=y_seg, mode="lines",
        line=dict(color=color, width=2),
        hoverinfo="skip",
    ))
    fig.update_layout(
        title=title, height=320,
        xaxis_title="Radio ε (km)", yaxis=dict(showticklabels=False, title="Feature"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig

def _persistence_fig(dgms, max_eps: float):
    """Diagrama de persistencia (birth vs death) para H0 y H1."""
    rows = []
    for dim, dgm in enumerate(dgms):
        for b, d in dgm:
            d_p = min(float(d), max_eps * 1.05) if np.isinf(d) else float(d)
            rows.append({"Dim": f"H{dim}", "Nacimiento": float(b), "Muerte": d_p,
                         "Persistencia": d_p - float(b)})
    if not rows:
        return go.Figure()
    df_p = pd.DataFrame(rows)
    fig = px.scatter(
        df_p, x="Nacimiento", y="Muerte", color="Dim",
        color_discrete_map={"H0": "#e74c3c", "H1": "#3498db"},
        size="Persistencia", size_max=16, opacity=0.75,
        hover_data={"Persistencia": ":.3f"},
        title="Diagrama de Persistencia",
        labels={"Nacimiento": "Nacimiento ε (km)", "Muerte": "Muerte ε (km)"},
        height=400,
    )
    lim = max_eps * 1.05
    fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                  line=dict(color="gray", dash="dot", width=1))
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig

@st.cache_data
def _compute_tda(pts_key: tuple):
    pts = np.array(pts_key)
    D = _dist_km(pts)
    res = _ripser_fn(D, distance_matrix=True, maxdim=1)
    return res["dgms"], D.tolist()

@st.cache_data
def _compute_coverage(pub_key: tuple, ageb_key: tuple):
    D = _dist_cross_km(np.array(pub_key), np.array(ageb_key))
    return D.min(axis=0).tolist()

@st.cache_data
def _compute_tda_full(pts_key: tuple):
    """TDA con cociclos — devuelve dgms, matriz D y centros aproximados de H1."""
    pts = np.array(pts_key)
    D   = _dist_km(pts)
    res = _ripser_fn(D, distance_matrix=True, maxdim=1, do_cocycles=True)
    dgms_serial = [d.tolist() for d in res["dgms"]]
    h1_centers  = []
    if len(res["cocycles"]) > 1:
        for cocycle in res["cocycles"][1]:
            if len(cocycle) > 0:
                verts     = sorted(set(int(v) for row in cocycle for v in row[:2]))
                cycle_pts = pts[verts]
                if len(verts) >= 3:
                    center = _chebyshev_center_deg(cycle_pts)
                else:
                    center = (float(cycle_pts[:, 0].mean()), float(cycle_pts[:, 1].mean()))
                h1_centers.append(center)
    return dgms_serial, D.tolist(), h1_centers


def _chebyshev_center_deg(cycle_pts_deg: np.ndarray) -> tuple:
    """
    Centro de Chebyshev del casco convexo de los vértices del ciclo H1.
    Es el centro del mayor círculo inscrito — más preciso que el centroide
    cuando los vértices están distribuidos asimétricamente alrededor del hueco.
    Opera en coordenadas km locales (isotrópicas) y devuelve (lat, lon) en grados.
    """
    from scipy.spatial import ConvexHull
    from scipy.optimize import linprog
    lat0   = float(cycle_pts_deg[:, 0].mean())
    LAT_KM = 111.0
    LON_KM = 111.0 * np.cos(np.radians(lat0))
    pts_km = np.column_stack([cycle_pts_deg[:, 0] * LAT_KM,
                               cycle_pts_deg[:, 1] * LON_KM])
    try:
        hull  = ConvexHull(pts_km)
        A     = hull.equations[:, :2]
        b_eq  = hull.equations[:, 2]
        norms = np.linalg.norm(A, axis=1, keepdims=True)
        # min -r  s.t.  [A | norms]*[x; r] <= -b_eq,  r >= 0
        res = linprog(
            np.array([0.0, 0.0, -1.0]),
            A_ub=np.hstack([A, norms]),
            b_ub=-b_eq,
            bounds=[(None, None), (None, None), (0, None)],
            method="highs",
        )
        if res.success:
            return float(res.x[0] / LAT_KM), float(res.x[1] / LON_KM)
    except Exception:
        pass
    return float(cycle_pts_deg[:, 0].mean()), float(cycle_pts_deg[:, 1].mean())


def _dist_pt_seg(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distancia mínima del punto p al segmento a–b (en coords locales km)."""
    ab = b - a
    t  = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
    return float(np.linalg.norm(p - (a + np.clip(t, 0, 1) * ab)))

def _border_dists(centers: list, pts_deg: np.ndarray) -> np.ndarray:
    """
    Distancia (km) de cada centro de hueco H₁ al borde del casco convexo
    de pts_deg. Permite detectar huecos que son artefactos del efecto de borde.
    """
    from scipy.spatial import ConvexHull
    if len(pts_deg) < 4 or len(centers) == 0:
        return np.zeros(len(centers))
    lat0   = pts_deg[:, 0].mean()
    LAT_KM = 111.0
    LON_KM = 111.0 * np.cos(np.radians(lat0))
    pts_km = np.column_stack([pts_deg[:, 0] * LAT_KM, pts_deg[:, 1] * LON_KM])
    try:
        hverts = pts_km[ConvexHull(pts_km).vertices]
        n      = len(hverts)
        return np.array([
            min(_dist_pt_seg(
                    np.array([lat * LAT_KM, lon * LON_KM]),
                    hverts[i], hverts[(i + 1) % n]
                ) for i in range(n))
            for lat, lon in centers
        ])
    except Exception:
        return np.zeros(len(centers))

def _betti_curves(h0: np.ndarray, h1: np.ndarray, eps_range: np.ndarray):
    """Números de Betti β₀ y β₁ para cada ε."""
    d0 = np.where(np.isinf(h0[:, 1]), 1e9, h0[:, 1])
    b0 = np.array([(d0 > e).sum() for e in eps_range], dtype=int)
    if len(h1):
        b1 = np.array([
            int(((h1[:, 0] <= e) & ((h1[:, 1] > e) | np.isinf(h1[:, 1]))).sum())
            for e in eps_range
        ], dtype=int)
    else:
        b1 = np.zeros(len(eps_range), dtype=int)
    return b0, b1

# ── Carga de datos ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    denue    = pd.read_csv("data/processed/denue_salud_cdmx_clean.csv")
    coneval  = pd.read_csv("data/processed/coneval_ageb_cdmx_limpio.csv")
    ids      = pd.read_csv("data/processed/ids_ageb_cdmx_limpio.csv")
    pob_raw  = pd.read_csv("data/processed/base_combinada_limpia.csv")

    # Limpiar encoding del CSV de población
    pob_raw.columns = pob_raw.columns.str.strip()
    for col in pob_raw.select_dtypes("object").columns:
        pob_raw[col] = pob_raw[col].apply(
            lambda x: x.encode("latin1").decode("utf-8") if isinstance(x, str) else x
        )
    # Agregar por municipio (sumar numéricas)
    num_cols = pob_raw.select_dtypes("number").columns.tolist()
    pob = pob_raw.groupby("Municipio")[num_cols].sum().reset_index()

    # Orden grados rezago
    rezago_order = ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]
    coneval["grado_rezago_social"] = pd.Categorical(
        coneval["grado_rezago_social"], categories=rezago_order, ordered=True
    )
    ids_order = ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]
    ids["grado_ids"] = pd.Categorical(
        ids["grado_ids"], categories=ids_order, ordered=True
    )
    return denue, coneval, ids, pob, pob_raw

denue, coneval, ids, pob, pob_raw = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("TDA · Salud CDMX")
    st.caption("Análisis Topológico de Datos aplicado al sector salud en la Ciudad de México")
    st.divider()

    st.subheader("Bases de datos")

    info_bases = [
        {
            "nombre": "DENUE Salud CDMX",
            "fuente": "INEGI – DENUE 2025",
            "filas": f"{len(denue):,}",
            "desc": "Unidades económicas del sector salud geocodificadas. Incluye sector, subsector, tamaño y ubicación.",
            "color": "#4e8df5",
        },
        {
            "nombre": "CONEVAL – Rezago Social",
            "fuente": "CONEVAL 2020 (AGEB)",
            "filas": f"{len(coneval):,}",
            "desc": "Índice y grado de rezago social por AGEB para CDMX. Incluye área y centroides.",
            "color": "#e74c3c",
        },
        {
            "nombre": "IDS – Desarrollo Social",
            "fuente": "EVALUA CDMX (AGEB)",
            "filas": f"{len(ids):,}",
            "desc": "Índice de Desarrollo Social por AGEB. Incluye grado, población y proporción de necesidades básicas insatisfechas.",
            "color": "#f39c12",
        },
        {
            "nombre": "Población Municipal",
            "fuente": "INEGI – Censo 2020",
            "filas": f"{len(pob_raw):,}",
            "desc": "Demografía y derechohabiencia en salud por localidad/municipio en CDMX.",
            "color": "#2ecc71",
        },
    ]

    for b in info_bases:
        with st.expander(f"**{b['nombre']}**"):
            st.markdown(
                f"<span style='color:{b['color']};font-weight:bold'>●</span> "
                f"**Fuente:** {b['fuente']}  \n"
                f"**Registros:** {b['filas']}  \n"
                f"{b['desc']}",
                unsafe_allow_html=True,
            )

    st.divider()
    st.caption("**Enfoque:** Identificar huecos de cobertura en salud pública correlacionados con rezago social y bajo desarrollo en la CDMX mediante TDA (Čech / Vietoris-Rips).")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_datos, tab_complejos, tab_persist, tab_prior, tab_concl = st.tabs([
    "Datos", "Complejos Simpliciales", "Persistencia Homológica",
    "Priorización de Huecos", "Conclusiones y Estrategia",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB DATOS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_datos:

    st.header("Visualización Descriptiva Inicial")
    st.caption(
        "Las cuatro bases preprocesadas para el análisis TDA del sector salud en CDMX."
    )

    # ── KPIs generales ────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Unidades económicas salud", f"{len(denue):,}", "DENUE 2025")
    k2.metric("AGEBs con rezago (CONEVAL)", f"{len(coneval):,}", "CDMX")
    k3.metric("AGEBs con IDS", f"{len(ids):,}", "EVALUA CDMX")
    k4.metric("Municipios CDMX", "16", "Censo 2020")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 1. DENUE
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("1 · DENUE – Unidades Económicas de Salud")

    col_a, col_b = st.columns(2)

    # 1a. Por municipio
    with col_a:
        mun_counts = (
            denue.groupby("municipio")
            .size()
            .reset_index(name="establecimientos")
            .sort_values("establecimientos", ascending=True)
        )
        fig = px.bar(
            mun_counts,
            x="establecimientos",
            y="municipio",
            orientation="h",
            title="Unidades económicas por municipio",
            color="establecimientos",
            color_continuous_scale="Blues",
            labels={"municipio": "", "establecimientos": "Establecimientos"},
        )
        fig.update_layout(coloraxis_showscale=False, height=420, margin=dict(l=0, r=10, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # 1b. Sector + Subsector
    with col_b:
        sector_counts = denue["sector"].value_counts().reset_index()
        sector_counts.columns = ["sector", "count"]
        fig2 = px.pie(
            sector_counts,
            names="sector",
            values="count",
            title="Distribución por sector",
            color="sector",
            color_discrete_map=COLORS,
            hole=0.45,
        )
        fig2.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # Municipio público y privado por separado
    col_mun_pub, col_mun_priv = st.columns(2)

    with col_mun_pub:
        mun_pub = (
            denue[denue["sector"] == "Público"]
            .groupby("municipio").size()
            .reset_index(name="establecimientos")
            .sort_values("establecimientos", ascending=True)
        )
        fig_mp = px.bar(
            mun_pub, x="establecimientos", y="municipio", orientation="h",
            title="Unidades públicas por municipio",
            color="establecimientos", color_continuous_scale="Reds",
            labels={"municipio": "", "establecimientos": "Establecimientos"},
        )
        fig_mp.update_layout(coloraxis_showscale=False, height=420, margin=dict(l=0, r=10, t=40, b=0))
        st.plotly_chart(fig_mp, use_container_width=True)

    with col_mun_priv:
        mun_priv = (
            denue[denue["sector"] == "Privado"]
            .groupby("municipio").size()
            .reset_index(name="establecimientos")
            .sort_values("establecimientos", ascending=True)
        )
        fig_mv = px.bar(
            mun_priv, x="establecimientos", y="municipio", orientation="h",
            title="Unidades privadas por municipio",
            color="establecimientos", color_continuous_scale="Blues",
            labels={"municipio": "", "establecimientos": "Establecimientos"},
        )
        fig_mv.update_layout(coloraxis_showscale=False, height=420, margin=dict(l=0, r=10, t=40, b=0))
        st.plotly_chart(fig_mv, use_container_width=True)

    col_c = st.container()

    # 1c. Subsector
    with col_c:
        sub_counts = denue["subsector"].value_counts().reset_index()
        sub_counts.columns = ["subsector", "count"]
        fig3 = px.bar(
            sub_counts,
            x="subsector",
            y="count",
            title="Unidades por subsector",
            color="subsector",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            labels={"subsector": "", "count": "Establecimientos"},
        )
        fig3.update_layout(showlegend=False, height=320, margin=dict(l=0, r=0, t=40, b=60))
        fig3.update_xaxes(tickangle=-15)
        st.plotly_chart(fig3, use_container_width=True)

    _hidden_tab_datos = '''
    # 1d. Tamaño (personal ocupado)
    with st.container():
        per_order = [
            "0 a 5 personas", "6 a 10 personas", "11 a 30 personas",
            "31 a 50 personas", "51 a 100 personas",
            "101 a 250 personas", "251 y más personas",
        ]
        per_counts = (
            denue["per_ocu"]
            .value_counts()
            .reindex(per_order)
            .reset_index()
        )
        per_counts.columns = ["personal_ocupado", "count"]
        fig4 = px.bar(
            per_counts,
            x="personal_ocupado",
            y="count",
            title="Tamaño de establecimientos (personal ocupado)",
            color="count",
            color_continuous_scale="Purples",
            labels={"personal_ocupado": "", "count": "Establecimientos"},
        )
        fig4.update_layout(coloraxis_showscale=False, height=320, margin=dict(l=0, r=0, t=40, b=60))
        fig4.update_xaxes(tickangle=-20)
        st.plotly_chart(fig4, use_container_width=True)

    # Tamaño público y privado por separado
    col_tam_pub, col_tam_priv = st.columns(2)

    with col_tam_pub:
        per_pub = (
            denue[denue["sector"] == "Público"]["per_ocu"]
            .value_counts().reindex(per_order).reset_index()
        )
        per_pub.columns = ["personal_ocupado", "count"]
        fig_tp = px.bar(
            per_pub, x="personal_ocupado", y="count",
            title="Tamaño establecimientos – Público",
            color="count", color_continuous_scale="Reds",
            labels={"personal_ocupado": "", "count": "Establecimientos"},
        )
        fig_tp.update_layout(coloraxis_showscale=False, height=320, margin=dict(l=0, r=0, t=40, b=60))
        fig_tp.update_xaxes(tickangle=-20)
        st.plotly_chart(fig_tp, use_container_width=True)

    with col_tam_priv:
        per_priv = (
            denue[denue["sector"] == "Privado"]["per_ocu"]
            .value_counts().reindex(per_order).reset_index()
        )
        per_priv.columns = ["personal_ocupado", "count"]
        fig_tv = px.bar(
            per_priv, x="personal_ocupado", y="count",
            title="Tamaño establecimientos – Privado",
            color="count", color_continuous_scale="Blues",
            labels={"personal_ocupado": "", "count": "Establecimientos"},
        )
        fig_tv.update_layout(coloraxis_showscale=False, height=320, margin=dict(l=0, r=0, t=40, b=60))
        fig_tv.update_xaxes(tickangle=-20)
        st.plotly_chart(fig_tv, use_container_width=True)
    '''

    # 1e. Desglose público vs privado
    st.markdown("**Desglose por sector: Público vs Privado**")
    col_e1 = st.container()

    with col_e1:
        sub_sector = (
            denue.groupby(["subsector", "sector"])
            .size()
            .reset_index(name="count")
        )
        fig_sub_sec = px.bar(
            sub_sector,
            x="subsector",
            y="count",
            color="sector",
            color_discrete_map=COLORS,
            barmode="group",
            title="Subsector por tipo de sector",
            labels={"subsector": "", "count": "Establecimientos", "sector": "Sector"},
        )
        fig_sub_sec.update_layout(height=360, margin=dict(l=0, r=0, t=40, b=60))
        fig_sub_sec.update_xaxes(tickangle=-15)
        st.plotly_chart(fig_sub_sec, width="stretch")


#######
    _hidden_tab_datos = '''
    with st.container():
        per_sector = (
            denue.groupby(["per_ocu", "sector"])
            .size()
            .reset_index(name="count")
        )
        per_sector["per_ocu"] = pd.Categorical(per_sector["per_ocu"], categories=per_order, ordered=True)
        per_sector = per_sector.sort_values("per_ocu")
        fig_per_sec = px.bar(
            per_sector,
            x="per_ocu",
            y="count",
            color="sector",
            color_discrete_map=COLORS,
            barmode="group",
            title="Tamaño de establecimiento por sector",
            labels={"per_ocu": "", "count": "Establecimientos", "sector": "Sector"},
        )
        fig_per_sec.update_layout(height=360, margin=dict(l=0, r=0, t=40, b=60))
        fig_per_sec.update_xaxes(tickangle=-20)
        st.plotly_chart(fig_per_sec, width="stretch")
    '''

    # 1f. Mapa de puntos DENUE
    st.markdown("**Distribución espacial de unidades económicas de salud**")
    denue_map = denue.dropna(subset=["latitud", "longitud"])
    fig_map1 = px.scatter_mapbox(
        denue_map,
        lat="latitud",
        lon="longitud",
        color="sector",
        color_discrete_map=COLORS,
        hover_name="nom_estab",
        hover_data={"municipio": True, "subsector": True, "per_ocu": True, "latitud": False, "longitud": False},
        zoom=10,
        height=480,
        title="Unidades de salud CDMX – DENUE 2025",
        opacity=0.6,
    )
    fig_map1.update_layout(
        mapbox_style="carto-positron",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_map1, use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 2. CONEVAL – Rezago Social
    # ══════════════════════════════════════════════════════════════════════════
    _hidden_tab_datos = '''
    st.subheader("2 · CONEVAL – Rezago Social por AGEB")

    col_e, col_f = st.columns(2)

    with col_e:
        rez_counts = (
            coneval["grado_rezago_social"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        rez_counts.columns = ["grado", "ageb_count"]
        fig5 = px.bar(
            rez_counts,
            x="grado",
            y="ageb_count",
            title="AGEBs por grado de rezago social",
            color="grado",
            color_discrete_map=COLORS,
            labels={"grado": "Grado de rezago", "ageb_count": "Número de AGEBs"},
        )
        fig5.update_layout(showlegend=False, height=360, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig5, use_container_width=True)

    with col_f:
        fig6 = px.histogram(
            coneval,
            x="rezago_norm",
            nbins=40,
            title="Distribución del índice de rezago normalizado",
            color_discrete_sequence=["#e74c3c"],
            labels={"rezago_norm": "Rezago social normalizado (0–1)"},
        )
        fig6.update_layout(height=360, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig6, use_container_width=True)

    # Mapa CONEVAL
    st.markdown("**Distribución espacial del rezago social por AGEB**")
    coneval_map = coneval.dropna(subset=["centroide_lat", "centroide_lon"])
    fig_map2 = px.scatter_mapbox(
        coneval_map,
        lat="centroide_lat",
        lon="centroide_lon",
        color="grado_rezago_social",
        color_discrete_map=COLORS,
        hover_data={"rezago_norm": ":.3f", "area_km2": ":.3f", "centroide_lat": False, "centroide_lon": False},
        zoom=10,
        height=460,
        title="Rezago social por AGEB – CONEVAL 2020",
        opacity=0.75,
        size_max=6,
    )
    fig_map2.update_layout(
        mapbox_style="carto-positron",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_map2, use_container_width=True)

    st.divider()
    '''

    # ══════════════════════════════════════════════════════════════════════════
    # 3. IDS – Índice de Desarrollo Social
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("2 · IDS – Índice de Desarrollo Social por AGEB")

    col_g = st.container()

    with col_g:
        ids_counts = (
            ids["grado_ids"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        ids_counts.columns = ["grado", "ageb_count"]
        fig7 = px.bar(
            ids_counts,
            x="grado",
            y="ageb_count",
            title="AGEBs por grado de desarrollo social",
            color="grado",
            color_discrete_map=COLORS_IDS,
            labels={"grado": "Grado IDS", "ageb_count": "Número de AGEBs"},
        )
        fig7.update_layout(showlegend=False, height=360, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig7, use_container_width=True)

    _hidden_tab_datos = '''
    with st.container():
        fig8 = px.scatter(
            ids,
            x="ids_norm",
            y="prop_nbi_norm",
            color="grado_ids",
            color_discrete_map=COLORS_IDS,
            opacity=0.6,
            title="IDS normalizado vs Proporción de NBI",
            labels={
                "ids_norm": "IDS normalizado",
                "prop_nbi_norm": "Prop. Necesidades Básicas Insatisfechas (norm.)",
                "grado_ids": "Grado IDS",
            },
            height=360,
        )
        fig8.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig8, use_container_width=True)

    # Distribución IDS
    fig9 = px.histogram(
        ids,
        x="ids",
        color="grado_ids",
        color_discrete_map=COLORS_IDS,
        nbins=50,
        title="Distribución del IDS por grado",
        labels={"ids": "Índice de Desarrollo Social", "count": "AGEBs"},
        barmode="overlay",
        opacity=0.75,
        height=320,
    )
    fig9.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig9, use_container_width=True)
    '''

    # Mapa IDS — centroides desde CONEVAL
    st.markdown("**Distribución espacial del IDS por AGEB**")
    ids_map = ids.merge(
        coneval[["cvegeo", "centroide_lat", "centroide_lon"]],
        on="cvegeo",
        how="left",
    ).dropna(subset=["centroide_lat", "centroide_lon"])
    fig_map_ids = px.scatter_mapbox(
        ids_map,
        lat="centroide_lat",
        lon="centroide_lon",
        color="grado_ids",
        color_discrete_map=COLORS_IDS,
        hover_data={
            "ids": ":.3f",
            "prop_nbi": ":.3f",
            "centroide_lat": False,
            "centroide_lon": False,
        },
        zoom=10,
        height=460,
        title="Índice de Desarrollo Social por AGEB",
        opacity=0.75,
    )
    fig_map_ids.update_layout(
        mapbox_style="carto-positron",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_map_ids, use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 4. POBLACIÓN POR MUNICIPIO
    # ══════════════════════════════════════════════════════════════════════════
    _hidden_tab_datos = '''
    st.subheader("4 · Población y Derechohabiencia en Salud por Municipio")

    # Filtrar solo los 16 municipios clave (los que tienen pob > 0)
    pob_mun = pob[pob["Población total"] > 0].sort_values("Población total", ascending=False)

    col_i, col_j = st.columns(2)

    with col_i:
        fig10 = px.bar(
            pob_mun.sort_values("Población total"),
            x="Población total",
            y="Municipio",
            orientation="h",
            title="Población total por municipio",
            color="Población total",
            color_continuous_scale="Teal",
            labels={"Municipio": ""},
        )
        fig10.update_layout(coloraxis_showscale=False, height=420, margin=dict(l=0, r=10, t=40, b=0))
        st.plotly_chart(fig10, use_container_width=True)

    with col_j:
        # Derechohabiencia: columnas relevantes
        derecho_cols = {
            "IMSS": "Derechohabiente IMSS",
            "ISSSTE Federal": "Derechohabiente ISSSTE federal",
            "ISSSTE Estatal": "Derechohabiente ISSSTE estatal",
            "Seguro Popular/Bienestar": "Afiliado Seguro Popular/Bienestar",
            "IMSS-Bienestar": "Derechohabiente IMSS-Bienestar",
            "Privada": "Afiliado institución privada",
        }
        derecho_totals = {
            k: pob_mun[v].sum() for k, v in derecho_cols.items() if v in pob_mun.columns
        }
        df_der = pd.DataFrame(
            list(derecho_totals.items()), columns=["Institución", "Derechohabientes"]
        )
        fig11 = px.pie(
            df_der,
            names="Institución",
            values="Derechohabientes",
            title="Derechohabiencia en salud (CDMX)",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig11.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig11, use_container_width=True)

    # Sin derechohabiencia por municipio
    if "Sin derechohabiencia" in pob_mun.columns and "Población total" in pob_mun.columns:
        pob_mun = pob_mun.copy()
        pob_mun["pct_sin_derecho"] = (
            pob_mun["Sin derechohabiencia"] / pob_mun["Población total"] * 100
        ).round(1)
        fig12 = px.bar(
            pob_mun.sort_values("pct_sin_derecho", ascending=True),
            x="pct_sin_derecho",
            y="Municipio",
            orientation="h",
            title="% Población sin derechohabiencia en salud por municipio",
            color="pct_sin_derecho",
            color_continuous_scale="Reds",
            labels={"pct_sin_derecho": "% sin derechohabiencia", "Municipio": ""},
        )
        fig12.update_layout(coloraxis_showscale=False, height=380, margin=dict(l=0, r=10, t=40, b=0))
        st.plotly_chart(fig12, use_container_width=True)

    st.divider()
    '''

    # ══════════════════════════════════════════════════════════════════════════
    # 5. SECCIÓN INTEGRADA – Contexto del análisis TDA
    # ══════════════════════════════════════════════════════════════════════════
    _hidden_tab_datos = '''
    st.subheader("5 · Contexto Integrado: Oferta de Salud y Rezago Social")
    st.caption(
        "Esta sección vincula las bases para contextualizar el análisis topológico: "
        "dónde se concentra la oferta pública de salud y dónde el rezago es mayor."
    )

    col_k, col_l = st.columns(2)

    # 5a. Sector público vs privado por municipio
    with col_k:
        sector_mun = (
            denue.groupby(["municipio", "sector"])
            .size()
            .reset_index(name="count")
        )
        fig13 = px.bar(
            sector_mun.sort_values("count", ascending=False),
            x="municipio",
            y="count",
            color="sector",
            color_discrete_map=COLORS,
            title="Oferta de salud pública vs privada por municipio",
            labels={"municipio": "", "count": "Establecimientos", "sector": "Sector"},
            barmode="stack",
        )
        fig13.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=80))
        fig13.update_xaxes(tickangle=-35)
        st.plotly_chart(fig13, use_container_width=True)

    # 5b. Rezago medio por municipio (proxy: AGEBs × rezago_norm)
    with col_l:
        # Estimar municipio a partir de cvegeo (dígitos 5-7 = cve_mun en CDMX)
        coneval_copy = coneval.copy()
        coneval_copy["cve_mun_str"] = coneval_copy["cvegeo"].astype(str).str[2:5]
        coneval_copy["municipio"] = coneval_copy["cve_mun_str"].map(MUN_MAP)
        rez_mun = (
            coneval_copy.groupby("municipio")["rezago_norm"]
            .mean()
            .reset_index()
            .rename(columns={"rezago_norm": "rezago_medio"})
            .sort_values("rezago_medio", ascending=True)
            .dropna()
        )
        fig14 = px.bar(
            rez_mun,
            x="rezago_medio",
            y="municipio",
            orientation="h",
            title="Rezago social medio por municipio (CONEVAL)",
            color="rezago_medio",
            color_continuous_scale="Reds",
            labels={"rezago_medio": "Rezago normalizado promedio", "municipio": ""},
        )
        fig14.update_layout(coloraxis_showscale=False, height=400, margin=dict(l=0, r=10, t=40, b=0))
        st.plotly_chart(fig14, use_container_width=True)

    # 5c. Scatter integrado: unidades de salud pública vs rezago medio
    denue_pub_mun = (
        denue[denue["sector"] == "Público"]
        .groupby("municipio")
        .size()
        .reset_index(name="unidades_publicas")
    )
    denue_pub_mun["municipio_norm"] = denue_pub_mun["municipio"].str.strip()
    rez_mun["municipio_norm"] = rez_mun["municipio"].str.strip()
    merged = denue_pub_mun.merge(rez_mun, on="municipio_norm", how="inner")

    if not merged.empty:
        fig15 = px.scatter(
            merged,
            x="rezago_medio",
            y="unidades_publicas",
            text="municipio_norm",
            size="unidades_publicas",
            color="rezago_medio",
            color_continuous_scale="RdYlGn_r",
            title="Unidades de salud públicas vs Rezago social promedio por municipio",
            labels={
                "rezago_medio": "Rezago social normalizado promedio",
                "unidades_publicas": "Unidades de salud públicas",
            },
            height=440,
        )
        fig15.update_traces(textposition="top center", marker=dict(opacity=0.85))
        fig15.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig15, use_container_width=True)
        st.caption(
            "Este gráfico resume la tensión central del proyecto: "
            "¿los municipios con mayor rezago tienen suficiente oferta pública de salud? "
            "Los huecos identificados con TDA (Čech/Vietoris-Rips) permitirán cuantificar esta brecha espacialmente."
        )
    '''

# ═══════════════════════════════════════════════════════════════════════════════
# TAB COMPLEJOS SIMPLICIALES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_complejos:

    st.header("Complejos de Vietoris-Rips y Homología Persistente")
    st.caption(
        "Red de salud **pública** de CDMX analizada con TDA. "
        "Los círculos crecen (radio ε) y cuando se tocan forman una arista del complejo. "
        "Los **huecos persistentes** en H₁ sugieren posibles vacíos estructurales de cobertura."
    )

    # ── Controles fila 1 ─────────────────────────────────────────────────────
    cc1, cc2, cc3 = st.columns([2, 3, 2])
    with cc1:
        mun_list = ["CDMX completa"] + sorted(denue["municipio"].unique().tolist())
        mun_sel = st.selectbox("Alcaldía", mun_list, key="mun_comp")
    with cc2:
        eps = st.slider(
            "Radio ε (km) — radio de cada círculo",
            min_value=0.25, max_value=8.0, value=1.5, step=0.01,
        )
    with cc3:
        show_rezago_layer = st.toggle("Mostrar capa de rezago AGEBs", value=True)

    # ── Controles fila 2 ─────────────────────────────────────────────────────
    cc4, cc5, cc6 = st.columns([2, 3, 2])
    with cc4:
        sector_opts = sorted(denue["sector"].dropna().unique().tolist())
        sector_sel = st.selectbox("Sector", sector_opts, index=sector_opts.index("Público") if "Público" in sector_opts else 0, key="sector_comp")
    with cc5:
        subsec_opts = ["Todos los subsectores"] + sorted(
            denue[denue["sector"] == sector_sel]["subsector"].dropna().unique().tolist()
        )
        subsec_sel = st.selectbox("Subsector a analizar", subsec_opts, key="subsec_comp")
    with cc6:
        max_pts = st.slider("Máximo de puntos", min_value=50, max_value=2500, value=2100, step=50,
                            help="Limita el número de unidades para el cómputo TDA")

    # ── Filtrado unidades públicas ────────────────────────────────────────────
    pub_all = denue[denue["sector"] == sector_sel].dropna(subset=["latitud", "longitud"])
    pub_f   = pub_all if mun_sel == "CDMX completa" else pub_all[pub_all["municipio"] == mun_sel]
    if subsec_sel != "Todos los subsectores":
        pub_f = pub_f[pub_f["subsector"] == subsec_sel]

    n_universe = len(pub_f)
    sampled = False
    if n_universe > max_pts:
        pub_f   = pub_f.sample(max_pts, random_state=42)
        sampled = True

    info_parts = [f"Sector: **{sector_sel}**"]
    if subsec_sel != "Todos los subsectores":
        info_parts.append(f"Subsector: **{subsec_sel}**")
    if sampled:
        info_parts.append(f"muestra {max_pts} de {n_universe} unidades")
    if info_parts:
        st.caption(" " + " · ".join(info_parts))

    pts   = pub_f[["latitud", "longitud"]].values
    lats  = pts[:, 0].tolist()
    lons  = pts[:, 1].tolist()

    # ── Cómputo TDA (cacheado) ────────────────────────────────────────────────
    with st.spinner("Calculando homología persistente…"):
        pts_key       = tuple(map(tuple, pts))
        dgms, D_list  = _compute_tda(pts_key)
        D             = np.array(D_list)
        _, _, h1_centers_map = _compute_tda_full(pts_key)  # cached, costo ~0

    h0, h1     = dgms[0], dgms[1]
    max_finite = float(D.max()) if D.size > 0 else 15.0
    max_finite = min(max_finite, 15.0)

    # ── Cómputo cobertura AGEBs ───────────────────────────────────────────────
    if mun_sel == "CDMX completa":
        coneval_sel = coneval
    else:
        cve_codes = coneval["cvegeo"].astype(str).str[2:5]
        coneval_sel = coneval[cve_codes == MUN_INV.get(mun_sel, "000")]

    ageb_geo = coneval_sel.dropna(subset=["centroide_lat", "centroide_lon"])
    ageb_pts = ageb_geo[["centroide_lat", "centroide_lon"]].values

    if len(pts) > 0 and len(ageb_pts) > 0:
        pub_key_cov  = tuple(map(tuple, pts))
        ageb_key_cov = tuple(map(tuple, ageb_pts))
        min_dists    = np.array(_compute_coverage(pub_key_cov, ageb_key_cov))
    else:
        min_dists = np.array([])

    # ── KPIs ──────────────────────────────────────────────────────────────────
    h0_deaths = np.where(np.isinf(h0[:, 1]), max_finite * 2, h0[:, 1])
    n_comp    = int((h0_deaths > eps).sum())

    if len(h1) > 0:
        alive_h1 = (h1[:, 0] <= eps) & ((h1[:, 1] > eps) | np.isinf(h1[:, 1]))
        n_holes  = int(alive_h1.sum())
    else:
        n_holes = 0

    if min_dists.size > 0:
        pct_cov    = float((min_dists <= eps).mean() * 100)
        n_uncov    = int((min_dists > eps).sum())
    else:
        pct_cov, n_uncov = 0.0, 0

    km1, km2, km3, km4 = st.columns(4)
    km1.metric(f"Unidades {sector_sel.lower()}", f"{len(pub_f):,}", mun_sel)
    km2.metric("Componentes conexas H₀", n_comp,  f"ε = {eps} km")
    km3.metric("Huecos activos H₁",      n_holes, f"ε = {eps} km")
    km4.metric("AGEBs con cobertura",    f"{pct_cov:.1f}%", f"{n_uncov} sin cobertura")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # A. Mapa interactivo VR
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("A · Mapa Interactivo — Complejo de Vietoris-Rips")
    st.caption(
        f"**Azul**: zona de cobertura (radio ε/2 = {eps/2:.2f} km). "
        f"**Amarillo**: aristas del complejo (d ≤ {eps} km). "
        f"**Puntos**: unidades de salud {sector_sel.lower()} por subsector."
    )

    fig_vr = go.Figure()

    # Capa 0: centroides AGEB coloreados por rezago (fondo)
    if show_rezago_layer and len(ageb_pts) > 0:
        for grado, color in COLORS.items():
            sub_ag = ageb_geo[ageb_geo["grado_rezago_social"] == grado]
            if sub_ag.empty:
                continue
            fig_vr.add_trace(go.Scattermapbox(
                lat=sub_ag["centroide_lat"].tolist(),
                lon=sub_ag["centroide_lon"].tolist(),
                mode="markers",
                marker=dict(size=6, color=color, opacity=0.3),
                name=f"Rezago {grado}",
                hovertemplate=f"Rezago: {grado}<extra></extra>",
                legendgroup="rezago",
            ))

    # Capa 1: círculos de cobertura (radio ε/2)
    c_lats, c_lons = _circles(lats, lons, eps / 2)
    fig_vr.add_trace(go.Scattermapbox(
        lat=c_lats, lon=c_lons,
        mode="lines",
        line=dict(color="rgba(52,152,219,0.35)", width=1),
        fill="toself",
        fillcolor="rgba(52,152,219,0.07)",
        name=f"Cobertura ε/2={eps/2:.2f} km",
        hoverinfo="skip",
        legendgroup="vr",
    ))

    # Capa 2: aristas VR
    e_lats, e_lons = _edges(lats, lons, D, eps)
    if e_lats:
        fig_vr.add_trace(go.Scattermapbox(
            lat=e_lats, lon=e_lons,
            mode="lines",
            line=dict(color="rgba(241,196,15,0.55)", width=1),
            name="Aristas VR",
            hoverinfo="skip",
            legendgroup="vr",
        ))

    # Capa 3: unidades por subsector
    SUBSEC_COL = {
        "Servicios ambulatorios": "#e74c3c",
        "Asistencia social":      "#3498db",
        "Hospitales":             "#f39c12",
        "Residencias y cuidado":  "#9b59b6",
    }
    for subsec, sc in SUBSEC_COL.items():
        sub_pts = pub_f[pub_f["subsector"] == subsec]
        if sub_pts.empty:
            continue
        fig_vr.add_trace(go.Scattermapbox(
            lat=sub_pts["latitud"].tolist(),
            lon=sub_pts["longitud"].tolist(),
            mode="markers",
            marker=dict(size=7, color=sc),
            name=subsec,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Municipio: %{customdata[1]}<br>"
                "Personal: %{customdata[2]}<extra></extra>"
            ),
            customdata=sub_pts[["nom_estab", "municipio", "per_ocu"]].values.tolist(),
            legendgroup="unidades",
        ))



    center_lat = float(np.mean(lats)) if lats else 19.43
    center_lon = float(np.mean(lons)) if lons else -99.13
    fig_vr.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=11 if mun_sel != "CDMX completa" else 10,
        ),
        height=560,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", font=dict(size=11)),
    )
    st.plotly_chart(fig_vr, width="stretch")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # B. Homología Persistente
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("B · Homología Persistente")

    col_bc1, col_bc2 = st.columns(2)

    with col_bc1:
        fig_h0 = _barcode_fig(h0, "H₀ – Componentes Conexas (barcode)", "#e74c3c", max_finite)
        fig_h0.add_vline(x=eps, line_dash="dash", line_color="#333", opacity=0.7,
                         annotation_text=f"ε actual", annotation_position="top right")
        st.plotly_chart(fig_h0, width="stretch")
        st.caption(
            "Cada barra es un grupo de unidades que aún no se han conectado. "
            "Al aumentar ε las barras mueren. "
            "Una barra muy larga = grupo aislado geográficamente."
        )

    with col_bc2:
        fig_h1 = _barcode_fig(h1, "H₁ – Huecos / Loops (barcode)", "#3498db", max_finite)
        fig_h1.add_vline(x=eps, line_dash="dash", line_color="#333", opacity=0.7,
                         annotation_text=f"ε actual", annotation_position="top right")
        st.plotly_chart(fig_h1, width="stretch")
        st.caption(
            "Cada barra es un hueco topológico (zona rodeada de unidades sin cobertura interior). "
            "Una barra larga = hueco **persistente y candidato**. "
            "Una barra corta = ruido o irregularidad local."
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # C. Cobertura y Rezago Social
    # ══════════════════════════════════════════════════════════════════════════
    _hidden_tab_complejos = '''
    st.subheader("C · Huecos de Cobertura y Rezago Social")
    st.caption(
        f"AGEBs cuyo centroide está a más de {eps} km de cualquier unidad pública "
        f"de salud se consideran **sin cobertura** a este radio."
    )

    if min_dists.size > 0:
        ageb_cov = ageb_geo.copy()
        ageb_cov["dist_min"] = min_dists
        ageb_cov["cobertura"] = np.where(min_dists <= eps, "Cubierta", "Sin cobertura")

        col_cm, col_cr = st.columns(2)

        # Mapa cobertura
        with col_cm:
            fig_cmap = go.Figure()
            for status, sc, sz in [("Cubierta", "#2ecc71", 5), ("Sin cobertura", "#e74c3c", 8)]:
                sub_c = ageb_cov[ageb_cov["cobertura"] == status]
                fig_cmap.add_trace(go.Scattermapbox(
                    lat=sub_c["centroide_lat"].tolist(),
                    lon=sub_c["centroide_lon"].tolist(),
                    mode="markers",
                    marker=dict(size=sz, color=sc, opacity=0.8),
                    name=status,
                    hovertemplate=(
                        f"{status}<br>"
                        "Rezago: %{customdata[0]}<br>"
                        "Dist. más cercana: %{customdata[1]:.2f} km<extra></extra>"
                    ),
                    customdata=sub_c[["grado_rezago_social", "dist_min"]].values.tolist(),
                ))
            fig_cmap.update_layout(
                mapbox=dict(
                    style="carto-positron",
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=11 if mun_sel != "CDMX completa" else 10,
                ),
                height=440,
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(bgcolor="rgba(255,255,255,0.85)"),
                title=f"AGEBs con/sin cobertura (ε = {eps} km)",
            )
            st.plotly_chart(fig_cmap, width="stretch")

        # Rezago en cubiertas vs no cubiertas
        with col_cr:
            rez_order = ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]
            rez_cov_df = (
                ageb_cov.groupby(["cobertura", "grado_rezago_social"])
                .size().reset_index(name="n_agebs")
            )
            rez_cov_df["grado_rezago_social"] = pd.Categorical(
                rez_cov_df["grado_rezago_social"], categories=rez_order, ordered=True
            )
            fig_rcov = px.bar(
                rez_cov_df.sort_values("grado_rezago_social"),
                x="grado_rezago_social", y="n_agebs",
                color="cobertura",
                color_discrete_map={"Cubierta": "#2ecc71", "Sin cobertura": "#e74c3c"},
                barmode="group",
                title=f"Rezago de AGEBs: cubiertas vs sin cobertura (ε = {eps} km)",
                labels={
                    "grado_rezago_social": "Grado de rezago",
                    "n_agebs": "Número de AGEBs",
                    "cobertura": "",
                },
                height=440,
            )
            fig_rcov.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_rcov, width="stretch")

        # Resumen narrativo
        sinc = ageb_cov[ageb_cov["cobertura"] == "Sin cobertura"]
        if len(sinc) > 0:
            top_rezago = sinc["grado_rezago_social"].value_counts().idxmax()
            st.info(
                f"A ε = **{eps} km**, hay **{n_uncov} AGEBs sin cobertura** "
                f"({100 - pct_cov:.1f}% del total). "
                f"El grado de rezago más frecuente entre ellas es **{top_rezago}**. "
                f"Estos huecos de cobertura son los candidatos que el análisis TDA (H₁) "
                f"detecta como vacíos estructurales persistentes en la red de salud pública."
            )
    else:
        st.info("Selecciona una alcaldía con datos CONEVAL para ver el análisis de cobertura.")
    '''

# ═══════════════════════════════════════════════════════════════════════════════
# TAB PERSISTENCIA HOMOLÓGICA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_persist:

    st.header("Persistencia Homológica")
    st.caption(
        "Análisis detallado de las características topológicas persistentes de la red de salud pública. "
        "Se identifican componentes conexas (H₀), huecos (H₁) y su relación con la cobertura y el rezago social."
    )

    # ── Controles ─────────────────────────────────────────────────────────────
    pa, pb, pc, pd0 = st.columns([2, 2, 2, 2])
    with pa:
        mun_list_p = ["CDMX completa"] + sorted(denue["municipio"].unique().tolist())
        mun_p = st.selectbox("Alcaldía", mun_list_p, key="mun_persist")
    with pb:
        sector_opts_p = sorted(denue["sector"].dropna().unique().tolist())
        sector_p = st.selectbox("Sector", sector_opts_p, index=sector_opts_p.index("Público") if "Público" in sector_opts_p else 0, key="sector_persist")
    with pc:
        subsec_opts_p = ["Todos los subsectores"] + sorted(
            denue[denue["sector"] == sector_p]["subsector"].dropna().unique().tolist()
        )
        subsec_p = st.selectbox("Subsector", subsec_opts_p, key="subsec_persist")
    with pd0:
        max_pts_p = st.slider("Máx. puntos", 50, 2500, 2100, 50, key="maxpts_persist")

    # ── Filtrado ──────────────────────────────────────────────────────────────
    pub_p = denue[denue["sector"] == sector_p].dropna(subset=["latitud", "longitud"])
    if mun_p != "CDMX completa":
        pub_p = pub_p[pub_p["municipio"] == mun_p]
    if subsec_p != "Todos los subsectores":
        pub_p = pub_p[pub_p["subsector"] == subsec_p]
    n_univ_p = len(pub_p)
    if n_univ_p > max_pts_p:
        pub_p = pub_p.sample(max_pts_p, random_state=42)
        st.caption(f" Muestra de {max_pts_p} de {n_univ_p} unidades.")

    pts_p  = pub_p[["latitud", "longitud"]].values
    if len(pts_p) < 5:
        st.warning("Muy pocas unidades para el análisis. Amplía el filtro.")
        st.stop()

    # ── Cómputo TDA completo (antes del slider para calcular μ) ──────────────
    with st.spinner("Calculando persistencia…"):
        pts_p_key = tuple(map(tuple, pts_p))
        dgms_raw, D_p_list, h1_centers = _compute_tda_full(pts_p_key)
        D_p = np.array(D_p_list)

    h0_p = np.array(dgms_raw[0])
    h1_p = np.array(dgms_raw[1]) if len(dgms_raw[1]) else np.empty((0, 2))
    max_ep = min(float(D_p.max()) if D_p.size else 15.0, 15.0)

    h1_pers = (h1_p[:, 1] - h1_p[:, 0]) if len(h1_p) else np.array([])

    # ── Estadísticas de vida de los huecos ────────────────────────────────────
    h1_pers_fin = h1_pers[~np.isinf(h1_pers)] if len(h1_pers) else np.array([])
    if len(h1_pers_fin) >= 2:
        mu_pers    = float(h1_pers_fin.mean())
        sigma_pers = float(h1_pers_fin.std())
        thresh_sug = mu_pers + 0.5 * sigma_pers   # μ + ½σ como umbral estadístico
        # Hueco representativo: el más cercano a la media
        idx_repr   = int(np.argmin(np.abs(h1_pers_fin - mu_pers)))
    else:
        mu_pers = sigma_pers = thresh_sug = 0.0
        idx_repr = 0

    # ── Controles fila 2 — umbral default = μ + ½σ vida H₁ ──────────────────
    _thresh_default = float(np.round(np.clip(thresh_sug, 0.1, 3.0), 1)) if thresh_sug > 0 else 0.5
    pd1, pd2 = st.columns([3, 2])
    with pd1:
        thresh = st.slider(
            "Umbral de persistencia significativa (km)",
            min_value=0.1, max_value=3.0, value=_thresh_default, step=0.1,
            help=f"Auto-calculado como μ + ½σ = {thresh_sug:.3f} km. Ajusta manualmente si lo requieres.",
        )
    with pd2:
        eps_p = st.slider("ε de referencia (km)", 0.25, 8.0, 1.5, 0.25, key="eps_persist")

    with st.expander("¿Cómo se calcula el umbral sugerido?", expanded=False):
        if len(h1_pers_fin) >= 2:
            _diff_p = thresh - thresh_sug
            _delta_p = f"Manual: {thresh:.2f} km ({'+' if _diff_p >= 0 else ''}{_diff_p:.2f})" if abs(_diff_p) > 0.01 else "*(= sugerido)*"
            st.markdown(f"""
**Fórmula:** $\\text{{Umbral}} = \\mu_{{H_1}} + \\tfrac{{1}}{{2}}\\,\\sigma_{{H_1}}$

| Estadístico | Valor |
|---|---|
| Media de vidas H₁ (μ) | **{mu_pers:.4f} km** |
| Desv. estándar (σ) | **{sigma_pers:.4f} km** |
| ½ σ | **{0.5 * sigma_pers:.4f} km** |
| **Umbral = μ + ½σ** | **{thresh_sug:.4f} km** |
| Umbral actual (slider) | **{thresh:.2f} km** — {_delta_p} |

La media μ es el promedio de vida de todos los huecos H₁. Sumar ½σ filtra el ruido topológico reteniendo solo los huecos más persistentes (~30–35%).
> Se **recalcula automáticamente** al cambiar los filtros.
""")
        else:
            st.info("No hay suficientes huecos H₁ para calcular estadísticas (se necesitan ≥ 2).")

    h1_sig  = h1_p[h1_pers >= thresh] if len(h1_p) else np.empty((0, 2))
    n_sig   = len(h1_sig)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    kp1, kp2, kp3, kp4, kp5 = st.columns(5)
    kp1.metric("Unidades analizadas", f"{len(pts_p):,}")
    kp2.metric("Componentes H₀", len(h0_p), help="Total de componentes conexas detectadas en la filtración")
    kp3.metric("Features H₁ totales", len(h1_p), help="Huecos detectados en toda la filtración")
    kp4.metric(f"H₁ significativos (≥{thresh} km)", n_sig)
    kp5.metric("Cavidades H₂", "0", help="Trivialmente vacío para datos geográficos 2D (lat/lon). H₂ requiere al menos una dimensión adicional.")
    if len(h1_pers):
        pass  # se muestra en KPIs estadísticos abajo
    st.caption(
        "**H₀** = componentes conexas (clústeres de unidades). "
        "**H₁** = huecos topológicos (vacíos de cobertura). "
        "**H₂** = cavidades — siempre cero para nubes de puntos 2D; "
        "requeriría datos volumétricos 3D para ser no trivial."
    )

    # ── KPIs estadísticos de vida ──────────────────────────────────────────────
    if len(h1_pers_fin) >= 2:
        sk1, sk2, sk3, sk4 = st.columns(4)
        sk1.metric("Vida media H₁ (μ)", f"{mu_pers:.3f} km")
        sk2.metric("Desv. estándar (σ)", f"{sigma_pers:.3f} km")
        sk3.metric("Umbral sugerido (μ + ½σ)", f"{thresh_sug:.3f} km",
                delta=f"{'↑' if thresh_sug > thresh else '↓'} vs manual {thresh:.2f} km",
                delta_color="off")
        sk4.metric("Hueco representativo", f"H₁ #{idx_repr + 1}",
                delta=f"persist. {h1_pers_fin[idx_repr]:.3f} km", delta_color="off")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # A. DIAGRAMA DE PERSISTENCIA (avanzado)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("A · Diagrama de Persistencia")

    rows_pers = []
    for b, d in h0_p:
        d_p = min(float(d), max_ep * 1.05) if np.isinf(d) else float(d)
        rows_pers.append({"Dim": "H₀ Componentes", "Nacimiento": float(b),
                        "Muerte": d_p, "Persistencia": d_p - float(b)})
    for b, d in h1_p:
        d_p = min(float(d), max_ep * 1.05) if np.isinf(d) else float(d)
        pers = d_p - float(b)
        rows_pers.append({"Dim": "H₁ Huecos", "Nacimiento": float(b),
                        "Muerte": d_p, "Persistencia": pers,
                        "Significativo": "Sí" if pers >= thresh else "No"})

    df_pers = pd.DataFrame(rows_pers)
    fig_pd = px.scatter(
        df_pers, x="Nacimiento", y="Muerte", color="Dim",
        color_discrete_map={"H₀ Componentes": "#e74c3c", "H₁ Huecos": "#3498db"},
        size="Persistencia", size_max=18, opacity=0.75,
        hover_data={"Persistencia": ":.3f"},
        title="Diagrama de Persistencia — H₀ y H₁",
        labels={"Nacimiento": "Nacimiento ε (km)", "Muerte": "Muerte ε (km)"},
        height=450,
    )
    lim = max_ep * 1.05
    # Diagonal (persistencia = 0, ruido)
    fig_pd.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                    line=dict(color="gray", dash="dot", width=1))
    # Umbral de significancia (persistencia = thresh)
    fig_pd.add_shape(type="line", x0=0, y0=thresh, x1=lim - thresh, y1=lim,
                    line=dict(color="#f39c12", dash="dash", width=1.5))
    # Línea ε referencia
    fig_pd.add_vline(x=eps_p, line_dash="dash", line_color="green", opacity=0.6,
                    annotation_text=f"ε={eps_p}km", annotation_position="top right")
    # Anotaciones de zonas
    fig_pd.add_annotation(x=lim * 0.08, y=lim * 0.97,
                        text="◀ Ruido (baja persistencia)", showarrow=False,
                        font=dict(color="gray", size=11))
    fig_pd.add_annotation(x=lim * 0.15, y=lim * 0.6,
                        text="Huecos<br>significativos ▶", showarrow=False,
                        font=dict(color="#f39c12", size=11))
    fig_pd.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_pd, width="stretch")
    st.caption(
        "**Diagonal punteada gris** = persistencia cero (ruido). "
        "**Línea naranja** = umbral de significancia. "
        "Los puntos azules (H₁) alejados de la diagonal son **huecos topológicos reales**."
    )

    st.divider()

    # # ══════════════════════════════════════════════════════════════════════════
    # # B. DISTRIBUCIÓN DE PERSISTENCIA (diagrama de barras)
    # # ══════════════════════════════════════════════════════════════════════════
    # st.subheader("B · Distribución de Persistencias — H₀ y H₁")
    # st.caption(
    #     "Histograma de los tiempos de vida de cada característica topológica. "
    #     "Las barras a la **derecha del umbral** son estructuras persistentes reales."
    # )

    # _rows_hist = []
    # for b, d in h0_p:
    #     _p = (min(float(d), max_ep) if np.isinf(d) else float(d)) - float(b)
    #     _rows_hist.append({"Persistencia (km)": _p, "Dimensión": "H₀ — Componentes conexas"})
    # for b, d in h1_p:
    #     _p = (min(float(d), max_ep) if np.isinf(d) else float(d)) - float(b)
    #     _rows_hist.append({"Persistencia (km)": _p, "Dimensión": "H₁ — Huecos"})
    # _df_hist = pd.DataFrame(_rows_hist)

    # _fig_hist = px.histogram(
    #     _df_hist, x="Persistencia (km)", color="Dimensión",
    #     color_discrete_map={"H₀ — Componentes conexas": "#e74c3c", "H₁ — Huecos": "#3498db"},
    #     barmode="overlay", opacity=0.7, nbins=40,
    #     title="Distribución de Persistencia — H₀ y H₁",
    #     labels={"Persistencia (km)": "Persistencia (km)", "count": "Cantidad de features"},
    #     height=370,
    # )
    # _fig_hist.add_vline(x=thresh, line_dash="dash", line_color="#f39c12",
    #                     annotation_text=f"Umbral {thresh} km",
    #                     annotation_position="top right")
    # _fig_hist.update_layout(margin=dict(l=0, r=0, t=40, b=0),
    #                         legend=dict(orientation="h", y=-0.15))
    # st.plotly_chart(_fig_hist, width="stretch")
    # st.caption(
    #     "**Barras rojas (H₀):** cada barra es una componente conexa con esa vida. "
    #     "**Barras azules (H₁):** cada barra es un hueco con esa persistencia. "
    #     "Las barras a la izquierda del umbral naranja son **ruido topológico**."
    # )

    # st.divider()

    # # ══════════════════════════════════════════════════════════════════════════
    # # C. COBERTURA DE AGEBs Y CONECTIVIDAD
    # # ══════════════════════════════════════════════════════════════════════════
    # st.subheader("C · Cobertura de AGEBs y Conectividad")
    # st.caption(
    #     f"Proporción de AGEBs con al menos una unidad de salud a ≤ ε = {eps_p} km. "
    #     "Vincula las características topológicas (H₀, H₁) con la cobertura territorial real."
    # )

    # if mun_p == "CDMX completa":
    #     _ageb_cov_p = coneval.dropna(subset=["centroide_lat", "centroide_lon"])
    # else:
    #     _cve_mask_p = coneval["cvegeo"].astype(str).str[2:5] == MUN_INV.get(mun_p, "000")
    #     _ageb_cov_p = coneval[_cve_mask_p].dropna(subset=["centroide_lat", "centroide_lon"])

    # if len(_ageb_cov_p) > 0 and len(pts_p) > 0:
    #     _ageb_pts_p   = _ageb_cov_p[["centroide_lat", "centroide_lon"]].values
    #     _ageb_key_p   = tuple(map(tuple, _ageb_pts_p))
    #     _md_p         = np.array(_compute_coverage(pts_p_key, _ageb_key_p))
    #     _pct_cov_p    = float((_md_p <= eps_p).mean() * 100)
    #     _n_uncov_p    = int((_md_p > eps_p).sum())

    #     # H₀ activas al ε de referencia (conectividad)
    #     _h0_deaths_p  = np.where(np.isinf(h0_p[:, 1]), max_ep * 2, h0_p[:, 1])
    #     _n_comp_eps   = int((_h0_deaths_p > eps_p).sum())

    #     _cc1, _cc2, _cc3, _cc4 = st.columns(4)
    #     _cc1.metric("AGEBs en área", len(_ageb_cov_p))
    #     _cc2.metric(f"AGEBs cubiertas (≤{eps_p} km)", f"{_pct_cov_p:.1f}%",
    #                 delta=f"{len(_ageb_cov_p) - _n_uncov_p} AGEBs", delta_color="off")
    #     _cc3.metric("AGEBs sin cobertura", _n_uncov_p,
    #                 delta=f"{100 - _pct_cov_p:.1f}% del total", delta_color="off")
    #     _cc4.metric(f"H₀ activas (ε={eps_p} km)", _n_comp_eps,
    #                 help="Componentes conexas aún separadas a este radio — indica fragmentación de la red")

    #     # Diagrama de barras: rezago vs cobertura
    #     _ageb_cov_p   = _ageb_cov_p.copy()
    #     _ageb_cov_p["dist_min"]  = _md_p
    #     _ageb_cov_p["cobertura"] = np.where(_md_p <= eps_p, "Cubierta", "Sin cobertura")

    #     _rez_order = ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]
    #     _df_rcov = (
    #         _ageb_cov_p.groupby(["cobertura", "grado_rezago_social"])
    #         .size().reset_index(name="n_agebs")
    #     )
    #     _df_rcov["grado_rezago_social"] = pd.Categorical(
    #         _df_rcov["grado_rezago_social"], categories=_rez_order, ordered=True
    #     )
    #     _fig_rcov = px.bar(
    #         _df_rcov.sort_values("grado_rezago_social"),
    #         x="grado_rezago_social", y="n_agebs",
    #         color="cobertura",
    #         color_discrete_map={"Cubierta": "#2ecc71", "Sin cobertura": "#e74c3c"},
    #         barmode="group",
    #         title=f"AGEBs cubiertas vs sin cobertura por grado de rezago social (ε = {eps_p} km)",
    #         labels={"grado_rezago_social": "Grado de rezago social",
    #                 "n_agebs": "Número de AGEBs", "cobertura": ""},
    #         height=380,
    #     )
    #     _fig_rcov.update_layout(margin=dict(l=0, r=0, t=40, b=0),
    #                             legend=dict(orientation="h", y=-0.2))
    #     st.plotly_chart(_fig_rcov, width="stretch")

    #     _sinc_p = _ageb_cov_p[_ageb_cov_p["cobertura"] == "Sin cobertura"]
    #     if len(_sinc_p) > 0:
    #         _top_rez_p = _sinc_p["grado_rezago_social"].value_counts().idxmax()
    #         st.info(
    #             f"A ε = **{eps_p} km**, el **{_pct_cov_p:.1f}%** de las AGEBs tienen cobertura "
    #             f"({_n_uncov_p} sin cobertura). "
    #             f"El rezago más frecuente en AGEBs sin cobertura es **{_top_rez_p}**. "
    #             f"La red presenta **{_n_comp_eps} componentes conexas (H₀)** separadas a este radio, "
    #             f"y **{n_sig} huecos H₁ persistentes** (≥ {thresh} km) que corresponden a "
    #             f"vacíos estructurales de cobertura."
    #         )
    # else:
    #     st.info("No hay datos CONEVAL disponibles para calcular cobertura en esta selección.")

    # st.divider()

    # ═══════════════════════════════════════════════════════════════════════
    # D. MAPA DE HUECOS PERSISTENTES
    # ═══════════════════════════════════════════════════════════════════════
    st.subheader("D · Mapa de Huecos Persistentes")
    st.caption(
        f"Ubicación geográfica de los **{n_sig} huecos H₁** con persistencia ≥ {thresh} km. "
        "Tamaño y color proporcionales a la persistencia (amarillo → rojo = mayor vida)."
    )

    # Distancias al borde para clasificar interior vs borde
    _bd_map = _border_dists(h1_centers[:len(h1_p)], pts_p) if len(h1_p) and len(h1_centers) else np.array([])

    _map_rows = []
    for _i, (b, d) in enumerate(h1_p):
        _pers = float(h1_pers[_i])
        if _pers < thresh:
            continue
        if _i >= len(h1_centers):
            continue
        _lat, _lon = h1_centers[_i]
        _tipo = "Interior" if (len(_bd_map) > _i and _bd_map[_i] >= eps_p) else "Borde"
        _map_rows.append({
            "lat": float(_lat), "lon": float(_lon),
            "Persistencia (km)": round(_pers, 3),
            "Nacimiento (km)": round(float(b), 3),
            "Muerte (km)": round(float(d) if not np.isinf(d) else 99.0, 3),
            "Tipo": _tipo,
            "label": f"H₁ #{_i + 1}",
        })

    df_map_p = pd.DataFrame(_map_rows)

    if len(df_map_p) > 0:
        _clat = float(pts_p[:, 0].mean())
        _clon = float(pts_p[:, 1].mean())
        _max_p = float(df_map_p["Persistencia (km)"].max())
        _min_p = float(df_map_p["Persistencia (km)"].min())

        df_map_p["marker_size"] = 12 + 28 * (
            (df_map_p["Persistencia (km)"] - _min_p) / max(_max_p - _min_p, 1e-9)
        )

        fig_map_pers = go.Figure()

        # Unidades de salud (fondo)
        fig_map_pers.add_trace(go.Scattermap(
            lat=pts_p[:, 0].tolist(),
            lon=pts_p[:, 1].tolist(),
            mode="markers",
            marker=dict(size=4, color="#3498db", opacity=0.35),
            name="Unidades de salud",
            hoverinfo="skip",
        ))

        # Huecos significativos por tipo
        for _tipo, _is_int in [("Interior", True), ("Borde", False)]:
            _sub = df_map_p[df_map_p["Tipo"] == _tipo]
            if len(_sub) == 0:
                continue
            fig_map_pers.add_trace(go.Scattermap(
                lat=_sub["lat"].tolist(),
                lon=_sub["lon"].tolist(),
                mode="markers+text",
                marker=dict(
                    size=_sub["marker_size"].tolist(),
                    color=_sub["Persistencia (km)"].tolist(),
                    colorscale="YlOrRd",
                    cmin=_min_p, cmax=_max_p,
                    showscale=_is_int,
                    colorbar=dict(title="Persistencia<br>(km)", x=1.01, thickness=14) if _is_int else {},
                    opacity=0.85 if _is_int else 0.55,
                ),
                text=_sub["label"].tolist(),
                textfont=dict(size=8, color="#2d3436"),
                name="H₁ Interior" if _is_int else "H₁ Borde (posible artefacto)",
                customdata=_sub[["Persistencia (km)", "Nacimiento (km)", "Muerte (km)", "Tipo"]].values,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Persistencia: %{customdata[0]:.3f} km<br>"
                    "Nacimiento ε: %{customdata[1]:.3f} km<br>"
                    "Muerte ε: %{customdata[2]:.3f} km<br>"
                    "Tipo: %{customdata[3]}<extra></extra>"
                ),
            ))

        fig_map_pers.update_layout(
            map=dict(style="carto-positron", center=dict(lat=_clat, lon=_clon), zoom=10),
            height=540,
            margin=dict(t=40, b=0, l=0, r=0),
            legend=dict(orientation="h", y=-0.05, font_size=12),
            title=f"{n_sig} huecos H₁ significativos — {mun_p} · umbral {thresh} km",
        )
        st.plotly_chart(fig_map_pers, width="stretch")
        st.caption(
            "**Círculos rellenos** = huecos interiores (más confiables). "
            "**Círculos semitransparentes** = huecos en el borde (posibles artefactos). "
            "**Tamaño y color** proporcionales a la persistencia del hueco."
        )
    else:
        st.info(
            f"No hay huecos con centros detectados y persistencia ≥ {thresh} km. "
            "Intenta reducir el umbral."
        )

# TAB PRIORIZACIÓN DE HUECOS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_prior:

    st.header("Priorización de Huecos H₁")
    st.caption(
        "Cada hueco topológico detectado se pondera combinando su persistencia matemática "
        "con variables territoriales de vulnerabilidad social, para pasar de un análisis "
        "puramente topológico a una lectura de política pública."
    )

    # ── Controles ─────────────────────────────────────────────────────────────
    pr1, pr2, pr4 = st.columns([2, 2, 1])
    with pr1:
        mun_pr = st.selectbox(
            "Alcaldía",
            ["CDMX completa"] + sorted(denue["municipio"].unique().tolist()),
            key="mun_pr",
        )
    with pr2:
        subsec_opts_pr = ["Todos los subsectores"] + sorted(
            denue[denue["sector"] == "Público"]["subsector"].dropna().unique().tolist()
        )
        subsec_pr = st.selectbox("Subsector público", subsec_opts_pr, key="subsec_pr")
    eps_pr = 1.5
    with pr4:
        max_pts_pr = st.slider("Máx. puntos", 50, 2500, 2100, 50, key="maxpts_prior")

    # ── Preparar datos ─────────────────────────────────────────────────────────
    # 1. Unidades públicas
    pub_pr = denue[denue["sector"] == "Público"].dropna(subset=["latitud", "longitud"])
    if mun_pr != "CDMX completa":
        pub_pr = pub_pr[pub_pr["municipio"] == mun_pr]
    if subsec_pr != "Todos los subsectores":
        pub_pr = pub_pr[pub_pr["subsector"] == subsec_pr]

    # 2. AGEB lookup: CONEVAL + IDS
    # La normalización de densidad se calcula sobre el universo GLOBAL (todas las AGEBs)
    # usando rango percentil → distribución uniforme [0,1], sin comprimir zonas de baja densidad
    ageb_pr = coneval.merge(
        ids[["cvegeo", "bajo_desarrollo_norm", "prop_nbi_norm",
             "grado_ids", "grado_ids_num", "poblacion_ids"]],
        on="cvegeo", how="left",
    ).copy()
    _dens_raw = ageb_pr["poblacion_ids"] / ageb_pr["area_km2"].replace(0, np.nan)
    ageb_pr["densidad_norm"] = _dens_raw.fillna(0).rank(pct=True)

    if mun_pr != "CDMX completa":
        _cve_pr = ageb_pr["cvegeo"].astype(str).str[2:5]
        ageb_pr = ageb_pr[_cve_pr == MUN_INV.get(mun_pr, "000")]
    ageb_pr = ageb_pr.dropna(subset=["centroide_lat", "centroide_lon"]).reset_index(drop=True)

    # 3. TDA
    n_univ_pr = len(pub_pr)
    if n_univ_pr > max_pts_pr:
        pub_pr_tda = pub_pr.sample(max_pts_pr, random_state=42)
        st.caption(f"Muestra de {max_pts_pr} de {n_univ_pr} unidades públicas.")
    else:
        pub_pr_tda = pub_pr
        st.caption(f"Se usan {n_univ_pr} unidades públicas para el cálculo TDA.")
    _pts_pr = pub_pr_tda[["latitud", "longitud"]].values

    if len(_pts_pr) < 5:
        st.warning("No hay suficientes unidades públicas con este filtro (mínimo 5).")
        st.stop()

    with st.spinner("Calculando TDA…"):
        _pts_pr_key         = tuple(map(tuple, _pts_pr))
        _dgms_pr, _, _h1c_pr = _compute_tda_full(_pts_pr_key)
        _h1_pr  = np.array(_dgms_pr[1]) if len(_dgms_pr[1]) else np.empty((0, 2))

    if len(_h1_pr) == 0:
        st.info("No se detectaron huecos H₁ con estos parámetros.")
        st.stop()

    _h1p_pr      = _h1_pr[:, 1] - _h1_pr[:, 0]
    _max_pers_pr = float(_h1p_pr.max()) or 1.0
    _h1p_norm    = _h1p_pr / _max_pers_pr

    # ── Umbral auto-calculado (μ + ½σ) ────────────────────────────────────────
    _h1p_pr_fin = _h1p_pr[~np.isinf(_h1p_pr)]
    if len(_h1p_pr_fin) >= 2:
        _mu_pr    = float(_h1p_pr_fin.mean())
        _sigma_pr = float(_h1p_pr_fin.std())
        _tsug_pr  = _mu_pr + 0.5 * _sigma_pr
    else:
        _mu_pr = _sigma_pr = _tsug_pr = 0.0
    _thresh_pr_default = float(np.round(np.clip(_tsug_pr, 0.1, 3.0), 1)) if _tsug_pr > 0 else 0.5

    _tc1, _tc2 = st.columns([3, 2])
    with _tc1:
        thresh_pr = st.slider(
            "Umbral de persistencia significativa (km)",
            min_value=0.1, max_value=3.0, value=_thresh_pr_default, step=0.1,
            key="thresh_pr",
            help=f"Auto-calculado como μ + ½σ = {_tsug_pr:.3f} km. Ajusta manualmente si lo requieres.",
        )
    with _tc2:
        st.metric("Umbral sugerido (μ + ½σ)", f"{_tsug_pr:.3f} km")

    _h1c_arr  = np.array(_h1c_pr[:len(_h1_pr)]) if len(_h1c_pr) else np.empty((0, 2))
    _bd_pr    = _border_dists(_h1c_arr, _pts_pr) if len(_h1c_arr) else np.array([])
    _int_mask = (
        (_h1p_pr >= thresh_pr) & (_bd_pr >= eps_pr)
        if len(_bd_pr) else (_h1p_pr >= thresh_pr)
    )

    _sig_idx = np.where(_h1p_pr >= thresh_pr)[0]
    if len(_sig_idx) == 0:
        st.info(f"No hay huecos con persistencia ≥ {thresh_pr} km. Baja el umbral.")
        st.stop()

    # 4. Construir tabla de huecos con índice de importancia
    _ageb_coords = ageb_pr[["centroide_lat", "centroide_lon"]].values if len(ageb_pr) else np.empty((0, 2))

    _holes = []
    for _idx in _sig_idx:
        _center     = _h1c_arr[_idx] if _idx < len(_h1c_arr) else None
        _is_int     = bool(_int_mask[_idx]) if _idx < len(_int_mask) else False
        _social     = {}
        _dist_ageb  = np.nan
        _ageb_label = "—"

        if _center is not None and len(_ageb_coords) > 0:
            _D_ha     = _dist_cross_km(np.array([_center]), _ageb_coords)[0]
            _near_idx = int(_D_ha.argmin())
            _dist_ageb = float(_D_ha[_near_idx])
            _row       = ageb_pr.iloc[_near_idx]
            _ageb_label = str(_row.get("cvegeo", "—"))
            _social = {
                "rezago_norm":        float(_row.get("rezago_norm", 0) or 0),
                "bajo_desarrollo_norm": float(_row.get("bajo_desarrollo_norm", 0) or 0),
                "prop_nbi_norm":      float(_row.get("prop_nbi_norm", 0) or 0),
                "densidad_norm":      float(_row.get("densidad_norm", 0) or 0),
                "grado_rezago":       str(_row.get("grado_rezago_social", "—")),
                "grado_ids":          str(_row.get("grado_ids", "—")),
            }

        P_i = float(_h1p_norm[_idx])
        D_i = _social.get("densidad_norm", 0.0)
        R_i = _social.get("rezago_norm", 0.0)
        B_i = _social.get("bajo_desarrollo_norm", 0.0)
        N_i = _social.get("prop_nbi_norm", 0.0)
        I_i = 0.40 * P_i + 0.15 * D_i + 0.10 * R_i + 0.25 * B_i + 0.10 * N_i

        _ctx = "Completo" if (not np.isnan(_dist_ageb) and _dist_ageb < eps_pr) \
               else "Aproximado" if not np.isnan(_dist_ageb) else "Sin datos"

        _holes.append({
            "lat": float(_center[0]) if _center is not None else np.nan,
            "lon": float(_center[1]) if _center is not None else np.nan,
            "birth":       float(_h1_pr[_idx, 0]),
            "death":       float(_h1_pr[_idx, 1]) if not np.isinf(_h1_pr[_idx, 1]) else _max_pers_pr,
            "persistence": float(_h1p_pr[_idx]),
            "P_i": P_i, "D_i": D_i, "R_i": R_i, "B_i": B_i, "N_i": N_i,
            "indice":      round(I_i, 4),
            "interior":    _is_int,
            "tipo":        "Interior" if _is_int else "Artefacto de borde",
            "dist_ageb_km": round(_dist_ageb, 3) if not np.isnan(_dist_ageb) else np.nan,
            "contexto":    _ctx,
            "grado_rezago": _social.get("grado_rezago", "—"),
            "grado_ids":    _social.get("grado_ids", "—"),
            "ageb_cvegeo":  _ageb_label,
        })

    df_holes = (
        pd.DataFrame(_holes)
        .sort_values("indice", ascending=False)
        .reset_index(drop=True)
    )
    df_holes["label"]  = [f"H{i+1:03d}" for i in range(len(df_holes))]
    df_holes["w_P"]    = (df_holes["P_i"] * 0.40).round(4)
    df_holes["w_D"]    = (df_holes["D_i"] * 0.15).round(4)
    df_holes["w_R"]    = (df_holes["R_i"] * 0.10).round(4)
    df_holes["w_B"]    = (df_holes["B_i"] * 0.25).round(4)
    df_holes["w_N"]    = (df_holes["N_i"] * 0.10).round(4)

    # ── Métricas resumen ───────────────────────────────────────────────────────
    _pm1, _pm2, _pm3, _pm4 = st.columns(4)
    _pm1.metric("Huecos significativos", len(df_holes))
    _pm2.metric("Interiores confirmados", int(df_holes["interior"].sum()))
    _pm3.metric("Índice máximo", f"{df_holes['indice'].max():.3f}")
    _pm4.metric("Persistencia máxima", f"{df_holes['persistence'].max():.2f} km")

    st.divider()

    # ── Nota metodológica ──────────────────────────────────────────────────────
    with st.expander("Metodología del índice de importancia"):
        st.markdown("""
**Fórmula:**

$$I(H_i) = 0.44 \\cdot P_i + 0.28 \\cdot B_i + 0.17 \\cdot D_i + 0.11 \\cdot N_i$$

| Variable | Descripción | Fuente | Peso |
|---|---|---|---|
| P_i | Persistencia topológica normalizada | Ripser / TDA | 0.44 |
| B_i | Bajo desarrollo social normalizado | IDS EVALUA CDMX | 0.28 |
| D_i | Densidad poblacional normalizada | IDS × CONEVAL | 0.17 |
| N_i | Proporción de NBI normalizada | IDS EVALUA CDMX | 0.11 |

**Ponderaciones del análisis de sensibilidad:**

| Escenario | P_i Persistencia | B_i Bajo desarrollo IDS | D_i Densidad | N_i NBI |
|---|---:|---:|---:|---:|
| Base | 0.44 | 0.28 | 0.17 | 0.11 |
| Topológico | 0.61 | 0.17 | 0.11 | 0.11 |
| Social | 0.33 | 0.33 | 0.17 | 0.17 |
| Densidad | 0.39 | 0.22 | 0.28 | 0.11 |

        """)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # A. MAPA
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("A · Mapa de Huecos Priorizados")

    _df_map = df_holes.dropna(subset=["lat", "lon"]).copy()
    if len(_df_map) > 0:
        _idx_range = _df_map["indice"].max() - _df_map["indice"].min()
        _df_map["marker_size"] = 12 + 28 * (
            (_df_map["indice"] - _df_map["indice"].min()) / (_idx_range + 1e-10)
        )

        # Cuartiles de importancia
        _n_q = min(4, len(_df_map["indice"].unique()))
        if _n_q > 1:
            _df_map["cuartil"] = pd.qcut(
                _df_map["indice"], q=_n_q,
                labels=[f"Q{i+1}" for i in range(_n_q)],
                duplicates="drop",
            ).astype(str)
        else:
            _df_map["cuartil"] = "Q1"
        _q_colors = {"Q1": "#74b9ff", "Q2": "#fdcb6e", "Q3": "#e17055", "Q4": "#d63031"}

        fig_map_pr = px.scatter_map(
            _df_map,
            lat="lat", lon="lon",
            color="cuartil",
            color_discrete_map=_q_colors,
            size="marker_size",
            size_max=40,
            hover_name="label",
            hover_data={
                "persistence": ":.3f", "indice": ":.3f",
                "grado_rezago": True, "grado_ids": True,
                "tipo": True, "marker_size": False, "cuartil": False,
            },
            zoom=10, height=520,
            title="Huecos H₁ priorizados (tamaño proporcional al índice de importancia)",
        )
        # Añadir etiquetas de texto
        fig_map_pr.add_trace(go.Scattermap(
            lat=_df_map["lat"].tolist(),
            lon=_df_map["lon"].tolist(),
            mode="text",
            text=_df_map["label"].tolist(),
            textfont=dict(size=9, color="#2d3436"),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig_map_pr.update_layout(
            map_style="carto-positron",
            margin=dict(t=40, b=0, l=0, r=0),
            legend_title_text="Cuartil de importancia",
        )
        st.plotly_chart(fig_map_pr, width="stretch")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # B. DESCOMPOSICIÓN DEL ÍNDICE
    # ══════════════════════════════════════════════════════════════════════════
    _comentado_prior_b = """
    st.subheader("B · Descomposición del Índice de Importancia")
    st.caption(
        "Cada barra muestra la contribución ponderada de cada componente al índice total. "
        "I(H) = 0.40·P + 0.25·B + 0.15·D + 0.10·R + 0.10·N"
    )

    _comp_defs = [
        ("w_P", "Persistencia topológica (×0.40)", "#2d3436"),
        ("w_B", "Bajo desarrollo IDS (×0.25)",      "#d63031"),
        ("w_D", "Densidad poblacional (×0.15)",      "#6c5ce7"),
        ("w_R", "Rezago social CONEVAL (×0.10)",     "#e17055"),
        ("w_N", "Necesidades básicas NBI (×0.10)",   "#fdcb6e"),
    ]

    _fig_decomp = go.Figure()
    for _col, _lbl, _col_hex in _comp_defs:
        _fig_decomp.add_bar(
            y=df_holes["label"],
            x=df_holes[_col],
            name=_lbl,
            orientation="h",
            marker_color=_col_hex,
        )
    _fig_decomp.update_layout(
        barmode="stack",
        xaxis_title="Contribución al índice (suma = I(H))",
        height=max(320, len(df_holes) * 30 + 120),
        margin=dict(t=20, b=40, l=70, r=20),
        legend=dict(orientation="h", y=-0.25, font_size=11),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(_fig_decomp, width="stretch")

    st.divider()
    """

    # ══════════════════════════════════════════════════════════════════════════
    # C. TABLA DE PRIORIZACIÓN
    # ══════════════════════════════════════════════════════════════════════════
    _comentado_prior_cd = """
    st.subheader("C · Tabla de Priorización")

    _df_show = df_holes[[
        "label", "tipo", "persistence", "indice",
        "w_P", "w_B", "w_D", "w_R", "w_N",
        "grado_rezago", "grado_ids", "contexto",
    ]].copy()
    _df_show.columns = [
        "Hueco", "Tipo", "Persistencia (km)", "Índice I(H)",
        "Pers.×0.40", "IDS bajo×0.25", "Densidad×0.15", "Rezago×0.10", "NBI×0.10",
        "Rezago social", "Grado IDS", "Contexto social",
    ]
    st.dataframe(
        _df_show.style.background_gradient(subset=["Índice I(H)"], cmap="RdYlGn"),
        width="stretch",
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # D. LECTURA INTERPRETATIVA
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("D · Lectura Interpretativa de los Huecos Principales")

    for _, _r in df_holes.head(min(5, len(df_holes))).iterrows():
        _P_lv  = "alta" if _r["P_i"] > 0.7 else "media" if _r["P_i"] > 0.4 else "baja"
        _tipo  = "interior confirmado" if _r["interior"] else "posible artefacto de borde"
        _rezago = _r.get("grado_rezago", "—")
        _ids_g  = _r.get("grado_ids", "—")

        _txt = (
            f"Persistencia topológica <b>{_P_lv}</b> ({_r['persistence']:.2f} km): "
            f"el hueco es {'muy estable y estructuralmente robusto' if _P_lv == 'alta' else 'moderadamente estable' if _P_lv == 'media' else 'poco persistente; interpretar con cautela'}. "
            f"Se clasifica como <b>{_tipo}</b>. "
        )
        if _rezago not in ("—", "nan"):
            _txt += (
                f"El AGEB más cercano presenta rezago social <b>{_rezago.lower()}</b> "
                f"y un IDS de nivel <b>{_ids_g.lower()}</b>. "
            )
        if _r["B_i"] > 0.6:
            _txt += "El alto nivel de bajo desarrollo social refuerza la urgencia de intervención. "
        if _r["N_i"] > 0.6:
            _txt += "La elevada proporción de necesidades básicas insatisfechas indica demanda significativa de servicios públicos. "
        if _r["D_i"] > 0.6:
            _txt += "La alta densidad poblacional amplía el impacto potencial de una nueva unidad de salud. "
        if _r["contexto"] == "Aproximado":
            _txt += "<i>(Contexto social asignado desde un AGEB a más de ε km del centroide; interpretar con cautela.)</i>"

        _border_col = (
            "#d63031" if _r["indice"] > 0.6 else
            "#e17055" if _r["indice"] > 0.4 else
            "#fdcb6e" if _r["indice"] > 0.25 else "#74b9ff"
        )
        st.markdown(
            f"<div style='background:#f8f9fa;border-left:4px solid {_border_col};"
            f"padding:12px 16px;border-radius:4px;margin-bottom:10px'>"
            f"<b>{_r['label']}</b>&nbsp;&nbsp;"
            f"Índice: <b>{_r['indice']:.3f}</b> &nbsp;·&nbsp; "
            f"Persistencia: <b>{_r['persistence']:.2f} km</b>"
            f"<br><small style='color:#444;line-height:1.5'>{_txt}</small></div>",
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
    """
    st.divider()

    # E. ANALISIS DE SENSIBILIDAD
    st.subheader("E · Análisis de Sensibilidad de Pesos")
    st.caption(
        "Este análisis evalúa si los huecos prioritarios dependen demasiado de una sola "
        "elección de pesos o si se mantienen robustos bajo criterios topológicos, sociales "
        "y de densidad."
    )

    top_k_sens = 5

    df_sens = df_holes.copy()
    _escenarios_sens = {
        "Base": {"P_i": 0.40, "B_i": 0.25, "D_i": 0.15, "R_i": 0.10, "N_i": 0.10},
        "Topológico": {"P_i": 0.55, "B_i": 0.15, "D_i": 0.10, "R_i": 0.10, "N_i": 0.10},
        "Social": {"P_i": 0.30, "B_i": 0.30, "D_i": 0.15, "R_i": 0.10, "N_i": 0.15},
        "Densidad": {"P_i": 0.35, "B_i": 0.20, "D_i": 0.25, "R_i": 0.10, "N_i": 0.10},
    }

    for _esc, _pesos in _escenarios_sens.items():
        _col_ind = f"I_{_esc}"
        df_sens[_col_ind] = sum(df_sens[_var] * _w for _var, _w in _pesos.items()).round(4)

    _top_k_eff = min(top_k_sens, len(df_sens))
    _tops_sens = {}
    _rows_top_sens = []

    for _esc in _escenarios_sens:
        _col_ind = f"I_{_esc}"
        _top_esc = (
            df_sens.sort_values(_col_ind, ascending=False)
            .head(_top_k_eff)
            .reset_index(drop=True)
        )
        _tops_sens[_esc] = _top_esc
        for _rank, (_, _r) in enumerate(_top_esc.iterrows(), start=1):
            _rows_top_sens.append({
                "Escenario": _esc,
                "Ranking": _rank,
                "Hueco": _r["label"],
                "Índice": _r[_col_ind],
                "Persistencia (km)": _r["persistence"],
                "Tipo": _r["tipo"],
                "Rezago social": _r["grado_rezago"],
                "Grado IDS": _r["grado_ids"],
                "Contexto social": _r["contexto"],
            })

    _df_top_sens = pd.DataFrame(_rows_top_sens)
    # st.dataframe(_df_top_sens)

    _labels_top_sens = [
        _label
        for _top_esc in _tops_sens.values()
        for _label in _top_esc["label"].tolist()
    ]
    _apariciones = pd.Series(_labels_top_sens).value_counts()
    _df_robustez = (
        df_sens[df_sens["label"].isin(_apariciones.index)][[
            "label", "persistence", "tipo", "grado_rezago", "grado_ids", "contexto"
        ]]
        .copy()
        .rename(columns={"label": "Hueco"})
    )
    _df_robustez["apariciones_top"] = _df_robustez["Hueco"].map(_apariciones).fillna(0).astype(int)
    _df_robustez["robustez_pct"] = (
        _df_robustez["apariciones_top"] / len(_escenarios_sens) * 100
    ).round(1)
    _df_robustez["clasificacion_robustez"] = _df_robustez["apariciones_top"].map({
        4: "Robusto integral",
        3: "Muy estable",
        2: "Estable parcial",
        1: "Dependiente del criterio",
    })
    _df_robustez = _df_robustez.sort_values(
        ["apariciones_top", "robustez_pct", "persistence"],
        ascending=[False, False, False],
    )
    _df_robustez = _df_robustez[[
        "Hueco", "apariciones_top", "robustez_pct", "clasificacion_robustez",
        "persistence", "tipo", "grado_rezago", "grado_ids", "contexto",
    ]]
    # st.dataframe(_df_robustez)

    if len(_df_robustez) > 0:
        _fig_robustez = px.bar(
            _df_robustez,
            x="Hueco",
            y="robustez_pct",
            color="clasificacion_robustez",
            title="Robustez de huecos ante cambios de pesos",
            labels={
                "robustez_pct": "Robustez (%)",
                "clasificacion_robustez": "Clasificación",
            },
            height=380,
        )
        _fig_robustez.update_layout(margin=dict(t=50, b=40, l=0, r=0))
        st.plotly_chart(_fig_robustez, width="stretch")

        _labels_robustos = _df_robustez["Hueco"].tolist()
        _df_indices_sens = df_sens[df_sens["label"].isin(_labels_robustos)][
            ["label"] + [f"I_{_esc}" for _esc in _escenarios_sens]
        ].copy()
        _df_indices_sens = _df_indices_sens.rename(columns={"label": "Hueco"})
        _df_indices_long = _df_indices_sens.melt(
            id_vars="Hueco",
            var_name="Escenario",
            value_name="Índice",
        )
        _df_indices_long["Escenario"] = _df_indices_long["Escenario"].str.replace("I_", "", regex=False)

        _fig_indices_sens = px.bar(
            _df_indices_long,
            x="Hueco",
            y="Índice",
            color="Escenario",
            barmode="group",
            title="Comparación del índice por escenario de pesos",
            height=420,
        )
        _fig_indices_sens.update_layout(margin=dict(t=50, b=40, l=0, r=0))
        st.plotly_chart(_fig_indices_sens, width="stretch")

        _base_rank = {
            _label: _rank
            for _rank, _label in enumerate(_tops_sens["Base"]["label"].tolist(), start=1)
        }

        def _huecos_que_suben(_escenario):
            _labels = []
            for _rank, _label in enumerate(_tops_sens[_escenario]["label"].tolist(), start=1):
                _rank_base = _base_rank.get(_label, _top_k_eff + 1)
                if _rank < _rank_base:
                    _labels.append(_label)
            return _labels

        _robustos_integrales = _df_robustez[
            _df_robustez["apariciones_top"] == len(_escenarios_sens)
        ]["Hueco"].tolist()
        _suben_topologico = _huecos_que_suben("Topológico")
        _suben_social = _huecos_que_suben("Social")

        _txt_robustos = (
            f"Huecos presentes en todos los escenarios: **{', '.join(_robustos_integrales)}**."
            if _robustos_integrales else
            "No hubo huecos completamente robustos en todos los escenarios."
        )
        _txt_topologico = (
            f"En el escenario **Topológico** suben en el ranking: **{', '.join(_suben_topologico)}**."
            if _suben_topologico else
            "En el escenario **Topológico** no hay huecos que suban respecto al ranking Base dentro del top seleccionado."
        )
        _txt_social = (
            f"En el escenario **Social** suben en el ranking: **{', '.join(_suben_social)}**."
            if _suben_social else
            "En el escenario **Social** no hay huecos que suban respecto al ranking Base dentro del top seleccionado."
        )
        st.info(
            f"{_txt_robustos}\n\n"
            f"{_txt_topologico}\n\n"
            f"{_txt_social}\n\n"
            "Los huecos robustos son mejores candidatos para intervención porque su prioridad "
            "no depende de una sola elección de pesos."
        )

# TAB HALLAZGOS — REPORTE DINÁMICO
# ═══════════════════════════════════════════════════════════════════════════════
_hidden_tab_report = '''
with tab_report:

    st.header("Reporte de Hallazgos Principales")
    st.caption("El reporte se genera automáticamente según los filtros aplicados.")

    # ── Controles ─────────────────────────────────────────────────────────────
    rp1, rp2, rp3, rp4 = st.columns([2, 2, 2, 2])
    with rp1:
        mun_r = st.selectbox("Alcaldía", ["CDMX completa"] + sorted(denue["municipio"].unique().tolist()), key="mun_r")
    with rp2:
        subsec_opts_r = ["Todos los subsectores"] + sorted(
            denue[denue["sector"] == "Público"]["subsector"].dropna().unique().tolist()
        )
        subsec_r = st.selectbox("Subsector", subsec_opts_r, key="subsec_r")
    with rp3:
        eps_r = st.slider("Radio ε (km)", 0.25, 8.0, 1.5, 0.25, key="eps_r")
    with rp4:
        thresh_r = st.slider("Umbral sig. (km)", 0.1, 3.0, 0.5, 0.1, key="thresh_r")

    from datetime import date

    # ── Preparar datos ────────────────────────────────────────────────────────
    # 1. Unidades públicas filtradas
    pub_r = denue[denue["sector"] == "Público"].dropna(subset=["latitud", "longitud"])
    if mun_r != "CDMX completa":
        pub_r = pub_r[pub_r["municipio"] == mun_r]
    if subsec_r != "Todos los subsectores":
        pub_r = pub_r[pub_r["subsector"] == subsec_r]

    # 2. DENUE completo (todos sectores) para contexto
    denue_ctx = denue.copy()
    if mun_r != "CDMX completa":
        denue_ctx = denue_ctx[denue_ctx["municipio"] == mun_r]
    if subsec_r != "Todos los subsectores":
        denue_ctx = denue_ctx[denue_ctx["subsector"] == subsec_r]

    # 3. AGEBs CONEVAL + IDS integrado
    ageb_full = coneval.merge(
        ids[["cvegeo", "ids", "grado_ids", "grado_ids_num"]], on="cvegeo", how="left"
    )
    if mun_r != "CDMX completa":
        cve_r = ageb_full["cvegeo"].astype(str).str[2:5]
        ageb_full = ageb_full[cve_r == MUN_INV.get(mun_r, "000")]

    ageb_geo_r = ageb_full.dropna(subset=["centroide_lat", "centroide_lon"])

    # 4. TDA sobre muestra (máx 2500)
    pts_r = pub_r[["latitud", "longitud"]].values
    if len(pts_r) > 2500:
        pts_r = pub_r.sample(2500, random_state=42)[["latitud", "longitud"]].values

    with st.spinner("Generando análisis…"):
        if len(pts_r) >= 5:
            pts_r_key = tuple(map(tuple, pts_r))
            dgms_r, D_r_list, h1c_r = _compute_tda_full(pts_r_key)
            D_r    = np.array(D_r_list)
            h0_r   = np.array(dgms_r[0])
            h1_r   = np.array(dgms_r[1]) if len(dgms_r[1]) else np.empty((0, 2))
            max_er = min(float(D_r.max()) if D_r.size else 15.0, 15.0)

            h1_pers_r  = (h1_r[:, 1] - h1_r[:, 0]) if len(h1_r) else np.array([])
            n_sig_r    = int((h1_pers_r >= thresh_r).sum()) if len(h1_pers_r) else 0
            max_pers_r = float(h1_pers_r.max()) if len(h1_pers_r) else 0.0
            # Efecto de borde: clasificar huecos significativos
            bd_r       = _border_dists(h1c_r[:len(h1_r)], pts_r) if len(h1c_r) else np.array([])
            sig_mask_r = (h1_pers_r >= thresh_r) if len(h1_pers_r) else np.array([], dtype=bool)
            n_sig_int_r = int(((sig_mask_r) & (bd_r >= eps_r)).sum()) if len(bd_r) else n_sig_r
            n_sig_brd_r = n_sig_r - n_sig_int_r

            eps_range_r    = np.linspace(0, max_er, 300)
            b0_r, b1_r     = _betti_curves(h0_r, h1_r, eps_range_r)
            peak_eps_r     = float(eps_range_r[b1_r.argmax()]) if b1_r.max() > 0 else 0.0
            peak_b1_r      = int(b1_r.max())

            # Coverage AGEBs
            ageb_pts_r = ageb_geo_r[["centroide_lat", "centroide_lon"]].values
            if len(ageb_pts_r) > 0:
                md_r = np.array(_compute_coverage(pts_r_key, tuple(map(tuple, ageb_pts_r))))
                ageb_geo_r = ageb_geo_r.copy()
                ageb_geo_r["dist_min"]  = md_r
                ageb_geo_r["cobertura"] = np.where(md_r <= eps_r, "Cubierta", "Sin cobertura")
                pct_cov_r = float((md_r <= eps_r).mean() * 100)
                n_unc_r   = int((md_r > eps_r).sum())
            else:
                pct_cov_r, n_unc_r = 100.0, 0
                ageb_geo_r["dist_min"]  = 0.0
                ageb_geo_r["cobertura"] = "Cubierta"
        else:
            st.warning("No hay suficientes unidades públicas con los filtros aplicados.")
            st.stop()

    # ── Encabezado del reporte ────────────────────────────────────────────────
    area_label = mun_r if mun_r != "CDMX completa" else "Ciudad de México (CDMX)"
    subsec_label = subsec_r if subsec_r != "Todos los subsectores" else "todos los subsectores"
    st.markdown(
        f"""
        <div style="background:#f0f4ff;border-left:5px solid #4e8df5;padding:16px 20px;border-radius:6px;margin-bottom:8px">
        <h3 style="margin:0;color:#1a2f6e">Reporte TDA · Sector Salud Pública</h3>
        <p style="margin:4px 0 0;color:#444">
        <b>Área:</b> {area_label} &nbsp;|&nbsp;
        <b>Subsector:</b> {subsec_label} &nbsp;|&nbsp;
        <b>Radio ε:</b> {eps_r} km &nbsp;|&nbsp;
        <b>Umbral sig.:</b> {thresh_r} km &nbsp;|&nbsp;
        <b>Fecha:</b> {date.today().strftime("%d/%m/%Y")}
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 1. PANORAMA DE LA OFERTA
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("1 · Panorama de la Oferta de Salud Pública")

    r1a, r1b, r1c, r1d = st.columns(4)
    r1a.metric("Unidades públicas", f"{len(pub_r):,}")
    r1b.metric("Unidades privadas", f"{len(denue_ctx[denue_ctx['sector']=='Privado']):,}")
    r1c.metric("No gubernamentales", f"{len(denue_ctx[denue_ctx['sector']=='No gubernamental']):,}")
    r1d.metric("Total sector salud", f"{len(denue_ctx):,}")

    col_of1, col_of2 = st.columns(2)

    with col_of1:
        mun_pub_r = pub_r.groupby("municipio").size().reset_index(name="n").sort_values("n", ascending=True)
        fig_of1 = px.bar(
            mun_pub_r, x="n", y="municipio", orientation="h",
            title="Unidades públicas por alcaldía",
            color="n", color_continuous_scale="Blues",
            labels={"municipio": "", "n": "Unidades"},
            height=300,
        )
        fig_of1.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=35, b=0))
        st.plotly_chart(fig_of1, width="stretch")

    with col_of2:
        sub_r_counts = pub_r["subsector"].value_counts().reset_index()
        sub_r_counts.columns = ["subsector", "n"]
        fig_of2 = px.pie(
            sub_r_counts, names="subsector", values="n",
            title="Distribución por subsector (público)",
            hole=0.42,
            color_discrete_sequence=px.colors.qualitative.Set2,
            height=300,
        )
        fig_of2.update_layout(margin=dict(l=0, r=0, t=35, b=0))
        st.plotly_chart(fig_of2, width="stretch")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 2. HALLAZGOS TOPOLÓGICOS
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("2 · Hallazgos del Análisis Topológico (TDA)")

    r2a, r2b, r2c, r2d = st.columns(4)
    r2a.metric("Huecos H₁ detectados", len(h1_r))
    r2b.metric(
        f"Huecos significativos (≥{thresh_r}km)",
        n_sig_r,
        f"{n_sig_int_r} interiores · {n_sig_brd_r} borde",
    )
    r2c.metric("Persistencia H₁ máxima", f"{max_pers_r:.2f} km")
    r2d.metric("ε crítico (pico β₁)", f"{peak_eps_r:.2f} km", f"β₁ máx = {peak_b1_r}")

    col_tda1, col_tda2 = st.columns([3, 2])

    with col_tda1:
        # Curva de Betti compacta
        df_b_r = pd.DataFrame({
            "ε (km)": np.concatenate([eps_range_r, eps_range_r]),
            "Valor": np.concatenate([b0_r, b1_r]),
            "Indicador": ["β₀ Componentes"] * len(eps_range_r) + ["β₁ Huecos"] * len(eps_range_r),
        })
        fig_b_r = px.line(
            df_b_r, x="ε (km)", y="Valor", color="Indicador",
            color_discrete_map={"β₀ Componentes": "#e74c3c", "β₁ Huecos": "#3498db"},
            title="Evolución topológica (Números de Betti)",
            height=280,
        )
        fig_b_r.add_vline(x=eps_r, line_dash="dash", line_color="green", opacity=0.6,
                          annotation_text=f"ε={eps_r}", annotation_position="top right")
        if peak_b1_r > 0:
            fig_b_r.add_vline(x=peak_eps_r, line_dash="dot", line_color="#3498db", opacity=0.5,
                              annotation_text=f"pico β₁", annotation_position="top left")
        fig_b_r.update_layout(margin=dict(l=0, r=0, t=35, b=0))
        st.plotly_chart(fig_b_r, width="stretch")

    with col_tda2:
        st.markdown("**Top huecos H₁ significativos**")
        if n_sig_r > 0:
            sig_mask = h1_pers_r >= thresh_r
            df_top = pd.DataFrame({
                "Nacimiento": h1_r[sig_mask, 0].round(3),
                "Muerte":     h1_r[sig_mask, 1].round(3),
                "Persist.":   h1_pers_r[sig_mask].round(3),
            }).sort_values("Persist.", ascending=False).head(8).reset_index(drop=True)
            df_top.index += 1
            st.dataframe(df_top, use_container_width=True, height=250)
        else:
            st.info(f"Sin huecos con persistencia ≥ {thresh_r} km en la selección.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 3. BRECHAS DE COBERTURA
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("3 · Brechas de Cobertura")

    r3a, r3b, r3c = st.columns(3)
    r3a.metric("AGEBs con cobertura", f"{pct_cov_r:.1f}%", f"ε = {eps_r} km")
    r3b.metric("AGEBs sin cobertura", f"{n_unc_r:,}", f"{100 - pct_cov_r:.1f}%")
    r3c.metric("Total AGEBs analizadas", f"{len(ageb_geo_r):,}")

    col_cov1, col_cov2 = st.columns(2)

    with col_cov1:
        rez_ord_r = ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]
        df_cov_r = (
            ageb_geo_r.groupby(["cobertura", "grado_rezago_social"])
            .size().reset_index(name="n")
        )
        df_cov_r["grado_rezago_social"] = pd.Categorical(
            df_cov_r["grado_rezago_social"], categories=rez_ord_r, ordered=True
        )
        totals_r = df_cov_r.groupby("cobertura")["n"].transform("sum")
        df_cov_r["pct"] = (df_cov_r["n"] / totals_r * 100).round(1)
        fig_cov_r = px.bar(
            df_cov_r.sort_values("grado_rezago_social"),
            x="grado_rezago_social", y="pct", color="cobertura",
            color_discrete_map={"Cubierta": "#2ecc71", "Sin cobertura": "#e74c3c"},
            barmode="group",
            title="% AGEBs por rezago: cubiertas vs sin cobertura",
            labels={"grado_rezago_social": "Rezago", "pct": "% dentro del grupo", "cobertura": ""},
            height=280,
        )
        fig_cov_r.update_layout(margin=dict(l=0, r=0, t=35, b=0))
        st.plotly_chart(fig_cov_r, width="stretch")

    with col_cov2:
        # Distancia media por grado de rezago
        df_dist_r = ageb_geo_r.groupby("grado_rezago_social")["dist_min"].mean().reset_index()
        df_dist_r.columns = ["Rezago", "dist"]
        df_dist_r["Rezago"] = pd.Categorical(df_dist_r["Rezago"], categories=rez_ord_r, ordered=True)
        df_dist_r = df_dist_r.sort_values("Rezago")
        fig_dist_r = px.bar(
            df_dist_r, x="Rezago", y="dist",
            color="Rezago", color_discrete_map=COLORS,
            title="Distancia media a unidad pública por rezago",
            labels={"dist": "km promedio", "Rezago": ""},
            height=280,
        )
        fig_dist_r.add_hline(y=eps_r, line_dash="dash", line_color="gray",
                             annotation_text=f"ε={eps_r}km", annotation_position="right")
        fig_dist_r.update_layout(showlegend=False, margin=dict(l=0, r=0, t=35, b=0))
        st.plotly_chart(fig_dist_r, width="stretch")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 4. ZONAS DE MAYOR VULNERABILIDAD
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("4 · Zonas de Mayor Vulnerabilidad")
    st.caption("AGEBs que combinan rezago social alto/muy alto, bajo IDS **y** están sin cobertura a ε.")

    vuln_r = ageb_geo_r[
        (ageb_geo_r["grado_rezago_num"] >= 3) &
        (ageb_geo_r["cobertura"] == "Sin cobertura")
    ].copy()

    col_v1, col_v2 = st.columns([2, 3])

    with col_v1:
        nv = len(vuln_r)
        pct_v = nv / len(ageb_geo_r) * 100 if len(ageb_geo_r) > 0 else 0
        st.metric("AGEBs vulnerables sin cobertura", nv, f"{pct_v:.1f}% del total")
        if len(vuln_r) > 0:
            st.metric("Distancia media (vulnerable)", f"{vuln_r['dist_min'].mean():.2f} km")
            ids_ok = vuln_r["ids"].dropna()
            if len(ids_ok):
                st.metric("IDS promedio (vulnerable)", f"{ids_ok.mean():.3f}")
            # Distribución por rezago dentro de vulnerables
            vul_rez = vuln_r["grado_rezago_social"].value_counts()
            st.markdown("**Rezago en zonas vulnerables:**")
            for g, n_g in vul_rez.items():
                pct_g = n_g / nv * 100
                st.markdown(f"- {g}: **{n_g}** AGEBs ({pct_g:.0f}%)")
        else:
            st.success(f"No hay AGEBs vulnerables sin cobertura a ε = {eps_r} km.")

    with col_v2:
        if len(vuln_r) > 0:
            fig_v = go.Figure()
            # AGEBs normales sin cobertura
            no_vuln = ageb_geo_r[(ageb_geo_r["cobertura"] == "Sin cobertura") &
                                  (ageb_geo_r["grado_rezago_num"] < 3)]
            fig_v.add_trace(go.Scattermapbox(
                lat=no_vuln["centroide_lat"].tolist(),
                lon=no_vuln["centroide_lon"].tolist(),
                mode="markers",
                marker=dict(size=6, color="#f39c12", opacity=0.7),
                name="Sin cobertura (rezago bajo/medio)",
            ))
            fig_v.add_trace(go.Scattermapbox(
                lat=vuln_r["centroide_lat"].tolist(),
                lon=vuln_r["centroide_lon"].tolist(),
                mode="markers",
                marker=dict(size=10, color="#c0392b", opacity=0.9),
                name="Vulnerable (rezago alto + sin cobertura)",
                hovertemplate="Rezago: %{customdata[0]}<br>Dist: %{customdata[1]:.2f}km<extra></extra>",
                customdata=vuln_r[["grado_rezago_social", "dist_min"]].values.tolist(),
            ))
            # Unidades públicas de fondo
            fig_v.add_trace(go.Scattermapbox(
                lat=pub_r["latitud"].tolist()[:500],
                lon=pub_r["longitud"].tolist()[:500],
                mode="markers",
                marker=dict(size=4, color="#2ecc71", opacity=0.5),
                name="Unidades públicas",
            ))
            cl_v = float(ageb_geo_r["centroide_lat"].mean())
            co_v = float(ageb_geo_r["centroide_lon"].mean())
            fig_v.update_layout(
                mapbox=dict(style="carto-positron",
                            center=dict(lat=cl_v, lon=co_v),
                            zoom=11 if mun_r != "CDMX completa" else 10),
                height=320,
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(bgcolor="rgba(255,255,255,0.85)", font=dict(size=10)),
            )
            st.plotly_chart(fig_v, width="stretch")
        else:
            st.info("No hay zonas vulnerables con los filtros actuales.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 5. HALLAZGOS CLAVE
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("5 · Hallazgos Clave")

    sinc_r    = ageb_geo_r[ageb_geo_r["cobertura"] == "Sin cobertura"]
    top_rez_r = sinc_r["grado_rezago_social"].value_counts().idxmax() if len(sinc_r) else "—"
    max_dist_rez_r = (
        ageb_geo_r.groupby("grado_rezago_social")["dist_min"].mean().idxmax()
        if len(ageb_geo_r) > 0 else "—"
    )

    hallazgos = [
        (
            " Oferta pública",
            f"En {area_label} hay **{len(pub_r):,}** unidades de salud pública "
            + (f"del subsector **{subsec_r}**" if subsec_r != "Todos los subsectores" else "de todos los subsectores")
            + f". El subsector con mayor presencia es **{pub_r['subsector'].value_counts().idxmax() if len(pub_r) else '—'}**."
        ),
        (
            " Estructura topológica",
            f"El análisis TDA identifica **{len(h1_r)} huecos topológicos**, de los cuales "
            f"**{n_sig_r} son significativos** (persistencia ≥ {thresh_r} km): "
            f"**{n_sig_int_r} interiores** (huecos candidatos) y **{n_sig_brd_r} posibles artefactos de borde**. "
            f"El radio crítico donde coexisten más huecos es **ε ≈ {peak_eps_r:.2f} km**."
        ),
        (
            " Cobertura geográfica",
            f"A un radio de **{eps_r} km**, el **{pct_cov_r:.1f}%** de las AGEBs cuenta con "
            f"al menos una unidad pública de salud cercana. Quedan **{n_unc_r} AGEBs sin cobertura** "
            f"({100 - pct_cov_r:.1f}% del total analizado)."
        ),
        (
            " Rezago y cobertura",
            f"El grado de rezago más frecuente entre las AGEBs sin cobertura es **{top_rez_r}**. "
            f"Las AGEBs con rezago **{max_dist_rez_r}** tienen la mayor distancia media al servicio público más cercano."
        ),
        (
            " Zonas vulnerables",
            f"Se identifican **{len(vuln_r)} AGEBs** que combinan rezago social alto/muy alto "
            f"y ausencia de cobertura a ε = {eps_r} km. "
            + (f"Representan el **{len(vuln_r)/len(ageb_geo_r)*100:.1f}%** de las AGEBs analizadas." if len(ageb_geo_r) > 0 else "")
        ),
        (
            " Conectividad",
            f"La red pública se conecta completamente (β₀ = 1) a partir de aproximadamente "
            f"**ε = {f'{float(eps_range_r[b0_r == 1][0]):.2f}' if (b0_r == 1).any() else f'>{max_er:.2f}'} km**. "
            f"La persistencia H₁ máxima es **{max_pers_r:.2f} km**, "
            f"indicando el tamaño del hueco estructural más grande."
        ),
    ]

    for icon_title, texto in hallazgos:
        texto_html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', texto)
        st.markdown(
            f"<div style='background:#f9f9f9;border-left:4px solid #4e8df5;"
            f"padding:10px 14px;border-radius:4px;margin-bottom:8px'>"
            f"<b>{icon_title}</b><br>{texto_html}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 6. RECOMENDACIONES
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("6 · Recomendaciones")

    recomendaciones = []

    if n_sig_int_r >= 3:
        recomendaciones.append(
            f"**Priorizar intervención en huecos H₁ interiores persistentes.** "
            f"Se detectaron {n_sig_int_r} vacíos estructurales interiores (no artefactos de borde) "
            f"con persistencia ≥ {thresh_r} km. Se recomienda evaluar la instalación de nuevas "
            f"unidades públicas en los centros aproximados de estos huecos."
        )
    elif n_sig_r >= 3 and n_sig_int_r < 3:
        recomendaciones.append(
            f"**Verificar huecos H₁ detectados cerca del borde del área.** "
            f"De los {n_sig_r} huecos significativos, {n_sig_brd_r} son probables artefactos de borde "
            f"(generados por la ausencia de datos de estados vecinos). Solo {n_sig_int_r} "
            f"son huecos interiores confirmados."
        )
    if len(vuln_r) > 0:
        recomendaciones.append(
            f"**Atención urgente a {len(vuln_r)} AGEBs vulnerables.** "
            f"Estas zonas combinan rezago social alto/muy alto con ausencia de cobertura "
            f"pública a {eps_r} km. Son las de mayor urgencia de intervención."
        )
    if pct_cov_r < 80:
        recomendaciones.append(
            f"**Ampliar cobertura: solo {pct_cov_r:.0f}% de AGEBs cubiertas.** "
            f"Incrementar la densidad de unidades públicas para alcanzar cobertura universal "
            f"dentro del radio de {eps_r} km establecido como estándar."
        )
    if peak_eps_r > 0:
        recomendaciones.append(
            f"**Radio de planificación sugerido: {peak_eps_r:.2f} km.** "
            f"Este es el radio donde la red presenta la mayor complejidad topológica (pico β₁). "
            f"Usarlo como referencia para el diseño de áreas de servicio."
        )
    recomendaciones.append(
        f"**Profundizar el análisis a nivel AGEB.** "
        f"El siguiente paso es construir el complejo de Čech con los parámetros identificados "
        f"y cruzar los generadores H₁ con datos de densidad poblacional y NBI para priorizar."
    )

    rec_colors = ["#e8f5e9", "#fff3e0", "#fce4ec", "#e3f2fd", "#f3e5f5"]
    rec_borders = ["#2ecc71", "#f39c12", "#e74c3c", "#3498db", "#9b59b6"]
    for i, rec in enumerate(recomendaciones):
        c = rec_colors[i % len(rec_colors)]
        b = rec_borders[i % len(rec_borders)]
        rec_html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', rec)
        st.markdown(
            f"<div style='background:{c};border-left:4px solid {b};"
            f"padding:10px 14px;border-radius:4px;margin-bottom:8px'>"
            f"<b>R{i+1}.</b> {rec_html}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Exportar reporte ──────────────────────────────────────────────────────
    st.subheader("Exportar Reporte")

    reporte_md = f"""# Reporte TDA — Sector Salud Pública
**Área:** {area_label}
**Subsector:** {subsec_label}
**Radio ε:** {eps_r} km · **Umbral significancia:** {thresh_r} km
**Fecha:** {date.today().strftime("%d/%m/%Y")}
**Fuentes:** DENUE 2025 (INEGI) · CONEVAL 2020 · IDS EVALUA CDMX

---

## 1. Oferta de Salud Pública
- Unidades públicas: {len(pub_r):,}
- Unidades privadas: {len(denue_ctx[denue_ctx['sector']=='Privado']):,}
- Total sector salud: {len(denue_ctx):,}

## 2. Hallazgos Topológicos (TDA)
- Huecos H₁ detectados: {len(h1_r)}
- Huecos significativos (≥{thresh_r} km): {n_sig_r} ({n_sig_int_r} interiores · {n_sig_brd_r} posibles artefactos de borde)
- Persistencia H₁ máxima: {max_pers_r:.2f} km
- Radio crítico (pico β₁): {peak_eps_r:.2f} km (β₁ máx = {peak_b1_r})

## 3. Cobertura
- AGEBs con cobertura a ε={eps_r} km: {pct_cov_r:.1f}%
- AGEBs sin cobertura: {n_unc_r:,} ({100-pct_cov_r:.1f}%)
- Total AGEBs analizadas: {len(ageb_geo_r):,}

## 4. Vulnerabilidad Social
- AGEBs con rezago alto + sin cobertura: {len(vuln_r)}
- Rezago más frecuente en AGEBs sin cobertura: {top_rez_r}
- Rezago con mayor distancia media al servicio: {max_dist_rez_r}

## 5. Hallazgos Clave
""" + "\n".join([f"- **{t}**: {txt}" for t, txt in hallazgos]) + "\n\n## 6. Recomendaciones\n" + "\n".join([f"R{i+1}. {r}" for i, r in enumerate(recomendaciones)])

    st.download_button(
        label="⬇️ Descargar reporte (.md)",
        data=reporte_md.encode("utf-8"),
        file_name=f"reporte_tda_{mun_r.replace(' ','_').lower()}_{date.today()}.md",
        mime="text/markdown",
    )

'''

# # ═══════════════════════════════════════════════════════════════════════════════
# # TAB COMPARACIÓN — TDA vs K-MEANS + PCA
# # ═══════════════════════════════════════════════════════════════════════════════
# with tab_compare:
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.decomposition import PCA
#     from sklearn.cluster import KMeans

#     st.header("Comparación: TDA vs K-Means + PCA")
#     st.caption(
#         "K-Means agrupa AGEBs por similitud de atributos socioeconómicos. "
#         "TDA detecta vacíos estructurales en la red de cobertura. "
#         "Aquí contrastamos ambas perspectivas sobre el mismo territorio."
#     )

#     # ── Controles ─────────────────────────────────────────────────────────────
#     cmp1, cmp2, cmp3, cmp4, cmp5 = st.columns([2, 2, 1, 1, 1])
#     with cmp1:
#         mun_cmp = st.selectbox(
#             "Alcaldía",
#             ["CDMX completa"] + sorted(denue["municipio"].unique().tolist()),
#             key="mun_cmp",
#         )
#     with cmp2:
#         subsec_opts_cmp = ["Todos los subsectores"] + sorted(
#             denue[denue["sector"] == "Público"]["subsector"].dropna().unique().tolist()
#         )
#         subsec_cmp = st.selectbox("Subsector público", subsec_opts_cmp, key="subsec_cmp")
#     with cmp3:
#         k_cmp = st.slider("K (clusters)", 2, 8, 4, key="k_cmp")
#     with cmp4:
#         eps_cmp = st.slider("ε TDA (km)", 0.25, 8.0, 1.5, 0.25, key="eps_cmp")
#     with cmp5:
#         thresh_cmp = st.slider("Umbral sig. (km)", 0.1, 3.0, 0.5, 0.1, key="thresh_cmp")

#     # ── Preparar datos ─────────────────────────────────────────────────────────
#     # 1. AGEBs con variables socioeconómicas
#     ageb_cmp = coneval.merge(
#         ids[["cvegeo", "ids_norm", "grado_ids_num"]], on="cvegeo", how="left"
#     ).dropna(subset=["centroide_lat", "centroide_lon", "rezago_norm"])

#     if mun_cmp != "CDMX completa":
#         _cve_cmp = ageb_cmp["cvegeo"].astype(str).str[2:5]
#         ageb_cmp = ageb_cmp[_cve_cmp == MUN_INV.get(mun_cmp, "000")]

#     # 2. Unidades públicas filtradas
#     pub_cmp = denue[denue["sector"] == "Público"].dropna(subset=["latitud", "longitud"])
#     if mun_cmp != "CDMX completa":
#         pub_cmp = pub_cmp[pub_cmp["municipio"] == mun_cmp]
#     if subsec_cmp != "Todos los subsectores":
#         pub_cmp = pub_cmp[pub_cmp["subsector"] == subsec_cmp]

#     # 3. Distancia AGEB → unidad pública más cercana
#     ageb_cmp = ageb_cmp.copy()
#     if len(ageb_cmp) > 0 and len(pub_cmp) > 0:
#         _ageb_pts  = ageb_cmp[["centroide_lat", "centroide_lon"]].values
#         _pub_pts   = pub_cmp[["latitud", "longitud"]].values
#         _D_ap      = _dist_cross_km(_ageb_pts, _pub_pts)
#         ageb_cmp["dist_nearest"] = _D_ap.min(axis=1)
#     else:
#         ageb_cmp["dist_nearest"] = 0.0

#     # 4. Matriz de características y limpieza
#     _feat_cols = ["centroide_lat", "centroide_lon", "rezago_norm", "ids_norm", "dist_nearest"]
#     ageb_feat = ageb_cmp.dropna(subset=_feat_cols).reset_index(drop=True)

#     if len(ageb_feat) < 10:
#         st.warning("Pocos AGEBs con este filtro. Amplía la selección.")
#         st.stop()

#     X_raw = ageb_feat[_feat_cols].values
#     _scaler = StandardScaler()
#     Xs = _scaler.fit_transform(X_raw)

#     # 5. PCA
#     _n_comp = min(5, Xs.shape[1])
#     _pca = PCA(n_components=_n_comp, random_state=42)
#     Xpca = _pca.fit_transform(Xs)

#     # 6. K-Means con k seleccionado
#     _km = KMeans(n_clusters=k_cmp, random_state=42, n_init=10)
#     _km_labels = _km.fit_predict(Xs)

#     ageb_feat = ageb_feat.copy()
#     ageb_feat["km_cluster"] = [f"C{lb+1}" for lb in _km_labels]
#     ageb_feat["pc1"] = Xpca[:, 0]
#     ageb_feat["pc2"] = Xpca[:, 1]

#     # 7. Curva del codo
#     _inertias = [
#         KMeans(n_clusters=ki, random_state=42, n_init=5).fit(Xs).inertia_
#         for ki in range(2, 9)
#     ]

#     # 8. TDA sobre unidades públicas (máx 2500, cached)
#     _pts_cmp = pub_cmp[["latitud", "longitud"]].values
#     if len(_pts_cmp) > 2500:
#         _pts_cmp = pub_cmp.sample(2500, random_state=42)[["latitud", "longitud"]].values

#     _cluster_colors = px.colors.qualitative.Set2

#     # ══════════════════════════════════════════════════════════════════════════
#     # A. PCA
#     # ══════════════════════════════════════════════════════════════════════════
#     st.subheader("A · Análisis de Componentes Principales (PCA)")
#     st.caption(
#         "5 variables por AGEB: ubicación geográfica (lat/lon), rezago social normalizado, "
#         "IDS normalizado y distancia a la unidad pública más cercana."
#     )

#     colA1, colA2 = st.columns(2)

#     with colA1:
#         _ev  = _pca.explained_variance_ratio_
#         _cev = np.cumsum(_ev)
#         fig_scree = go.Figure()
#         fig_scree.add_bar(
#             x=[f"PC{i+1}" for i in range(_n_comp)],
#             y=(_ev * 100).tolist(),
#             name="Varianza (%)", marker_color="#4e8df5",
#         )
#         fig_scree.add_scatter(
#             x=[f"PC{i+1}" for i in range(_n_comp)],
#             y=(_cev * 100).tolist(),
#             name="Acumulada (%)", mode="lines+markers",
#             line=dict(color="#e74c3c", width=2), marker=dict(size=7),
#         )
#         fig_scree.update_layout(
#             title="Varianza explicada por componente",
#             yaxis_title="% varianza", height=320,
#             legend=dict(orientation="h", y=-0.3),
#             margin=dict(t=40, b=50),
#         )
#         st.plotly_chart(fig_scree, width="stretch")

#     with colA2:
#         _rezago_order = ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]
#         fig_pca = px.scatter(
#             ageb_feat, x="pc1", y="pc2",
#             color="grado_rezago_social",
#             color_discrete_map=COLORS,
#             category_orders={"grado_rezago_social": _rezago_order},
#             labels={"pc1": "PC1", "pc2": "PC2", "grado_rezago_social": "Rezago social"},
#             title="Biplot PC1 vs PC2 (color = rezago social)",
#             height=320, opacity=0.65,
#         )
#         fig_pca.update_traces(marker_size=5)
#         fig_pca.update_layout(margin=dict(t=40, b=20))
#         st.plotly_chart(fig_pca, width="stretch")

#     with st.expander("Cargas (loadings) PCA"):
#         _feat_labels = ["Latitud", "Longitud", "Rezago norm.", "IDS norm.", "Dist. a unidad (km)"]
#         _df_loads = pd.DataFrame(
#             _pca.components_.T,
#             index=_feat_labels,
#             columns=[f"PC{i+1}" for i in range(_n_comp)],
#         ).round(3)
#         st.dataframe(_df_loads, width="stretch")

#     st.divider()

#     # ══════════════════════════════════════════════════════════════════════════
#     # B. K-MEANS
#     # ══════════════════════════════════════════════════════════════════════════
#     st.subheader("B · Clustering K-Means")

#     colB1, colB2 = st.columns([1, 2])

#     with colB1:
#         fig_elbow = go.Figure(go.Scatter(
#             x=list(range(2, 9)), y=_inertias,
#             mode="lines+markers",
#             line=dict(color="#f39c12", width=2),
#             marker=dict(size=8, color="#f39c12"),
#         ))
#         fig_elbow.add_vline(
#             x=k_cmp, line_dash="dash", line_color="#e74c3c",
#             annotation_text=f"k={k_cmp}", annotation_position="top right",
#         )
#         fig_elbow.update_layout(
#             title="Codo (inercia vs k)",
#             xaxis_title="k", yaxis_title="Inercia",
#             height=260, margin=dict(t=40, b=20),
#         )
#         st.plotly_chart(fig_elbow, width="stretch")

#         _prof_cols = {"rezago_norm": "Rezago", "ids_norm": "IDS", "dist_nearest": "Dist. (km)"}
#         _df_prof = (
#             ageb_feat.groupby("km_cluster")[list(_prof_cols.keys())]
#             .mean().round(3)
#             .rename(columns=_prof_cols)
#         )
#         _df_prof.index.name = "Cluster"
#         st.markdown("**Perfil medio por cluster**")
#         st.dataframe(_df_prof, width="stretch")

#     with colB2:
#         fig_km_map = px.scatter_map(
#             ageb_feat,
#             lat="centroide_lat", lon="centroide_lon",
#             color="km_cluster",
#             color_discrete_sequence=_cluster_colors,
#             hover_data={"grado_rezago_social": True, "dist_nearest": ":.2f", "km_cluster": True},
#             labels={"km_cluster": "Cluster"},
#             title=f"Clusters K-Means (k={k_cmp}) — AGEBs CDMX",
#             zoom=10, height=420,
#         )
#         fig_km_map.update_layout(
#             map_style="carto-positron",
#             margin=dict(t=40, b=0, l=0, r=0),
#         )
#         st.plotly_chart(fig_km_map, width="stretch")

#     st.divider()

#     # ══════════════════════════════════════════════════════════════════════════
#     # C. TDA — huecos H₁ sobre unidades públicas
#     # ══════════════════════════════════════════════════════════════════════════
#     st.subheader("C · Huecos Topológicos TDA (H₁) sobre red pública")

#     with st.spinner("Calculando TDA…"):
#         if len(_pts_cmp) >= 5:
#             _pts_cmp_key = tuple(map(tuple, _pts_cmp))
#             _dgms_cmp, _D_cmp_list, _h1c_raw = _compute_tda_full(_pts_cmp_key)
#             _D_cmp  = np.array(_D_cmp_list)
#             _h1_cmp = np.array(_dgms_cmp[1]) if len(_dgms_cmp[1]) else np.empty((0, 2))
#             _max_ec = min(float(_D_cmp.max()) if _D_cmp.size else 15.0, 15.0)
#             _h1p    = _h1_cmp[:, 1] - _h1_cmp[:, 0] if len(_h1_cmp) else np.array([])
#             _sig_c  = _h1p >= thresh_cmp
#             _h1c_arr = np.array(_h1c_raw[:len(_h1_cmp)]) if len(_h1c_raw) else np.empty((0, 2))
#             _bd_c   = _border_dists(_h1c_arr, _pts_cmp) if len(_h1c_arr) else np.array([])
#             _int_c  = (_sig_c & (_bd_c >= eps_cmp)) if len(_bd_c) else _sig_c.copy()
#             _brd_c  = _sig_c & ~_int_c
#         else:
#             _h1_cmp  = np.empty((0, 2))
#             _h1c_arr = np.empty((0, 2))
#             _max_ec  = 8.0
#             _sig_c   = np.array([], dtype=bool)
#             _int_c   = np.array([], dtype=bool)
#             _brd_c   = np.array([], dtype=bool)

#     _n_h1_c   = len(_h1_cmp)
#     _n_sig_c  = int(_sig_c.sum())
#     _n_int_c  = int(_int_c.sum())

#     colC1, colC2 = st.columns([1, 2])

#     with colC1:
#         st.metric("Huecos H₁ detectados", _n_h1_c)
#         st.metric("Significativos (pers. ≥ 0.5 km)", _n_sig_c)
#         st.metric("Interiores confirmados", _n_int_c,
#                   f"{_n_sig_c - _n_int_c} posibles artefactos de borde")
#         st.metric("Unidades públicas analizadas", len(_pts_cmp))
#         if _n_h1_c > 0:
#             _fig_bc = _barcode_fig(_h1_cmp, "Barcode H₁", "#e74c3c", _max_ec)
#             _fig_bc.update_layout(height=220, margin=dict(t=30, b=20))
#             st.plotly_chart(_fig_bc, width="stretch")

#     with colC2:
#         # Mapa de AGEBs coloreadas por K-Means + huecos TDA superpuestos
#         fig_overlay = px.scatter_map(
#             ageb_feat,
#             lat="centroide_lat", lon="centroide_lon",
#             color="km_cluster",
#             color_discrete_sequence=_cluster_colors,
#             opacity=0.5,
#             hover_data={"grado_rezago_social": True, "dist_nearest": ":.2f"},
#             labels={"km_cluster": "Cluster K-Means"},
#             title=f"Clusters K-Means + huecos TDA (ε={eps_cmp} km)",
#             zoom=10, height=440,
#         )
#         if _n_int_c > 0:
#             _int_centers = _h1c_arr[_int_c]
#             fig_overlay.add_trace(go.Scattermap(
#                 lat=_int_centers[:, 0].tolist(),
#                 lon=_int_centers[:, 1].tolist(),
#                 mode="markers",
#                 marker=dict(size=18, color="#e74c3c", symbol="x"),
#                 name="Hueco TDA interior",
#                 hovertemplate=(
#                     "<b>Hueco H₁ interior</b><br>"
#                     "lat=%{lat:.4f}<br>lon=%{lon:.4f}<extra></extra>"
#                 ),
#             ))
#         if int(_brd_c.sum()) > 0:
#             _brd_centers = _h1c_arr[_brd_c]
#             fig_overlay.add_trace(go.Scattermap(
#                 lat=_brd_centers[:, 0].tolist(),
#                 lon=_brd_centers[:, 1].tolist(),
#                 mode="markers",
#                 marker=dict(size=14, color="#aaa", symbol="x"),
#                 name="Artefacto de borde",
#                 hovertemplate=(
#                     "<b>Posible artefacto de borde</b><br>"
#                     "lat=%{lat:.4f}<br>lon=%{lon:.4f}<extra></extra>"
#                 ),
#             ))
#         fig_overlay.update_layout(
#             map_style="carto-positron",
#             margin=dict(t=40, b=0, l=0, r=0),
#         )
#         st.plotly_chart(fig_overlay, width="stretch")

#     st.divider()

#     # ══════════════════════════════════════════════════════════════════════════
#     # D. CONTRASTE — ¿Qué gana TDA?
#     # ══════════════════════════════════════════════════════════════════════════
#     st.subheader("D · ¿Qué gana TDA sobre K-Means?")

#     # Marcar AGEBs dentro del radio de algún hueco interior
#     ageb_feat = ageb_feat.copy()
#     if _n_int_c > 0:
#         _int_centers = _h1c_arr[_int_c]
#         _ageb_coords = ageb_feat[["centroide_lat", "centroide_lon"]].values
#         _D_hole      = _dist_cross_km(_ageb_coords, _int_centers)
#         ageb_feat["dist_to_hole"] = _D_hole.min(axis=1)
#         ageb_feat["en_hueco_tda"] = ageb_feat["dist_to_hole"] < eps_cmp
#     else:
#         ageb_feat["dist_to_hole"] = np.nan
#         ageb_feat["en_hueco_tda"] = False

#     colD1, colD2 = st.columns(2)

#     with colD1:
#         # Scatter PCA coloreado por K-Means; ✕ = AGEB dentro de hueco TDA
#         fig_contrast = px.scatter(
#             ageb_feat, x="pc1", y="pc2",
#             color="km_cluster",
#             color_discrete_sequence=_cluster_colors,
#             symbol="en_hueco_tda",
#             symbol_map={True: "x", False: "circle"},
#             labels={
#                 "pc1": "PC1", "pc2": "PC2",
#                 "km_cluster": "Cluster", "en_hueco_tda": "En hueco TDA",
#             },
#             title="Espacio PCA — clusters K-Means (✕ = dentro de hueco TDA)",
#             height=380, opacity=0.75,
#         )
#         fig_contrast.update_traces(marker_size=6)
#         fig_contrast.update_layout(margin=dict(t=40, b=20))
#         st.plotly_chart(fig_contrast, width="stretch")

#     with colD2:
#         # Boxplot distancia a unidad pública por cluster
#         fig_box = px.box(
#             ageb_feat, x="km_cluster", y="dist_nearest",
#             color="km_cluster",
#             color_discrete_sequence=_cluster_colors,
#             points="outliers",
#             labels={"km_cluster": "Cluster", "dist_nearest": "Dist. a unidad pública (km)"},
#             title="Distancia a unidad pública por cluster K-Means",
#             height=380,
#         )
#         fig_box.update_layout(showlegend=False, margin=dict(t=40, b=20))
#         st.plotly_chart(fig_box, width="stretch")

#     # Tabla: huecos por cluster
#     if _n_int_c > 0:
#         _df_hc = ageb_feat.groupby("km_cluster")["en_hueco_tda"].agg(
#             en_hueco="sum", total="count"
#         ).reset_index()
#         _df_hc["% en hueco"] = (_df_hc["en_hueco"] / _df_hc["total"] * 100).round(1)
#         _df_hc.columns = ["Cluster", "AGEBs en hueco TDA", "Total AGEBs", "% en hueco"]
#         st.markdown("**AGEBs dentro de huecos topológicos por cluster K-Means**")
#         st.dataframe(_df_hc, width="stretch")
#         st.caption(
#             "Un hueco TDA puede cruzar varios clusters de K-Means, "
#             "demostrando que el vacío de cobertura es estructural y no depende del perfil socioeconómico."
#         )

#     # Tabla comparativa de métodos
#     st.markdown("### ¿Qué detecta cada método?")
#     _cmp_df = pd.DataFrame({
#         "Método":                        ["K-Means",                                            "PCA",                                               "TDA (H₁)"],
#         "¿Qué detecta?":                 ["Grupos de AGEBs con perfil socioeconómico similar",  "Dimensiones latentes de variación territorial",     "Vacíos estructurales en la red de cobertura"],
#         "¿Detecta huecos de cobertura?": ["No",                                                 "No",                                                "Sí"],
#         "¿Usa topología espacial?":      ["Parcial (lat/lon como variable)",                    "No directamente",                                   "Sí (intrínsecamente)"],
#         "Parámetro clave":               [f"k = {k_cmp} clusters",                              f"{_n_comp} componentes",                            f"ε = {eps_cmp} km"],
#     })
#     st.dataframe(_cmp_df, width="stretch")

#     # Insight cards
#     _insights = [
#         ("#e8f5e9", "#2ecc71", "Lo que K-Means sí puede",
#          "Identificar territorios con perfiles similares de rezago, IDS y acceso. "
#          "Útil para priorizar zonas de alta vulnerabilidad socioeconómica y asignar recursos de manera eficiente."),
#         ("#fce4ec", "#e74c3c", "Lo que K-Means no puede",
#          "Detectar huecos topológicos en la red de servicios. Un cluster de baja vulnerabilidad "
#          "puede contener un posible vacío estructural de cobertura si la red de salud no conecta esa zona con ninguna unidad pública."),
#         ("#e3f2fd", "#3498db", f"El valor agregado de TDA ({_n_int_c} hueco{'s' if _n_int_c != 1 else ''} interior{'es' if _n_int_c != 1 else ''} confirmado{'s' if _n_int_c != 1 else ''})",
#          f"TDA identifica {_n_int_c} hueco{'s' if _n_int_c != 1 else ''} interior{'es' if _n_int_c != 1 else ''} confirmado{'s' if _n_int_c != 1 else ''} "
#          f"que representan zonas sin acceso estructural a salud pública independientemente del perfil "
#          f"socioeconómico del área. Estos vacíos trascienden los límites de los clusters de K-Means y "
#          f"sólo son visibles con análisis topológico."),
#     ]
#     for _bg, _bd, _title, _text in _insights:
#         st.markdown(
#             f"<div style='background:{_bg};border-left:4px solid {_bd};"
#             f"padding:10px 14px;border-radius:4px;margin-bottom:8px'>"
#             f"<b>{_title}</b><br>{_text}</div>",
#             unsafe_allow_html=True,
#         )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB CONCLUSIONES Y ESTRATEGIA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_concl:

    st.header("Conclusiones y Estrategia de Intervención")
    st.caption(
        "Síntesis ejecutiva del análisis TDA aplicado a la red pública de salud en CDMX: "
        "hallazgos principales, estrategia de intervención y justificación metodológica."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 0. CÁLCULO DE MÉTRICAS GLOBALES
    # ══════════════════════════════════════════════════════════════════════════

    _pub_exec = denue[denue["sector"] == "Público"].dropna(subset=["latitud", "longitud"])
    _pts_exec = _pub_exec[["latitud", "longitud"]].values

    if len(_pts_exec) > 2100:
        _rng_exec = np.random.default_rng(42)
        _pts_exec = _pts_exec[_rng_exec.choice(len(_pts_exec), 2100, replace=False)]

    with st.spinner("Calculando métricas globales…"):
        _dgms_exec, _, _h1c_exec = _compute_tda_full(tuple(map(tuple, _pts_exec)))

    _h1_exec = np.array(_dgms_exec[1]) if len(_dgms_exec[1]) else np.empty((0, 2))
    _h1p_exec = (_h1_exec[:, 1] - _h1_exec[:, 0]) if len(_h1_exec) else np.array([])
    _h1p_fin_exec = _h1p_exec[~np.isinf(_h1p_exec)] if len(_h1p_exec) else np.array([])

    if len(_h1p_fin_exec) >= 2:
        _mu_exec = float(_h1p_fin_exec.mean())
        _sig_exec = float(_h1p_fin_exec.std())
        _thr_exec = _mu_exec + 0.5 * _sig_exec
        _n_sig_exec = int((_h1p_exec >= _thr_exec).sum())
        _max_p_exec = float(_h1p_fin_exec.max())
    else:
        _thr_exec = 0.5
        _n_sig_exec = 0
        _max_p_exec = 0.0

    # ── Índice topológico por alcaldía con 4 escenarios de pesos ────────────

    _ageb_exec = coneval.merge(
        ids[["cvegeo", "bajo_desarrollo_norm", "prop_nbi_norm",
             "grado_ids", "grado_ids_num", "poblacion_ids"]],
        on="cvegeo", how="left",
    ).copy()

    _dens_raw_e = _ageb_exec["poblacion_ids"] / _ageb_exec["area_km2"].replace(0, np.nan)
    _ageb_exec["densidad_norm"] = _dens_raw_e.fillna(0).rank(pct=True)
    _ageb_exec = _ageb_exec.dropna(subset=["centroide_lat", "centroide_lon"]).reset_index(drop=True)
    _ageb_exec["mun_code"] = _ageb_exec["cvegeo"].astype(str).str[2:5]
    _ageb_exec["municipio"] = _ageb_exec["mun_code"].map(MUN_MAP)
    _ageb_coords_e = np.deg2rad(_ageb_exec[["centroide_lat", "centroide_lon"]].values)

    _h1p_max_e = float(_h1p_fin_exec.max()) if len(_h1p_fin_exec) else 1.0

    _escenarios_concl = {
        "Base":        {"P_i": 0.40, "B_i": 0.25, "D_i": 0.15, "R_i": 0.10, "N_i": 0.10},
        "Topológico":  {"P_i": 0.55, "B_i": 0.15, "D_i": 0.10, "R_i": 0.10, "N_i": 0.10},
        "Social":      {"P_i": 0.30, "B_i": 0.30, "D_i": 0.15, "R_i": 0.10, "N_i": 0.15},
        "Densidad":    {"P_i": 0.35, "B_i": 0.20, "D_i": 0.25, "R_i": 0.10, "N_i": 0.10},
    }

    _hole_rows = []

    if len(_h1c_exec) > 0 and len(_h1p_exec) > 0:
        _sig_mask = (_h1p_exec >= _thr_exec) & ~np.isinf(_h1p_exec)

        for _i, (_clat, _clon) in enumerate(_h1c_exec):
            if _i >= len(_sig_mask) or not _sig_mask[_i]:
                continue

            _hpt = np.deg2rad(np.array([[_clat, _clon]]))
            _dh = np.arcsin(np.sqrt(
                np.sin((_ageb_coords_e[:, 0] - _hpt[0, 0]) / 2) ** 2
                + np.cos(_hpt[0, 0]) * np.cos(_ageb_coords_e[:, 0])
                * np.sin((_ageb_coords_e[:, 1] - _hpt[0, 1]) / 2) ** 2
            )) * 2 * 6371

            _ni = int(_dh.argmin())
            _row_a = _ageb_exec.iloc[_ni]

            P_i = float(_h1p_exec[_i]) / _h1p_max_e
            D_i = float(_row_a.get("densidad_norm", 0) or 0)
            R_i = float(_row_a.get("rezago_norm", 0) or 0)
            B_i = float(_row_a.get("bajo_desarrollo_norm", 0) or 0)
            N_i = float(_row_a.get("prop_nbi_norm", 0) or 0)

            _entry = {
                "municipio": str(_row_a["municipio"]) if pd.notna(_row_a["municipio"]) else "Desconocido",
                "persistencia": float(_h1p_exec[_i]),
                "P_i": P_i,
                "B_i": B_i,
                "D_i": D_i,
                "R_i": R_i,
                "N_i": N_i,
            }

            for _esc, _w in _escenarios_concl.items():
                _entry[f"I_{_esc}"] = sum(_entry[v] * ww for v, ww in _w.items())

            _hole_rows.append(_entry)

    if _hole_rows:
        _df_h = pd.DataFrame(_hole_rows)

        _agg_cols = {f"I_{e}": "sum" for e in _escenarios_concl}
        _agg_cols["persistencia"] = ["count", "sum", "mean"]

        _df_rank = _df_h.groupby("municipio").agg(_agg_cols)
        _df_rank.columns = (
            [f"I_{e}" for e in _escenarios_concl]
            + ["n_huecos", "persist_total", "persist_media"]
        )
        _df_rank = _df_rank.reset_index()

        for _esc in _escenarios_concl:
            _col = f"I_{_esc}"
            _mx = _df_rank[_col].max()
            _df_rank[f"N_{_esc}"] = (_df_rank[_col] / _mx * 100).round(1) if _mx > 0 else 0.0

        for _esc in _escenarios_concl:
            _df_rank[f"rk_{_esc}"] = _df_rank[f"N_{_esc}"].rank(
                ascending=False, method="min"
            ).astype(int)

        _df_rank["top3_count"] = sum(
            (_df_rank[f"rk_{e}"] <= 3).astype(int) for e in _escenarios_concl
        )

        _df_rank = _df_rank.sort_values("N_Base", ascending=False).reset_index(drop=True)
        _worst_mun = str(_df_rank.loc[0, "municipio"])
    else:
        _df_rank = pd.DataFrame()
        _worst_mun = "Iztapalapa"

    # ══════════════════════════════════════════════════════════════════════════
    # 1. RESUMEN EJECUTIVO
    # ══════════════════════════════════════════════════════════════════════════

    st.markdown(
        "<div style='background:linear-gradient(135deg,#2d3436,#0984e3);"
        "color:#fff;padding:24px 30px;border-radius:10px;margin-bottom:20px'>"
        "<b style='font-size:1.2em;letter-spacing:.5px'>RESUMEN EJECUTIVO</b>"
        "<p style='margin:8px 0 0;font-size:0.95em;opacity:.9'>"
        "Este tablero convierte datos públicos de salud en una herramienta para detectar brechas "
        "de cobertura, medir su importancia y priorizar posibles intervenciones territoriales."
        "</p></div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unidades públicas analizadas", f"{len(_pts_exec):,}")
    c2.metric("Huecos H₁ persistentes", _n_sig_exec)
    c3.metric("Persistencia máxima", f"{_max_p_exec:.2f} km")
    c4.metric("Alcaldía más vulnerable", _worst_mun)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 2. HALLAZGOS PRINCIPALES
    # ══════════════════════════════════════════════════════════════════════════

    st.subheader("1 · Hallazgos principales")

    _hallazgos = [
        ("#d63031", "1. Existen brechas estructurales de cobertura"),
        ("#0984e3", "2. La cobertura pública no está distribuida de forma uniforme"),
        ("#6c5ce7", f"3. {_worst_mun}, Xochimilco y Tlalpan concentran la mayor prioridad de intervención"),
    ]

    for _color, _title in _hallazgos:
        st.markdown(
            f"""
            <div style="
                border-left:5px solid {_color};
                background:#f8f9fa;
                padding:18px;
                border-radius:8px;
                margin-bottom:12px;
            ">
                <span style="
                    color:{_color};
                    font-weight:700;
                    font-size:1.08rem;
                ">
                    {_title}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 3. RANKING POR ALCALDÍA
    # ══════════════════════════════════════════════════════════════════════════

    st.subheader("2 · Ranking de alcaldías por vulnerabilidad topológica")
    st.caption(
        "El ranking agrupa los huecos por alcaldía y evalúa su prioridad bajo cuatro escenarios: "
        "Base, Topológico, Social y Densidad."
    )

    if not _df_rank.empty:
        _esc_colors = {
            "Base": "#0984e3",
            "Topológico": "#6c5ce7",
            "Social": "#00b894",
            "Densidad": "#e17055",
        }

        _tbl_head = (
            "<tr style='background:#2d3436;color:#fff'>"
            "<th style='padding:8px 10px'>#</th>"
            "<th style='padding:8px 10px;text-align:left'>Alcaldía</th>"
            "<th style='padding:8px 10px'>H₁</th>"
            "<th style='padding:8px 10px'>P. media</th>"
            + "".join(
                f"<th style='padding:8px 10px;color:{_esc_colors[e]}'>{e}</th>"
                for e in _escenarios_concl
            )
            + "<th style='padding:8px 10px'>Top-3</th>"
            "</tr>"
        )

        _tbl_rows = ""

        for _ri, _row in _df_rank.iterrows():
            _rank_n = _ri + 1
            _bg = "#fff5f5" if _rank_n == 1 else ("#f8f9fa" if _ri % 2 == 0 else "#fff")
            _fw = "700" if _rank_n == 1 else "400"
            _medal = "#1" if _rank_n == 1 else ("#2" if _rank_n == 2 else ("#3" if _rank_n == 3 else str(_rank_n)))

            _cells = (
                f"<td style='padding:7px 10px;text-align:center'>{_medal}</td>"
                f"<td style='padding:7px 10px;font-weight:{_fw}'>{_row['municipio']}</td>"
                f"<td style='padding:7px 10px;text-align:center'>{int(_row['n_huecos'])}</td>"
                f"<td style='padding:7px 10px;text-align:center'>{_row['persist_media']:.2f} km</td>"
            )

            for _esc in _escenarios_concl:
                _v = _row[f"N_{_esc}"]
                _rk = int(_row[f"rk_{_esc}"])
                _c = _esc_colors[_esc]
                _bw = max(3, int(_v * 0.6))

                _cells += (
                    f"<td style='padding:7px 10px'>"
                    f"<div style='display:flex;align-items:center;gap:5px'>"
                    f"<div style='width:{_bw}px;background:{_c};height:12px;border-radius:2px'></div>"
                    f"<span style='color:{_c};font-size:0.85em'>{_v} <sup>#{_rk}</sup></span>"
                    f"</div></td>"
                )

            _t3 = int(_row["top3_count"])
            _t3c = "#d63031" if _t3 == 4 else ("#e17055" if _t3 >= 3 else ("#fdcb6e" if _t3 >= 2 else "#b2bec3"))
            _cells += f"<td style='padding:7px 10px;text-align:center;color:{_t3c};font-weight:700'>{_t3}/4</td>"

            _tbl_rows += f"<tr style='background:{_bg}'>{_cells}</tr>"

        st.markdown(
            f"<table style='width:100%;border-collapse:collapse;font-size:0.87em'>"
            f"<thead>{_tbl_head}</thead><tbody>{_tbl_rows}</tbody></table>",
            unsafe_allow_html=True,
        )

        st.caption(
            "Los valores de cada escenario están normalizados a 100. El superíndice indica la posición "
            "en el ranking de ese escenario. Top-3 muestra en cuántos escenarios la alcaldía aparece "
            "entre las tres más prioritarias."
        )
    else:
        st.info("No se encontraron huecos significativos para construir el ranking.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 4. ESTRATEGIA DE INTERVENCIÓN
    # ══════════════════════════════════════════════════════════════════════════

    st.subheader("3 · Estrategia de intervención")

    st.markdown(
        """
La propuesta no es simplemente construir más clínicas, sino usar el tablero como una **ruta de decisión**:
identificar dónde están las brechas, medir cuáles son más importantes y decidir dónde conviene intervenir primero.
        """
    )

    e1, e2, e3 = st.columns(3)

    with e1:
        st.markdown(
            "<div style='background:#f8f9fa;border-left:5px solid #00b894;"
            "padding:15px;border-radius:6px;height:170px'>"
            "<b style='color:#00b894'>¿A quién va dirigida?</b><br><br>"
            "A gobierno, alcaldías e instituciones de salud que necesitan decidir dónde invertir recursos limitados."
            "</div>",
            unsafe_allow_html=True,
        )

    with e2:
        st.markdown(
            "<div style='background:#f8f9fa;border-left:5px solid #0984e3;"
            "padding:15px;border-radius:6px;height:170px'>"
            "<b style='color:#0984e3'>¿Qué ofrece?</b><br><br>"
            "Un ranking de zonas prioritarias basado en huecos persistentes, desarrollo social, "
            "densidad, rezago y necesidades básicas."
            "</div>",
            unsafe_allow_html=True,
        )

    with e3:
        st.markdown(
            "<div style='background:#f8f9fa;border-left:5px solid #6c5ce7;"
            "padding:15px;border-radius:6px;height:170px'>"
            "<b style='color:#6c5ce7'>¿Por qué es útil?</b><br><br>"
            "Porque convierte un análisis complejo en una herramienta clara, actualizable y defendible "
            "para tomar decisiones públicas."
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
**Ruta propuesta**

1. Detectar huecos H₁ persistentes en la red pública de salud.  
2. Cruzarlos con IDS, rezago, NBI y densidad poblacional.  
3. Priorizar las zonas con mayor índice I(H).  
4. Validar en campo qué intervención conviene: nueva unidad, ampliación de servicios, brigadas móviles o mejor conexión territorial.  
5. Actualizar el análisis con nuevas bases públicas para dar seguimiento.
        """
    )

    st.success(
        "La propuesta de valor es convertir el TDA en una herramienta práctica para priorizar inversión en salud pública "
        "con evidencia territorial y social."
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 5. ¿POR QUÉ TDA?
    # ══════════════════════════════════════════════════════════════════════════

    st.subheader("4 · ¿Por qué TDA es el método indicado?")

    st.markdown(
        """
TDA es el método indicado porque no solo mide distancia entre una zona y una unidad de salud; 
analiza la **forma completa de la red**. Esto permite detectar huecos de cobertura, medir si son estables 
al cambiar el radio y después cruzarlos con variables sociales para priorizar intervención.
        """
    )

    ctda1, ctda2, ctda3 = st.columns(3)

    with ctda1:
        st.markdown(
            "<div style='background:#f8f9fa;border-left:5px solid #d63031;"
            "padding:14px;border-radius:6px;height:135px'>"
            "<b style='color:#d63031'>Detecta huecos</b><br><br>"
            "Identifica vacíos en la red que un buffer tradicional puede pasar por alto."
            "</div>",
            unsafe_allow_html=True,
        )

    with ctda2:
        st.markdown(
            "<div style='background:#f8f9fa;border-left:5px solid #0984e3;"
            "padding:14px;border-radius:6px;height:135px'>"
            "<b style='color:#0984e3'>Mide estabilidad</b><br><br>"
            "La persistencia permite distinguir huecos relevantes de ruido del modelo."
            "</div>",
            unsafe_allow_html=True,
        )

    with ctda3:
        st.markdown(
            "<div style='background:#f8f9fa;border-left:5px solid #00b894;"
            "padding:14px;border-radius:6px;height:135px'>"
            "<b style='color:#00b894'>Prioriza con contexto</b><br><br>"
            "Combina evidencia topológica con rezago, IDS, densidad y NBI."
            "</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Ver tabla comparativa de métodos", expanded=False):

        _comp_headers = [
            "Capacidad",
            "Buffer / isocrona",
            "K-Means",
            "Regresión espacial",
            "TDA",
        ]

        _comp_rows = [
            ("Detecta huecos en la red de servicios", "No", "No", "Parcial", "Sí"),
            ("Evalúa estabilidad al cambiar el radio", "No", "No", "Varía", "Sí, por persistencia"),
            ("No depende de clusters circulares", "Sí", "No", "Parcial", "Sí"),
            ("Mide la gravedad del hueco", "No", "No", "Parcial", "Sí, por vida del feature"),
            ("Detecta huecos que cruzan varios clusters", "No", "No", "No", "Sí"),
            ("Integra variables sociales", "Parcial", "Sí", "Sí", "Sí, con índice I(H)"),
            ("Sirve para priorizar intervención", "Parcial", "Parcial", "Parcial", "Sí"),
        ]

        _s_yes = "background:#d4edda;color:#155724;font-weight:600;border-radius:4px;padding:2px 7px"
        _s_no = "background:#f8d7da;color:#721c24;border-radius:4px;padding:2px 7px"
        _s_part = "background:#fff3cd;color:#856404;border-radius:4px;padding:2px 7px"

        def _badge(v):
            if v.startswith("Sí"):
                return f"<span style='{_s_yes}'>{v}</span>"
            elif v == "No":
                return f"<span style='{_s_no}'>{v}</span>"
            else:
                return f"<span style='{_s_part}'>{v}</span>"

        _th = "".join(
            f"<th style='background:#2d3436;color:#fff;padding:8px 10px;text-align:left'>{h}</th>"
            for h in _comp_headers
        )

        _tb = ""

        for _i, (_cap, *_vals) in enumerate(_comp_rows):
            _bg = "#f8f9fa" if _i % 2 == 0 else "#ffffff"
            _tb += f"<tr style='background:{_bg}'><td style='padding:7px 10px'>{_cap}</td>"
            _tb += "".join(
                f"<td style='padding:7px 10px;text-align:center'>{_badge(v)}</td>"
                for v in _vals
            )
            _tb += "</tr>"

        st.markdown(
            f"<table style='width:100%;border-collapse:collapse;font-size:0.88em'>"
            f"<thead><tr>{_th}</tr></thead><tbody>{_tb}</tbody></table>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='background:#0984e3;color:#fff;padding:14px 18px;"
            "border-radius:8px;margin-top:12px'>"
            "<b>Lectura clave:</b> Buffer mide cercanía, K-Means agrupa zonas parecidas "
            "y la regresión explica relaciones entre variables. TDA es el método indicado porque "
            "detecta huecos en la forma de la red, mide su estabilidad y permite priorizarlos con datos sociales."
            "</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 6. CIERRE EJECUTIVO
    # ══════════════════════════════════════════════════════════════════════════

    st.subheader("5 · Cierre ejecutivo")

    st.markdown(
        f"<div style='background:#2d3436;color:#dfe6e9;padding:20px 24px;"
        f"border-radius:8px;text-align:center'>"
        f"<b>Síntesis final</b><br>"
        f"<span style='font-size:0.92em'>El tablero demuestra que TDA permite pasar de un mapa descriptivo "
        f"de unidades de salud a una estrategia de intervención. Su valor está en detectar brechas estructurales, "
        f"medir su persistencia y priorizarlas con variables sociales para orientar decisiones públicas con evidencia.</span>"
        f"<br><br><small>Fuentes: DENUE 2025 (INEGI) · CONEVAL 2020 · IDS EVALUA CDMX · Censo 2020 (INEGI)</small>"
        f"</div>",
        unsafe_allow_html=True,
    )


#deffer
