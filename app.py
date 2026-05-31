import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TDA Salud CDMX",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paleta / estilo ────────────────────────────────────────────────────────────
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

# IDS invertido: Muy alto = mejor situación = verde, Muy bajo = peor = rojo
COLORS_IDS = {
    "Muy bajo":  "#c0392b",
    "Bajo":      "#f0854f",
    "Medio":     "#f9e07a",
    "Alto":      "#a8e6a3",
    "Muy alto":  "#2ecc71",
}

# ── Carga de datos ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    denue    = pd.read_csv("data/processed/denue_salud_cdmx_clean.csv")
    coneval  = pd.read_csv("data/processed/coneval_ageb_cdmx_limpio.csv")
    ids      = pd.read_csv("data/processed/ids_ageb_cdmx_limpio_densidad.csv")
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
tab_datos, = st.tabs(["📊 Datos"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB DATOS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_datos:

    st.header("Visualización Descriptiva Inicial")
    st.caption(
        "Las cuatro bases preprocesadas que alimentan el análisis TDA del sector salud en CDMX."
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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig2, width="stretch")

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
        st.plotly_chart(fig_mp, width="stretch")

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
        st.plotly_chart(fig_mv, width="stretch")

    col_c, col_d = st.columns(2)

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
        st.plotly_chart(fig3, width="stretch")

    # 1d. Tamaño (personal ocupado)
    with col_d:
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
        st.plotly_chart(fig4, width="stretch")

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
        st.plotly_chart(fig_tp, width="stretch")

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
        st.plotly_chart(fig_tv, width="stretch")

    # 1e. Desglose público vs privado
    st.markdown("**Desglose por sector: Público vs Privado**")
    col_e1, col_e2 = st.columns(2)

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

    with col_e2:
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
    st.plotly_chart(fig_map1, width="stretch")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 2. CONEVAL – Rezago Social
    # ══════════════════════════════════════════════════════════════════════════
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
        st.plotly_chart(fig5, width="stretch")

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
        st.plotly_chart(fig6, width="stretch")

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
    st.plotly_chart(fig_map2, width="stretch")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 3. IDS – Índice de Desarrollo Social
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("3 · IDS – Índice de Desarrollo Social por AGEB")

    col_g, col_h = st.columns(2)

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
        st.plotly_chart(fig7, width="stretch")

    with col_h:
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
        st.plotly_chart(fig8, width="stretch")

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
    st.plotly_chart(fig9, width="stretch")

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
    st.plotly_chart(fig_map_ids, width="stretch")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 4. POBLACIÓN POR MUNICIPIO
    # ══════════════════════════════════════════════════════════════════════════
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
        st.plotly_chart(fig10, width="stretch")

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
        st.plotly_chart(fig11, width="stretch")

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
        st.plotly_chart(fig12, width="stretch")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 5. SECCIÓN INTEGRADA – Contexto del análisis TDA
    # ══════════════════════════════════════════════════════════════════════════
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
        st.plotly_chart(fig13, width="stretch")

    # 5b. Rezago medio por municipio (proxy: AGEBs × rezago_norm)
    with col_l:
        # Estimar municipio a partir de cvegeo (dígitos 5-7 = cve_mun en CDMX)
        coneval_copy = coneval.copy()
        coneval_copy["cve_mun_str"] = coneval_copy["cvegeo"].astype(str).str[2:5]
        mun_map = {
            "002": "Azcapotzalco", "003": "Coyoacán", "004": "Cuajimalpa de Morelos",
            "005": "Gustavo A. Madero", "006": "Iztacalco", "007": "Iztapalapa",
            "008": "La Magdalena Contreras", "009": "Milpa Alta", "010": "Álvaro Obregón",
            "011": "Tláhuac", "012": "Tlalpan", "013": "Xochimilco",
            "014": "Benito Juárez", "015": "Cuauhtémoc", "016": "Miguel Hidalgo",
            "017": "Venustiano Carranza",
        }
        coneval_copy["municipio"] = coneval_copy["cve_mun_str"].map(mun_map)
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
        st.plotly_chart(fig14, width="stretch")

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
        st.plotly_chart(fig15, width="stretch")
        st.caption(
            "Este gráfico resume la tensión central del proyecto: "
            "¿los municipios con mayor rezago tienen suficiente oferta pública de salud? "
            "Los huecos identificados con TDA (Čech/Vietoris-Rips) permitirán cuantificar esta brecha espacialmente."
        )
