from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
from unidecode import unidecode


# =========================
# 1. RUTAS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "coneval"

OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_GEOJSON = OUT_DIR / "coneval_ageb_cdmx_limpio.geojson"
OUT_GPKG = OUT_DIR / "coneval_ageb_cdmx_limpio.gpkg"
OUT_CSV = OUT_DIR / "coneval_ageb_cdmx_limpio.csv"

# =========================
# 2. FUNCIONES
# =========================

def limpiar_columna(col):
    col = str(col)
    col = unidecode(col)
    col = col.strip().lower()
    col = col.replace(" ", "_")
    col = col.replace("-", "_")
    col = col.replace(".", "")
    col = col.replace("/", "_")
    while "__" in col:
        col = col.replace("__", "_")
    return col


def minmax(s):
    s = pd.to_numeric(s, errors="coerce")
    minimo = s.min()
    maximo = s.max()

    if pd.isna(minimo) or pd.isna(maximo) or maximo == minimo:
        return pd.Series(0, index=s.index)

    return (s - minimo) / (maximo - minimo)


def grado_a_numero(x):
    if pd.isna(x):
        return np.nan

    x = unidecode(str(x)).strip().lower()

    mapa = {
        "muy bajo": 1,
        "bajo": 2,
        "medio": 3,
        "alto": 4,
        "muy alto": 5,
    }

    return mapa.get(x, np.nan)


def encontrar_columna(columnas, candidatos):
    columnas = list(columnas)

    for c in candidatos:
        if c in columnas:
            return c

    for col in columnas:
        for c in candidatos:
            if c in col:
                return col

    return None


def buscar_o_extraer_shapefile(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Buscar .shp ya descomprimido, incluso dentro de subcarpetas
    shp_files = sorted(raw_dir.rglob("*.shp"))

    if len(shp_files) > 0:
        shp_path = shp_files[0]
        print(f"Leyendo shapefile encontrado: {shp_path}")
        return shp_path

    # Si no hay .shp, buscar ZIP
    zip_files = sorted(raw_dir.rglob("*.zip"))

    if len(zip_files) == 0:
        raise FileNotFoundError(
            f"No encontré ningún .shp ni .zip en {raw_dir}.\n"
            "Mete ahí el ZIP completo de CONEVAL o descomprímelo."
        )

    # Extraer ZIP
    for zip_path in zip_files:
        print(f"Extrayendo ZIP: {zip_path}")
        extract_dir = raw_dir / zip_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

    # Volver a buscar .shp
    shp_files = list(raw_dir.rglob("*.shp"))

    if len(shp_files) == 0:
        raise FileNotFoundError(
            f"Extraí el ZIP, pero no encontré ningún .shp dentro de {raw_dir}."
        )

    shp_path = shp_files[0]
    print(f"Leyendo shapefile después de extraer ZIP: {shp_path}")
    return shp_path


# =========================
# 3. CARGAR SHAPEFILE
# =========================

SHP_PATH = buscar_o_extraer_shapefile(RAW_DIR)

gdf = gpd.read_file(SHP_PATH)

print("\nShape original:", gdf.shape)
print("CRS original:", gdf.crs)
print("\nColumnas originales:")
print(gdf.columns.tolist())


# =========================
# 4. LIMPIAR COLUMNAS
# =========================

gdf.columns = [limpiar_columna(c) for c in gdf.columns]

print("\nColumnas limpias:")
print(gdf.columns.tolist())


# =========================
# 5. DETECTAR COLUMNAS CLAVE
# =========================

col_cvegeo = encontrar_columna(
    gdf.columns,
    ["cvegeo", "cve_geo", "clave_geoestadistica", "clavegeo", "cve_ageb"]
)

col_clave_entidad = encontrar_columna(
    gdf.columns,
    ["cve_ent", "cve_entidad", "clv_ntd"]
)

col_entidad = encontrar_columna(
    gdf.columns,
    ["entidad", "entdd_f", "nom_ent"]
)

col_localidad = encontrar_columna(
    gdf.columns,
    ["clv_lcl", "cve_loc", "cve_localidad", "localidad"]
)

col_ageb = encontrar_columna(
    gdf.columns,
    ["ageb"]
)

col_grado = encontrar_columna(
    gdf.columns,
    ["grado_rezago_social", "grado_de_rezago_social", "g_rezago", "grs", "grado", "gm_2020"]
)

col_indice = encontrar_columna(
    gdf.columns,
    ["indice_rezago_social", "indice_de_rezago_social", "irs", "i_rezago", "ind_rez_soc", "ir_2020"]
)

print("\nColumnas detectadas:")
print("CVEGEO:", col_cvegeo)
print("Clave entidad:", col_clave_entidad)
print("Entidad:", col_entidad)
print("Localidad:", col_localidad)
print("AGEB:", col_ageb)
print("Grado rezago:", col_grado)
print("Índice rezago:", col_indice)


# =========================
# 6. FILTRAR CDMX
# =========================

if col_cvegeo is not None:
    gdf[col_cvegeo] = gdf[col_cvegeo].astype(str).str.strip()
    gdf[col_cvegeo] = gdf[col_cvegeo].str.replace(".0", "", regex=False)
    gdf[col_cvegeo] = gdf[col_cvegeo].str.zfill(13)

    gdf = gdf.rename(columns={col_cvegeo: "cvegeo"})
    gdf_cdmx = gdf[gdf["cvegeo"].str.startswith("09")].copy()

elif col_clave_entidad is not None:
    gdf[col_clave_entidad] = gdf[col_clave_entidad].astype(str).str.strip()
    gdf[col_clave_entidad] = gdf[col_clave_entidad].str.replace(".0", "", regex=False)
    gdf[col_clave_entidad] = gdf[col_clave_entidad].str.zfill(2)

    gdf_cdmx = gdf[gdf[col_clave_entidad] == "09"].copy()

    if col_localidad is not None and col_ageb is not None:
        localidad = (
            gdf_cdmx[col_localidad]
            .astype(str)
            .str.strip()
            .str.replace(".0", "", regex=False)
            .str.zfill(9)
        )
        ageb = (
            gdf_cdmx[col_ageb]
            .astype(str)
            .str.strip()
            .str.replace(".0", "", regex=False)
            .str.zfill(4)
        )
        gdf_cdmx["cvegeo"] = localidad + ageb
    elif "cvegeo" not in gdf_cdmx.columns:
        gdf_cdmx["cvegeo"] = np.nan

elif col_entidad is not None:
    entidad_normalizada = (
        gdf[col_entidad]
        .astype(str)
        .map(lambda x: unidecode(x).strip().lower())
    )

    gdf_cdmx = gdf[
        entidad_normalizada.isin(["ciudad de mexico", "cdmx", "distrito federal"])
    ].copy()

    if "cvegeo" not in gdf_cdmx.columns:
        gdf_cdmx["cvegeo"] = np.nan

else:
    raise ValueError("No encontré CVEGEO ni entidad. Revisa las columnas impresas arriba.")

print("\nShape CDMX:", gdf_cdmx.shape)

if gdf_cdmx.shape[0] == 0:
    raise ValueError(
        "Después de filtrar CDMX quedaron 0 filas. "
        "Revisa el formato de la columna geográfica."
    )


# =========================
# 7. CREAR VARIABLES DE REZAGO
# =========================

if col_grado is not None and col_grado in gdf_cdmx.columns:
    gdf_cdmx = gdf_cdmx.rename(columns={col_grado: "grado_rezago_social"})
else:
    gdf_cdmx["grado_rezago_social"] = np.nan

if col_indice is not None and col_indice in gdf_cdmx.columns:
    gdf_cdmx = gdf_cdmx.rename(columns={col_indice: "indice_rezago_social"})
else:
    gdf_cdmx["indice_rezago_social"] = np.nan

gdf_cdmx["grado_rezago_num"] = gdf_cdmx["grado_rezago_social"].apply(grado_a_numero)

gdf_cdmx["indice_rezago_social"] = pd.to_numeric(
    gdf_cdmx["indice_rezago_social"],
    errors="coerce"
)

if gdf_cdmx["indice_rezago_social"].notna().sum() > 0:
    gdf_cdmx["rezago_norm"] = minmax(gdf_cdmx["indice_rezago_social"])
else:
    gdf_cdmx["rezago_norm"] = minmax(gdf_cdmx["grado_rezago_num"])


# =========================
# 8. CRS, ÁREAS Y CENTROIDES
# =========================

if gdf_cdmx.crs is None:
    print("\nAdvertencia: el shapefile no trae CRS. Asumimos EPSG:4326.")
    gdf_cdmx = gdf_cdmx.set_crs(epsg=4326)

gdf_web = gdf_cdmx.to_crs(epsg=4326)

try:
    gdf_metric = gdf_cdmx.to_crs(epsg=6372)
except Exception:
    gdf_metric = gdf_cdmx.to_crs(epsg=32614)

gdf_web["area_km2"] = gdf_metric.geometry.area / 1_000_000

centroides_web = gdf_metric.geometry.centroid.to_crs(epsg=4326)
gdf_web["centroide_lon"] = centroides_web.x
gdf_web["centroide_lat"] = centroides_web.y


# =========================
# 9. COLUMNAS FINALES
# =========================

columnas_utiles = [
    "cvegeo",
    "grado_rezago_social",
    "grado_rezago_num",
    "indice_rezago_social",
    "rezago_norm",
    "area_km2",
    "centroide_lon",
    "centroide_lat",
    "geometry",
]

columnas_existentes = [c for c in columnas_utiles if c in gdf_web.columns]

gdf_final = gdf_web[columnas_existentes].copy()


# =========================
# 10. VALIDACIONES
# =========================

print("\nColumnas finales:")
print(gdf_final.columns.tolist())

print("\nPrimeras filas:")
print(gdf_final.head())

print("\nDistribución grado rezago:")
print(gdf_final["grado_rezago_social"].value_counts(dropna=False))

print("\nResumen rezago_norm:")
print(gdf_final["rezago_norm"].describe())


# =========================
# 11. GUARDAR
# =========================

gdf_final.to_file(OUT_GEOJSON, driver="GeoJSON")
gdf_final.to_file(OUT_GPKG, driver="GPKG")

df_sin_geom = pd.DataFrame(gdf_final.drop(columns="geometry"))
df_sin_geom.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("\nArchivos guardados:")
print(OUT_GEOJSON)
print(OUT_GPKG)
print(OUT_CSV)
