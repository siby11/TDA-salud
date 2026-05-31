from pathlib import Path
import numpy as np
import pandas as pd
from unidecode import unidecode


# =========================
# 1. RUTAS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ids"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "ids_ageb_cdmx_limpio.csv"


# =========================
# 2. FUNCIONES
# =========================

def minmax(s):
    s = pd.to_numeric(s, errors="coerce")
    minimo = s.min()
    maximo = s.max()

    if pd.isna(minimo) or pd.isna(maximo) or maximo == minimo:
        return pd.Series(0, index=s.index)

    return (s - minimo) / (maximo - minimo)


def limpiar_texto(x):
    if pd.isna(x):
        return np.nan
    return unidecode(str(x)).strip().lower()


def buscar_csv(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    csvs = sorted(raw_dir.rglob("*.csv"))

    if len(csvs) == 0:
        raise FileNotFoundError(
            f"No encontré ningún CSV en {raw_dir}. "
            "Mete ahí el archivo ids_ageb.csv."
        )

    print(f"Leyendo CSV: {csvs[0]}")
    return csvs[0]


def grado_ids_a_numero(x):
    """
    En IDS, un valor alto significa mejor desarrollo.
    Por eso:
    Muy bajo = 1
    Bajo = 2
    Medio = 3
    Alto = 4
    Muy alto = 5
    """
    x = limpiar_texto(x)

    mapa = {
        "muy bajo": 1,
        "bajo": 2,
        "medio": 3,
        "alto": 4,
        "muy alto": 5,
        "sin informacion": np.nan,
    }

    return mapa.get(x, np.nan)


# =========================
# 3. CARGAR BASE
# =========================

CSV_PATH = buscar_csv(RAW_DIR)

df = pd.read_csv(CSV_PATH, encoding="latin1")

print("\nShape original:", df.shape)
print("\nColumnas originales:")
print(df.columns.tolist())


# =========================
# 4. LIMPIEZA BÁSICA
# =========================

df.columns = [str(c).strip().lower() for c in df.columns]

# folio_ageb ya viene como clave AGEB completa
df["cvegeo"] = (
    df["folio_ageb"]
    .astype(str)
    .str.strip()
    .str.replace(".0", "", regex=False)
    .str.zfill(13)
)

# Filtrar CDMX
df = df[df["cvegeo"].str.startswith("09")].copy()

print("\nShape CDMX:", df.shape)


# =========================
# 5. LIMPIAR VARIABLES IDS
# =========================

df["ids"] = pd.to_numeric(df["idsm"], errors="coerce")
df["poblacion_ids"] = pd.to_numeric(df["pob"], errors="coerce")
df["pob_nbi"] = pd.to_numeric(df["pob_nbi"], errors="coerce")

df["grado_ids"] = df["e_idsm"]
df["grado_ids_num"] = df["grado_ids"].apply(grado_ids_a_numero)

# Normalizamos IDS continuo.
# IDS alto = mejor desarrollo social.
df["ids_norm"] = minmax(df["ids"])

# Para priorizar huecos queremos lo contrario:
# menor IDS = mayor vulnerabilidad.
df["bajo_desarrollo_norm"] = 1 - df["ids_norm"]

# También normalizamos la categoría, por si queremos usarla.
df["grado_ids_norm"] = minmax(df["grado_ids_num"])

# Proporción de población con necesidades básicas insatisfechas
df["prop_nbi"] = df["pob_nbi"] / df["poblacion_ids"]
df.loc[df["poblacion_ids"] <= 0, "prop_nbi"] = np.nan
df["prop_nbi_norm"] = minmax(df["prop_nbi"])


# =========================
# 6. COLUMNAS FINALES
# =========================

columnas_finales = [
    "cvegeo",
    "ids",
    "ids_norm",
    "bajo_desarrollo_norm",
    "grado_ids",
    "grado_ids_num",
    "grado_ids_norm",
    "poblacion_ids",
    "pob_nbi",
    "prop_nbi",
    "prop_nbi_norm",
]

df_final = df[columnas_finales].copy()

# Quitar filas sin IDS
df_final = df_final.dropna(subset=["ids"])

# Quitar duplicados por seguridad
df_final = df_final.drop_duplicates(subset=["cvegeo"])


# =========================
# 7. VALIDACIONES
# =========================

print("\nColumnas finales:")
print(df_final.columns.tolist())

print("\nPrimeras filas:")
print(df_final.head())

print("\nDistribución grado IDS:")
print(df_final["grado_ids"].value_counts(dropna=False))

print("\nResumen IDS:")
print(df_final["ids"].describe())

print("\nResumen bajo_desarrollo_norm:")
print(df_final["bajo_desarrollo_norm"].describe())

print("\nResumen prop_nbi:")
print(df_final["prop_nbi"].describe())


# =========================
# 8. GUARDAR
# =========================

df_final.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("\nArchivo guardado:")
print(OUT_CSV)