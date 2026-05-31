# TDA-salud

## Estructura de datos

```text
data/
  raw/
    coneval/
      grs_ageb_2020_shp/
    ids/
      ids_ageb_2020.csv
  processed/
    coneval_ageb_cdmx_limpio.csv
    coneval_ageb_cdmx_limpio.geojson
    coneval_ageb_cdmx_limpio.gpkg
    ids_ageb_cdmx_limpio.csv
```

## Scripts

```text
scripts/limpiar_coneval.py
scripts/limpiar_ids.py
```

Para procesar IDS:

```powershell
python scripts/limpiar_ids.py
```
