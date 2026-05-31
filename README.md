# TDA-salud

Análisis topológico de datos (TDA) aplicado al sector salud de la Ciudad de México, usando datos del DENUE INEGI (mayo 2025).

---

## Datos fuente

| Archivo | Descripción |
|---------|-------------|
| `denue_inegi_09_.csv` | DENUE completo para la entidad 09 (CDMX), ~460k establecimientos, todos los sectores |
| `denue_diccionario_de_datos.csv` | Diccionario de columnas del DENUE |
| `metadatos_denue.txt` | Metadatos del dataset (INEGI, mayo 2025) |

---

## Notebooks

### `01-Preparación.ipynb`

Limpieza y preprocesamiento del DENUE para dejar únicamente el sector salud listo para análisis.

**Pasos:**

1. **Carga** — CSV con encoding `latin-1`, todo como `str` para preservar ceros a la izquierda en claves geográficas
2. **Filtro sector salud** — `codigo_act` con prefijo `62` (SCIAN) → ~22,800 registros
3. **Descarte de actividades irrelevantes** — se eliminan ambulancias (`621910`) y enfermería a domicilio (`621610`)
4. **Limpieza de texto** — strip de espacios en todos los campos de texto; strings vacíos → `NaN`
5. **Tipos de datos**
   - `latitud` / `longitud` → `float`
   - `fecha_alta` → `datetime` (formato `YYYY-MM`)
   - `per_ocu` → categórica ordinal (de "0 a 5 personas" a "251 y más personas")
6. **Columna `subsector`** — basada en los primeros 3 dígitos del código SCIAN:

   | Código | Subsector |
   |--------|-----------|
   | 621 | Servicios ambulatorios |
   | 622 | Hospitales |
   | 623 | Residencias y cuidado |
   | 624 | Asistencia social |

7. **Columna `sector`** — clasificación público / privado / no gubernamental:
   - Se infiere del texto de `nombre_act` ("del sector privado" / "del sector público")
   - Optometría (`621320`) → Privado
   - Agrupaciones de autoayuda AA (`624191`) → No gubernamental

8. **Validación de coordenadas** — registros fuera del bbox de CDMX (lat 19.05–19.60, lon −99.40–−98.95) se marcan con `NaN` sin eliminar el registro
9. **Duplicados** — revisión de IDs duplicados; coordenadas duplicadas se conservan intencionalmente (un mismo inmueble puede albergar varios consultorios)
10. **Visualizaciones** — distribución por alcaldía, subsector y comparativa público vs privado
11. **Exportación** — `denue_salud_cdmx_clean.csv` en UTF-8

**Output:** `denue_salud_cdmx_clean.csv` — 29 columnas, encoding UTF-8

---

## Estructura del proyecto

```
TDA-salud/
├── 01-Preparación.ipynb          # Limpieza y preprocesamiento
├── denue_inegi_09_.csv           # Datos crudos DENUE CDMX
├── denue_diccionario_de_datos.csv
├── metadatos_denue.txt
└── denue_salud_cdmx_clean.csv    # Datos limpios (generado)
```

---

## Fuente

INEGI — Directorio Estadístico Nacional de Unidades Económicas (DENUE), mayo 2025.
Entidad: 09 Ciudad de México.
