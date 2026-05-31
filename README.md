# ITER 2020 — Ciudad de México: Base de Datos Combinada

Construcción y análisis de una base de datos integrada a partir del **Inventario Nacional de Viviendas (ITER) 2020** del INEGI para la Ciudad de México, con énfasis en datos demográficos, de empleo y de servicios de salud.

---

## Archivos del proyecto

| Archivo | Descripción |
|---|---|
| `base_combinada.ipynb` | Notebook principal: carga, combinación, limpieza y visualización |
| `base_combinada.csv` | Base de datos final limpia y exportada |
| `graficas_iter2020.png` | Visualizaciones generadas |

### Datos fuente (no incluidos en el repositorio)

| Archivo | Contenido |
|---|---|
| `ITER2020 - 09 Ciudad de México (1).csv` | Datos geográficos, demográficos y de empleo |
| `ITER2020 - 09 Ciudad de México (2).csv` | Datos de servicios de salud y seguridad social |

> Fuente: INEGI — ITER 2020

---

## Estructura del notebook

### 1. Carga de datos
Se cargan los dos archivos CSV con `encoding='latin1'` para manejar correctamente los caracteres especiales del español.

### 2. Construcción de la base combinada
Se unen ambos archivos por las llaves geográficas `ENTIDAD`, `MUN` y `LOC`.

De cada archivo se toman:
- **Archivo 1** — completo: geografía, demografía y empleo
- **Archivo 2** — solo columnas exclusivas de servicios de salud

Las columnas de edad se agrupan en 4 categorías:

| Grupo | Rango | Cómo se calcula |
|---|---|---|
| Niños | 0–11 años | 0-2 + 3-5 + 6-11 |
| Jóvenes | 12–17 años | 12-14 + 15-17 |
| Adultos | 18–59 años | 18 y más − 60 y más |
| Tercera edad | 60+ años | directo de P_60YMAS |

Todas las columnas son renombradas con nombres descriptivos en español.

### 3. Vista previa
Exploración inicial con `df.head()` y `df.info()`.

### 4. Limpieza de datos

| Paso | Acción |
|---|---|
| 1 | Reemplazar cadenas `"N/A"` por `NaN` real |
| 2 | Eliminar filas de totales y resúmenes (claves 0, 9998, 9999) |
| 3 | Eliminar duplicados |
| 4 | Convertir columnas a tipo numérico |
| 5 | Eliminar filas con valores nulos restantes |
| 6 | Resetear índice |

### 5. Visualizaciones

1. Distribución por grupos de edad
2. Grupos de edad por sexo (mujeres vs hombres)
3. Cobertura de derechohabiencia
4. Población por tipo de servicio de salud
5. Top 10 municipios por población total
6. Situación laboral

### 6. Exportar a CSV
La base limpia se exporta como `base_combinada.csv` con `encoding='utf-8-sig'` para compatibilidad con Excel.

---

## Columnas de la base final

### Geografía
`Clave entidad`, `Entidad`, `Clave municipio`, `Municipio`, `Clave localidad`, `Localidad`, `Longitud`, `Latitud`, `Altitud`

### Población general
`Población total`, `Población femenina`, `Población masculina`, `Relación hombres-mujeres`

### Acumulados de referencia
`Pob. 3 años y más` (total, F, M) — `Pob. 18 años y más` (total, F, M)

### Grupos de edad
`Niños`, `Jóvenes`, `Adultos`, `Tercera edad` — cada uno con total, (F) y (M)

### Empleo
`Pob. económicamente activa`, `Pob. económicamente inactiva`, `Pob. ocupada` — cada uno con total, (F) y (M)

### Servicios de salud
`Sin derechohabiencia`, `Con derechohabiencia`, `Derechohabiente IMSS`, `Derechohabiente ISSSTE federal`, `Derechohabiente ISSSTE estatal`, `Afiliado Seguro Popular/Bienestar`, `Seguro Popular o Médico Siglo XXI`, `Derechohabiente IMSS-Bienestar`, `Afiliado institución privada`, `Afiliado otra institución`

---

## Requisitos

```
pandas
matplotlib
```

Instalar con:

```bash
pip install pandas matplotlib
```

---

## Autora
Regina — 5to semestre, Portafolio TDA Salud
