# Mapa interactivo de crecimiento distrital del Perú

Este repositorio contiene el código, los datos de entrada y el HTML publicado en
[GitHub Pages](https://gsaco.github.io/policy_brief/). No requiere Jupyter ni
archivos externos para reconstruir el mapa.

## Ejecutar

```bash
git clone https://github.com/gsaco/policy_brief.git
cd policy_brief
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build_growth_interactive_map.py
```

El último comando genera dos copias idénticas:

- `docs/index.html`: pagina que publica GitHub Pages.
- `figures/tasa_crecimiento_promedio_distritos_interactivo.html`: copia de trabajo.

Para revisar el resultado localmente:

```bash
python -m http.server 8000 --directory docs
```

Luego abre <http://localhost:8000>.

## Modificar el mapa

- Datos y geometría: reemplaza los archivos de `data/` manteniendo su estructura.
- Colores, clasificaciones y serie inicial: edita las constantes al inicio de
  `scripts/build_growth_interactive_map.py`.
- Textos, controles y diseño: edita el HTML, CSS y JavaScript incluidos en ese
  mismo script.

También se pueden indicar inputs alternativos sin cambiar el código:

```bash
python scripts/build_growth_interactive_map.py \
  --excel ruta/datos.xlsx \
  --districts ruta/distritos.geojson \
  --countries ruta/paises.geojson
```

## Inputs incluidos

- `data/Data_Final_CEMS.xlsx`: hojas `Datos_Dist` y `Datos_Prov` con las series
  distritales y provinciales usadas por el mapa.
- `data/peru_districts.geojson`: límites de 1.834 distritos, incluida la
  geometría completa de los distritos utilizados.
- `data/south_america_countries.geojson`: contexto geográfico de Sudamérica,
  derivado de Natural Earth (dominio público).

El script valida las columnas esenciales y se detiene con un mensaje claro si
falta algún input.
