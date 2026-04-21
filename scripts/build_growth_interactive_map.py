from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from cartopy.io import shapereader


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HTML_NAME = "tasa_crecimiento_promedio_distritos_interactivo.html"
DEFAULT_PAGES_DIR = ROOT / "docs"
LOWERCASE_WORDS = {"de", "del", "la", "las", "los", "y", "e", "o", "u", "da", "das", "do", "dos"}
ROMAN_NUMERAL_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
DEFAULT_CLASSIFICATION_MODE = "deciles"
DEFAULT_COLOR_PRESET = "red_green"
CLASSIFICATION_MODES = {
    "deciles": {
        "label": "Deciles",
        "singular": "Decil",
        "prefix": "D",
        "classes": 10,
    },
    "quintiles": {
        "label": "Quintiles",
        "singular": "Quintil",
        "prefix": "Q",
        "classes": 5,
    },
    "terciles": {
        "label": "Terciles",
        "singular": "Tercil",
        "prefix": "T",
        "classes": 3,
    },
}
COLOR_PRESETS = {
    "red_green": {
        "label": "Rojo → verde",
        # Diverging muted ramp inspired by cartographic classed choropleths.
        "colors": [
            "#7f0000",
            "#a70f1f",
            "#c73635",
            "#e36a4b",
            "#f2a176",
            "#f5d3a8",
            "#d6e8b2",
            "#9ece82",
            "#5fa85a",
            "#1f6b35",
        ],
    },
    "brown_green": {
        "label": "Marrón → verde",
        "colors": [
            "#5b3a1e",
            "#714b23",
            "#896028",
            "#a07430",
            "#b78a3b",
            "#9da146",
            "#7f943f",
            "#607f37",
            "#3f682c",
            "#1f4d1e",
        ],
    },
    "greens": {
        "label": "Escala de verdes",
        # Sequential green ramp close to ColorBrewer recommendations for choropleths.
        "colors": [
            "#f4fbef",
            "#e5f4dd",
            "#d0ebc4",
            "#b4dda2",
            "#93cb7d",
            "#6eb05b",
            "#4f9446",
            "#34773a",
            "#1d5d2e",
            "#0b3f21",
        ],
    },
}


def resolve_existing_path(relative_candidates: list[Path]) -> Path:
    roots = [ROOT, ROOT.parent, Path.cwd(), Path.cwd().parent]
    for root in roots:
        for relative in relative_candidates:
            candidate = (root / relative).resolve()
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No se encontro ninguno de los paths: {relative_candidates}")


def _clean_geo_value(value):
    if pd.isna(value):
        return np.nan
    text = re.sub(r"\s+", " ", str(value).strip())
    if not text or text.upper() == "NAN":
        return np.nan
    return text


def _capitalize_piece(piece: str, *, is_first: bool) -> str:
    if not piece:
        return piece
    if ROMAN_NUMERAL_RE.fullmatch(piece):
        return piece.upper()
    lower = piece.lower()
    if not is_first and lower in LOWERCASE_WORDS:
        return lower
    return lower[:1].upper() + lower[1:]


def _pretty_geo_name(value) -> str:
    text = _clean_geo_value(value)
    if pd.isna(text):
        return ""
    pretty_words = []
    for word_idx, word in enumerate(text.split(" ")):
        parts = word.split("-")
        pretty_parts = [
            _capitalize_piece(part, is_first=(word_idx == 0 and part_idx == 0))
            for part_idx, part in enumerate(parts)
        ]
        pretty_words.append("-".join(pretty_parts))
    return " ".join(pretty_words)


def _format_ubigeo(series: pd.Series, digits: int = 6) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    mask = numeric.notna()
    out.loc[mask] = numeric.loc[mask].astype(int).astype(str).str.zfill(digits)
    return out


def _compute_quantile_bins(values: pd.Series, q: int = 10) -> np.ndarray:
    series = pd.Series(values).dropna().astype(float)
    if series.empty:
        return np.array([0.0, 1.0])

    quantiles = min(q, int(series.nunique()))
    if quantiles <= 1:
        value = float(series.iloc[0])
        return np.array([value - 1e-9, value + 1e-9])

    _, bins = pd.qcut(series, q=quantiles, retbins=True, duplicates="drop")
    bins = np.unique(np.asarray(bins, dtype=float))
    if len(bins) <= 1:
        value = float(series.iloc[0])
        return np.array([value - 1e-9, value + 1e-9])
    return bins


def _assign_quantile_index(values: pd.Series, bins: np.ndarray) -> pd.Series:
    idx = pd.cut(values, bins=bins, labels=False, include_lowest=True)
    return idx.astype("Int64")


def _build_discrete_colorscale(colors: list[str]) -> list[list[float | str]]:
    if not colors:
        raise ValueError("La escala discreta necesita al menos un color.")

    colorscale: list[list[float | str]] = []
    n_colors = len(colors)
    for idx, color in enumerate(colors):
        start = idx / n_colors
        end = (idx + 1) / n_colors
        colorscale.append([start, color])
        colorscale.append([end, color])
    colorscale[-1][0] = 1.0
    return colorscale


def _select_palette_steps(colors: list[str], n_classes: int) -> list[str]:
    if n_classes <= 0:
        raise ValueError("n_classes debe ser mayor que cero.")
    if len(colors) < n_classes:
        raise ValueError("La paleta no tiene suficientes colores para la clasificacion requerida.")
    if len(colors) == n_classes:
        return list(colors)

    idx = np.linspace(0, len(colors) - 1, n_classes)
    idx = np.rint(idx).astype(int)
    idx = np.clip(idx, 0, len(colors) - 1)
    idx = np.maximum.accumulate(idx)
    idx[-1] = len(colors) - 1
    return [colors[i] for i in idx]


def _normalize_geo_key(value):
    text = _clean_geo_value(value)
    if pd.isna(text):
        return pd.NA
    text = "".join(
        char for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    ).upper()
    text = re.sub(r"[^A-Z0-9]+", "", text)
    aliases = {
        "LIMAPROVINCE": "LIMA",
    }
    return aliases.get(text, text)


def _fill_missing_district_geometries(district_geom: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    enriched = district_geom.copy()
    missing_mask = enriched.geometry.isna()
    if not missing_mask.any():
        return enriched

    for column, key in [("Departamento", "dep_key"), ("Provincia", "prov_key"), ("Distrito", "dist_key")]:
        enriched[key] = enriched[column].map(_normalize_geo_key)

    gadm_path = resolve_existing_path([Path("notebooks/_cache_cartopy/gadm41_PER_3.json")])
    gadm = gpd.read_file(gadm_path).to_crs(4326).rename(
        columns={
            "NAME_1": "Departamento",
            "NAME_2": "Provincia",
            "NAME_3": "Distrito",
        }
    )
    for column, key in [("Departamento", "dep_key"), ("Provincia", "prov_key"), ("Distrito", "dist_key")]:
        gadm[key] = gadm[column].map(_normalize_geo_key)

    geoboundaries_path = resolve_existing_path([Path("notebooks/_cache_cartopy/geoBoundaries-PER-ADM3.geojson")])
    geoboundaries = gpd.read_file(geoboundaries_path).to_crs(4326).rename(columns={"shapeName": "Distrito"})
    geoboundaries["dist_key"] = geoboundaries["Distrito"].map(_normalize_geo_key)
    geoboundaries_unique = geoboundaries.loc[
        geoboundaries["dist_key"].notna()
        & ~geoboundaries["dist_key"].duplicated(keep=False)
    ].copy()

    for idx, row in enriched.loc[missing_mask].iterrows():
        gadm_match = gadm.loc[
            gadm["dep_key"].eq(row["dep_key"])
            & gadm["prov_key"].eq(row["prov_key"])
            & gadm["dist_key"].eq(row["dist_key"])
        ]
        if len(gadm_match) == 1:
            enriched.at[idx, "geometry"] = gadm_match.geometry.iloc[0]
            continue

        geoboundaries_match = geoboundaries_unique.loc[
            geoboundaries_unique["dist_key"].eq(row["dist_key"])
        ]
        if len(geoboundaries_match) == 1:
            enriched.at[idx, "geometry"] = geoboundaries_match.geometry.iloc[0]

    return enriched.drop(columns=["dep_key", "prov_key", "dist_key"], errors="ignore")


def load_district_growth_map(
    *,
    excel_path: Path | None = None,
    geo_path: Path | None = None,
) -> gpd.GeoDataFrame:
    excel_path = excel_path or resolve_existing_path([Path("contribuciones.xlsx")])
    geo_path = geo_path or resolve_existing_path(
        [
            Path("notebooks/_cache_cartopy/peru_distrital_simple.geojson"),
            Path("_cache_cartopy/peru_distrital_simple.geojson"),
        ]
    )

    district_geom = gpd.read_file(geo_path).to_crs(4326).rename(
        columns={
            "NOMBDEP": "Departamento",
            "NOMBPROV": "Provincia",
            "NOMBDIST": "Distrito",
            "IDDIST": "ubigeo",
        }
    )
    district_geom["ubigeo"] = district_geom["ubigeo"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)

    for column in ["Departamento", "Provincia", "Distrito"]:
        district_geom[column] = district_geom[column].map(_pretty_geo_name)

    district_geom = _fill_missing_district_geometries(district_geom)

    dist_levels_raw = pd.read_excel(excel_path, sheet_name="PIB_Dist")
    dist_levels_raw["ubigeo"] = _format_ubigeo(dist_levels_raw["Ubigeo"])
    dist_levels_raw[1993] = pd.to_numeric(dist_levels_raw[1993], errors="coerce")
    dist_levels_raw[2018] = pd.to_numeric(dist_levels_raw[2018], errors="coerce")

    dist_levels = dist_levels_raw.loc[
        dist_levels_raw["Distrito"].map(_clean_geo_value).fillna("").str.upper().ne("NACIONAL")
        & dist_levels_raw["Distrito"].notna(),
        ["ubigeo", 1993, 2018],
    ].copy()

    positive_mask = dist_levels[1993].gt(0) & dist_levels[2018].gt(0)
    dist_levels = dist_levels.loc[positive_mask].copy()
    dist_levels["avg_growth_9318"] = (
        np.log(dist_levels[2018]) - np.log(dist_levels[1993])
    ) / (2018 - 1993 + 1)

    map_gdf = district_geom.merge(
        dist_levels[["ubigeo", "avg_growth_9318"]],
        on="ubigeo",
        how="left",
    )
    return gpd.GeoDataFrame(map_gdf, geometry="geometry", crs=4326).dropna(subset=["geometry"]).copy()


def load_district_pib_trajectories(*, excel_path: Path | None = None) -> tuple[pd.DataFrame, list[int]]:
    excel_path = excel_path or resolve_existing_path([Path("contribuciones.xlsx")])
    dist_levels_raw = pd.read_excel(excel_path, sheet_name="PIB_Dist")
    dist_levels_raw["ubigeo"] = _format_ubigeo(dist_levels_raw["Ubigeo"])

    year_columns = [year for year in range(1993, 2019) if year in dist_levels_raw.columns]
    keep_columns = ["ubigeo", "Departamento", "Provincia", "Distrito", *year_columns]
    trajectories = dist_levels_raw.loc[
        dist_levels_raw["Distrito"].map(_clean_geo_value).fillna("").str.upper().ne("NACIONAL")
        & dist_levels_raw["Distrito"].notna(),
        keep_columns,
    ].copy()

    for column in ["Departamento", "Provincia", "Distrito"]:
        trajectories[column] = trajectories[column].map(_pretty_geo_name)
    trajectories[year_columns] = trajectories[year_columns].apply(pd.to_numeric, errors="coerce")
    return trajectories, year_columns


def prepare_growth_ranking(map_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    required_cols = {"ubigeo", "geometry", "avg_growth_9318"}
    missing = required_cols.difference(map_gdf.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas para el mapa interactivo: {sorted(missing)}")

    renamed = map_gdf.rename(
        columns={
            "NOMBDEP": "Departamento",
            "NOMBPROV": "Provincia",
            "NOMBDIST": "Distrito",
            "IDDIST": "ubigeo",
        }
    ).copy()
    renamed = renamed.drop(columns=["growth_rank", "rank_label"], errors="ignore")

    for column in ["Departamento", "Provincia", "Distrito"]:
        if column in renamed.columns:
            renamed[column] = renamed[column].map(_pretty_geo_name)
        else:
            renamed[column] = ""

    renamed["ubigeo"] = renamed["ubigeo"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    renamed["avg_growth_9318"] = pd.to_numeric(renamed["avg_growth_9318"], errors="coerce")
    renamed = renamed.dropna(subset=["geometry", "avg_growth_9318"]).copy()

    ranking_order = renamed.sort_values(
        ["avg_growth_9318", "Departamento", "Provincia", "Distrito", "ubigeo"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)
    ranking_order["growth_rank"] = np.arange(1, len(ranking_order) + 1)

    ranked = renamed.merge(
        ranking_order[["ubigeo", "growth_rank"]],
        on="ubigeo",
        how="left",
        validate="one_to_one",
    )
    ranked["growth_rank"] = ranked["growth_rank"].astype(int)
    ranked["rank_label"] = ranked["growth_rank"].map(lambda value: f"{value:,}".replace(",", "."))
    return gpd.GeoDataFrame(ranked, geometry="geometry", crs=4326)


def add_growth_deciles(map_gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    decile_bins = _compute_quantile_bins(map_gdf["avg_growth_9318"], q=10)
    decile_idx = _assign_quantile_index(map_gdf["avg_growth_9318"], decile_bins)
    n_deciles = max(len(decile_bins) - 1, 1)

    enriched = map_gdf.copy()
    enriched["growth_decile"] = decile_idx.astype(int) + 1
    enriched["growth_decile_label"] = enriched["growth_decile"].map(lambda value: f"D{int(value)}")
    enriched["growth_decile_range"] = enriched["growth_decile"].map(
        lambda value: (
            f"{decile_bins[int(value) - 1]:.4f} a {decile_bins[int(value)]:.4f}"
            if 1 <= int(value) <= n_deciles
            else ""
        )
    )
    return gpd.GeoDataFrame(enriched, geometry="geometry", crs=4326), decile_bins


def build_classification_presets(map_gdf: gpd.GeoDataFrame) -> dict:
    presets = {}
    ordered = map_gdf.copy()

    for mode_key, mode in CLASSIFICATION_MODES.items():
        bins = _compute_quantile_bins(ordered["avg_growth_9318"], q=mode["classes"])
        class_idx = _assign_quantile_index(ordered["avg_growth_9318"], bins).astype(int) + 1
        n_classes = max(len(bins) - 1, 1)
        tick_text = [f"{mode['prefix']}{idx}" for idx in range(1, n_classes + 1)]
        by_ubigeo = {}

        for ubigeo, idx in zip(ordered["ubigeo"], class_idx):
            idx_int = int(idx)
            lo = bins[idx_int - 1]
            hi = bins[idx_int]
            by_ubigeo[str(ubigeo)] = {
                "index": idx_int,
                "label": f"{mode['prefix']}{idx_int}",
                "range": f"{lo:.4f} a {hi:.4f}",
            }

        presets[mode_key] = {
            "key": mode_key,
            "label": mode["label"],
            "singular": mode["singular"],
            "prefix": mode["prefix"],
            "classes": n_classes,
            "bins": [float(value) for value in bins.tolist()],
            "tick_text": tick_text,
            "by_ubigeo": by_ubigeo,
        }

    return presets


def _geojson_payload(gdf: gpd.GeoDataFrame, keep_columns: list[str]) -> dict:
    payload = gdf[keep_columns + ["geometry"]].copy()
    return json.loads(payload.to_json())


def _build_context_layers(
    map_gdf: gpd.GeoDataFrame,
    *,
    department_bounds: gpd.GeoDataFrame | None = None,
) -> dict:
    countries_path = shapereader.natural_earth(
        resolution="10m",
        category="cultural",
        name="admin_0_countries",
    )
    countries = gpd.read_file(countries_path).to_crs(4326)
    name_column = "NAME_LONG" if "NAME_LONG" in countries.columns else "ADMIN"
    countries = countries.rename(columns={name_column: "country_name"})
    context_countries = countries.loc[
        countries["CONTINENT"].fillna("").eq("South America")
    ].copy()

    if department_bounds is None:
        department_bounds = (
            map_gdf[["Departamento", "geometry"]]
            .dissolve(by="Departamento", aggfunc="first")
            .reset_index()
        )
    department_bounds = gpd.GeoDataFrame(department_bounds, geometry="geometry", crs=4326).dropna(subset=["geometry"])

    peru_outline = map_gdf[["geometry"]].dissolve().reset_index(drop=True)
    peru_outline = gpd.GeoDataFrame(peru_outline, geometry="geometry", crs=4326)

    return {
        "countries_fill": _geojson_payload(context_countries, ["country_name"]),
        "countries_line": _geojson_payload(context_countries, ["country_name"]),
        "department_bounds": _geojson_payload(department_bounds, ["Departamento"]),
        "peru_outline": _geojson_payload(peru_outline, []),
    }


def _build_trajectory_payload(
    ranked_map: gpd.GeoDataFrame,
    *,
    excel_path: Path | None = None,
) -> dict:
    trajectories, years = load_district_pib_trajectories(excel_path=excel_path)
    trajectories = trajectories[["ubigeo", *years]].copy()
    metadata_cols = [
        "ubigeo",
        "Distrito",
        "Provincia",
        "Departamento",
        "growth_rank",
        "avg_growth_9318",
    ]
    merged = trajectories.merge(
        ranked_map[metadata_cols],
        on="ubigeo",
        how="inner",
        validate="one_to_one",
    )

    payload = {}
    for _, row in merged.iterrows():
        payload[str(row["ubigeo"])] = {
            "ubigeo": str(row["ubigeo"]),
            "distrito": row["Distrito"],
            "provincia": row["Provincia"],
            "departamento": row["Departamento"],
            "growth_rank": int(row["growth_rank"]),
            "avg_growth_9318": float(row["avg_growth_9318"]),
            "years": years,
            "values": [
                None if pd.isna(row[year]) else float(row[year])
                for year in years
            ],
        }

    return payload


def _build_interactive_dashboard_html(
    figure: go.Figure,
    *,
    config: dict,
    trajectory_payload: dict,
    classification_modes: dict,
) -> str:
    plot_div = pio.to_html(
        figure,
        include_plotlyjs="cdn",
        full_html=False,
        config=config,
        div_id="district-map",
    )
    trajectory_json = json.dumps(trajectory_payload, ensure_ascii=False)
    classification_json = json.dumps(classification_modes, ensure_ascii=False)
    palette_json = json.dumps(COLOR_PRESETS, ensure_ascii=False)
    classification_options = "\n".join(
        [
            f'<option value="{key}"{" selected" if key == DEFAULT_CLASSIFICATION_MODE else ""}>{meta["label"]}</option>'
            for key, meta in CLASSIFICATION_MODES.items()
        ]
    )
    palette_options = "\n".join(
        [
            f'<option value="{key}"{" selected" if key == DEFAULT_COLOR_PRESET else ""}>{meta["label"]}</option>'
            for key, meta in COLOR_PRESETS.items()
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mapa interactivo distrital</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --panel: rgba(255, 255, 255, 0.96);
      --panel-border: #d8d0c1;
      --text: #0f172a;
      --muted: #5f6b76;
      --green: #1f4d1e;
      --brown: #8a6028;
      --shadow: 0 18px 44px rgba(15, 23, 42, 0.14);
      --radius: 20px;
      --edge-gap: clamp(12px, 1.8vw, 20px);
      --panel-gap: clamp(12px, 2vw, 22px);
      --controls-width: min(292px, calc(100vw - (var(--edge-gap) * 2)), calc(100vh * 0.42));
      --years-width: min(348px, calc(100vw - (var(--edge-gap) * 2)), calc(100vh * 0.5));
      --left-stack-width: max(var(--controls-width), var(--years-width));
      --left-stack-bottom: max(96px, calc(env(safe-area-inset-bottom) + 30px));
    }}
    * {{
      box-sizing: border-box;
    }}
    html, body {{
      margin: 0;
      height: 100%;
      background: radial-gradient(circle at 18% 12%, #f7f3eb 0%, var(--bg) 58%, #ece5d8 100%);
      font-family: "Avenir Next", "Segoe UI", Arial, sans-serif;
      color: var(--text);
    }}
    body {{
      min-height: 100vh;
    }}
    .app-shell {{
      position: relative;
      min-height: 100vh;
      min-height: 100dvh;
      width: 100%;
      overflow: hidden;
    }}
    .map-shell {{
      position: relative;
      min-height: 100vh;
      min-height: 100dvh;
      width: 100%;
      padding: 0;
    }}
    .left-overlay-stack {{
      position: absolute;
      top: max(var(--edge-gap), calc(env(safe-area-inset-top) + 10px));
      left: max(var(--edge-gap), calc(env(safe-area-inset-left) + 10px));
      bottom: var(--left-stack-bottom);
      z-index: 30;
      width: var(--left-stack-width);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      align-items: flex-start;
      gap: 18px;
      pointer-events: none;
    }}
    .controls-shell {{
      position: relative;
      z-index: 1;
      width: min(var(--controls-width), 100%);
      background: rgba(255, 255, 255, 0.95);
      border: 1px solid rgba(216, 208, 193, 0.96);
      border-radius: 16px;
      box-shadow: 0 16px 40px rgba(15, 23, 42, 0.12);
      backdrop-filter: blur(10px);
      padding: 10px 10px 9px;
      pointer-events: auto;
    }}
    .controls-title {{
      margin: 0 0 8px;
      font-size: 10px;
      line-height: 1;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      font-weight: 800;
      color: var(--muted);
    }}
    .controls-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}
    .control-field {{
      display: grid;
      gap: 5px;
    }}
    .control-label {{
      font-size: 10px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .control-select {{
      width: 100%;
      border: 1px solid #d7cfbf;
      border-radius: 12px;
      background: #fbf8f2;
      color: var(--text);
      padding: 9px 10px;
      font-size: 13px;
      font-weight: 700;
      outline: none;
      transition: border-color 120ms ease, box-shadow 120ms ease, background 120ms ease;
    }}
    .control-select:hover {{
      background: #f8f2e8;
    }}
    .control-select:focus {{
      border-color: #7f943f;
      box-shadow: 0 0 0 3px rgba(127, 148, 63, 0.16);
    }}
    .controls-footnote {{
      margin: 8px 2px 0;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
    }}
    .controls-note-secondary {{
      margin: 6px 2px 0;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
    }}
    .years-shell {{
      position: relative;
      z-index: 1;
      width: min(var(--years-width), 100%);
      background: rgba(255, 255, 255, 0.95);
      border: 1px solid rgba(216, 208, 193, 0.96);
      border-radius: 18px;
      box-shadow: 0 16px 40px rgba(15, 23, 42, 0.12);
      backdrop-filter: blur(10px);
      padding: 12px 12px 10px;
      overflow: visible;
      scrollbar-width: none;
      pointer-events: auto;
    }}
    .years-title {{
      margin: 0 0 4px;
      font-size: 10px;
      line-height: 1;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      font-weight: 800;
      color: var(--muted);
    }}
    .years-subtitle {{
      margin: 0 0 9px;
      color: var(--text);
      font-size: 13px;
      font-weight: 700;
      line-height: 1.35;
    }}
    .years-values {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 9px;
    }}
    .year-pill {{
      display: grid;
      gap: 3px;
      border: 1px solid rgba(216, 208, 193, 0.92);
      background: #faf7f1;
      border-radius: 14px;
      padding: 8px 10px;
    }}
    .year-pill-label {{
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
      color: var(--muted);
    }}
    .year-pill-value {{
      font-size: 17px;
      font-weight: 800;
      color: var(--text);
      line-height: 1;
    }}
    .dual-range {{
      position: relative;
      height: 34px;
      margin: 0 2px 9px;
    }}
    .range-track-bg,
    .range-track-active {{
      position: absolute;
      left: 0;
      right: 0;
      top: 15px;
      height: 5px;
      border-radius: 999px;
    }}
    .range-track-bg {{
      background: #e4dfd4;
    }}
    .range-track-active {{
      background: linear-gradient(90deg, #8c6239 0%, #c79b58 48%, #3d6f4a 100%);
    }}
    .year-range {{
      position: absolute;
      left: 0;
      top: 0;
      width: 100%;
      height: 34px;
      margin: 0;
      pointer-events: none;
      -webkit-appearance: none;
      appearance: none;
      background: none;
    }}
    .year-range::-webkit-slider-runnable-track {{
      height: 5px;
      background: transparent;
    }}
    .year-range::-moz-range-track {{
      height: 5px;
      background: transparent;
    }}
    .year-range::-webkit-slider-thumb {{
      -webkit-appearance: none;
      appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 999px;
      background: #ffffff;
      border: 2px solid #2f4f5f;
      box-shadow: 0 3px 10px rgba(15, 23, 42, 0.18);
      pointer-events: auto;
      cursor: pointer;
      margin-top: -5px;
    }}
    .year-range::-moz-range-thumb {{
      width: 16px;
      height: 16px;
      border-radius: 999px;
      background: #ffffff;
      border: 2px solid #2f4f5f;
      box-shadow: 0 3px 10px rgba(15, 23, 42, 0.18);
      pointer-events: auto;
      cursor: pointer;
    }}
    .formula-box {{
      border: 1px solid rgba(216, 208, 193, 0.92);
      background: #fbf8f2;
      border-radius: 14px;
      padding: 10px 10px 8px;
      margin-bottom: 8px;
    }}
    .formula-label {{
      margin: 0 0 4px;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
      color: var(--muted);
    }}
    .formula-main {{
      margin: 0 0 5px;
      font-size: 12.5px;
      line-height: 1.4;
      color: var(--text);
      font-weight: 700;
    }}
    .formula-current {{
      margin: 0;
      font-size: 10px;
      line-height: 1.4;
      color: var(--muted);
      white-space: nowrap;
    }}
    #district-map {{
      width: 100%;
      height: 100vh;
      height: 100dvh;
    }}
    .panel {{
      position: absolute;
      top: var(--panel-gap);
      right: var(--panel-gap);
      width: min(420px, calc(100vw - (var(--panel-gap) * 2)));
      max-height: calc(100vh - (var(--panel-gap) * 2));
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      overflow: hidden;
      z-index: 20;
      transition: transform 180ms ease, opacity 180ms ease;
    }}
    .panel.is-hidden {{
      opacity: 0;
      transform: translateX(22px);
      pointer-events: none;
    }}
    .panel-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 14px;
      padding: 18px 18px 14px;
      border-bottom: 1px solid rgba(216, 208, 193, 0.82);
      background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(249,246,240,0.96) 100%);
    }}
    .panel-kicker {{
      margin: 0 0 6px;
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 700;
    }}
    .panel-title {{
      margin: 0;
      font-size: 22px;
      line-height: 1.05;
      font-weight: 800;
    }}
    .panel-subtitle {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.3;
    }}
    .panel-close {{
      border: 0;
      background: #f2ede4;
      color: #354150;
      width: 34px;
      height: 34px;
      border-radius: 999px;
      cursor: pointer;
      font-size: 18px;
      line-height: 1;
      transition: background 120ms ease, transform 120ms ease;
    }}
    .panel-close:hover {{
      background: #e8dfd0;
      transform: translateY(-1px);
    }}
    .panel-body {{
      padding: 16px 18px 18px;
      overflow: auto;
      max-height: calc(100vh - 118px);
    }}
    .panel-placeholder {{
      display: grid;
      gap: 14px;
      padding: 18px 4px 4px;
      min-height: 280px;
      align-content: center;
    }}
    .placeholder-title {{
      font-size: 20px;
      font-weight: 800;
      margin: 0;
    }}
    .placeholder-copy {{
      margin: 0;
      font-size: 14px;
      line-height: 1.5;
      color: var(--muted);
    }}
    .placeholder-chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      padding: 9px 12px;
      border-radius: 999px;
      background: #f2ede4;
      color: #354150;
      font-size: 12px;
      font-weight: 700;
    }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .metric-card {{
      border: 1px solid rgba(216, 208, 193, 0.92);
      border-radius: 14px;
      padding: 12px 12px 10px;
      background: #faf7f1;
    }}
    .metric-label {{
      margin: 0 0 6px;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
    }}
    .metric-value {{
      margin: 0;
      font-size: 16px;
      font-weight: 800;
      line-height: 1.15;
    }}
    #series-plot {{
      width: 100%;
      min-height: 320px;
    }}
    .panel-note {{
      margin: 10px 2px 0;
      color: var(--muted);
      font-size: 12.5px;
      line-height: 1.45;
    }}
    @media (max-width: 980px) {{
      .left-overlay-stack {{
        top: max(14px, calc(env(safe-area-inset-top) + 6px));
        left: max(14px, calc(env(safe-area-inset-left) + 6px));
        bottom: max(64px, calc(env(safe-area-inset-bottom) + 20px));
        width: min(304px, calc(100vw - 28px));
      }}
      .controls-shell {{
        width: min(274px, calc(100vw - 28px));
      }}
      .years-shell {{
        width: min(320px, calc(100vw - 28px));
      }}
      .panel {{
        top: auto;
        right: 12px;
        left: 12px;
        bottom: 12px;
        width: auto;
        max-height: 56vh;
      }}
      .panel-body {{
        max-height: calc(56vh - 96px);
      }}
      #district-map {{
        height: 100vh;
        height: 100dvh;
      }}
    }}
    @media (max-height: 900px) {{
      .left-overlay-stack {{
        bottom: max(84px, calc(env(safe-area-inset-bottom) + 26px));
        gap: 14px;
      }}
      .controls-shell {{
        width: min(276px, calc(100vw - (var(--edge-gap) * 2)), calc(100vh * 0.44));
        padding: 9px 9px 8px;
      }}
      .controls-footnote,
      .controls-note-secondary {{
        font-size: 10px;
        line-height: 1.3;
      }}
      .years-shell {{
        width: min(320px, calc(100vw - (var(--edge-gap) * 2)), calc(100vh * 0.52));
        padding: 11px 11px 9px;
      }}
      .years-subtitle {{
        font-size: 12px;
        margin-bottom: 8px;
      }}
      .year-pill {{
        padding: 7px 9px;
      }}
      .year-pill-value {{
        font-size: 16px;
      }}
      .dual-range {{
        margin-bottom: 8px;
      }}
      .formula-box {{
        padding: 9px 9px 7px;
      }}
      .formula-main {{
        font-size: 11.5px;
      }}
    }}
    @media (max-height: 780px) {{
      .left-overlay-stack {{
        top: max(10px, calc(env(safe-area-inset-top) + 4px));
        bottom: max(76px, calc(env(safe-area-inset-bottom) + 22px));
        gap: 12px;
      }}
      .controls-shell {{
        width: min(256px, calc(100vw - (var(--edge-gap) * 2)), calc(100vh * 0.44));
      }}
      .control-select {{
        padding: 8px 9px;
        font-size: 12px;
      }}
      .controls-title,
      .control-label,
      .years-title,
      .year-pill-label,
      .formula-label {{
        font-size: 9px;
      }}
      .controls-footnote,
      .controls-note-secondary {{
        font-size: 9.5px;
      }}
      .years-shell {{
        width: min(300px, calc(100vw - (var(--edge-gap) * 2)), calc(100vh * 0.54));
        padding: 9px 9px 8px;
      }}
      .years-subtitle {{
        font-size: 11px;
        margin-bottom: 7px;
      }}
      .year-pill-value {{
        font-size: 15px;
      }}
      .dual-range {{
        height: 30px;
        margin-bottom: 7px;
      }}
      .range-track-bg,
      .range-track-active {{
        top: 13px;
      }}
      .formula-main {{
        font-size: 10.5px;
      }}
      .formula-current {{
        font-size: 9px;
      }}
      .panel {{
        width: min(400px, calc(100vw - (var(--panel-gap) * 2)));
      }}
      .panel-title {{
        font-size: 20px;
      }}
    }}
    .left-overlay-stack.is-tight {{
      gap: 12px;
    }}
    .left-overlay-stack.is-tight .controls-shell {{
      width: min(264px, 100%);
      padding: 9px 9px 8px;
    }}
    .left-overlay-stack.is-tight .years-shell {{
      width: min(308px, 100%);
      padding: 10px 10px 8px;
    }}
    .left-overlay-stack.is-tight .control-select {{
      padding: 8px 9px;
      font-size: 12px;
    }}
    .left-overlay-stack.is-tight .controls-title,
    .left-overlay-stack.is-tight .control-label,
    .left-overlay-stack.is-tight .years-title,
    .left-overlay-stack.is-tight .year-pill-label,
    .left-overlay-stack.is-tight .formula-label {{
      font-size: 9px;
    }}
    .left-overlay-stack.is-tight .controls-footnote,
    .left-overlay-stack.is-tight .controls-note-secondary,
    .left-overlay-stack.is-tight .years-subtitle,
    .left-overlay-stack.is-tight .formula-main {{
      font-size: 11px;
      line-height: 1.3;
    }}
    .left-overlay-stack.is-tight .year-pill {{
      padding: 7px 9px;
    }}
    .left-overlay-stack.is-tight .year-pill-value {{
      font-size: 15px;
    }}
    .left-overlay-stack.is-tight .formula-current {{
      font-size: 9px;
    }}
    .left-overlay-stack.is-compact {{
      gap: 10px;
    }}
    .left-overlay-stack.is-compact .controls-shell {{
      width: min(248px, 100%);
      padding: 8px 8px 7px;
    }}
    .left-overlay-stack.is-compact .years-shell {{
      width: min(288px, 100%);
      padding: 8px 8px 7px;
    }}
    .left-overlay-stack.is-compact .controls-grid {{
      grid-template-columns: 1fr;
      gap: 7px;
    }}
    .left-overlay-stack.is-compact .control-select {{
      padding: 7px 8px;
      font-size: 11px;
    }}
    .left-overlay-stack.is-compact .controls-title,
    .left-overlay-stack.is-compact .control-label,
    .left-overlay-stack.is-compact .years-title,
    .left-overlay-stack.is-compact .year-pill-label,
    .left-overlay-stack.is-compact .formula-label {{
      font-size: 8.5px;
    }}
    .left-overlay-stack.is-compact .controls-footnote,
    .left-overlay-stack.is-compact .controls-note-secondary,
    .left-overlay-stack.is-compact .years-subtitle,
    .left-overlay-stack.is-compact .formula-main {{
      font-size: 10px;
      line-height: 1.25;
    }}
    .left-overlay-stack.is-compact .years-values {{
      gap: 6px;
      margin-bottom: 7px;
    }}
    .left-overlay-stack.is-compact .year-pill {{
      padding: 6px 8px;
      gap: 2px;
    }}
    .left-overlay-stack.is-compact .year-pill-value {{
      font-size: 14px;
    }}
    .left-overlay-stack.is-compact .dual-range {{
      height: 28px;
      margin-bottom: 6px;
    }}
    .left-overlay-stack.is-compact .range-track-bg,
    .left-overlay-stack.is-compact .range-track-active {{
      top: 12px;
    }}
    .left-overlay-stack.is-compact .formula-box {{
      padding: 8px 8px 6px;
      margin-bottom: 0;
    }}
    .left-overlay-stack.is-compact .formula-current {{
      font-size: 8.5px;
      white-space: normal;
    }}
    @media (max-width: 640px) {{
      .controls-grid {{
        grid-template-columns: 1fr;
      }}
      .years-values {{
        grid-template-columns: 1fr 1fr;
      }}
      .metrics-grid {{
        grid-template-columns: 1fr;
      }}
      .panel-title {{
        font-size: 19px;
      }}
    }}
  </style>
</head>
<body>
  <div class="app-shell">
    <div class="map-shell">
      <div id="left-overlay-stack" class="left-overlay-stack">
        <div id="controls-shell" class="controls-shell">
          <p class="controls-title">Clasificación cartográfica</p>
          <div class="controls-grid">
            <label class="control-field">
              <span class="control-label">Agrupación</span>
              <select id="classification-mode" class="control-select">
                {classification_options}
              </select>
            </label>
            <label class="control-field">
              <span class="control-label">Escala de color</span>
              <select id="color-preset" class="control-select">
                {palette_options}
              </select>
            </label>
          </div>
          <p class="controls-footnote">Base inicial: deciles con paleta rojo → verde. Puedes cambiar ambas opciones sin recargar el mapa.</p>
          <p class="controls-note-secondary">Pasa el cursor sobre un distrito para ver su información. Haz clic para abrir el gráfico con la trayectoria del PBI distrital.</p>
        </div>
        <div id="years-shell" class="years-shell">
          <p class="years-title">Rango temporal del cálculo</p>
          <p class="years-subtitle">Elige los años usados para recalcular la tasa de crecimiento promedio anual.</p>
          <div class="years-values">
            <div class="year-pill">
              <span class="year-pill-label">Año inicial</span>
              <span id="year-start-value" class="year-pill-value">1993</span>
            </div>
            <div class="year-pill">
              <span class="year-pill-label">Año final</span>
              <span id="year-end-value" class="year-pill-value">2018</span>
            </div>
          </div>
          <div class="dual-range">
            <div class="range-track-bg"></div>
            <div id="range-track-active" class="range-track-active"></div>
            <input id="year-range-start" class="year-range" type="range" min="1993" max="2018" step="1" value="1993" />
            <input id="year-range-end" class="year-range" type="range" min="1993" max="2018" step="1" value="2018" />
          </div>
          <div class="formula-box">
            <p class="formula-label">Fórmula usada</p>
            <p class="formula-main">Tasa promedio anual = (ln(PBI final) − ln(PBI inicial)) / (año final − año inicial + 1)</p>
            <p id="formula-current" class="formula-current">Actualmente: (ln(PBI 2018) − ln(PBI 1993)) / (2018 − 1993 + 1)</p>
          </div>
        </div>
      </div>
      {plot_div}
    </div>
    <aside id="district-panel" class="panel is-hidden">
      <div class="panel-header">
        <div>
          <p class="panel-kicker">Trayectoria del PBI</p>
          <h2 id="panel-title" class="panel-title">Selecciona un distrito</h2>
          <p id="panel-subtitle" class="panel-subtitle">Haz clic sobre cualquier distrito del mapa para abrir su serie 1993-2018.</p>
        </div>
        <button id="panel-close" class="panel-close" type="button" aria-label="Cerrar panel">×</button>
      </div>
      <div class="panel-body">
        <div id="panel-placeholder" class="panel-placeholder">
          <span class="placeholder-chip">Clic en el mapa</span>
          <p class="placeholder-title">Serie anual del PBI distrital</p>
          <p class="placeholder-copy">Al seleccionar un distrito se mostrará su trayectoria anual de PBI entre 1993 y 2018, junto con su clase activa de crecimiento y su ranking nacional.</p>
        </div>
        <div id="panel-content" style="display:none;">
          <div class="metrics-grid">
            <div class="metric-card">
              <p class="metric-label">Ranking nacional</p>
              <p id="metric-rank" class="metric-value"></p>
            </div>
            <div class="metric-card">
              <p id="metric-class-label" class="metric-label">Decil</p>
              <p id="metric-class" class="metric-value"></p>
            </div>
            <div class="metric-card">
              <p class="metric-label">Tasa promedio</p>
              <p id="metric-growth" class="metric-value"></p>
            </div>
          </div>
          <div id="series-plot"></div>
          <p id="panel-note" class="panel-note"></p>
        </div>
      </div>
    </aside>
  </div>

  <script>
    const DISTRICT_SERIES = {trajectory_json};
    const CLASSIFICATION_PRESETS = {classification_json};
    const COLOR_PRESETS = {palette_json};
    const DEFAULT_MODE = '{DEFAULT_CLASSIFICATION_MODE}';
    const DEFAULT_PALETTE = '{DEFAULT_COLOR_PRESET}';
    const AVAILABLE_YEARS = Object.values(DISTRICT_SERIES)[0].years.slice();
    const state = {{
      mode: DEFAULT_MODE,
      palette: DEFAULT_PALETTE,
      selectedUbigeo: null,
      startYear: AVAILABLE_YEARS[0],
      endYear: AVAILABLE_YEARS[AVAILABLE_YEARS.length - 1],
      currentMetrics: null
    }};
    const mapPlot = document.getElementById('district-map');
    const leftOverlayStack = document.getElementById('left-overlay-stack');
    const controlsShell = document.getElementById('controls-shell');
    const yearsShell = document.getElementById('years-shell');
    const classificationSelect = document.getElementById('classification-mode');
    const colorPresetSelect = document.getElementById('color-preset');
    const yearRangeStart = document.getElementById('year-range-start');
    const yearRangeEnd = document.getElementById('year-range-end');
    const yearStartValue = document.getElementById('year-start-value');
    const yearEndValue = document.getElementById('year-end-value');
    const rangeTrackActive = document.getElementById('range-track-active');
    const formulaCurrent = document.getElementById('formula-current');
    const panelTitle = document.getElementById('panel-title');
    const panelSubtitle = document.getElementById('panel-subtitle');
    const districtPanel = document.getElementById('district-panel');
    const panelPlaceholder = document.getElementById('panel-placeholder');
    const panelContent = document.getElementById('panel-content');
    const panelClose = document.getElementById('panel-close');
    const metricRank = document.getElementById('metric-rank');
    const metricClassLabel = document.getElementById('metric-class-label');
    const metricClass = document.getElementById('metric-class');
    const metricGrowth = document.getElementById('metric-growth');
    const panelNote = document.getElementById('panel-note');
    let mapEventsBound = false;

    function formatInteger(value) {{
      return new Intl.NumberFormat('es-PE').format(value);
    }}

    function formatGrowth(value) {{
      return value.toFixed(4) + ' log puntos/año';
    }}

    function formatPBI(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) {{
        return 'NA';
      }}
      return new Intl.NumberFormat('es-PE', {{
        maximumFractionDigits: 0,
      }}).format(value);
    }}

    function getActiveScheme() {{
      return CLASSIFICATION_PRESETS[state.mode];
    }}

    function getActivePalette() {{
      return COLOR_PRESETS[state.palette];
    }}

    function getMapLocations() {{
      return Array.from(mapPlot.data[0].locations).map(value => String(value));
    }}

    function getYearIndex(year) {{
      return AVAILABLE_YEARS.indexOf(Number(year));
    }}

    function computeGrowthForRange(district, startYear, endYear) {{
      const startIdx = getYearIndex(startYear);
      const endIdx = getYearIndex(endYear);
      if (startIdx === -1 || endIdx === -1 || endIdx < startIdx) {{
        return null;
      }}
      const startValue = district.values[startIdx];
      const endValue = district.values[endIdx];
      if (startValue === null || endValue === null || !Number.isFinite(startValue) || !Number.isFinite(endValue) || startValue <= 0 || endValue <= 0) {{
        return null;
      }}
      return (Math.log(endValue) - Math.log(startValue)) / (endYear - startYear + 1);
    }}

    function quantileSorted(sortedValues, q) {{
      if (!sortedValues.length) {{
        return [0, 1];
      }}
      if (sortedValues.length === 1) {{
        return [sortedValues[0] - 1e-9, sortedValues[0] + 1e-9];
      }}
      const bins = [];
      for (let i = 0; i <= q; i += 1) {{
        const p = i / q;
        const pos = (sortedValues.length - 1) * p;
        const lower = Math.floor(pos);
        const upper = Math.ceil(pos);
        const weight = pos - lower;
        const value = sortedValues[lower] + (sortedValues[upper] - sortedValues[lower]) * weight;
        bins.push(value);
      }}
      const uniqueBins = bins.filter((value, idx) => idx === 0 || Math.abs(value - bins[idx - 1]) > 1e-12);
      if (uniqueBins.length <= 1) {{
        return [uniqueBins[0] - 1e-9, uniqueBins[0] + 1e-9];
      }}
      return uniqueBins;
    }}

    function assignClassIndex(value, bins) {{
      for (let idx = 0; idx < bins.length - 1; idx += 1) {{
        const upper = bins[idx + 1];
        if (value <= upper || idx === bins.length - 2) {{
          return idx + 1;
        }}
      }}
      return bins.length - 1;
    }}

    function buildCurrentMetrics() {{
      const scheme = getActiveScheme();
      const rows = Object.values(DISTRICT_SERIES).map((district) => {{
        const growth = computeGrowthForRange(district, state.startYear, state.endYear);
        return {{
          ubigeo: district.ubigeo,
          distrito: district.distrito,
          provincia: district.provincia,
          departamento: district.departamento,
          growth,
        }};
      }}).filter((row) => row.growth !== null && Number.isFinite(row.growth));

      rows.sort((a, b) => {{
        if (b.growth !== a.growth) return b.growth - a.growth;
        if (a.departamento !== b.departamento) return a.departamento.localeCompare(b.departamento, 'es');
        if (a.provincia !== b.provincia) return a.provincia.localeCompare(b.provincia, 'es');
        if (a.distrito !== b.distrito) return a.distrito.localeCompare(b.distrito, 'es');
        return a.ubigeo.localeCompare(b.ubigeo, 'es');
      }});
      rows.forEach((row, idx) => {{
        row.rank = idx + 1;
      }});

      const bins = quantileSorted(rows.map((row) => row.growth).sort((a, b) => a - b), scheme.classes);
      const classes = Math.max(bins.length - 1, 1);
      const byUbigeo = {{}};
      rows.forEach((row) => {{
        const classIndex = assignClassIndex(row.growth, bins);
        byUbigeo[row.ubigeo] = {{
          index: classIndex,
          label: scheme.prefix + classIndex,
          range: bins[classIndex - 1].toFixed(4) + ' a ' + bins[classIndex].toFixed(4),
          rank: row.rank,
          growth: row.growth
        }};
      }});
      return {{
        scheme,
        bins,
        classes,
        tick_text: Array.from({{ length: classes }}, (_, idx) => scheme.prefix + (idx + 1)),
        byUbigeo
      }};
    }}

    function buildDiscreteColorscale(colors) {{
      const colorscale = [];
      const n = colors.length;
      colors.forEach((color, idx) => {{
        const start = idx / n;
        const end = (idx + 1) / n;
        colorscale.push([start, color], [end, color]);
      }});
      colorscale[colorscale.length - 1][0] = 1;
      return colorscale;
    }}

    function samplePalette(colors, nClasses) {{
      if (colors.length === nClasses) {{
        return colors.slice();
      }}
      const selected = [];
      for (let i = 0; i < nClasses; i += 1) {{
        const raw = i * (colors.length - 1) / Math.max(nClasses - 1, 1);
        selected.push(colors[Math.round(raw)]);
      }}
      selected[selected.length - 1] = colors[colors.length - 1];
      return selected;
    }}

    function buildCustomdata(locations) {{
      const currentMetrics = state.currentMetrics;
      const scheme = currentMetrics.scheme;
      return locations.map((ubigeo) => {{
        const district = DISTRICT_SERIES[String(ubigeo)];
        const klass = currentMetrics.byUbigeo[String(ubigeo)];
        return [
          district.distrito,
          district.provincia,
          district.departamento,
          klass.rank,
          String(ubigeo),
          klass.label,
          klass.range,
          klass.growth
        ];
      }});
    }}

    function buildHovertemplate() {{
      const scheme = state.currentMetrics.scheme;
      return (
        '<b>%{{customdata[0]}}</b><br>' +
        '<span style="color:#475569">%{{customdata[1]}}, %{{customdata[2]}}</span><br>' +
        scheme.singular + ': %{{customdata[5]}}<br>' +
        'Rango de clase: %{{customdata[6]}}<br>' +
        'Ranking nacional: %{{customdata[3]}}<br>' +
        'Ubigeo: %{{customdata[4]}}<br>' +
        'Tasa promedio: %{{customdata[7]:.4f}} log puntos/año' +
        '<extra></extra>'
      );
    }}

    function buildColorbar() {{
      const currentMetrics = state.currentMetrics;
      const scheme = currentMetrics.scheme;
      return {{
        title: {{
          text: scheme.label + '<br>de crecimiento',
          side: 'top'
        }},
        tickvals: Array.from({{ length: currentMetrics.classes }}, (_, idx) => idx + 1),
        ticktext: currentMetrics.tick_text,
        thickness: 22,
        len: 0.62,
        x: 0.978,
        xanchor: 'right',
        y: 0.5,
        outlinecolor: '#d6d3c8',
        outlinewidth: 0.8,
        bgcolor: 'rgba(255,255,255,0.95)',
        tickfont: {{ size: 12, color: '#0f172a' }}
      }};
    }}

    function updateYearRangeUI() {{
      yearStartValue.textContent = state.startYear;
      yearEndValue.textContent = state.endYear;
      formulaCurrent.textContent = 'Actualmente: (ln(PBI ' + state.endYear + ') − ln(PBI ' + state.startYear + ')) / (' + state.endYear + ' − ' + state.startYear + ' + 1)';

      const minYear = Number(yearRangeStart.min);
      const maxYear = Number(yearRangeStart.max);
      const left = ((state.startYear - minYear) / (maxYear - minYear)) * 100;
      const right = ((state.endYear - minYear) / (maxYear - minYear)) * 100;
      rangeTrackActive.style.left = left + '%';
      rangeTrackActive.style.right = (100 - right) + '%';
    }}

    function syncLeftOverlayDensity() {{
      if (!leftOverlayStack || !controlsShell || !yearsShell) {{
        return;
      }}
      leftOverlayStack.classList.remove('is-tight', 'is-compact');
      const gap = parseFloat(window.getComputedStyle(leftOverlayStack).gap || '18');
      const availableHeight = leftOverlayStack.getBoundingClientRect().height;
      const naturalHeight = controlsShell.getBoundingClientRect().height + yearsShell.getBoundingClientRect().height + gap;
      if (naturalHeight > availableHeight - 4) {{
        leftOverlayStack.classList.add('is-tight');
      }}
      const tightGap = parseFloat(window.getComputedStyle(leftOverlayStack).gap || '12');
      const tightenedHeight = controlsShell.getBoundingClientRect().height + yearsShell.getBoundingClientRect().height + tightGap;
      if (tightenedHeight > availableHeight - 4) {{
        leftOverlayStack.classList.add('is-compact');
      }}
    }}

    function applyMapEncoding() {{
      state.currentMetrics = buildCurrentMetrics();
      const currentMetrics = state.currentMetrics;
      const palette = getActivePalette();
      const locations = getMapLocations();
      const sampledColors = samplePalette(palette.colors, currentMetrics.classes);
      const zValues = locations.map((ubigeo) => currentMetrics.byUbigeo[String(ubigeo)].index);
      const customdata = buildCustomdata(locations);
      updateYearRangeUI();

      Plotly.restyle(
        mapPlot,
        {{
          z: [zValues],
          colorscale: [buildDiscreteColorscale(sampledColors)],
          zmin: [0.5],
          zmax: [currentMetrics.classes + 0.5],
          customdata: [customdata],
          hovertemplate: [buildHovertemplate()],
          colorbar: [buildColorbar()]
        }},
        [0]
      );

      if (state.selectedUbigeo) {{
        renderSeriesPanel(state.selectedUbigeo);
      }}
      window.requestAnimationFrame(syncLeftOverlayDensity);
    }}

    function closePanel() {{
      state.selectedUbigeo = null;
      districtPanel.classList.add('is-hidden');
      panelTitle.textContent = 'Selecciona un distrito';
      panelSubtitle.textContent = 'Haz clic sobre cualquier distrito del mapa para abrir su serie 1993-2018.';
      panelPlaceholder.style.display = 'grid';
      panelContent.style.display = 'none';
      Plotly.purge('series-plot');
    }}

    function renderSeriesPanel(ubigeo) {{
      const district = DISTRICT_SERIES[String(ubigeo)];
      if (!district) {{
        return;
      }}
      const currentMetrics = state.currentMetrics || buildCurrentMetrics();
      state.currentMetrics = currentMetrics;
      const scheme = currentMetrics.scheme;
      const klass = currentMetrics.byUbigeo[String(ubigeo)];

      state.selectedUbigeo = String(ubigeo);
      districtPanel.classList.remove('is-hidden');
      panelTitle.textContent = district.distrito;
      panelSubtitle.textContent = district.provincia + ', ' + district.departamento;
      panelPlaceholder.style.display = 'none';
      panelContent.style.display = 'block';

      metricRank.textContent = '#' + formatInteger(klass.rank);
      metricClassLabel.textContent = scheme.singular;
      metricClass.textContent = klass.label;
      metricGrowth.textContent = formatGrowth(klass.growth);
      panelNote.textContent = 'Rango del ' + scheme.singular.toLowerCase() + ': ' + klass.range + '. Serie mostrada para ' + state.startYear + '–' + state.endYear + '; el gráfico incluye toda la trayectoria anual disponible.';

      const values = district.values.map(value => value === null ? null : Number(value));
      const trace = {{
        x: district.years,
        y: values,
        type: 'scatter',
        mode: 'lines+markers',
        connectgaps: false,
        line: {{
          color: '#1f4d1e',
          width: 3,
          shape: 'linear'
        }},
        marker: {{
          size: 7,
          color: '#b78a3b',
          line: {{
            color: '#f8f4ec',
            width: 1.4
          }}
        }},
        hovertemplate: 'Año %{{x}}<br>PBI: %{{y:,.0f}}<extra></extra>'
      }};

      const seriesLayout = {{
        margin: {{ l: 54, r: 14, t: 14, b: 42 }},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: '#fbf8f2',
        font: {{
          family: 'Avenir Next, Segoe UI, Arial, sans-serif',
          color: '#0f172a'
        }},
        xaxis: {{
          title: '',
          tickmode: 'array',
          tickvals: [1993, 1998, 2003, 2008, 2013, 2018],
          tickfont: {{ size: 11 }},
          gridcolor: 'rgba(127, 136, 145, 0.12)',
          zeroline: false,
          linecolor: 'rgba(127, 136, 145, 0.18)'
        }},
        yaxis: {{
          title: 'PBI',
          tickfont: {{ size: 11 }},
          gridcolor: 'rgba(127, 136, 145, 0.14)',
          zeroline: false,
          separatethousands: true
        }},
        hoverlabel: {{
          bgcolor: 'rgba(255,255,255,0.96)',
          bordercolor: '#0f172a',
          font: {{
            color: '#0f172a',
            size: 12
          }}
        }},
        showlegend: false
      }};

      Plotly.react('series-plot', [trace], seriesLayout, {{
        responsive: true,
        displayModeBar: false
      }});
    }}

    function bindMapEvents() {{
      if (mapEventsBound || !mapPlot || typeof mapPlot.on !== 'function') {{
        return;
      }}
      mapPlot.on('plotly_click', function(event) {{
        const point = event && event.points && event.points[0];
        if (!point || !point.customdata) {{
          return;
        }}
        const ubigeo = point.customdata[4];
        renderSeriesPanel(ubigeo);
      }});
      mapEventsBound = true;
    }}

    function initializeInteractiveMap() {{
      if (!window.Plotly || !mapPlot || typeof mapPlot.on !== 'function' || !mapPlot._fullLayout || !mapPlot.data || !mapPlot.data.length) {{
        window.setTimeout(initializeInteractiveMap, 60);
        return;
      }}
      bindMapEvents();
      classificationSelect.value = DEFAULT_MODE;
      colorPresetSelect.value = DEFAULT_PALETTE;
      yearRangeStart.value = state.startYear;
      yearRangeEnd.value = state.endYear;
      state.currentMetrics = buildCurrentMetrics();
      updateYearRangeUI();
      window.requestAnimationFrame(syncLeftOverlayDensity);
      window.setTimeout(syncLeftOverlayDensity, 120);
    }}

    panelClose.addEventListener('click', closePanel);
    classificationSelect.addEventListener('change', function(event) {{
      state.mode = event.target.value;
      applyMapEncoding();
    }});
    colorPresetSelect.addEventListener('change', function(event) {{
      state.palette = event.target.value;
      applyMapEncoding();
    }});
    yearRangeStart.addEventListener('input', function(event) {{
      let nextStart = Number(event.target.value);
      if (nextStart >= state.endYear) {{
        nextStart = state.endYear - 1;
      }}
      state.startYear = Math.max(AVAILABLE_YEARS[0], nextStart);
      yearRangeStart.value = state.startYear;
      updateYearRangeUI();
    }});
    yearRangeEnd.addEventListener('input', function(event) {{
      let nextEnd = Number(event.target.value);
      if (nextEnd <= state.startYear) {{
        nextEnd = state.startYear + 1;
      }}
      state.endYear = Math.min(AVAILABLE_YEARS[AVAILABLE_YEARS.length - 1], nextEnd);
      yearRangeEnd.value = state.endYear;
      updateYearRangeUI();
    }});
    yearRangeStart.addEventListener('change', function() {{
      applyMapEncoding();
    }});
    yearRangeEnd.addEventListener('change', function() {{
      applyMapEncoding();
    }});

    window.addEventListener('resize', syncLeftOverlayDensity);
    if (window.visualViewport) {{
      window.visualViewport.addEventListener('resize', syncLeftOverlayDensity);
      window.visualViewport.addEventListener('scroll', syncLeftOverlayDensity);
    }}
    if (document.fonts && typeof document.fonts.ready === 'object') {{
      document.fonts.ready.then(syncLeftOverlayDensity).catch(function() {{}});
    }}

    initializeInteractiveMap();
  </script>
</body>
</html>
"""


def build_interactive_growth_figure(
    map_gdf: gpd.GeoDataFrame,
    *,
    classification_presets: dict,
    department_bounds: gpd.GeoDataFrame | None = None,
    default_mode: str = DEFAULT_CLASSIFICATION_MODE,
    default_palette: str = DEFAULT_COLOR_PRESET,
) -> go.Figure:
    if {"growth_rank", "rank_label"}.issubset(map_gdf.columns):
        ranked = map_gdf.copy()
    else:
        ranked = prepare_growth_ranking(map_gdf)
    context_layers = _build_context_layers(ranked, department_bounds=department_bounds)
    scheme = classification_presets[default_mode]
    n_classes = int(scheme["classes"])
    palette_colors = _select_palette_steps(COLOR_PRESETS[default_palette]["colors"], n_classes)
    colorscale = _build_discrete_colorscale(palette_colors)
    tick_values = np.arange(1, n_classes + 1)
    tick_text = scheme["tick_text"]
    minx, miny, maxx, maxy = ranked.total_bounds
    default_class_lookup = scheme["by_ubigeo"]
    initial_customdata = np.array(
        [
            [
                row["Distrito"],
                row["Provincia"],
                row["Departamento"],
                int(row["growth_rank"]),
                str(row["ubigeo"]),
                default_class_lookup[str(row["ubigeo"])]["label"],
                default_class_lookup[str(row["ubigeo"])]["range"],
                float(row["avg_growth_9318"]),
            ]
            for _, row in ranked.iterrows()
        ],
        dtype=object,
    )
    hovertemplate = (
        "<b>%{customdata[0]}</b><br>"
        "<span style='color:#475569'>%{customdata[1]}, %{customdata[2]}</span><br>"
        f"{scheme['singular']}: " + "%{customdata[5]}<br>"
        "Rango de clase: %{customdata[6]}<br>"
        "Ranking nacional: %{customdata[3]}<br>"
        "Ubigeo: %{customdata[4]}<br>"
        "Tasa promedio: %{customdata[7]:.4f} log puntos/año"
        "<extra></extra>"
    )

    figure = go.Figure(
        go.Choroplethmap(
            geojson=_geojson_payload(ranked, ["ubigeo"]),
            locations=ranked["ubigeo"],
            z=[default_class_lookup[str(ubigeo)]["index"] for ubigeo in ranked["ubigeo"]],
            featureidkey="properties.ubigeo",
            colorscale=colorscale,
            zmin=0.5,
            zmax=n_classes + 0.5,
            marker={"line": {"color": "rgba(255,255,255,0.58)", "width": 0.26}},
            customdata=initial_customdata,
            hovertemplate=hovertemplate,
            hoverlabel={
                "bgcolor": "rgba(255,255,255,0.96)",
                "bordercolor": "#0f172a",
                "font": {"color": "#0f172a", "size": 13},
                "namelength": -1,
            },
            colorbar={
                "title": {"text": f"{scheme['label']}<br>de crecimiento", "side": "top"},
                "tickvals": tick_values,
                "ticktext": tick_text,
                "thickness": 22,
                "len": 0.62,
                "x": 0.978,
                "xanchor": "right",
                "y": 0.50,
                "outlinecolor": "#d6d3c8",
                "outlinewidth": 0.8,
                "bgcolor": "rgba(255,255,255,0.95)",
                "tickfont": {"size": 12, "color": "#0f172a"},
            },
        )
    )

    figure.update_layout(
        height=920,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="#f3efe6",
        plot_bgcolor="#f3efe6",
        clickmode="event+select",
        font={"family": "Avenir Next, Segoe UI, Arial, sans-serif", "color": "#0f172a"},
        hovermode="closest",
        uirevision="district_avg_growth_interactive",
        map_style="white-bg",
        map_zoom=4.28,
        map_center={"lat": float((miny + maxy) / 2) - 0.22, "lon": float((minx + maxx) / 2) - 0.05},
        map_layers=[
            {
                "source": context_layers["countries_fill"],
                "type": "fill",
                "color": "#efe6d1",
                "opacity": 0.36,
                "below": "traces",
            },
            {
                "source": context_layers["countries_line"],
                "type": "line",
                "color": "#b4b7ba",
                "line": {"width": 1.0},
                "below": "traces",
            },
            {
                "source": context_layers["department_bounds"],
                "type": "line",
                "color": "#304351",
                "line": {"width": 1.95},
            },
            {
                "source": context_layers["peru_outline"],
                "type": "line",
                "color": "#0f172a",
                "line": {"width": 2.3},
            },
        ],
        annotations=[],
    )

    return figure


def build_interactive_growth_artifacts(
    *,
    map_gdf: gpd.GeoDataFrame | None = None,
    department_bounds: gpd.GeoDataFrame | None = None,
    excel_path: Path | None = None,
    geo_path: Path | None = None,
    figures_dir: Path | None = None,
    docs_dir: Path | None = None,
    html_name: str = DEFAULT_HTML_NAME,
) -> dict:
    if map_gdf is None:
        map_gdf = load_district_growth_map(excel_path=excel_path, geo_path=geo_path)

    figures_dir = figures_dir or resolve_existing_path([Path("figures")])
    figures_dir.mkdir(exist_ok=True)
    docs_dir = (docs_dir or DEFAULT_PAGES_DIR).resolve()
    docs_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_growth_ranking(map_gdf)
    classification_presets = build_classification_presets(prepared)
    figure = build_interactive_growth_figure(
        prepared,
        classification_presets=classification_presets,
        department_bounds=department_bounds,
    )
    html_path = figures_dir / html_name

    config = {
        "responsive": True,
        "displaylogo": False,
        "scrollZoom": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": html_path.stem,
            "scale": 2,
        },
        "modeBarButtonsToRemove": [
            "select2d",
            "lasso2d",
            "zoomIn2d",
            "zoomOut2d",
            "autoScale2d",
            "toggleSpikelines",
        ],
    }
    trajectory_payload = _build_trajectory_payload(
        prepared,
        excel_path=excel_path,
    )
    html_string = _build_interactive_dashboard_html(
        figure,
        config=config,
        trajectory_payload=trajectory_payload,
        classification_modes=CLASSIFICATION_MODES,
    )
    html_path.write_text(html_string, encoding="utf-8")
    pages_path = docs_dir / "index.html"
    pages_path.write_text(html_string, encoding="utf-8")
    nojekyll_path = docs_dir / ".nojekyll"
    nojekyll_path.write_text("GitHub Pages: deploy static HTML without Jekyll.\n", encoding="utf-8")

    ordered = prepared.sort_values("growth_rank").reset_index(drop=True)
    top_row = ordered.iloc[0]
    bottom_row = ordered.iloc[-1]
    summary = pd.DataFrame(
        [
            {
                "archivo_html": html_path.name,
                "distritos": len(prepared),
                "min": prepared["avg_growth_9318"].min(),
                "p50": prepared["avg_growth_9318"].median(),
                "max": prepared["avg_growth_9318"].max(),
                "top_1": f"{top_row['Distrito']} | {top_row['Provincia']} | {top_row['Departamento']}",
                "bottom_1": f"{bottom_row['Distrito']} | {bottom_row['Provincia']} | {bottom_row['Departamento']}",
            }
        ]
    )

    return {
        "figure": figure,
        "config": config,
        "html_path": html_path,
        "pages_path": pages_path,
        "nojekyll_path": nojekyll_path,
        "summary": summary,
        "map_df": prepared,
    }


def main() -> None:
    artifacts = build_interactive_growth_artifacts()
    summary = artifacts["summary"].iloc[0]
    print(f"HTML interactivo escrito en: {artifacts['html_path']}")
    print(f"HTML para GitHub Pages escrito en: {artifacts['pages_path']}")
    print(f"Distritos: {int(summary['distritos'])}")
    print(f"Top 1: {summary['top_1']}")
    print(f"Bottom 1: {summary['bottom_1']}")


if __name__ == "__main__":
    main()
