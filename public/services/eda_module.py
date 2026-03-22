"""
eda_module.py — Módulo de análisis exploratorio descriptivo (EDA).

Responsabilidad: generar estadísticas descriptivas y un resumen
estructurado del DataFrame ya limpiado.
"""

import pandas as pd


def get_summary(df: pd.DataFrame, column_types: dict) -> dict:
    """
    Genera un resumen estadístico completo del DataFrame.

    Incluye conteos globales, estadísticas descriptivas para columnas
    numéricas y distribución de frecuencias para columnas categóricas.

    Args:
        df (pd.DataFrame): DataFrame limpio.
        column_types (dict): Diccionario ``{col: tipo}`` generado por
            ``detect_column_types``.

    Returns:
        dict: Resumen con las claves:
            - ``filas``            (int) — número de filas.
            - ``columnas``         (int) — número de columnas.
            - ``tipos``            (dict) — mapa col → tipo.
            - ``nulos_por_columna`` (dict) — conteo de nulos por col.
            - ``porcentaje_nulos`` (dict) — porcentaje de nulos por col.
            - ``estadisticas``     (dict) — estadísticas por col numérica.
            - ``categorias``       (dict) — top-10 por col categórica.
    """
    summary = {
        'filas': len(df),
        'columnas': len(df.columns),
        'tipos': column_types,
        'nulos_por_columna': df.isnull().sum().to_dict(),
        'porcentaje_nulos': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
    }

    # ── Estadísticas descriptivas numéricas ───────────────────────────────────
    numeric_cols = [c for c, t in column_types.items() if t == 'numeric' and c in df.columns]
    if numeric_cols:
        desc = df[numeric_cols].describe().round(2)
        # Renombrar percentiles para uso fácil en plantillas
        desc = desc.rename(index={'25%': 'q25', '50%': 'q50', '75%': 'q75'})
        summary['estadisticas'] = desc.to_dict()

    # ── Frecuencias categóricas ───────────────────────────────────────────────
    cat_cols = [c for c, t in column_types.items() if t == 'categorical' and c in df.columns]
    summary['categorias'] = {}
    for col in cat_cols:
        vc = df[col].value_counts().head(10)
        summary['categorias'][col] = {
            'total_categorias': int(df[col].nunique()),
            'top_10': vc.to_dict(),
        }

    return summary
