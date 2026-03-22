"""
data_validator.py — Módulo de limpieza e imputación de datos.

Responsabilidad: eliminar duplicados, columnas vacías e imputar
valores nulos con estrategias estadísticas apropiadas.

Estrategia de imputación
------------------------
- Variables numéricas  → mediana  (robusta ante valores atípicos)
- Variables categóricas → moda    (categoría más frecuente)
"""

import pandas as pd


def clean_data(df: pd.DataFrame) -> tuple:
    """
    Limpia el DataFrame aplicando imputación y eliminación de ruido.

    Pasos en orden:
        1. Eliminar filas completamente duplicadas.
        2. Eliminar columnas donde el 100 % de los valores son nulos.
        3. Imputar nulos numéricos con la mediana de cada columna.
        4. Imputar nulos categóricos con la moda de cada columna.

    Args:
        df (pd.DataFrame): DataFrame original. No se modifica in-place.

    Returns:
        tuple: ``(df_limpio, reporte)`` donde *reporte* es un dict con:
            - ``filas_originales``          (int)
            - ``columnas_originales``       (int)
            - ``duplicados_eliminados``     (int)
            - ``columnas_vacias_eliminadas`` (list[str])
            - ``valores_nulos_rellenados``  (int)
            - ``filas_finales``             (int)
            - ``columnas_finales``          (int)
    """
    df = df.copy()

    report = {
        'filas_originales': len(df),
        'columnas_originales': len(df.columns),
        'duplicados_eliminados': 0,
        'columnas_vacias_eliminadas': [],
        'valores_nulos_rellenados': 0,
    }

    # ── 1. Duplicados ─────────────────────────────────────────────────────────
    n_before = len(df)
    df = df.drop_duplicates()
    report['duplicados_eliminados'] = n_before - len(df)

    # ── 2. Columnas completamente vacías ──────────────────────────────────────
    empty_cols = df.columns[df.isnull().all()].tolist()
    df = df.drop(columns=empty_cols)
    report['columnas_vacias_eliminadas'] = empty_cols

    # ── 3 & 4. Imputación ─────────────────────────────────────────────────────
    null_count = 0
    for col in df.columns:
        n_nulls = int(df[col].isnull().sum())
        if n_nulls == 0:
            continue
        null_count += n_nulls
        if pd.api.types.is_numeric_dtype(df[col]):
            # Imputar con mediana (robusta ante outliers)
            df[col] = df[col].fillna(df[col].median())
        else:
            # Imputar con moda (valor más frecuente)
            moda = df[col].mode()
            fill_val = moda.iloc[0] if not moda.empty else 'Sin dato'
            df[col] = df[col].fillna(fill_val)

    report['valores_nulos_rellenados'] = null_count
    report['filas_finales'] = len(df)
    report['columnas_finales'] = len(df.columns)

    return df, report
