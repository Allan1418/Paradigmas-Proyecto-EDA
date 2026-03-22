"""
data_loader.py — Módulo de ingesta de datos.

Responsabilidad: cargar archivos CSV/Excel en un DataFrame de pandas
y detectar automáticamente el tipo semántico de cada columna
(numérico, categórico, temporal, booleano).
"""

import os
import pandas as pd


def load_file(filepath: str) -> pd.DataFrame:
    """
    Carga un archivo CSV o Excel en un DataFrame de pandas.

    Para CSV intenta múltiples codificaciones (UTF-8, Latin-1, CP1252).
    Para Excel detecta automáticamente la fila de encabezado correcta
    cuando las primeras filas contienen metadatos o títulos.

    Args:
        filepath (str): Ruta absoluta al archivo (CSV, XLSX o XLS).

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.

    Raises:
        ValueError: Si la extensión del archivo no es compatible.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.csv':
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return pd.read_csv(filepath, encoding=encoding)
            except UnicodeDecodeError:
                continue
        # Último recurso: reemplazar caracteres no decodificables
        return pd.read_csv(filepath, encoding='utf-8', errors='replace')

    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath, engine='openpyxl')
        # Detectar si hay filas de metadatos antes del encabezado real
        # (común en Excel con filas de título sobre la tabla de datos)
        for skip in range(1, 6):
            unnamed_ratio = (
                sum(1 for c in df.columns if str(c).startswith('Unnamed:'))
                / max(len(df.columns), 1)
            )
            if unnamed_ratio <= 0.5:
                break
            df2 = pd.read_excel(filepath, engine='openpyxl', header=skip)
            unnamed2 = (
                sum(1 for c in df2.columns if str(c).startswith('Unnamed:'))
                / max(len(df2.columns), 1)
            )
            if unnamed2 < unnamed_ratio:
                df = df2
        return df

    raise ValueError(f"Formato no soportado: '{ext}'. Use CSV, XLSX o XLS.")


def detect_column_types(df: pd.DataFrame) -> dict:
    """
    Detecta automáticamente el tipo semántico de cada columna.

    Orden de detección:
        1. **Temporal** — columnas de tipo ``object`` parseables como fecha.
        2. **Boolean**  — columnas de dtype bool.
        3. **Numeric**  — cualquier dtype numérico de pandas.
        4. **Temporal** — columnas de dtype datetime64.
        5. **Categorical** — todo lo demás.

    Args:
        df (pd.DataFrame): DataFrame a analizar.

    Returns:
        dict: Mapa ``{nombre_columna: tipo_str}`` donde ``tipo_str`` ∈
        ``{'numeric', 'categorical', 'temporal', 'boolean'}``.
    """
    types = {}
    for col in df.columns:
        # pandas ≤2.x usa dtype 'object' para strings;
        # pandas 3.0+ usa dtype 'str'/'string' (StringDtype nativo)
        dtype_str = str(df[col].dtype)
        if dtype_str in ('object', 'str', 'string'):
            try:
                pd.to_datetime(df[col])
                types[col] = 'temporal'
                continue
            except (ValueError, TypeError):
                pass

        if pd.api.types.is_bool_dtype(df[col]):
            types[col] = 'boolean'
        elif pd.api.types.is_numeric_dtype(df[col]):
            types[col] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = 'temporal'
        else:
            types[col] = 'categorical'

    return types
