"""
ai_module.py — Módulo de Inteligencia Artificial.

Responsabilidad: aplicar técnicas de Machine Learning para clustering,
detección de valores atípicos, análisis de correlación e importancia
de variables.

Algoritmos implementados
------------------------
Clustering     : K-Means (principal), DBSCAN (alternativo)
Outliers       : IQR (principal), Z-score, Isolation Forest
Correlaciones  : Pearson
Importancia    : Random Forest (feature_importances_)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from scipy import stats


# ── Clustering ────────────────────────────────────────────────────────────────

def run_clustering(df: pd.DataFrame, column_types: dict, n_clusters: int = 3) -> dict:
    """
    Ejecuta K-Means y DBSCAN sobre las columnas numéricas del DataFrame.

    K-Means es el método principal (grupos bien definidos y de tamaño
    similar). DBSCAN complementa detectando grupos de densidad variable
    y puntos de ruido sin necesidad de especificar el número de clusters.

    Se aplica PCA a 2 dimensiones para la visualización del dispersograma.

    Args:
        df (pd.DataFrame): DataFrame limpio.
        column_types (dict): Mapa ``{col: tipo}`` de ``detect_column_types``.
        n_clusters (int): Número de grupos para K-Means. Defecto: 3.

    Returns:
        dict: Con claves ``exito``, ``metodo``, ``n_clusters``, ``labels``,
        ``x_2d``, ``y_2d``, ``columnas_usadas``, ``resumen_clusters``,
        ``insight`` y ``dbscan``. Si falla, retorna ``{'exito': False, ...}``.
    """
    numeric_cols = [c for c, t in column_types.items() if t == 'numeric' and c in df.columns]

    if len(numeric_cols) < 2:
        return {'exito': False, 'mensaje': 'Se necesitan al menos 2 columnas numéricas.'}

    X = df[numeric_cols].dropna()
    if len(X) < n_clusters:
        return {'exito': False, 'mensaje': 'No hay suficientes filas para clustering.'}

    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ── K-Means ───────────────────────────────────────────────────────────
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

        # ── PCA 2D para dispersograma ─────────────────────────────────────────
        n_components = min(2, X_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_2d = pca.fit_transform(X_scaled)
        x_2d = X_2d[:, 0].tolist()
        y_2d = (X_2d[:, 1].tolist() if X_2d.shape[1] > 1 else [0.0] * len(X_2d))

        # ── Resumen por cluster ───────────────────────────────────────────────
        df_result = X.copy()
        df_result['cluster'] = labels
        resumen_clusters = {}
        for cid in sorted(set(labels)):
            grupo = df_result[df_result['cluster'] == cid]
            resumen_clusters[f'Grupo {cid + 1}'] = {
                'cantidad': len(grupo),
                'porcentaje': round(len(grupo) / len(df_result) * 100, 1),
                'promedios': grupo[numeric_cols].mean().round(2).to_dict(),
            }

        cols_text = ', '.join(numeric_cols[:3])
        insight = (
            f"Se detectaron {n_clusters} agrupaciones en los datos según {cols_text}. "
            f"El grupo más grande contiene "
            f"{max(v['cantidad'] for v in resumen_clusters.values())} registros."
        )

        # ── DBSCAN ────────────────────────────────────────────────────────────
        dbscan_result = _run_dbscan(X_scaled)

        return {
            'exito': True,
            'metodo': 'K-Means',
            'n_clusters': n_clusters,
            'labels': labels.tolist(),
            'x_2d': x_2d,
            'y_2d': y_2d,
            'columnas_usadas': numeric_cols,
            'resumen_clusters': resumen_clusters,
            'insight': insight,
            'dbscan': dbscan_result,
        }

    except Exception as e:
        return {'exito': False, 'mensaje': str(e)}


def _run_dbscan(X_scaled: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> dict:
    """
    Ejecuta DBSCAN sobre datos ya normalizados (StandardScaler).

    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    no requiere especificar el número de clusters; detecta grupos por
    densidad y clasifica puntos aislados como ruido (etiqueta -1).

    Args:
        X_scaled (np.ndarray): Matriz normalizada.
        eps (float): Radio de vecindad. Defecto: 0.5.
        min_samples (int): Mínimo de puntos para formar un cluster. Defecto: 5.

    Returns:
        dict: Con ``exito``, ``n_clusters``, ``n_ruido`` e ``insight``.
    """
    try:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_ruido = int(np.sum(labels == -1))
        insight = (
            f"DBSCAN identificó {n_clusters} grupos por densidad "
            f"({n_ruido} punto(s) clasificado(s) como ruido)."
        )
        return {'exito': True, 'n_clusters': n_clusters, 'n_ruido': n_ruido, 'insight': insight}
    except Exception as e:
        return {'exito': False, 'mensaje': str(e)}


# ── Detección de outliers ─────────────────────────────────────────────────────

def detect_outliers(df: pd.DataFrame, column_types: dict) -> dict:
    """
    Detecta valores atípicos usando tres métodos complementarios.

    Métodos
    -------
    - **IQR** (Rango Intercuartílico): límites ``[Q1 − 1.5·IQR, Q3 + 1.5·IQR]``.
      Robusto ante distribuciones asimétricas. Resultado principal.
    - **Z-score**: ``|z| > 3``. Asume distribución aproximadamente normal.
    - **Isolation Forest**: aprendizaje no supervisado; eficaz en alta dimensión.

    Los resultados de IQR son el resultado principal para mantener
    compatibilidad con el dashboard. Z-score e Isolation Forest se
    incluyen en la clave ``zscore`` e ``isolation_forest`` respectivamente.

    Args:
        df (pd.DataFrame): DataFrame limpio.
        column_types (dict): Mapa ``{col: tipo}``.

    Returns:
        dict: Con ``exito``, ``metodo``, ``total_outliers``,
        ``porcentaje_total``, ``por_columna``, ``insight``,
        ``zscore`` e ``isolation_forest``.
    """
    numeric_cols = [c for c, t in column_types.items() if t == 'numeric' and c in df.columns]

    if not numeric_cols:
        return {'exito': False, 'mensaje': 'No se encontraron columnas numéricas.'}

    X = df[numeric_cols].dropna()
    iqr_result = _detect_iqr(X, numeric_cols)
    zscore_result = _detect_zscore(X, numeric_cols)
    isof_result = _detect_isolation_forest(X, numeric_cols)

    total = iqr_result['total']
    pct = round(total / max(len(X), 1) * 100, 1)

    col_max = max(iqr_result['por_columna'], key=lambda c: iqr_result['por_columna'][c]['cantidad'])
    insight = (
        f"El {pct}% de los registros presentan valores atípicos (IQR), "
        f"especialmente en '{col_max}'. "
        f"Z-score detectó {zscore_result['total']} atípicos; "
        f"Isolation Forest detectó {isof_result['total']}."
    )

    return {
        'exito': True,
        'metodo': 'IQR',
        'total_outliers': total,
        'porcentaje_total': pct,
        'por_columna': iqr_result['por_columna'],
        'insight': insight,
        'zscore': zscore_result,
        'isolation_forest': isof_result,
    }


def _detect_iqr(X: pd.DataFrame, numeric_cols: list) -> dict:
    """
    Detección de outliers mediante Rango Intercuartílico (IQR).

    Args:
        X (pd.DataFrame): Subconjunto numérico sin nulos.
        numeric_cols (list): Columnas a analizar.

    Returns:
        dict: ``total`` (int) y ``por_columna`` (dict).
    """
    por_columna = {}
    total = 0
    for col in numeric_cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask = (X[col] < lower) | (X[col] > upper)
        n = int(mask.sum())
        total += n
        por_columna[col] = {
            'cantidad': n,
            'porcentaje': round(n / len(X) * 100, 2),
            'rango_normal': f"[{round(lower, 2)}, {round(upper, 2)}]",
        }
    return {'total': total, 'por_columna': por_columna}


def _detect_zscore(X: pd.DataFrame, numeric_cols: list, threshold: float = 3.0) -> dict:
    """
    Detección de outliers mediante Z-score (|z| > threshold).

    Args:
        X (pd.DataFrame): Subconjunto numérico sin nulos.
        numeric_cols (list): Columnas a analizar.
        threshold (float): Umbral de desviaciones estándar. Defecto: 3.0.

    Returns:
        dict: ``total`` (int), ``detalle`` (list) y ``umbral`` (float).
    """
    total = 0
    detalle = []
    for col in numeric_cols:
        z = np.abs(stats.zscore(X[col].values))
        n = int((z > threshold).sum())
        total += n
        detalle.append({'columna': col, 'cantidad': n})
    return {'total': total, 'detalle': detalle, 'umbral': threshold}


def _detect_isolation_forest(X: pd.DataFrame, numeric_cols: list,
                              contamination: float = 0.05) -> dict:
    """
    Detección de outliers mediante Isolation Forest.

    Isolation Forest entrena un bosque de árboles de aislamiento;
    los puntos que requieren menos particiones para ser aislados
    se clasifican como atípicos.

    Args:
        X (pd.DataFrame): Subconjunto numérico sin nulos.
        numeric_cols (list): Columnas a analizar.
        contamination (float): Fracción esperada de outliers. Defecto: 0.05.

    Returns:
        dict: ``total`` (int) y ``contaminacion`` (float).
    """
    try:
        if len(X) < 10:
            return {'total': 0, 'contaminacion': contamination, 'error': 'Insuficientes datos'}
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(X[numeric_cols].values)
        n_out = int((preds == -1).sum())
        return {'total': n_out, 'contaminacion': contamination}
    except Exception as e:
        return {'total': 0, 'contaminacion': contamination, 'error': str(e)}


# ── Correlaciones ─────────────────────────────────────────────────────────────

def compute_correlations(df: pd.DataFrame, column_types: dict) -> dict:
    """
    Calcula la matriz de correlación de Pearson entre variables numéricas.

    El coeficiente de Pearson mide la relación lineal entre dos variables
    continuas; su valor oscila entre −1 (correlación negativa perfecta)
    y +1 (correlación positiva perfecta).

    Args:
        df (pd.DataFrame): DataFrame limpio.
        column_types (dict): Mapa ``{col: tipo}``.

    Returns:
        dict: Con ``exito``, ``pearson`` (matriz), ``top_correlaciones``
        (top-10 pares), ``insights`` (lista de strings) y ``columnas``.
        Si falla, retorna ``{'exito': False, 'mensaje': str}``.
    """
    numeric_cols = [c for c, t in column_types.items() if t == 'numeric' and c in df.columns]

    if len(numeric_cols) < 2:
        return {'exito': False, 'mensaje': 'Se necesitan al menos 2 columnas numéricas.'}

    try:
        corr_matrix = df[numeric_cols].corr(method='pearson').round(3)

        top_correlations = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:
                    r = float(corr_matrix.loc[col1, col2])
                    top_correlations.append({
                        'variable_1': col1,
                        'variable_2': col2,
                        'pearson': round(r, 3),
                        'fuerza': _classify_correlation(abs(r)),
                    })

        top_correlations.sort(key=lambda x: abs(x['pearson']), reverse=True)

        insights = []
        for corr in top_correlations[:3]:
            r = corr['pearson']
            direction = 'positiva' if r > 0 else 'negativa'
            insights.append(
                f"Correlación {corr['fuerza']} {direction} (r={r}) entre "
                f"'{corr['variable_1']}' y '{corr['variable_2']}'."
            )

        return {
            'exito': True,
            'pearson': corr_matrix.to_dict(),
            'top_correlaciones': top_correlations[:10],
            'insights': insights,
            'columnas': numeric_cols,
        }

    except Exception as e:
        return {'exito': False, 'mensaje': str(e)}


def _classify_correlation(r: float) -> str:
    """Clasifica la fuerza de una correlación por su valor absoluto."""
    if r >= 0.8:   return 'muy fuerte'
    elif r >= 0.6: return 'fuerte'
    elif r >= 0.4: return 'moderada'
    elif r >= 0.2: return 'débil'
    else:          return 'muy débil'


# ── Importancia de variables ──────────────────────────────────────────────────

def compute_feature_importance(df: pd.DataFrame, column_types: dict) -> dict:
    """
    Estima la importancia de cada variable usando Random Forest.

    Para cada columna numérica como variable objetivo, entrena un
    ``RandomForestRegressor`` con las demás columnas numéricas como
    predictores y extrae ``feature_importances_``. Se genera la frase:
    "'{X}' explica el {Y}% de la variación en '{Z}' (R²={R}%)."

    Args:
        df (pd.DataFrame): DataFrame limpio.
        column_types (dict): Mapa ``{col: tipo}``.

    Returns:
        dict: Con ``exito``, ``resultados`` (list ordenada por R² desc)
        e ``insights`` (list de strings). Si falla o no hay datos
        suficientes, retorna ``{'exito': False, 'razon': str}``.
    """
    numeric_cols = [c for c, t in column_types.items() if t == 'numeric' and c in df.columns]

    if len(numeric_cols) < 2:
        return {'exito': False, 'razon': 'Se necesitan al menos 2 columnas numéricas.'}

    try:
        df_num = df[numeric_cols].dropna()
        if len(df_num) < 10:
            return {'exito': False, 'razon': 'Insuficientes filas para calcular importancia.'}

        resultados = []
        for target in numeric_cols:
            features = [c for c in numeric_cols if c != target]
            X = df_num[features].values
            y = df_num[target].values

            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            r2 = max(0.0, float(rf.score(X, y)))

            importancias = dict(zip(features, rf.feature_importances_.tolist()))
            top_feat = max(importancias, key=importancias.get)
            top_pct = round(importancias[top_feat] * 100, 1)

            resultados.append({
                'objetivo': target,
                'top_variable': top_feat,
                'importancia_pct': top_pct,
                'r2_pct': round(r2 * 100, 1),
                'importancias': {k: round(v * 100, 1) for k, v in importancias.items()},
            })

        resultados.sort(key=lambda x: x['r2_pct'], reverse=True)

        insights = [
            f"'{r['top_variable']}' explica el {r['importancia_pct']}% "
            f"de la variación en '{r['objetivo']}' (R²={r['r2_pct']}%)."
            for r in resultados[:3]
        ]

        return {'exito': True, 'resultados': resultados, 'insights': insights}

    except Exception as e:
        return {'exito': False, 'razon': str(e)}


# ── Orquestador principal ─────────────────────────────────────────────────────

def run_full_analysis(df: pd.DataFrame, column_types: dict) -> dict:
    """
    Ejecuta todos los análisis de IA sobre el DataFrame.

    Orquesta clustering (K-Means + DBSCAN), detección de outliers
    (IQR + Z-score + Isolation Forest), correlaciones de Pearson
    e importancia de variables (Random Forest). Consolida todos los
    hallazgos en una lista de insights estructurados.

    Args:
        df (pd.DataFrame): DataFrame limpio.
        column_types (dict): Mapa ``{col: tipo}``.

    Returns:
        dict: Con claves ``clustering``, ``outliers``, ``correlaciones``,
        ``feature_importance`` e ``insights``.
    """
    results = {
        'clustering':         run_clustering(df, column_types),
        'outliers':           detect_outliers(df, column_types),
        'correlaciones':      compute_correlations(df, column_types),
        'feature_importance': compute_feature_importance(df, column_types),
    }

    insights = []

    if results['clustering'].get('exito'):
        insights.append({
            'tipo': 'analisis', 'titulo': 'Agrupamientos',
            'texto': results['clustering']['insight'],
        })
        dbscan = results['clustering'].get('dbscan', {})
        if dbscan.get('exito'):
            insights.append({
                'tipo': 'info', 'titulo': 'DBSCAN',
                'texto': dbscan['insight'],
            })

    if results['outliers'].get('exito'):
        tipo = 'critico' if results['outliers']['porcentaje_total'] > 5 else 'info'
        insights.append({
            'tipo': tipo, 'titulo': 'Valores Atípicos',
            'texto': results['outliers']['insight'],
        })

    if results['correlaciones'].get('exito'):
        for ins in results['correlaciones']['insights']:
            insights.append({'tipo': 'info', 'titulo': 'Correlación', 'texto': ins})

    if results['feature_importance'].get('exito'):
        for ins in results['feature_importance']['insights']:
            insights.append({'tipo': 'analisis', 'titulo': 'Importancia de Variables', 'texto': ins})

    results['insights'] = insights
    return results
