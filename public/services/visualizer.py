"""
visualizer.py — Módulo de visualizaciones interactivas.

Responsabilidad: generar gráficos Plotly en formato HTML embebible
para el dashboard y el reporte exportable.

Gráficos disponibles
--------------------
- Mapa de calor de correlaciones (Pearson)
- Diagramas de caja (boxplots)
- Histogramas de frecuencia
- Dispersograma de clusters (K-Means)
- Barras de outliers por variable
- Barras de variables categóricas
- Barras de importancia de variables (Random Forest)
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd


COLORS = ['#10b981', '#3b82f6', '#8b5cf6', '#ef4444', '#f59e0b', '#06b6d4']

LAYOUT_BASE = dict(
    paper_bgcolor='#ffffff',
    plot_bgcolor='#f8fafc',
    font=dict(family='Arial, sans-serif', color='#1e293b', size=12),
    margin=dict(l=50, r=30, t=50, b=50),
    xaxis=dict(gridcolor='#e2e8f0', zerolinecolor='#e2e8f0'),
    yaxis=dict(gridcolor='#e2e8f0', zerolinecolor='#e2e8f0'),
)


def correlation_heatmap(corr_data: dict) -> str:
    """
    Genera un mapa de calor de la matriz de correlación de Pearson.

    Args:
        corr_data (dict): Resultado de ``compute_correlations``.
            Debe contener ``exito``, ``pearson`` y ``columnas``.

    Returns:
        str: HTML embebible o cadena vacía si no hay datos válidos.
    """
    if not corr_data.get('exito'):
        return None

    corr_matrix = pd.DataFrame(corr_data['pearson'])
    cols = corr_data['columnas']

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=cols,
        y=cols,
        colorscale=[[0, '#ef4444'], [0.5, '#ffffff'], [1, '#10b981']],
        zmin=-1, zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont=dict(size=11),
        colorbar=dict(title='r'),
    ))

    fig.update_layout(**LAYOUT_BASE, title='Matriz de Correlación (Pearson)', height=450)
    return fig


def boxplots(df: pd.DataFrame, column_types: dict) -> str:
    """
    Genera diagramas de caja para las variables numéricas.

    Muestra hasta 6 columnas en una cuadrícula de 3 columnas.
    Incluye media (boxmean=True) además de la mediana.

    Args:
        df (pd.DataFrame): DataFrame limpio.
        column_types (dict): Mapa ``{col: tipo}``.

    Returns:
        str: HTML embebible o cadena vacía si no hay columnas numéricas.
    """
    numeric_cols = [c for c, t in column_types.items() if t == 'numeric' and c in df.columns]
    if not numeric_cols:
        return None

    cols_to_plot = numeric_cols[:6]
    n_cols = min(3, len(cols_to_plot))
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cols_to_plot)

    for i, col in enumerate(cols_to_plot):
        fig.add_trace(
            go.Box(y=df[col].dropna(), name=col,
                   marker_color=COLORS[i % len(COLORS)], boxmean=True),
            row=i // n_cols + 1, col=i % n_cols + 1
        )

    fig.update_layout(**LAYOUT_BASE, title='Diagramas de Caja',
                      height=320 * n_rows, showlegend=False)
    return fig


def histograms(df: pd.DataFrame, column_types: dict) -> str:
    """
    Genera histogramas de distribución para las variables numéricas.

    Muestra hasta 6 columnas en una cuadrícula de 3 columnas.

    Args:
        df (pd.DataFrame): DataFrame limpio.
        column_types (dict): Mapa ``{col: tipo}``.

    Returns:
        str: HTML embebible o cadena vacía si no hay columnas numéricas.
    """
    numeric_cols = [c for c, t in column_types.items() if t == 'numeric' and c in df.columns]
    if not numeric_cols:
        return None

    cols_to_plot = numeric_cols[:6]
    n_cols = min(3, len(cols_to_plot))
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cols_to_plot)

    for i, col in enumerate(cols_to_plot):
        fig.add_trace(
            go.Histogram(x=df[col].dropna(), name=col,
                         marker_color=COLORS[i % len(COLORS)]),
            row=i // n_cols + 1, col=i % n_cols + 1
        )

    fig.update_layout(**LAYOUT_BASE, title='Histogramas — Distribución de Frecuencias',
                      height=320 * n_rows, showlegend=False)
    return fig


def cluster_scatter(cluster_data: dict) -> str:
    """
    Genera un dispersograma 2D con los grupos detectados por K-Means.

    Las coordenadas son componentes PCA calculadas en ``run_clustering``.
    Cada grupo se colorea de forma distinta; DBSCAN no se grafica aquí.

    Args:
        cluster_data (dict): Resultado de ``run_clustering``.
            Debe contener ``exito``, ``labels``, ``x_2d`` e ``y_2d``.

    Returns:
        str: HTML embebible o cadena vacía si no hay datos de clustering.
    """
    if not cluster_data.get('exito'):
        return None

    labels = cluster_data['labels']
    x = cluster_data['x_2d']
    y = cluster_data['y_2d']

    fig = go.Figure()
    for label in sorted(set(labels)):
        mask = [i for i, l in enumerate(labels) if l == label]
        name = f"Grupo {label + 1}" if label >= 0 else "Sin grupo"
        color = COLORS[label % len(COLORS)] if label >= 0 else '#94a3b8'
        fig.add_trace(go.Scatter(
            x=[x[i] for i in mask], y=[y[i] for i in mask],
            mode='markers', name=name,
            marker=dict(size=6, color=color, opacity=0.7),
        ))

    fig.update_layout(**LAYOUT_BASE, title='Grupos Detectados (K-Means)',
                      xaxis_title='Componente 1', yaxis_title='Componente 2', height=420)
    return fig


def outlier_chart(_df: pd.DataFrame, _column_types: dict, outlier_data: dict) -> str:
    """
    Genera un gráfico de barras con la cantidad de outliers por variable (IQR).

    Las barras se colorean en verde (<2%), ámbar (2–5%) o rojo (>5%).

    Args:
        _df (pd.DataFrame): Reservado para compatibilidad con la API del dashboard.
        _column_types (dict): Reservado para compatibilidad con la API del dashboard.
        outlier_data (dict): Resultado de ``detect_outliers``.
            Debe contener ``exito``, ``por_columna`` y ``metodo``.

    Returns:
        str: HTML embebible o cadena vacía si no hay datos de outliers.
    """
    if not outlier_data.get('exito'):
        return None

    por_columna = outlier_data['por_columna']
    cols = list(por_columna.keys())
    counts = [por_columna[c]['cantidad'] for c in cols]
    pcts = [por_columna[c]['porcentaje'] for c in cols]

    bar_colors = ['#ef4444' if p > 5 else '#f59e0b' if p > 2 else '#10b981' for p in pcts]

    fig = go.Figure(data=go.Bar(
        x=cols, y=counts, marker_color=bar_colors,
        text=[f"{p}%" for p in pcts], textposition='outside',
    ))

    fig.update_layout(**LAYOUT_BASE,
                      title=f'Valores Atípicos por Variable (método {outlier_data["metodo"]})',
                      xaxis_title='Variable', yaxis_title='Cantidad', height=380)
    return fig


def category_bar_chart(df: pd.DataFrame, column_types: dict) -> str:
    """
    Genera barras de frecuencia para las variables categóricas.

    Muestra hasta 4 columnas con las 8 categorías más frecuentes cada una.

    Args:
        df (pd.DataFrame): DataFrame limpio.
        column_types (dict): Mapa ``{col: tipo}``.

    Returns:
        str: HTML embebible o cadena vacía si no hay columnas categóricas.
    """
    cat_cols = [c for c, t in column_types.items() if t == 'categorical' and c in df.columns]
    if not cat_cols:
        return None

    cols_to_plot = cat_cols[:4]
    n_cols = min(2, len(cols_to_plot))
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cols_to_plot)

    for i, col in enumerate(cols_to_plot):
        vc = df[col].value_counts().head(8)
        fig.add_trace(
            go.Bar(x=vc.index.astype(str), y=vc.values, name=col,
                   marker_color=COLORS[i % len(COLORS)]),
            row=i // n_cols + 1, col=i % n_cols + 1
        )

    fig.update_layout(**LAYOUT_BASE, title='Variables Categóricas',
                      height=320 * n_rows, showlegend=False)
    return fig


def feature_importance_chart(fi_data: dict) -> str:
    """
    Genera un gráfico de barras horizontales con la importancia de variables.

    Muestra los predictores del objetivo con mayor R² (la relación más
    explicable del dataset) ordenados de mayor a menor importancia.

    Args:
        fi_data (dict): Resultado de ``compute_feature_importance``.
            Debe contener ``exito`` y ``resultados``.

    Returns:
        str: HTML embebible o cadena vacía si no hay datos válidos.
    """
    if not fi_data.get('exito') or not fi_data.get('resultados'):
        return None

    # Usar el resultado con mayor R² para el gráfico
    top_result = fi_data['resultados'][0]
    importancias = top_result['importancias']

    features = list(importancias.keys())
    values = [importancias[f] for f in features]

    # Ordenar de mayor a menor
    sorted_pairs = sorted(zip(values, features), reverse=True)
    values_sorted = [v for v, _ in sorted_pairs]
    features_sorted = [f for _, f in sorted_pairs]

    fig = go.Figure(data=go.Bar(
        x=values_sorted,
        y=features_sorted,
        orientation='h',
        marker_color=COLORS[0],
        text=[f"{v}%" for v in values_sorted],
        textposition='outside',
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Importancia de Variables para '{top_result['objetivo']}' (R²={top_result['r2_pct']}%)",
        xaxis_title='Importancia (%)',
        yaxis_title='Variable',
        height=max(300, 60 * len(features_sorted) + 100),
    )
    return fig
