import asyncio
import os
import warnings
from pyscript import document, display, window
from pyodide.ffi import create_proxy
from js import Uint8Array, Blob, URL, exportar_a_pdf_js
import micropip

# ==========================================
# SILENCIAR ADVERTENCIAS
# ==========================================
warnings.filterwarnings("ignore")

# ==========================================
# OCULTAR PANTALLA DE CARGA INICIAL
# ==========================================
document.getElementById("pyscript-loading").classList.add("hidden")


# ==========================================
# DESCARGA DE LIBRERIAS EN SEGUNDO PLANO
# ==========================================
async def cargar_librerias_fondo():
    await asyncio.sleep(1) 
    try:
        print("Iniciando descarga asincronica de librerias...")
        await micropip.install(["pandas", "scikit-learn", "scipy", "plotly", "openpyxl", "Jinja2"])
        print("Librerias listas y en cache.")
    except Exception as e:
        print(f"Error en descarga de fondo: {e}")

tarea_instalacion = asyncio.create_task(cargar_librerias_fondo())


# ==========================================
# VARIABLES GLOBALES
# ==========================================
archivo_seleccionado = None
estado_global = {}

# ==========================================
# MANEJO DE LA INTERFAZ
# ==========================================
def manejar_seleccion_archivo(evento):
    global archivo_seleccionado
    
    if hasattr(evento, 'dataTransfer') and evento.dataTransfer.files.length > 0:
        archivo = evento.dataTransfer.files.item(0)
    elif hasattr(evento.target, 'files') and evento.target.files.length > 0:
        archivo = evento.target.files.item(0)
    else:
        return
    
    nombre_archivo = archivo.name.lower()
    tipo_mime = archivo.type
    
    mimes_validos = [
        'text/csv', 
        'application/vnd.ms-excel', 
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    ]
    
    es_valido = False
    if tipo_mime in mimes_validos:
        es_valido = True
    elif nombre_archivo.endswith('.csv') or nombre_archivo.endswith('.xlsx') or nombre_archivo.endswith('.xls'):
        es_valido = True

    if not es_valido:
        window.alert(f"Formato no válido: {archivo.name}\n\nPor favor, sube únicamente archivos de datos (.csv, .xlsx, .xls).")
        return

    archivo_seleccionado = archivo
    
    display_name = document.getElementById("file-name-display")
    display_name.innerText = f"Archivo cargado: {archivo.name}"
    display_name.classList.remove("hidden")
    
    document.getElementById("btn-procesar").classList.remove("hidden")

def manejar_dragover(evento):
    evento.preventDefault()
    document.getElementById("drop-zone").classList.add("bg-blue-50", "border-blue-500")

def manejar_dragleave(evento):
    evento.preventDefault()
    document.getElementById("drop-zone").classList.remove("bg-blue-50", "border-blue-500")

def manejar_drop(evento):
    evento.preventDefault()
    document.getElementById("drop-zone").classList.remove("bg-blue-50", "border-blue-500")
    manejar_seleccion_archivo(evento)


# ==========================================
# FILTRO PARA ARREGLAR MARGENES Y TEXTOS SUPERPUESTOS
# ==========================================
def optimizar_grafico(fig):
    if fig is None: return None
    
    # 1. Inteligencia básica: Inspeccionar los datos del eje X
    necesita_rotacion = False
    
    # Plotly guarda las líneas/barras del gráfico en 'fig.data'
    if fig.data:
        for trace in fig.data:
            # Revisamos si el gráfico tiene un eje X
            if hasattr(trace, 'x') and trace.x is not None:
                for etiqueta in trace.x:
                    # Si la etiqueta es TEXTO (no un número) y tiene más de 10 letras
                    if isinstance(etiqueta, str) and len(etiqueta) > 10:
                        necesita_rotacion = True
                        break # Al encontrar uno largo, dejamos de buscar
            if necesita_rotacion:
                break
    
    # 2. Aplicar márgenes: Damos más espacio abajo (b=120) SOLO si vamos a rotar
    margen_inferior = 120 if necesita_rotacion else 50
    fig.update_layout(margin=dict(t=90, l=150, b=margen_inferior, r=20))
    
    # 3. Aplicar rotación condicional
    if necesita_rotacion:
        fig.update_xaxes(tickangle=-45, automargin=True)
    else:
        fig.update_xaxes(tickangle=0, automargin=True)
        
    fig.update_yaxes(automargin=True)
    
    if hasattr(fig.layout, 'annotations') and fig.layout.annotations:
        for ann in fig.layout.annotations:
            if hasattr(ann, 'text') and ann.text and isinstance(ann.text, str):
                # Si el título tiene más de 14 caracteres, lo truncamos
                if len(ann.text) > 14:
                    ann.text = ann.text[:12] + "..."
    
    return fig


# ==========================================
# LOGICA PRINCIPAL DEL BOTON
# ==========================================
async def procesar_archivo(evento):
    global archivo_seleccionado
    if not archivo_seleccionado:
        window.alert("Por favor selecciona un archivo primero.")
        return

    document.getElementById("btn-procesar").classList.add("hidden")
    document.getElementById("loading-indicator").classList.remove("hidden")
    document.getElementById("results-container").classList.add("hidden")
    
    await asyncio.sleep(0.1) 
    
    try:
        # ==========================================
        # PUNTO DE SINCRONIZACION
        # ==========================================
        await tarea_instalacion
        
        # Importamos los servicios matematicos
        from services.data_loader import load_file, detect_column_types
        from services.data_validator import clean_data
        from services.eda_module import get_summary
        from services.ai_module import run_full_analysis
        from services import visualizer
        import random
        
        async def actualizar_estado(mensaje):
            document.getElementById("loading-progress").innerText = mensaje
            await asyncio.sleep(0.1)
        
        # Escribir archivo al disco duro fantasma
        nombre_archivo = archivo_seleccionado.name
        array_buffer = await archivo_seleccionado.arrayBuffer()
        arreglo_bytes = bytearray(Uint8Array.new(array_buffer))
        
        ruta_virtual = f"/{nombre_archivo}"
        with open(ruta_virtual, "wb") as f:
            f.write(arreglo_bytes)
            
        
        # Ejecutar logica matematica con actualizaciones de estado
        await actualizar_estado("Leyendo el archivo en memoria...")
        df = load_file(ruta_virtual)
        
        await actualizar_estado("Detectando tipos de variables...")
        column_types = detect_column_types(df)
        
        await actualizar_estado("Limpiando datos e imputando valores...")
        df, cleaning_report = clean_data(df)
        
        await actualizar_estado("Calculando estadísticas descriptivas...")
        summary = get_summary(df, column_types)
        
        
        await actualizar_estado("Preparando muestra para modelos predictivos...")
        LIMITE_IA = 5000
        if len(df) > LIMITE_IA:
            df_ia = df.sample(n=LIMITE_IA, random_state=42)
        else:
            df_ia = df
            
        await actualizar_estado("Entrenando Machine Learning...")
        analysis = run_full_analysis(df_ia, column_types)
        
        estado_global["df"] = df
        estado_global["column_types"] = column_types
        estado_global["summary"] = summary
        estado_global["analysis"] = analysis
        estado_global["filename"] = nombre_archivo
        estado_global["cleaning_report"] = cleaning_report
        
        
        # Inyectar Resultados de Texto
        await actualizar_estado("Generando reporte visual...")
        renderizar_resumen(summary)
        renderizar_insights(analysis)
        
        
        # limitacion de la cantidad de puntos en graficos
        MAX_PUNTOS = 3000
        if len(df) > MAX_PUNTOS:
            df_graficos = df.sample(n=MAX_PUNTOS, random_state=42)
        else:
            df_graficos = df
            
        cluster_data = analysis.get('clustering', {})
        if cluster_data and cluster_data.get('exito') and len(cluster_data.get('labels', [])) > MAX_PUNTOS:
            indices = random.sample(range(len(cluster_data['labels'])), MAX_PUNTOS)
            cluster_graficos = {
                'exito': True,
                'labels': [cluster_data['labels'][i] for i in indices],
                'x_2d': [cluster_data['x_2d'][i] for i in indices],
                'y_2d': [cluster_data['y_2d'][i] for i in indices]
            }
        else:
            cluster_graficos = cluster_data
        
        estado_global["df_graficos"] = df_graficos
        estado_global["cluster_graficos"] = cluster_graficos
        
        # Mostrar panel final
        document.getElementById("results-container").classList.remove("hidden")
        
        # Renderizar Graficos
        document.getElementById("chart-heatmap").innerHTML = "" 
        display(optimizar_grafico(visualizer.correlation_heatmap(analysis.get('correlaciones', {}))), target="chart-heatmap")
        
        document.getElementById("chart-scatter").innerHTML = ""
        display(optimizar_grafico(visualizer.cluster_scatter(cluster_graficos)), target="chart-scatter")
        
        document.getElementById("chart-outliers").innerHTML = ""
        display(optimizar_grafico(visualizer.outlier_chart(df_graficos, column_types, analysis.get('outliers', {}))), target="chart-outliers")
        
        document.getElementById("chart-histograms").innerHTML = ""
        display(optimizar_grafico(visualizer.histograms(df_graficos, column_types)), target="chart-histograms")
        
        document.getElementById("chart-boxplots").innerHTML = ""
        display(optimizar_grafico(visualizer.boxplots(df_graficos, column_types)), target="chart-boxplots")
        
        document.getElementById("chart-feature-importance").innerHTML = ""
        display(optimizar_grafico(visualizer.feature_importance_chart(analysis.get('feature_importance', {}))), target="chart-feature-importance")
        
        
        
    except Exception as e:
        window.alert(f"Ocurrió un error al procesar el archivo:\n{str(e)}")
    finally:
        document.getElementById("loading-indicator").classList.add("hidden")
        document.getElementById("btn-procesar").classList.remove("hidden")
        if 'ruta_virtual' in locals() and os.path.exists(ruta_virtual):
            os.remove(ruta_virtual)


def renderizar_resumen(summary):
    tipos = summary.get('tipos', {})
    
    num_cols = sum(1 for t in tipos.values() if t == 'numeric')
    cat_cols = sum(1 for t in tipos.values() if t == 'categorical')
    tem_cols = sum(1 for t in tipos.values() if t == 'temporal')
    
    html = f"""
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
        <div class="bg-blue-50/50 p-4 rounded-lg border border-blue-100 text-center">
            <span class="block text-xs text-blue-500 font-bold uppercase tracking-wider mb-1">Total Registros</span>
            <strong class="text-3xl text-blue-700">{summary.get('filas', 0):,}</strong>
        </div>
        <div class="bg-emerald-50/50 p-4 rounded-lg border border-emerald-100 text-center">
            <span class="block text-xs text-emerald-500 font-bold uppercase tracking-wider mb-1">Total Variables</span>
            <strong class="text-3xl text-emerald-700">{summary.get('columnas', 0)}</strong>
        </div>
        <div class="bg-purple-50/50 p-4 rounded-lg border border-purple-100 text-center">
            <span class="block text-xs text-purple-500 font-bold uppercase tracking-wider mb-1">Numéricas</span>
            <strong class="text-3xl text-purple-700">{num_cols}</strong>
        </div>
        <div class="bg-amber-50/50 p-4 rounded-lg border border-amber-100 text-center">
            <span class="block text-xs text-amber-500 font-bold uppercase tracking-wider mb-1">Categóricas</span>
            <strong class="text-3xl text-amber-700">{cat_cols}</strong>
        </div>
    </div>
    """
    
    html += """
    <h4 class="font-semibold text-slate-700 mb-3 uppercase text-xs tracking-wider">Diccionario de Variables Encontradas</h4>
    <div class="overflow-hidden rounded-lg border border-slate-200">
        <table class="min-w-full text-left text-sm text-slate-600">
            <thead class="bg-slate-50 border-b border-slate-200">
                <tr>
                    <th class="px-4 py-3 font-semibold text-slate-700">Nombre de la Variable</th>
                    <th class="px-4 py-3 font-semibold text-slate-700">Tipo Detectado</th>
                    <th class="px-4 py-3 font-semibold text-slate-700">Información / Ejemplo</th>
                </tr>
            </thead>
            <tbody class="divide-y divide-slate-100">
    """
    
    for col, tipo in tipos.items():
        if tipo == 'numeric':
            tipo_html = '<span class="px-2 py-1 text-[11px] font-bold uppercase rounded-full bg-purple-100 text-purple-700">Numérica</span>'
            
            promedio = summary.get('estadisticas', {}).get(col, {}).get('mean', 'N/A')
            info_extra = f"Promedio global: <strong>{promedio}</strong>" if promedio != 'N/A' else "Medidas y cantidades"
            
        elif tipo == 'categorical':
            tipo_html = '<span class="px-2 py-1 text-[11px] font-bold uppercase rounded-full bg-amber-100 text-amber-700">Categórica</span>'
            
            top_cats = summary.get('categorias', {}).get(col, {}).get('top_10', {})
            ejemplos = list(top_cats.keys())[:3]
            info_extra = f"Ej: {', '.join(map(str, ejemplos))}" if ejemplos else "Textos y categorías"
            
        elif tipo == 'temporal':
            tipo_html = '<span class="px-2 py-1 text-[11px] font-bold uppercase rounded-full bg-sky-100 text-sky-700">Fecha/Tiempo</span>'
            info_extra = "Marcas de tiempo"
        else:
            tipo_html = '<span class="px-2 py-1 text-[11px] font-bold uppercase rounded-full bg-slate-100 text-slate-700">Otro</span>'
            info_extra = "Datos booleanos o mixtos"

        html += f"""
                <tr class="hover:bg-slate-50 transition-colors">
                    <td class="px-4 py-3 font-medium text-slate-800">{col}</td>
                    <td class="px-4 py-3">{tipo_html}</td>
                    <td class="px-4 py-3 text-slate-500">{info_extra}</td>
                </tr>
        """
        
    html += """
            </tbody>
        </table>
    </div>
    """
    
    document.getElementById("summary-content").innerHTML = html


def renderizar_insights(analysis):
    html = ""
    insights = analysis.get('insights', [])
    
    if not insights:
        html = """
        <div class="col-span-full py-8 text-center bg-slate-50 rounded-lg border border-dashed border-slate-300">
            <p class="text-slate-500 italic">No se detectaron hallazgos significativos en este dataset.</p>
        </div>
        """
    else:
        for insight in insights:
            tipo = insight.get('tipo', 'info')
            if tipo == 'critico':
                bg, border, text, icon_color = "bg-red-50", "border-red-500", "text-red-700", "text-red-400"
            elif tipo == 'info':
                bg, border, text, icon_color = "bg-blue-50", "border-blue-500", "text-blue-700", "text-blue-400"
            else: # success / positivo
                bg, border, text, icon_color = "bg-emerald-50", "border-emerald-500", "text-emerald-700", "text-emerald-400"

            html += f"""
            <div class="{bg} {border} border-l-4 p-4 rounded-r-lg shadow-sm transition-all hover:shadow-md">
                <div class="flex items-start gap-3">
                    <div class="flex-1">
                        <strong class="{text} block text-xs uppercase tracking-widest mb-1">{insight['titulo']}</strong>
                        <p class="text-slate-600 leading-relaxed font-medium">{insight['texto']}</p>
                    </div>
                </div>
            </div>
            """
    document.getElementById("insights-content").innerHTML = html

def generar_html_reporte():
    from jinja2 import Template
    import plotly.io as pio
    from services import visualizer
    
    with open("report.html", "r", encoding="utf-8") as f:
        template_text = f.read()
        
    template = Template(template_text)
    
    tipos = estado_global["summary"].get('tipos', {})
    counts = {
        'numeric': sum(1 for t in tipos.values() if t == 'numeric'),
        'categorical': sum(1 for t in tipos.values() if t == 'categorical'),
        'temporal': sum(1 for t in tipos.values() if t == 'temporal')
    }
    
    def get_chart_html(fig):
        if fig is None: return ""
        fig = optimizar_grafico(fig)
        return pio.to_html(fig, full_html=False, include_plotlyjs=False)
    
    analysis = estado_global["analysis"]
    df_ligero = estado_global.get("df_graficos", estado_global["df"])
    cluster_ligero = estado_global.get("cluster_graficos", analysis.get('clustering', {}))
    
    charts = {
        "outlier_chart": get_chart_html(visualizer.outlier_chart(
            df_ligero, estado_global["column_types"], analysis.get('outliers', {})
        )),
        "feature_importance": get_chart_html(visualizer.feature_importance_chart(
            analysis.get('feature_importance', {})
        )),
        "heatmap": get_chart_html(visualizer.correlation_heatmap(
            analysis.get('correlaciones', {})
        )),
        "cluster_scatter": get_chart_html(visualizer.cluster_scatter(
            cluster_ligero
        )),
        "histograms": get_chart_html(visualizer.histograms(
            df_ligero, estado_global["column_types"]
        )),
        "boxplots": get_chart_html(visualizer.boxplots(
            df_ligero, estado_global["column_types"]
        ))
    }
    
    html_final = template.render(
        filename=estado_global["filename"],
        summary=estado_global["summary"],
        analysis=analysis,
        charts=charts,
        cleaning_report=estado_global["cleaning_report"],
        counts=counts
    )
    
    return html_final

def exportar_html(evento):
    html_content = generar_html_reporte()
    blob = Blob.new([html_content], {"type": "text/html"})
    url = URL.createObjectURL(blob)
    a = document.createElement("a")
    a.href = url
    a.download = f"Reporte_EDA_{estado_global.get('filename', 'datos')}.html"
    a.click()
    URL.revokeObjectURL(url)


def exportar_pdf(evento):
    html_content = generar_html_reporte()
    filename = f"Reporte_EDA_{estado_global.get('filename', 'datos')}.pdf"
    exportar_a_pdf_js(html_content, filename)


# ==========================================
# CONECTAR EVENTOS AL HTML
# ==========================================
elemento_input = document.getElementById("file-upload")
elemento_input.addEventListener("change", create_proxy(manejar_seleccion_archivo))

zona_drop = document.getElementById("drop-zone")
zona_drop.addEventListener("dragover", create_proxy(manejar_dragover))
zona_drop.addEventListener("dragleave", create_proxy(manejar_dragleave))
zona_drop.addEventListener("drop", create_proxy(manejar_drop))

boton_procesar = document.getElementById("btn-procesar")
boton_procesar.addEventListener("click", create_proxy(procesar_archivo))

document.getElementById("btn-export-html").addEventListener("click", create_proxy(exportar_html))
document.getElementById("btn-export-pdf").addEventListener("click", create_proxy(exportar_pdf))