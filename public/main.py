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
        
        # Renderizar Graficos
        document.getElementById("chart-heatmap").innerHTML = "" 
        display(visualizer.correlation_heatmap(analysis.get('correlaciones', {})), target="chart-heatmap")
        
        document.getElementById("chart-scatter").innerHTML = ""
        display(visualizer.cluster_scatter(analysis.get('clustering', {})), target="chart-scatter")
        
        document.getElementById("chart-outliers").innerHTML = ""
        display(visualizer.outlier_chart(df, column_types, analysis.get('outliers', {})), target="chart-outliers")
        
        document.getElementById("chart-histograms").innerHTML = ""
        display(visualizer.histograms(df, column_types), target="chart-histograms")
        
        document.getElementById("chart-boxplots").innerHTML = ""
        display(visualizer.boxplots(df, column_types), target="chart-boxplots")
        
        document.getElementById("chart-feature-importance").innerHTML = ""
        display(visualizer.feature_importance_chart(analysis.get('feature_importance', {})), target="chart-feature-importance")
        
        # Mostrar panel final
        document.getElementById("results-container").classList.remove("hidden")
        
    except Exception as e:
        window.alert(f"Ocurrió un error al procesar el archivo:\n{str(e)}")
    finally:
        document.getElementById("loading-indicator").classList.add("hidden")
        document.getElementById("btn-procesar").classList.remove("hidden")
        if 'ruta_virtual' in locals() and os.path.exists(ruta_virtual):
            os.remove(ruta_virtual)


def renderizar_resumen(summary):
    html = f"""
    <ul class="space-y-2 text-slate-600">
        <li><span class="font-semibold text-slate-800">Filas analizadas:</span> {summary.get('filas', 0)}</li>
        <li><span class="font-semibold text-slate-800">Columnas procesadas:</span> {summary.get('columnas', 0)}</li>
    </ul>
    """
    document.getElementById("summary-content").innerHTML = html


def renderizar_insights(analysis):
    html = ""
    insights = analysis.get('insights', [])
    if not insights:
        html = "<p class='text-slate-500'>No se detectaron insights relevantes.</p>"
    else:
        for insight in insights:
            color = "red" if insight['tipo'] == 'critico' else "blue" if insight['tipo'] == 'info' else "emerald"
            html += f"""
            <div class="p-3 bg-{color}-50 border-l-4 border-{color}-500 rounded-r shadow-sm">
                <strong class="text-{color}-700 block mb-1">{insight['titulo']}</strong>
                <span class="text-slate-600">{insight['texto']}</span>
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
    
    def get_chart_html(fig):
        if fig is None: return ""
        return pio.to_html(fig, full_html=False, include_plotlyjs=False)
    
    analysis = estado_global["analysis"]
    charts = {
        "outlier_chart": get_chart_html(visualizer.outlier_chart(
            estado_global["df"], estado_global["column_types"], analysis.get('outliers', {})
        )),
        "feature_importance": get_chart_html(visualizer.feature_importance_chart(
            analysis.get('feature_importance', {})
        ))
    }
    
    html_final = template.render(
        filename=estado_global["filename"],
        summary=estado_global["summary"],
        analysis=analysis,
        charts=charts,
        cleaning_report=estado_global["cleaning_report"]
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