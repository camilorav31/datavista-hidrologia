import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter import font
import matplotlib.pyplot as plt 
from scipy.stats import gumbel_r, weibull_min, genextreme
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import os
import seaborn as sns

import sv_ttk 

from src.data_processing import cargar_datos, calcular_estadisticos,cargar_datos_enso
from src.visualization import *

# Función para seleccionar archivo
def seleccionar_archivo():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
    if file_path:
        label_imagen.pack_forget()  # Ocultar la imagen al cargar los datos
        btn_cargar.pack_forget()  # Ocultar el botón de selección de archivo
        procesar_datos(file_path)

# Función para mostrar los resultados en pantalla
def mostrar_resultados():
    root.state('zoomed')  # Maximizar ventana en modo convencional
    notebook.pack(fill="both", expand=True)

# Función que procesa los datos
def procesar_datos(file_path):
    try:
        datos = cargar_datos(file_path)
        estadisticos = calcular_estadisticos(datos)  # Calcular estadísticas
        crear_graficos_pestana1(datos, estadisticos)  # Gráficos en la primera pestaña
        crear_graficos_pestana2(datos)  # Gráficos en la segunda pestaña
        crear_graficos_pestana3(datos)
        mostrar_resultados()  # Mostrar los resultados
    except Exception as e:
        messagebox.showerror("Error", f"Error al procesar el archivo: {e}")

def crear_graficos_pestana1(datos, estadisticos):
    # Limpiar widgets previos
    for widget in frame_pestana1.winfo_children():
        widget.destroy()

    # Configurar grid de 3 filas y 3 columnas
    for i in range(3):
        frame_pestana1.grid_rowconfigure(i, weight=1, uniform="row")
        frame_pestana1.grid_columnconfigure(i, weight=1, uniform="col")

    # Tamaño fijo para todas las gráficas
    fig_size = (4.5, 3.5)  # Ancho y altura en pulgadas

    # Gráfico de serie temporal
    fig1, ax1 = plt.subplots(figsize=fig_size)
    datos['Fecha'] = pd.to_datetime(datos['Fecha'])
    ax1.plot(datos['Fecha'], datos['Valor'], label="Valor")
    ax1.set_title("Caudales Historicos")
    ax1.set_ylabel("Q [m3/s]")
    ax1.margins(x=0.05)
    fig1.tight_layout(pad=2)
    canvas1 = FigureCanvasTkAgg(fig1, master=frame_pestana1)
    canvas1.draw()
    canvas1.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    # Gráfico Boxplot
    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    fig2, ax2 = plt.subplots(figsize=fig_size)
    datos['Mes'] = datos['Fecha'].dt.month
    sns.boxplot(x=datos['Mes'], y=datos['Valor'], ax=ax2, whis=[0,100]).set_title("Ciclo Anual con Outlayers incorporados")
    ax2.set_xticks(range(12))
    ax2.set_xlabel("")
    ax2.set_ylabel("Q [m3/s]")
    ax2.set_xticklabels(meses)
    ax2.margins(x=0.05)
    fig2.tight_layout(pad=2)
    canvas2 = FigureCanvasTkAgg(fig2, master=frame_pestana1)
    canvas2.draw()
    canvas2.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

    # Gráfico de violín
    fig3, ax3 = plt.subplots(figsize=(4.5, 7))
    grafico_violin(datos, ax3)
    fig3.tight_layout(pad=2)
    canvas3 = FigureCanvasTkAgg(fig3, master=frame_pestana1)
    canvas3.draw()
    canvas3.get_tk_widget().grid(row=0, column=2, rowspan=2, sticky="nsew", padx=5, pady=5)

    # Gráfico de frecuencia relativa
    fig4, ax4 = plt.subplots(figsize=fig_size)
    sns.histplot(datos['Valor'], kde=False, stat='proportion', ax=ax4).set_title("Histograma de Frecuencias")
    ax4.margins(x=0.05)
    ax4.set_ylabel("Frecuencia relativa (Proporción)")
    ax4.set_xlabel("Q [m3/s]")
    fig4.tight_layout(pad=2)
    canvas4 = FigureCanvasTkAgg(fig4, master=frame_pestana1)
    canvas4.draw()
    canvas4.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    # Gráfico de curva de duración de caudales con Q10, Q50 y Q95 marcados como "X"
    fig5, ax5 = plt.subplots(figsize=(8, 6))

    # Calcular curva de duración de caudales
    datos_ordenados = datos['Valor'].sort_values(ascending=False).reset_index(drop=True)
    n = len(datos_ordenados)
    prob_excedencia = [(i / (n + 1)) * 100 for i in range(1, n + 1)]

    # Graficar curva de duración de caudales
    ax5.plot(prob_excedencia, datos_ordenados, label="Curva de Duración", linestyle='-')

    # Títulos y etiquetas
    ax5.set_title("Curva de Duración")
    ax5.set_xlabel("Probabilidad de Excedencia (%)")
    ax5.set_ylabel("Q [m³/s]")
    ax5.margins(x=0.05)
    ax5.grid(False)

    # Configurar minor ticks solo en el eje X
    ax5.minorticks_on()
    ax5.tick_params(axis='x', which='minor', length=4, color='gray')

    # Calcular Q10, Q50 y Q95
    q10_idx = next(i for i, p in enumerate(prob_excedencia) if p >= 10)
    q50_idx = next(i for i, p in enumerate(prob_excedencia) if p >= 50)
    q95_idx = next(i for i, p in enumerate(prob_excedencia) if p >= 95)

    q10 = datos_ordenados[q10_idx]
    q50 = datos_ordenados[q50_idx]
    q95 = datos_ordenados[q95_idx]

    # Marcar Q10, Q50 y Q95 con una "X", líneas y añadirlos a la leyenda
    for q, p, label, color in zip([q10, q50, q95], [10, 50, 95], ["Q10", "Q50", "Q95"], ["green", "red", "blue"]):
        # Línea desde el eje X al punto
        ax5.plot([p, p], [0, q], color=color, linestyle='--', linewidth=0.8)  
        ax5.plot([0, p], [q, q], color=color, linestyle='--', linewidth=0.8)
        # Marcar el punto con una "X"
        ax5.plot(p, q, marker='X', color=color, markersize=6, label=f"{label} : {q:.2f} m³/s")

    # Añadir leyenda
    ax5.legend()

    # Ajustar diseño
    fig5.tight_layout(pad=4)

    # Mostrar gráfico en la interfaz
    canvas5 = FigureCanvasTkAgg(fig5, master=frame_pestana1)
    canvas5.draw()
    canvas5.get_tk_widget().grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

#-//////////////////////////////////////////////////////////////////////////////////////////
    from scipy.stats import kendalltau, spearmanr

    # Cálculo del test de Mann-Kendall
    datos['Fecha'] = pd.to_datetime(datos['Fecha'])
    trend_slope_mk, p_value_mk = kendalltau(np.arange(len(datos)), datos['Valor'])

    # Cálculo del test de Spearman
    rho, p_value_spearman = spearmanr(np.arange(len(datos)), datos['Valor'])

    # Cálculo de la Pendiente de Sen (Zen)
    slopes = [(datos['Valor'][j] - datos['Valor'][i]) / (j - i) 
            for i in range(len(datos)) for j in range(i + 1, len(datos))]
    sen_slope = np.median(slopes)
    
    # Definir el resultado de la tendencia para Mann-Kendall
    if p_value_mk < 0.05:
        tendencia_mk = "Sí"
        direccion_mk = "Creciente" if sen_slope > 0 else "Decreciente"
    else:
        tendencia_mk = "No"
        direccion_mk = "N/A"

    # Definir el resultado de la tendencia para Spearman
    if p_value_spearman < 0.05:
        tendencia_spearman = "Sí"
        direccion_spearman = "Creciente" if rho > 0 else "Decreciente"
    else:
        tendencia_spearman = "No"
        direccion_spearman = "N/A"

    # Crear el frame para la tabla en la posición (3, 2)
    frame_tabla_resultados = tk.Frame(frame_pestana1, bd=2, relief="solid")
    frame_tabla_resultados.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)

    # Crear un estilo personalizado para la tabla
    style = ttk.Style()
    style.configure("Treeview", font=("Arial", 16))  # Ajuste de fuente para las filas
    style.configure("Treeview.Heading", font=("Arial", 18, "bold"))  # Encabezados en negrita y con mayor tamaño

    # Crear la tabla en Tkinter usando Treeview
    tree = ttk.Treeview(frame_tabla_resultados, columns=("Métrica", "Mann-Kendall", "Spearman"), show='headings', height=6)
    tree.heading("Métrica", text="Métrica", anchor='center')
    tree.heading("Mann-Kendall", text="Mann-Kendall", anchor='center')
    tree.heading("Spearman", text="Spearman", anchor='center')
    tree.column("Métrica", width=200, anchor='center')
    tree.column("Mann-Kendall", width=150, anchor='center')
    tree.column("Spearman", width=150, anchor='center')

    # Insertar los resultados en la tabla
    tree.insert("", "end", values=("p-value", f"{p_value_mk:.4f}", f"{p_value_spearman:.4f}"))
    tree.insert("", "end", values=("Tendencia", tendencia_mk, tendencia_spearman))
    tree.insert("", "end", values=("Dirección", direccion_mk, direccion_spearman))
    tree.insert("", "end", values=("Pendiente Sen", f"{trend_slope_mk:.3f} m3/s", f"{sen_slope:.3f} m3/s"))

    # Empaquetar la tabla para que ocupe todo el espacio dentro de su frame
    tree.pack(side=tk.LEFT, fill="both", expand=True, padx=10, pady=10)

    # Gráfico ENSO
    enso_data = cargar_datos_enso()  
    datos['Fecha'] = pd.to_datetime(datos['Fecha'])
    datos_combinados = pd.merge(datos, enso_data, on='Fecha', how='inner')

    # Paso 4: Clasificar cada período en fase ENSO (El Niño, La Niña, Neutral)
    def clasificar_fase(oni):
        if oni > 0.5:
            return 'El Niño'
        elif oni < -0.5:
            return 'La Niña'
        else:
            return 'Neutral'

    datos_combinados['Fase_ENSO'] = datos_combinados['ONI'].apply(clasificar_fase)

    # Paso 5: Agregar columna 'Mes' para calcular el ciclo anual
    datos_combinados['Mes'] = datos_combinados['Fecha'].dt.month

    # Paso 6: Calcular el ciclo anual promedio para cada fase ENSO
    ciclo_anual = datos_combinados.groupby(['Mes', 'Fase_ENSO'])['Valor'].mean().reset_index()

    # Paso 7: Crear el gráfico del ciclo anual en función de la fase ENSO
    ciclo_anual_std = datos_combinados.groupby(['Mes', 'Fase_ENSO']).agg(
        Valor=('Valor', 'mean'),
        Error=('Valor', 'std')  
    ).reset_index()

    fig6, ax = plt.subplots(figsize=(8, 6))
    colores_fase = {'La Niña': 'blue', 'Neutral': 'black', 'El Niño': 'red'}
    for fase in ciclo_anual_std['Fase_ENSO'].unique():
        datos_fase = ciclo_anual_std[ciclo_anual_std['Fase_ENSO'] == fase]
        ax.errorbar(
            datos_fase['Mes'], datos_fase['Valor'], yerr=datos_fase['Error'],
            fmt='-o', color=colores_fase[fase], label=fase, capsize=5  
        )
    promedio_mensual = datos_combinados.groupby('Mes')['Valor'].mean().reset_index()
    ax.bar(promedio_mensual['Mes'], promedio_mensual['Valor'], alpha=0.9)

    ax.set_title("Influencia del ENSO")
    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    ax.set_xticks(range(1, 13))  
    ax.set_xticklabels(meses)    
    ax.set_xlabel(None)    
    ax.set_ylabel("Q [m³/s]")
    ax.legend()
    fig6.tight_layout(pad=2)

    canvas = FigureCanvasTkAgg(fig6, master=frame_pestana1)
    canvas.draw()
    canvas.get_tk_widget().grid(row=2, column=1, sticky="nsew", padx=5, pady=5)
#//////////////////////////////////////////////////////////////////////////////////////////
    frame_tabla = tk.Frame(frame_pestana1, bd=2, relief="solid")
    frame_tabla.grid(row=2, column=2, sticky="nsew", padx=10, pady=10)  

    # Crear un estilo personalizado para la tabla
    style = ttk.Style()
    
    # Aumentar tamaño de la fuente en las filas
    style.configure("Treeview", font=("Arial", 16))
    
    # Poner encabezados en negrita y aumentar su tamaño
    style.configure("Treeview.Heading", font=("Arial", 22, "bold"))

    # Crear la tabla (Treeview)
    tree = ttk.Treeview(frame_tabla, columns=("Estadístico", "Valor"), show='headings', height=10)
    tree.heading("Estadístico", text="Estadístico", anchor='center')
    tree.heading("Valor", text="Valor", anchor='center')
    tree.column("Estadístico", width=200, anchor='center')  # Aumentamos el ancho de las columnas
    tree.column("Valor", width=150, anchor='center')

    # Añadir los valores de los estadísticos
    for estadistico, valor in estadisticos.items():
        tree.insert('', 'end', values=(estadistico, f"{valor:.2f}"))

    # Empaquetar la tabla
    tree.pack(side=tk.LEFT, fill="both", expand=True, padx=10, pady=10)


# Función para crear gráficos en la segunda pestaña
def crear_graficos_pestana2(datos):
    # Limpiar widgets previos
    for widget in frame_pestana2.winfo_children():
        widget.destroy()

    # Ajustar la cuadrícula
    frame_pestana2.grid_rowconfigure(0, weight=1)
    frame_pestana2.grid_rowconfigure(1, weight=1)
    frame_pestana2.grid_columnconfigure(0, weight=3)
    frame_pestana2.grid_columnconfigure(1, weight=2)
    frame_pestana2.grid_columnconfigure(2, weight=1)

    # Desestacionalización
    fig1, axs1 = plt.subplots(4, 1, figsize=(8, 6))  # Altura ajustada
    desestacionalizacion(datos, axs1)
    fig1.tight_layout()
    canvas1 = FigureCanvasTkAgg(fig1, master=frame_pestana2)
    canvas1.draw()
    canvas1.get_tk_widget().grid(row=0, column=0, rowspan=3, sticky="nsew", padx=5, pady=5)

    # Gráficos de ACF y PACF
    fig2, axs2 = plt.subplots(2, 1, figsize=(6, 4))
    grafico_acf_pacf(datos, axs2)
    fig2.tight_layout()
    canvas2 = FigureCanvasTkAgg(fig2, master=frame_pestana2)
    canvas2.draw()
    canvas2.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

    # Gráficos de lags (rezagos)
    fig3 = plot_lags(datos['Valor'], lags=15, nrows=3, ncols=5, figsize=(8, 5))
    canvas3 = FigureCanvasTkAgg(fig3, master=frame_pestana2)
    canvas3.draw()
    canvas3.get_tk_widget().grid(row=1, column=1, rowspan=2, sticky="nsew", padx=5, pady=5)

    # Mapa de calor de estacionalidad
    fig4, ax4 = plt.subplots(figsize=(6, 10))  # Aumentar el tamaño para que se ajuste a la altura
    heatmap_estacionalidad(datos, ax4)
    fig4.tight_layout()
    canvas4 = FigureCanvasTkAgg(fig4, master=frame_pestana2)
    canvas4.draw()
    canvas4.get_tk_widget().grid(row=0, column=2, rowspan=3, sticky="nsew", padx=5, pady=5)

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.visualization import plot_return_level_gumbel, plot_return_level_weibull, plot_return_level_frechet

def crear_graficos_pestana3(datos):
    # Limpiar widgets previos en la pestaña 3
    for widget in frame_pestana3.winfo_children():
        widget.destroy()

    # Configuración de grid
    for i in range(3):
        frame_pestana3.grid_rowconfigure(i, weight=1, uniform="row")
        frame_pestana3.grid_columnconfigure(i, weight=1, uniform="col")

    # [1,1] - Gráfico de máximos, mínimos y medios anuales
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    datos_anuales = datos.resample('YE', on='Fecha').agg({'Valor': ['max', 'min', 'mean']})
    ax1.plot(datos_anuales['Valor']['max'], label="Máximos", color='blue')
    ax1.plot(datos_anuales['Valor']['min'], label="Mínimos", color='red')
    ax1.plot(datos_anuales['Valor']['mean'], label="Medios", color='green')
    ax1.set_title("Máximos, Mínimos y Medios Anuales")
    ax1.legend()
    fig1.tight_layout()
    canvas1 = FigureCanvasTkAgg(fig1, master=frame_pestana3)
    canvas1.draw()
    canvas1.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    # [1,2] - Autocorrelación para las tres series
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    for serie, color, label in zip(['max', 'min', 'mean'], ['blue', 'red', 'green'], ['ACF Max', 'ACF Min', 'ACF Mean']):
        autocorr, lim_sup, lim_inf = calcular_autocorrelacion_pearson(datos_anuales['Valor'][serie], lags=20)
        ax2.plot(autocorr, color=color, label=label)
        ax2.fill_between(range(len(autocorr)), lim_inf, lim_sup, color='gray', alpha=0.2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title("Autocorrelación Series Anuales (Límites Pearson)")
    ax2.legend()
    fig2.tight_layout()
    canvas2 = FigureCanvasTkAgg(fig2, master=frame_pestana3)
    canvas2.draw()
    canvas2.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

    # [1,3] - Gráfico de distribuciones
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    sns.kdeplot(datos_anuales['Valor']['max'], label="Distribución Máximos", color="blue", ax=ax3)
    sns.kdeplot(datos_anuales['Valor']['min'], label="Distribución Mínimos", color="red", ax=ax3)
    sns.kdeplot(datos_anuales['Valor']['mean'], label="Distribución Medios", color="green", ax=ax3)
    ax3.set_title("Distribución de Máximos, Mínimos y Medios")
    ax3.legend()
    fig3.tight_layout()
    canvas3 = FigureCanvasTkAgg(fig3, master=frame_pestana3)
    canvas3.draw()
    canvas3.get_tk_widget().grid(row=0, column=2, sticky="nsew", padx=5, pady=5)


    def cambiar_distribucion(distribucion):
        # Limpiar únicamente el gráfico Q-Q en la posición [3,2] para evitar acumulaciones
        for widget in frame_pestana3.grid_slaves(row=2, column=2):
            widget.destroy()

        # Selección de la función de distribución y la función específica de visualización
        if distribucion == "gumbel":
            plot_return_level = plot_return_level_gumbel
            dist_obj = gumbel_r
        elif distribucion == "weibull":
            plot_return_level = plot_return_level_weibull
            dist_obj = weibull_min
        elif distribucion == "frechet":
            plot_return_level = plot_return_level_frechet
            dist_obj = invweibull
        elif distribucion == "lognormal":
            plot_return_level = plot_return_level_lognormal
            dist_obj = lognorm

        # [2,2] - Gráfico de niveles de retorno para los valores mínimos (1/T)
        fig_max, ax_max = plt.subplots(figsize=(6, 5))
        plot_return_level(datos_anuales["Valor"]["min"], ax_max, "Mínimos", "red", exceedance=True)
        ax_max.set_title(f"Niveles de Retorno Mínimos - {distribucion.capitalize()}")
        fig_max.tight_layout()
        canvas_max = FigureCanvasTkAgg(fig_max, master=frame_pestana3)
        canvas_max.draw()
        canvas_max.get_tk_widget().grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        # [2,3] - Gráfico de niveles de retorno para los valores máximos (1 - 1/T)
        fig_min, ax_min = plt.subplots(figsize=(6, 5))
        plot_return_level(datos_anuales["Valor"]["max"], ax_min, "Máximos", "blue", exceedance=False)
        ax_min.set_title(f"Niveles de Retorno Máximos - {distribucion.capitalize()}")
        fig_min.tight_layout()
        canvas_min = FigureCanvasTkAgg(fig_min, master=frame_pestana3)
        canvas_min.draw()
        canvas_min.get_tk_widget().grid(row=1, column=2, sticky="nsew", padx=5, pady=5)

        # [3,2] - Generación del gráfico Q-Q con la distribución seleccionada
        fig_qq, ax_qq = plt.subplots(figsize=(5, 4))
        # Ajustamos la distribución seleccionada a los datos para obtener los parámetros
        params = dist_obj.fit(datos_anuales["Valor"]["max"])
        # Generamos el gráfico Q-Q usando la distribución y parámetros obtenidos
        grafico_qq_personalizado(datos_anuales["Valor"]["max"], ax_qq, distribucion=dist_obj, params=params)
        ax_qq.set_title(f"Gráfico Q-Q - {distribucion.capitalize()}")
        fig_qq.tight_layout()
        canvas_qq = FigureCanvasTkAgg(fig_qq, master=frame_pestana3)
        canvas_qq.draw()
        canvas_qq.get_tk_widget().grid(row=2, column=1, sticky="nsew", padx=5, pady=5)

        # [2,1] - Actualizar la tabla de valores de retorno con la distribución seleccionada
        mostrar_tabla_retorno(datos_anuales, dist_obj)


    def mostrar_tabla_retorno(datos_anuales, distribucion):
        # Crear frame para la tabla
        frame_tabla = tk.Frame(frame_pestana3, bd=2, relief="solid")
        frame_tabla.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        # Crear un estilo personalizado para la tabla
        style = ttk.Style()
        style.configure("Treeview", font=("Arial", 16))
        style.configure("Treeview.Heading", font=("Arial", 22, "bold"))

        # Crear la tabla (Treeview) sin la columna de "Mean"
        tree = ttk.Treeview(frame_tabla, columns=("Tr (Años)", "Max", "Min"), show='headings', height=6)
        tree.heading("Tr (Años)", text="Tr (Años)", anchor='center')
        tree.heading("Max", text="Max", anchor='center')
        tree.heading("Min", text="Min", anchor='center')
        
        tree.column("Tr (Años)", width=100, anchor='center')
        tree.column("Max", width=100, anchor='center')
        tree.column("Min", width=100, anchor='center')

        # Rellenar la tabla con datos usando las probabilidades ajustadas para Max y Min
        for periodo in [2.33, 5, 10, 15, 20, 25, 50, 75, 100]:
            fila_valores = [str(periodo)]
            
            # Cálculo para 'Max' (probabilidad de no excedencia, 1 - 1/T)
            serie_max = datos_anuales['Valor']['max']
            params_max = distribucion.fit(serie_max)
            prob_max = 1 - 1 / periodo
            if len(params_max) == 3:
                c, loc, scale = params_max
                valor_max = distribucion.ppf(prob_max, c, loc=loc, scale=scale)
            else:
                loc, scale = params_max
                valor_max = distribucion.ppf(prob_max, loc=loc, scale=scale)
            fila_valores.append(f"{valor_max:.2f}")

            # Cálculo para 'Min' (probabilidad de excedencia, 1/T)
            serie_min = datos_anuales['Valor']['min']
            params_min = distribucion.fit(serie_min)
            prob_min = 1 / periodo
            if len(params_min) == 3:
                c, loc, scale = params_min
                valor_min = distribucion.ppf(prob_min, c, loc=loc, scale=scale)
            else:
                loc, scale = params_min
                valor_min = distribucion.ppf(prob_min, loc=loc, scale=scale)
            fila_valores.append(f"{valor_min:.2f}")
            
            # Inserta la fila en la tabla
            tree.insert('', 'end', values=fila_valores)

        # Empaquetar la tabla
        tree.pack(fill="both", expand=True, padx=10, pady=10)

    # Crear botones en [2,1]
    botones_frame = tk.Frame(frame_pestana3)
    botones_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

    # Configuración de los botones con estilos personalizados
    button_gumbel = tk.Button(botones_frame, text="Gumbel", command=lambda: cambiar_distribucion('gumbel'),
                              bg="#00c0ff", fg="white", font=("Roboto", 20, "bold"))
    button_weibull = tk.Button(botones_frame, text="Weibull", command=lambda: cambiar_distribucion('weibull'),
                               bg="#00c0ff", fg="white", font=("Roboto", 20, "bold"))
    button_frechet = tk.Button(botones_frame, text="Frechet", command=lambda: cambiar_distribucion('frechet'),
                               bg="#00c0ff", fg="white", font=("Roboto", 20, "bold"))
    button_lognormal = tk.Button(botones_frame, text="Lognormal", command=lambda: cambiar_distribucion('lognormal'),
                                bg="#00c0ff", fg="white", font=("Roboto", 20, "bold"))
    
    button_gumbel.pack(side="top", expand=True, fill="both", padx=2)
    button_weibull.pack(side="top", expand=True, fill="both", padx=2)
    button_frechet.pack(side="top", expand=True, fill="both", padx=2)
    button_lognormal.pack(side="top", expand=True, fill="both", padx=2)
    # Gráficos iniciales (Gumbel por defecto)
    cambiar_distribucion('gumbel')

#===============
def centrar_ventana(ventana, ancho, alto):
    ventana.update_idletasks()

    # Obtener el tamaño de la pantalla
    ancho_pantalla = ventana.winfo_screenwidth()
    alto_pantalla = ventana.winfo_screenheight()

    # Calcular las coordenadas para centrar la ventana
    x = (ancho_pantalla // 2) - (ancho // 2)
    y = (alto_pantalla // 2) - (alto // 2)

    # Establecer la geometría de la ventana con tamaño y posición centrada
    ventana.geometry(f'{ancho}x{alto}+{x}+{y}')

# Interfaz gráfica
root = tk.Tk()

sv_ttk.set_theme("dark")
root.title("DataVista Hidrología de Caudales - Todos los derechos reservados ©")
notebook = ttk.Notebook(root)
notebook.pack_forget()  # Oculto al inicio

ancho_inicial = 600
alto_inicial = 500

# Configura la ventana antes de centrarla
root.update_idletasks()

import sys
import os

def resource_path(relative_path):
    """Obtiene la ruta absoluta del recurso, para PyInstaller y entorno local."""
    try:
        # PyInstaller usa _MEIPASS para la ruta de los archivos temporales
        base_path = sys._MEIPASS
    except AttributeError:
        # En modo desarrollo, la ruta base es el directorio actual
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

icono_path = resource_path("assets/icon.png")
welcome_image_path = resource_path("assets/welcome.png")

# Cargar la imagen del icono (debe ser de formato .png)
icon_path = icono_path
icon_img = Image.open(icon_path)
icon_photo = ImageTk.PhotoImage(icon_img)
root.iconphoto(False, icon_photo)

# Llamar a la función para centrar la ventana
centrar_ventana(root, ancho_inicial,alto_inicial)
# Creación de las pestañas
frame_pestana1 = tk.Frame(notebook)
frame_pestana2 = tk.Frame(notebook)
frame_pestana3 = tk.Frame(notebook)

notebook.add(frame_pestana1, text="DESCRIPTIVA")
notebook.add(frame_pestana2, text="ESTACIONALIDAD")
notebook.add(frame_pestana3, text="EXTREMOS")

btn_cargar = tk.Button(root, text="SELECCIONAR ARCHIVO", 
                       command=seleccionar_archivo, 
                       bg="#00c0ff",
                       fg="white",
                       font=("Roboto", 12, "bold"))
btn_cargar.pack(pady=10)

# Cargar la imagen de bienvenida
image_path = welcome_image_path
img = Image.open(image_path)
width, height = img.size
new_width = 500
new_height = int(new_width * height / width)

img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
photo = ImageTk.PhotoImage(img)
label_imagen = tk.Label(root, image=photo)
label_imagen.pack(pady=10)

root.mainloop()