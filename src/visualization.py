import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.offsetbox import AnchoredText
import statsmodels.api as sm
from scipy.stats import norm
from scipy.stats import genextreme, gumbel_r, weibull_min
from scipy.stats import gaussian_kde

# Función para graficar un solo rezago
def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    x_ = x.shift(lag)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)

    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
        color="black"
    )
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_, y=y_, scatter_kws=scatter_kws, line_kws=line_kws, lowess=True, ax=ax, **kwargs)

    # Añadir texto con la correlación
    at = AnchoredText(f"{corr:.2f}", prop=dict(size="large"), frameon=True, loc="upper left")
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax

# Función para graficar múltiples rezagos en forma de matriz
def plot_lags(x, y=None, lags=6, nrows=1, ncols=None, lagplot_kwargs={}, **kwargs):
    import math
    if ncols is None:
        ncols = math.ceil(lags / nrows)
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', ncols)
    kwargs.setdefault('figsize', (ncols * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)

    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k + 1 <= lags:
            ax = lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig


# Función para generar gráficos incrustados en Tkinter
def generar_graficos_embedded(datos, frame):
    # Limpiar cualquier gráfico anterior del frame
    for widget in frame.winfo_children():
        widget.destroy()
    
    # Crear la figura de matplotlib
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    # Serie temporal con media móvil
    datos['Valor'].plot(ax=axs[0, 0], label='Original')
    datos['Valor'].rolling(window=12).mean().plot(ax=axs[0, 0], label='Media Móvil', color='red')
    axs[0, 0].set_title('Serie Temporal con Media Móvil')
    axs[0, 0].legend()

    # Ciclo anual (Boxplot por mes)
    datos['Mes'] = datos['Fecha'].dt.month
    sns.boxplot(x='Mes', y='Valor', data=datos, ax=axs[0, 1])
    axs[0, 1].set_title('Ciclo Anual (Boxplot por Mes)')

    # Histograma de frecuencias relativas
    sns.histplot(datos['Valor'], kde=False, stat='probability', ax=axs[1, 0])
    axs[1, 0].set_title('Histograma de Frecuencias Relativas')

    # Función de Probabilidad Acumulada
    sns.ecdfplot(datos['Valor'], ax=axs[1, 1])
    axs[1, 1].set_title('Función de Probabilidad Acumulada')

    # Ajustar el diseño de los gráficos
    fig.tight_layout()

    # Crear un canvas de Tkinter con la figura
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Función para generar el gráfico de violín
def grafico_violin(datos, ax):
    sns.violinplot(y=datos['Valor'], ax=ax, inner="box")
    ax.set_xlabel("Q [m3/s]")
    ax.set_ylabel(None)
    ax.set_title("Generalización de los datos")

# Función para generar el gráfico de distribución normal vs empírica
def grafico_distribucion(datos, ax):
    sns.kdeplot(datos['Valor'], color="red", ax=ax, label='Dist. Empírica')
    mu, std = norm.fit(datos['Valor'])
    x = pd.Series(sorted(datos['Valor']))
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'b-', label='Dist. Normal')
    ax.set_xlim(left=0)  # Iniciar desde 0
    ax.set_title("Dist. Normal vs. Empírica")
    ax.set_xlabel("Densidad")
    ax.set_xlabel("Q [m3/s]")
    ax.legend()

# Función para generar el gráfico Q-Q
def grafico_qq(datos, ax):
    sm.qqplot(datos['Valor'], line='s', ax=ax, markersize=3)
    ax.set_title("Gráfico Q-Q")
    ax.set_ylabel("Cuantiles Muestrales")
    ax.set_xlabel("Cuantiles Teoricos")

# Otras funciones para gráficos como ACF, PACF, etc.
def grafico_acf_pacf(datos, axs):
    sm.graphics.tsa.plot_acf(datos['Valor'], ax=axs[0])
    axs[0].set_title("ACF")
    sm.graphics.tsa.plot_pacf(datos['Valor'], ax=axs[1])
    axs[1].set_title("PACF")

# Función para gráficos de desestacionalización
def desestacionalizacion(datos, axs):
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(datos['Valor'], model='additive', period=12)
    result.observed.plot(ax=axs[0], title="Original")
    result.trend.plot(ax=axs[1], title="Tendencia")
    result.seasonal.plot(ax=axs[2], title="Estacionalidad")
    result.resid.plot(ax=axs[3], title="Residual")

# Función para el mapa de calor de estacionalidad
def heatmap_estacionalidad(datos, ax):
    """Genera un mapa de calor para la estacionalidad"""
    datos['Mes'] = datos['Fecha'].dt.month
    datos['Año'] = datos['Fecha'].dt.year
    monthly_avg = datos.groupby(['Año', 'Mes'])['Valor'].mean().reset_index()
    pivot_table = monthly_avg.pivot(index="Año", columns="Mes", values="Valor")

    sns.heatmap(pivot_table, cmap="coolwarm", ax=ax, cbar_kws={'label': 'Valor Promedio'})
    ax.set_title('Mapa de Calor de Estacionalidad')
    ax.set_xlabel('Mes')
    ax.set_ylabel('Año')

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

def graficar_acf_multiples_series(series_dict, ax, nlags=20, alpha=0.05):
    """
    Graficar ACF para múltiples series en un solo gráfico.
    
    Parámetros:
    - series_dict: Diccionario donde las llaves son los nombres de las series y los valores son las series de datos.
    - ax: El objeto `ax` de matplotlib donde se generará el gráfico.
    - nlags: Número de rezagos para calcular la ACF.
    - alpha: Nivel de significancia para los intervalos de confianza.
    """
    # Colores diferentes para cada serie
    colores = ['blue', 'green', 'red']
    
    for i, (nombre, serie) in enumerate(series_dict.items()):
        # Calcular ACF
        acf_vals, confint = acf(serie, nlags=nlags, alpha=alpha)
        confint = confint - acf_vals[:, None]
        
        # Graficar ACF
        ax.vlines(range(nlags + 1), [0], acf_vals, color=colores[i], label=f'ACF {nombre}')
        ax.fill_between(range(nlags + 1), confint[:, 0], confint[:, 1], color=colores[i], alpha=0.1)
    
    ax.axhline(y=0, linestyle='--', color='black')
    ax.set_title('ACF Múltiples Series')
    ax.legend()

def calcular_autocorrelacion_pearson(series, lags):
    autocorr = [series.autocorr(lag=i) for i in range(lags + 1)]
    N = len(series)
    limites_superior = [2 / np.sqrt(N - i) for i in range(lags + 1)]
    limites_inferior = [-2 / np.sqrt(N - i) for i in range(lags + 1)]
    return autocorr, limites_superior, limites_inferior

from scipy.stats import gumbel_r
import numpy as np
from scipy.stats import gumbel_r

def plot_return_level_gumbel(serie, ax, label_serie, color, exceedance=True):
    params = gumbel_r.fit(serie)
    loc, scale = params[0], params[1]
    return_periods = np.array([2.33, 5, 10, 15, 20, 25, 50, 75, 100])
    prob = 1 / return_periods if exceedance else 1 - 1 / return_periods
    return_levels = gumbel_r.ppf(prob, loc=loc, scale=scale)
    ax.plot(return_periods, return_levels, label=f'Return Level ({label_serie})', color=color, lw=2)
    ax.set_xlabel('Período de Retorno (Años)')
    ax.set_ylabel('Nivel de Retorno')
    ax.set_title(f'Nivel de Retorno - {label_serie} (Gumbel)')
    ax.legend()
    ax.grid(True)

def plot_return_level_weibull(serie, ax, label_serie, color, exceedance=True):
    params = weibull_min.fit(serie)
    shape, loc, scale = params[0], params[1], params[2]
    return_periods = np.array([2.33, 5, 10, 15, 20, 25, 50, 75, 100])
    prob = 1 / return_periods if exceedance else 1 - 1 / return_periods
    return_levels = weibull_min.ppf(prob, shape, loc=loc, scale=scale)
    ax.plot(return_periods, return_levels, label=f'Return Level ({label_serie})', color=color, lw=2)
    ax.set_xlabel('Período de Retorno (Años)')
    ax.set_ylabel('Nivel de Retorno')
    ax.set_title(f'Nivel de Retorno - {label_serie} (Weibull)')
    ax.legend()
    ax.grid(True)

from scipy.stats import invweibull

def plot_return_level_frechet(serie, ax, label_serie, color, exceedance=True):
    # Ajustar la serie de datos a la distribución invweibull (Fréchet)
    shape, _, scale = invweibull.fit(serie, floc=0)
    
    # Definir períodos de retorno
    return_periods = np.array([2.33, 5, 10, 15, 20, 25, 50, 75, 100])
    
    # Calcular la probabilidad de excedencia o no excedencia
    prob = 1 / return_periods if exceedance else 1 - 1 / return_periods
    
    # Calcular niveles de retorno usando invweibull
    return_levels = invweibull.ppf(prob, shape, loc=0, scale=scale)
    
    # Graficar los niveles de retorno
    ax.plot(return_periods, return_levels, label=f'Return Level ({label_serie})', color=color, lw=2)
    ax.set_xlabel('Período de Retorno (Años)')
    ax.set_ylabel('Nivel de Retorno')
    ax.set_title(f'Nivel de Retorno - {label_serie} (Fréchet)')
    ax.legend()
    ax.grid(True)

from scipy.stats import lognorm

def plot_return_level_lognormal(serie, ax, label_serie, color, exceedance=True):
    # Ajuste de parámetros para la distribución Lognormal
    shape, loc, scale = lognorm.fit(serie, floc=0)  # Se ajusta floc=0 para Lognormal
    return_periods = np.array([2.33, 5, 10, 15, 20, 25, 50, 75, 100])
    prob = 1 / return_periods if exceedance else 1 - 1 / return_periods
    return_levels = lognorm.ppf(prob, shape, loc=loc, scale=scale)
    
    # Graficar los niveles de retorno
    ax.plot(return_periods, return_levels, label=f'Return Level ({label_serie})', color=color, lw=2)
    ax.set_xlabel('Período de Retorno (Años)')
    ax.set_ylabel('Nivel de Retorno')
    ax.set_title(f'Nivel de Retorno - {label_serie} (Lognormal)')
    ax.legend()
    ax.grid(True)

import statsmodels.api as sm

def grafico_qq_personalizado(datos, ax, distribucion, params):
    """
    Genera un gráfico Q-Q en el eje dado (ax) usando una distribución personalizada.
    
    Parámetros:
    - datos: Serie de datos a comparar.
    - ax: Eje de matplotlib donde se genera el gráfico.
    - distribucion: Distribución de scipy.stats a utilizar para el Q-Q plot.
    - params: Parámetros de la distribución para el Q-Q plot.
    """
    # Separar parámetros
    if len(params) == 3:
        c, loc, scale = params
        distargs = (c,)  
    else:
        loc, scale = params
        distargs = ()

    # Ajustar la distribución en el gráfico Q-Q
    sm.qqplot(datos, dist=distribucion, line='45', ax=ax, distargs=distargs, loc=loc, scale=scale)
    ax.set_title("Gráfico Q-Q Personalizado")






