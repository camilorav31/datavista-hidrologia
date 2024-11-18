# **DataVista: Visualización y análisis hidrológico**

DataVista es una herramienta interactiva diseñada para la visualización y el análisis de datos hidrológicos. Este proyecto combina potentes librerías de procesamiento de datos y generación de gráficos en Python con una interfaz gráfica de usuario amigable basada en **Tkinter**. Está orientado a facilitar el análisis de series temporales, estadísticas descriptivas y patrones estacionales en caudales históricos.

---

## **Características principales**
- **Procesamiento avanzado de datos**:
  - Carga y validación de archivos CSV y Excel con datos hidrológicos.
  - Cálculo de estadísticos descriptivos básicos y avanzados, como curtosis y varianza.
- **Visualizaciones personalizadas**:
  - Gráficos de series temporales, histogramas, boxplots y gráficos de violín.
  - Análisis de patrones estacionales mediante mapas de calor y descomposición estacional.
  - Curvas de duración de caudales con cálculos de excedencia y niveles de retorno (Gumbel, Weibull, Fréchet).
- **Análisis ENSO**:
  - Correlación de datos hidrológicos con fases ENSO (El Niño, La Niña, Neutral) y análisis de su impacto.
- **Interfaz gráfica de usuario**:
  - Selección de archivos mediante un explorador integrado.
  - Tablas interactivas y visualizaciones incrustadas.

---

## **Requisitos**
Este proyecto fue desarrollado en Python 3.9 y depende de las siguientes bibliotecas:
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `statsmodels`
- `tkinter`
- `Pillow`

Asegúrate de instalar las dependencias con:
```bash
pip install -r requirements.txt
