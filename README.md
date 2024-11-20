# 🌊 **DataVista: Visualización y análisis hidrológico** 🌟

DataVista es una herramienta interactiva diseñada para la **visualización** y el **análisis** de datos hidrológicos. Este proyecto combina potentes librerías de procesamiento de datos y generación de gráficos en Python con una interfaz gráfica de usuario amigable basada en **Tkinter**. 🖥️ Está orientado a facilitar el análisis de series temporales, estadísticas descriptivas y patrones estacionales en caudales históricos.

####  📄 En este repositorio se incluye un archivo PDF titulado **`instrucciones.pdf`**, donde se detalla el funcionamiento del aplicativo y las consideraciones necesarias para su correcto uso.  

---

## ✨ **Características principales**
- 🧮 **Procesamiento avanzado de datos**:
  - ✅ Carga y validación de archivos **CSV** y **Excel** con datos hidrológicos.
  - 📊 Cálculo de estadísticos descriptivos básicos y avanzados, como **curtosis** y **varianza**.
- 📈 **Visualizaciones personalizadas**:
  - 📅 Gráficos de **series temporales**, histogramas, boxplots y gráficos de violín.
  - 🌡️ Análisis de patrones estacionales mediante mapas de calor y descomposición estacional.
  - 🌊 **Curvas de duración de caudales** con cálculos de **excedencia** y niveles de retorno (**Gumbel, Weibull, Fréchet,Pearson, Gamma, Lognormal**).
- 🌍 **Análisis ENSO**:
  - 🔄 Correlación de datos hidrológicos con fases ENSO (**El Niño, La Niña, Neutral**) y análisis de su impacto.
- 🖱️ **Interfaz gráfica de usuario**:
  - 📂 Selección de archivos mediante un explorador integrado.
  - 📋 Tablas interactivas y visualizaciones incrustadas.

---

## 🛠️ **Requisitos**
Este proyecto fue desarrollado en **Python 3.9** y depende de las siguientes bibliotecas:

- 📦 `pandas`
- 🎨 `matplotlib`
- 🖌️ `seaborn`
- 📐 `scipy`
- 📊 `statsmodels`
- 🖼️ `tkinter`
- 🖼️ `Pillow`

---

## 🐳 **Ejecución en Docker**

Si deseas ejecutar esta aplicación en un entorno aislado usando **Docker**, sigue estos pasos:

### 🛠️ **Crear y ejecutar el contenedor**
Ejecuta este comando para crear y correr el contenedor en **Windows PowerShell**. **Asegúrate de ajustar la ruta local de tu proyecto en la opción `-v`**:

```powershell
docker run -it --name datavista_container -e DISPLAY=host.docker.internal:0 -v "RUTA_LOCAL_DEL_PROYECTO:/app" -v /tmp/.X11-unix:/tmp/.X11-unix -p 8000:8000 python:3.11-slim /bin/bash
```

### 🚀 **Instrucciones dentro del contenedor**
Instala las dependencias del proyecto:

```bash
pip install --no-cache-dir -r /app/requirements.txt
apt-get update && apt-get install -y python3-tk libx11-6

```
Ejecuta la aplicación:

```bash
python /app/app.py
```
### 🔎 Notas
📂 Asegúrate de reemplazar `RUTA_LOCAL_DEL_PROYECTO` con la ruta absoluta de tu carpeta local.
Por ejemplo:
```powershell
-v "C:\Users\MiUsuario\MiProyecto:/app"
```

---

### 🖋️ **Créditos**

Desarrollado con dedicación por **Juan Camilo Ramírez Vidales** durante su pasantía profesional en **Servicios Hidrogeológicos Integrales S.A.S.**  
💡 Este proyecto tiene como objetivo facilitar el análisis hidrológico mediante herramientas de visualización avanzadas.  

📅 **Todos los derechos reservados © 2024.**  
🔒 **Este software no es libre ni de código abierto**. Su uso, distribución o reproducción están estrictamente prohibidos sin el consentimiento expreso de **Servicios Hidrogeológicos Integrales S.A.S.**  

🙌 Agradecimientos al equipo de Servicios Hidrogeológicos Integrales S.A.S. por su apoyo y guía.  

📬 **Contacto:** [juancramirez0331@gmail.com](mailto:juancramirez0331@gmail.com)  