# Usa una imagen base oficial de Python
FROM python:3.11-slim

# Instala las dependencias del sistema necesarias para tkinter
RUN apt-get update && apt-get install -y \
    python3-tk \
    libx11-6 \
    && apt-get clean

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos de tu proyecto al contenedor
COPY . /app

# Instala las dependencias
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expone el puerto que tu aplicación utiliza (ajusta según sea necesario)
EXPOSE 8000

# Comando por defecto para ejecutar el contenedor
CMD ["python", "app.py"]  