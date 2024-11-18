import pandas as pd
from scipy import stats

def calcular_estadisticos(datos):
    # Cálculo de estadísticos descriptivos básicos
    estadisticos = datos['Valor'].describe().round(2).to_dict()
    
    # Cálculo de otros estadísticos como varianza y curtosis
    estadisticos['varianza'] = round(datos['Valor'].var(), 2)
    estadisticos['curtosis'] = round(stats.kurtosis(datos['Valor'], fisher=False), 2)
    
    return estadisticos

def cargar_datos(file_path):
    if file_path.endswith('.csv'):
        datos = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        datos = pd.read_excel(file_path)
    else:
        raise ValueError("Formato de archivo no soportado")
    
    # Validaciones
    if 'Fecha' not in datos.columns or 'Valor' not in datos.columns:
        raise ValueError("El archivo debe contener las columnas 'Fecha' y 'Valor'")
    
    # Convertir a formato de fecha
    datos['Fecha'] = pd.to_datetime(datos['Fecha'])
    
    return datos


def cargar_datos_enso(ruta_csv='assets/enso.csv'):
    """
    Carga y transforma el archivo ENSO CSV en formato largo con columnas 'Fecha' y 'ONI'.
    
    Parámetros:
    ruta_csv (str): Ruta al archivo CSV que contiene los datos ENSO en formato matriz sin encabezados.
    
    Retorna:
    pd.DataFrame: DataFrame con columnas 'Fecha' (YYYY-MM-DD) y 'ONI' en formato largo.
    """
    
    # Cargar el archivo CSV sin encabezados
    enso_df = pd.read_csv(ruta_csv, header=None)
    
    # Nombrar columnas (la primera columna es 'Año' y las demás son los meses de enero a diciembre)
    enso_df.columns = ['Año', 'Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    # Transformar el DataFrame a formato largo (Fecha y ONI)
    enso_long = enso_df.melt(id_vars=['Año'], var_name='Mes', value_name='ONI')
    
    # Mapear los nombres de los meses a números para crear una columna de fecha
    mes_a_numero = {'Ene': 1, 'Feb': 2, 'Mar': 3, 'Abr': 4, 'May': 5, 'Jun': 6,
                    'Jul': 7, 'Ago': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dic': 12}
    enso_long['Mes'] = enso_long['Mes'].map(mes_a_numero)
    
    # Renombrar las columnas para que pandas interprete correctamente los datos de fecha
    enso_long = enso_long.rename(columns={'Año': 'year', 'Mes': 'month'})
    
    # Crear la columna de fecha
    enso_long['Fecha'] = pd.to_datetime(enso_long[['year', 'month']].assign(day=1))
    
    # Filtrar datos faltantes (-99.90) y reorganizar
    enso_long = enso_long[enso_long['ONI'] != -99.90].drop(columns=['year', 'month'])
    
    # Ordenar por fecha y reiniciar índice
    enso_long = enso_long.sort_values(by='Fecha').reset_index(drop=True)
    
    return enso_long