import pandas as pd
import numpy as np
import holidays
import io
import csv
from datetime import timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import glob
import os
import logging

# Configuración global
HORIZONTE_PRONOSTICO = 25  # Semanas a pronosticar
RUTA_RESULTADOS = 'resultados'
RUTA_DATOS = (r'C:\Users\seelro06\OneDrive - Arca Continental S.A.B. de C.V'
              r'\Documentos\uni\del-negociante\data\competidores'
              r'\Competidor*.csv')
AÑOS_FERIADOS = (2019, 2022)  # Rango de años para feriados

# Configurar logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Precalcular datos comunes: feriados en México
FERIADOS_MX = holidays.Mexico(years=range(*AÑOS_FERIADOS)).items()
DF_FERIADOS = pd.DataFrame(FERIADOS_MX, columns=['ds', 'holiday'])
DF_FERIADOS['ds'] = pd.to_datetime(DF_FERIADOS['ds'])

import pandas as pd
import csv

def cargar_datos(ruta_archivo: str) -> pd.DataFrame:
    """Carga y procesa datos de un archivo CSV de competidor, corrigiendo el header."""
    # Intentar leer con distintos separadores y detectar encabezados
    df = None
    separators = [',', ';', '\t']  # Posibles delimitadores
    
    for sep in separators:
        try:
            with open(ruta_archivo, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip().replace('"', '')  # Limpiar comillas dobles
            
            columns = first_line.split(sep)
            df = pd.read_csv(ruta_archivo, sep=sep, names=columns, skiprows=1, dtype={'YMD': str}, on_bad_lines='skip')

            if df.shape[1] > 1:  # Asegurar que hay más de una columna
                break
        except Exception:
            continue

    if df is None or df.shape[1] <= 1:
        print(f"Error al procesar {ruta_archivo}: No se pudieron detectar columnas correctamente.")
        return

    # Convertir la columna de fecha
    df['YMD'] = pd.to_datetime(df['YMD'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['YMD'])  # Eliminar filas con fechas no válidas

    return df


# Función auxiliar para preprocesar datos para Prophet
def preprocesar_prophet(s: pd.DataFrame) -> pd.DataFrame:
    """
    Restablece el índice, renombra la columna de fechas a 'ds' y
    la columna de producto (la primera que encuentre) a 'y'.
    """
    df = s.reset_index()  # Convierte el índice en columna
    # Buscar la columna de fecha, asumiendo que ya fue convertida (por ejemplo, 'YMD' o la detectada)
    date_cols = [col for col in df.columns if 'date' in col.lower() or col.lower() == 'ymd']
    if not date_cols:
        raise ValueError("No se encontró columna de fecha")
    # Usamos la primera columna de fecha encontrada
    fecha_col = date_cols[0]
    # Buscar columnas de producto (excluyendo la columna de fecha)
    product_cols = [col for col in df.columns if col not in [fecha_col]]
    if not product_cols:
        raise ValueError("No se encontró columna de producto")
    # Renombrar: la columna de fecha a 'ds' y la primera columna de producto a 'y'
    df = df.rename(columns={fecha_col: 'ds', product_cols[0]: 'y'})
    # Convertir 'ds' a datetime y 'y' a numérico
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['ds', 'y'])
    return df

# Definición de experimentos
CONFIG_EXPERIMENTOS = {
    # Holt-Winters
    # 'Holt-Winters Original': {
    #     'preprocesar': lambda s: s,
    #     'modelo': lambda t: ExponentialSmoothing(t, trend='add',
    #                                              seasonal='add',
    #                                              seasonal_periods=52).fit(),
    #     'pronosticar': lambda m, h: m.forecast(h)
    # },
    
    # Prophet Base
    'Prophet Base': {
        'preprocesar': preprocesar_prophet,
        'modelo': lambda t: Prophet().fit(t),
        'pronosticar': lambda m, h: m.make_future_dataframe(periods=h, freq='W').pipe(
            lambda df: m.predict(df)['yhat'].tail(h)
        )
    },
    
    # Prophet Avanzado
    # 'Prophet Avanzado': {
    #     'preprocesar': preprocesar_prophet,
    #     'modelo': lambda t: Prophet(
    #         seasonality_mode='multiplicative',
    #         changepoint_prior_scale=0.2,
    #         holidays=DF_FERIADOS
    #     ).add_seasonality(name='mensual', period=30.5, fourier_order=5).fit(t),
    #     'pronosticar': lambda m, h: m.make_future_dataframe(periods=h, freq='W').pipe(
    #         lambda df: m.predict(df)['yhat'].tail(h)
    #     )
    # },
    
    # # SARIMA
    # 'SARIMA Estacional': {
    #     'preprocesar': lambda s: s,
    #     'modelo': lambda t: SARIMAX(t, order=(2,1,1),
    #                                 seasonal_order=(1,1,1,52)).fit(disp=False),
    #     'pronosticar': lambda m, h: m.forecast(h)
    # }
}

def procesar_competidor(ruta_archivo: str) -> list:
    """Procesa un archivo de competidor y genera pronósticos"""
    try:
        datos = cargar_datos(ruta_archivo)
        if datos.empty:
            return []
        
        # Asegurar que la columna de fecha esté como índice
        if 'YMD' in datos.columns:
            datos['YMD'] = pd.to_datetime(datos['YMD'], errors='coerce')
            datos = datos.set_index('YMD')

        nombre_competidor = os.path.basename(ruta_archivo).split('.')[0]
        # Seleccionar columnas que comiencen con 'Product'
        productos = [col for col in datos.columns if col.lower().startswith('product')]
        resultados = []

        for producto in productos:
            # Re-muestrear a semanal y rellenar
            serie = datos[producto].resample('W').last().ffill()
            train = (serie.iloc[:-HORIZONTE_PRONOSTICO]
                     if len(serie) > HORIZONTE_PRONOSTICO else serie)
            test = (serie.iloc[-HORIZONTE_PRONOSTICO:]
                    if len(serie) > HORIZONTE_PRONOSTICO else None)

            for nombre_exp, config in CONFIG_EXPERIMENTOS.items():
                try:
                    # Preprocesamiento: se convierte la serie en DataFrame
                    datos_procesados = config['preprocesar'](train.to_frame())
                    # Modelado
                    modelo = config['modelo'](datos_procesados)
                    # Pronóstico
                    pronostico = config['pronosticar'](modelo, HORIZONTE_PRONOSTICO)
                    
                    # Cálculo de métricas
                    metricas = {}
                    if test is not None:
                        y_true = test.values
                        y_pred = pronostico[:len(test)]
                        metricas = {
                            'MAE': mean_absolute_error(y_true, y_pred),
                            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
                        }
                    
                    # Guardar resultados
                    resultados.append({
                        'Experimento': nombre_exp,
                        'Competidor': nombre_competidor,
                        'Producto': producto,
                        'Metricas': metricas,
                        'Pronostico': pronostico,
                        'Real': serie  # Datos históricos completos
                    })

                    # Generar gráfico
                    generar_grafico(
                        nombre_competidor, producto,
                        nombre_exp, serie, pronostico
                    )

                except Exception as e:
                    logging.error(f"Error en {nombre_exp}: {str(e)}")

        return resultados

    except Exception as e:
        logging.error(f"Error procesando 2 {ruta_archivo}: {str(e)}")
        return []

def generar_grafico(competidor: str, producto: str, experimento: str,
                    real: pd.Series, pronostico: pd.Series):
    """Genera y guarda gráficos comparativos"""
    plt.figure(figsize=(12, 6), tight_layout=True)
    plt.title(f"{competidor} - {producto} ({experimento})")
    
    # Graficar datos históricos
    plt.plot(real.index, real.values, label='Ventas Reales')
    
    # Generar fechas futuras alineadas con los datos históricos
    ultima_fecha = real.index[-1]
    fechas_futuras = pd.date_range(
        start=ultima_fecha + timedelta(weeks=1),
        periods=len(pronostico),
        freq='W'
    )
    
    # Graficar pronóstico
    plt.plot(fechas_futuras, pronostico.values, label='Pronóstico', linestyle='--')
    
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{RUTA_RESULTADOS}/{competidor}_{producto}_{experimento.replace(' ', '_')}.png", dpi=100)
    plt.close()

def generar_reporte(resultados: list):
    """Genera reportes consolidados de resultados"""
    # Métricas globales
    metricas_globales = pd.DataFrame([
        {**res['Metricas'], 'Experimento': res['Experimento']}
        for res in resultados if res['Metricas']
    ])

    if not metricas_globales.empty:
        resumen = metricas_globales.groupby('Experimento').agg(
            {'MAE': 'mean', 'RMSE': 'mean'}
        )
        logging.info("\nResumen de Métricas Globales:\n%s", resumen.round(2))

    # Tendencias
    for res in resultados:
        if res['Pronostico'] is not None and not res['Pronostico'].empty:
            tendencia = (
                'creciente'
                if res['Pronostico'].iloc[-1] > res['Pronostico'].iloc[0]
                else 'decreciente'
            )
            mae = res['Metricas'].get('MAE', None)
            logging.info(
                f"\n{res['Competidor']} - {res['Producto']} ({res['Experimento']}): "
                f"Tendencia {tendencia} | MAE: {mae:.2f}" if mae is not None else ""
            )

# Función para generar estructura de datos para Power BI
def generar_datos_powerbi(resultados):
    """Genera datasets estructurados para Power BI"""
    pronosticos_lista = []
    metricas_lista = []

    for res in resultados:
        # Datos básicos
        competidor = res['Competidor']
        producto = res['Producto']
        experimento = res['Experimento']
        
        # 1. Procesar datos históricos y pronósticos
        try:
            # Datos históricos
            historico = res['Real'].reset_index(name='Valor')
            historico.columns = ['Fecha', 'Valor']
            historico['Tipo'] = 'Real'
            historico['Experimento'] = experimento
            
            # Generar fechas futuras
            ultima_fecha = historico['Fecha'].max()
            fechas_futuras = pd.date_range(
                start=ultima_fecha + timedelta(weeks=1),
                periods=HORIZONTE_PRONOSTICO,
                freq='W'
            )
            
            # Datos de pronóstico
            pronostico_df = pd.DataFrame({
                'Fecha': fechas_futuras,
                'Valor': res['Pronostico'].values,
                'Tipo': 'Pronóstico',
                'Experimento': experimento
            })
            
            # Combinar datos
            df_full = pd.concat([historico, pronostico_df])
            df_full.insert(0, 'Competidor', competidor)
            df_full.insert(1, 'Producto', producto)
            
            pronosticos_lista.append(df_full)
            
        except Exception as e:
            logging.error(f"Error procesando pronósticos: {str(e)}")

        # 2. Calcular métricas de negocio
        if res['Metricas']:
            try:
                ventas_promedio = res['Real'].mean()
                ultimo_real = res['Real'].iloc[-1]
                primer_pronostico = res['Pronostico'].iloc[0]
                
                metricas_lista.append({
                    'Competidor': competidor,
                    'Producto': producto,
                    'Experimento': experimento,
                    'MAE': res['Metricas'].get('MAE', None),
                    'RMSE': res['Metricas'].get('RMSE', None),
                    'Confianza (%)': (1 - (res['Metricas']['RMSE']/ventas_promedio)) * 100,
                    'Variación Esperada (%)': ((primer_pronostico/ultimo_real - 1) * 100),
                    'Riesgo': min(5, max(1, int(res['Metricas']['MAE']/(ventas_promedio/10))))
                })
            except Exception as e:
                logging.error(f"Error calculando métricas: {str(e)}")

    # Consolidar datos
    pronosticos = pd.concat(pronosticos_lista, ignore_index=True)
    metricas = pd.DataFrame(metricas_lista)
    
    # 3. Calcular tendencias
    try:
        tendencias = pronosticos.groupby(
            ['Competidor', 'Producto', 'Experimento', 'Tipo'], 
            as_index=False
        )['Valor'].last()
        
        tendencias_pivot = tendencias.pivot_table(
            index=['Competidor', 'Producto', 'Experimento'],
            columns='Tipo',
            values='Valor'
        ).reset_index()
        
        tendencias_pivot['Recomendación'] = np.where(
            tendencias_pivot['Pronóstico'] > tendencias_pivot['Real'],
            'Aumentar producción',
            'Mantener niveles'
        )
    except Exception as e:
        logging.error(f"Error generando tendencias: {str(e)}")
        tendencias_pivot = pd.DataFrame()

    return {
        'Pronosticos': pronosticos,
        'Metricas': metricas,
        'Tendencias': tendencias_pivot
    }

if __name__ == "__main__":
    os.makedirs(RUTA_RESULTADOS, exist_ok=True)
    archivos = glob.glob(RUTA_DATOS)
    todos_resultados = []

    for archivo in archivos:
        todos_resultados.extend(procesar_competidor(archivo))

    generar_reporte(todos_resultados)

    datos_pbi = generar_datos_powerbi(todos_resultados)

    # Exportar todos los componentes
    datos_pbi['Pronosticos'].to_csv('pronosticos_ventas.csv', index=False)
    datos_pbi['Metricas'].to_csv('metricas_modelos.csv', index=False)
    datos_pbi['Tendencias'].to_csv('tendencias_clave.csv', index=False)
