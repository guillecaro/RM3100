# Análisis de fluctuaciones con respecto a la temperatura

# Guillermo Caro
# Ingeniería Física, Universidad de Santiago de Chile
# Abril, 2025

# ---------------------
# Funciones
# ---------------------

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import re
import os
from scipy import stats
from scipy import odr
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from math import log10, floor

# Filtrar outliers usando z-score
def filtrar_outliers(df, columnas):
    z_scores = np.abs(stats.zscore(df[columnas]))
    df_filtrado = df[(z_scores < 3).all(axis=1)]  # Filtrar filas con z-scores < 3
    return df_filtrado

# Normalización por Z-score
def normalizar_datos(datos):
    # Medidas de dispersión
    media = datos.mean()
    desviacion_estandar = datos.std()
    datos_normalizados = (datos - media) / desviacion_estandar
    return datos_normalizados, media, desviacion_estandar

# Ajuste de curva lineal
def ajuste_lineal(x, y):
    def modelo_lineal(x, a, b):
        return a * x + b
    
    coeficientes, _ = curve_fit(modelo_lineal, x, y)
    pendiente, intercepto = coeficientes
    return pendiente, intercepto

# Ajuste de curva polinómica
def ajuste_polinomio(x, y, grado):
    coeficientes = np.polyfit(x, y, grado)
    polinomio = np.poly1d(coeficientes)
    return coeficientes, polinomio

# Función para calcular R^2
def calcular_r2(y_real, y_pred):
    ss_res = np.sum((y_real - y_pred)**2)  # Suma de los residuos al cuadrado
    ss_tot = np.sum((y_real - np.mean(y_real))**2)  # Suma total de los cuadrados
    r2 = 1 - (ss_res / ss_tot)
    return r2

# Coeficiente de Pearson
def pearson_coefficient(y_real, y_pred):
    r, _ = pearsonr(y_real, y_pred)  # Retorna r y el p-valor
    return r

# Método ODR
def ajuste_odr(x, y, error_a, error_t):
    # Extraer los datos del DataFrame
    x = x.values
    y = y.values
    
    # Crear array de errores para x (constante para todos los puntos)
    error_x = np.full_like(x, error_t)
    
    # Definir el modelo lineal y = a*x + b
    def modelo_lineal(params, x):
        a, b = params
        return a * x + b
    
    # Crear el objeto Model para ODR
    modelo_odr = odr.Model(modelo_lineal)
    
    # Crear el objeto RealData con los errores en ambos ejes
    datos_reales = odr.RealData(x, y, sx=error_x, sy=error_a)
    
    # Crear el objeto ODR con los datos y el modelo
    odr_instancia = odr.ODR(datos_reales, modelo_odr, beta0=[-1e-4, 1.0])
    
    # Ejecutar el ajuste
    resultado_odr = odr_instancia.run()
    
    return resultado_odr

# Graficar campo 2 vs. campo 1
def graficar_campo_campo(x, y, eje, nombre_archivo):
    global ecuaciones  # Usar la lista global de ecuaciones
    
    # Normalizar los datos
    x_normalizados, media_x, desviacion_x = normalizar_datos(x)
    y_normalizados, media_y, desviacion_y = normalizar_datos(y)
    
    # Gráfico
    fig, ax = plt.subplots(figsize=(16, 10))
    c = ax.hist2d(x_normalizados, y_normalizados, bins=60, cmap='plasma')
    fig.colorbar(c[3], ax=ax, label='Densidad de puntos')
    
    # Ajuste lineal
    pendiente, intercepto = ajuste_lineal(x_normalizados, y_normalizados)
    y_lineal = pendiente * x_normalizados + intercepto
    ax.plot(x_normalizados, y_lineal, color='lime', linestyle='-', linewidth=3,
             label=f'Ajuste lineal:\ny(x) = {pendiente:.3f}x')
    
    # Ajuste ODR
    e_x = 7/desviacion_x
    e_y = 7/desviacion_y
    
    # Llamar a la función ODR
    resultado_odr = ajuste_odr(x_normalizados, y_normalizados, e_y, e_x)
    
    # Extraer parámetros y errores
    pendiente_odr = resultado_odr.beta[0]
    intercepto_odr = resultado_odr.beta[1]
    
    y_ajuste_odr = pendiente_odr * x_normalizados + intercepto_odr
    pearson_odr = pearson_coefficient(y_normalizados, y_ajuste_odr)
    
    # Guardar la ecuación lineal en la lista
    ecuaciones.append({
        'archivo': nombre_archivo,
        'eje': eje,
        'tipo_ajuste': 'lineal',
        'variable': f'B{eje}1(cuentas)',  # Nombre de la variable
        'ecuacion': f'y(x) = {pendiente_odr:.3f}x + {intercepto_odr:.3f}',
        'r': pearson_odr,
        'pendiente': pendiente,  # Guardar la pendiente
        'intercepto': intercepto  # Guardar el intercepto
    })
    
    # Configuración del gráfico
    ax.set_xlabel('B$_{REF}$ normalizado', fontsize=16)
    ax.set_ylabel('B$_{SUT}$ normalizado', fontsize=16)
    ax.set_title(f'Campo mag. SUT vs. campo mag. REF\nEje {eje}\n({hora_inicio} - {hora_fin})\n', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=14, loc='upper left')
    
    # Mostrar gráfico
    plt.show()
    plt.close()
    
# Graficar campo 1 y 2 vs. tiempo
def graficar_campo_tiempo(df, campos, ejes, nombre_archivo):
    # Configuración de colores y etiquetas
    colores_campo = ['red', 'blue']
    
    # Obtener rango temporal para el título
    hora_inicio = df['Tiempo'].iloc[0]
    hora_fin = df['Tiempo'].iloc[-1]
    
    # Iterar sobre cada eje
    for i, eje in enumerate(ejes):
        # Calcular estadísticas
        campo_ref = campos[0]
        campo_sut = campos[1]
        datos_ref = df[campo_ref]
        datos_sut = df[campo_sut]
        
        # Estadísticas datos sin normalizar
        promedio_ref = np.mean(datos_ref)
        promedio_sut = np.mean(datos_sut)
        
        # Mínimos y máximos
        min_ref = datos_ref.min()
        max_ref = datos_ref.max()
        dif_ref = max_ref - min_ref
        min_sut = datos_sut.min()
        max_sut = datos_sut.max()
        dif_sut = max_sut - min_sut
        
        # Normalización de datos
        datos_ref_norm, media_ref, desv_ref = normalizar_datos(datos_ref)
        datos_sut_norm, media_sut, desv_sut = normalizar_datos(datos_sut)
        
        # Crear figura con 3 subplots
        fig, axs = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'Campo magnético vs. tiempo\nEje {eje}\n({hora_inicio} - {hora_fin})\n', fontsize=18)
        
        # Subplot 1:
        ax = axs[0]
        
        label_text_ref = (f'$B_{{REF}}$ (min. = {min_ref:.0f} nT,'
                          + f' máx. = {max_ref:.0f} nT, dif. = {dif_ref:.0f} nT):\n' + f'$\sigma$ = {desv_ref:.0f} nT\n'
                          + r'$\bar{B}_{REF} = $' + f'{promedio_ref:.0f} ' + r'$\pm$ ' + f'{u_B1} nT')
        ax.plot(df['Tiempo'], datos_ref_norm, color=colores_campo[0], linewidth=1, label=label_text_ref)
        
        # Configurar formato de tiempo en eje x (h:min:s)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        ax.set_ylabel(f'$B_{{{eje}}}$ normalizado', fontsize=16)
        ax.set_title('REF', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(fontsize=14)
        ax.tick_params(labelbottom=False)  # Ocultar etiquetas del eje x
        
        # Subplot 2:
        ax = axs[1]
        
        label_text_sut = (f'$B_{{SUT}}$ (min. = {min_sut:.0f} nT,'
                          + f' máx. = {max_sut:.0f} nT, dif. = {dif_sut:.0f} nT):\n' + f'$\sigma$ = {desv_sut:.0f} nT\n'
                          + r'$\bar{B}_{SUT} = $' + f'{promedio_sut:.0f} ' + r'$\pm$ ' + f'{u_B2} nT')
        ax.plot(df['Tiempo'], datos_ref_norm, color=colores_campo[1], linewidth=1, label=label_text_sut)
        
        # Mismo formato para eje x
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        ax.set_ylabel(f'$B_{{{eje}}}$ normalizado', fontsize=16)
        ax.set_xlabel('t (h:min:s)', fontsize=16)
        ax.set_title('SUT', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(fontsize=14)
        
        # Ajustar límites del eje x para todos los subplots
        for ax in axs:
            ax.set_xlim([hora_inicio, hora_fin])
        
        plt.tight_layout()
        plt.show()
        plt.close()
    
# Calcular frecuencias de sampleo
def calcular_frecuencia(data, columna, factor=1):
    tiempos = data[columna].values
    diferencias = np.diff(tiempos) / 1000 # Calcular diferencias entre datos consecutivos
    frecuencias = factor / diferencias  # Calcular frecuencia
    return frecuencias
    
# Graficar frecuencia de sampleo vs. tiempo
def graficar_frecuencia_tiempo(df, frecuencia1, frecuencia2, nombre_archivo):
    # Configuración estética
    colores_frec = ['red', 'blue', 'limegreen']
    
    # Obtener rango temporal
    hora_inicio = df['Tiempo'].iloc[0]
    hora_fin = df['Tiempo'].iloc[-1]
    
    # Ajustar longitud del tiempo si es necesario
    tiempo = df['Tiempo'].iloc[:len(frecuencia1)]
    
    # Crear figura con 3 subplots compartiendo eje x
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Frecuencia de muestreo vs tiempo\n({hora_inicio} - {hora_fin})\n', fontsize=18)
    
    # Promedios frecuencia
    promedio_frecuencia1 = frecuencia1.mean()
    promedio_frecuencia2 = frecuencia2.mean()
    desviacion_frecuencia1 = frecuencia1.std()
    desviacion_frecuencia2 = frecuencia2.std()
    
    # --- Subplot 1: Frecuencia Sensor REF ---    
    ax1.plot(tiempo, frecuencia1, color=colores_frec[0], linewidth=1,
             label=f'f$_{{REF}}$ ($\sigma$ = {desviacion_frecuencia1:.3f} Hz):\n' + r'$\bar{f}_{REF}$'
             + f' = {promedio_frecuencia1:.3f} Hz')
    ax1.set_ylabel('f (Hz)', fontsize=16)
    ax1.set_title('REF', fontsize=16)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.legend(fontsize=14)
    ax1.tick_params(labelbottom=False)  # Ocultar etiquetas del eje x
    
    # --- Subplot 2: Frecuencia Sensor SUT ---    
    ax2.plot(tiempo, frecuencia2, color=colores_frec[1], linewidth=1,
             label=f'f$_{{SUT}}$ ($\sigma$ = {desviacion_frecuencia2:.3f} Hz):\n' + r'$\bar{f}_{SUT}$'
             + f' = {promedio_frecuencia2:.3f} Hz')
    ax2.set_ylabel('f (Hz)', fontsize=16)
    ax2.set_title('SUT', fontsize=16)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.legend(fontsize=14)
    ax2.tick_params(labelbottom=False)  # Ocultar etiquetas del eje x
    
    # --- Subplot 3: Diferencia de frecuencias ---
    diferencia = frecuencia1 - frecuencia2
    media_diferencia = diferencia.mean()
    desviacion_diferencia = diferencia.std()
    
    ax3.plot(tiempo, diferencia, color=colores_frec[2], linewidth=1,
             label=f'$\Delta$f = f$_{{REF}}$ - f$_{{SUT}}$ ($\sigma = {to_sci_notation(desviacion_diferencia)}$ Hz):\n'
             + r'$\bar{\Delta f}$ = ' + f'${to_sci_notation(media_diferencia)}$ Hz')
    ax3.set_xlabel('t (h:min:s)', fontsize=16)
    ax3.set_ylabel('f (Hz)', fontsize=16)
    ax3.set_title('Diferencias', fontsize=16)
    ax3.tick_params(axis='both', labelsize=14)
    ax3.legend(fontsize=14, loc='lower left')
    
    # Formatear solo el último eje x (compartido por los 3 subplots)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # Ajustar límites temporales
    ax1.set_xlim([hora_inicio, hora_fin])
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
# Graficar temperatura vs. tiempo
def graficar_temperatura_tiempo(df, nombre_archivo):
    # Calcular estadísticas
    promedio_temperatura = df['Temperatura(C)'].mean()
    desviacion_temperatura = df['Temperatura(C)'].std()
    tiempo = df['Tiempo']  # Usar tiempo real (datetime)

    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Graficar temperatura
    ax.plot(tiempo, df['Temperatura(C)'], color='magenta', linewidth=1, alpha=1,
            label='$T_{{SUT}}$ ($\sigma$ = ' + f'{desviacion_temperatura:.2f}' + ' °C)')
    
    # Ajuste lineal
    tiempo_seg = (tiempo - tiempo.iloc[0]).dt.total_seconds().values  # Convertir a segundos para ajuste
    pendiente_lineal, intercepto_lineal = ajuste_lineal(tiempo_seg, df['Temperatura(C)'])
    r2_lineal = calcular_r2(df['Temperatura(C)'], pendiente_lineal * tiempo_seg + intercepto_lineal)
    
    ecuaciones.append({
        'archivo': nombre_archivo,
        'tipo_ajuste': 'lineal',
        'variable': 'Temperatura(C)',
        'ecuacion': f'y(x) = {pendiente_lineal:.2e}x + {intercepto_lineal:.2f}',
        'r': r2_lineal,
        'promedio': promedio_temperatura
    })
    
    # Ajuste polinómico
    coeficientes = np.polyfit(tiempo_seg, df['Temperatura(C)'], 2)
    polinomio = np.poly1d(coeficientes)
    r2_polinomio = calcular_r2(df['Temperatura(C)'], polinomio(tiempo_seg))
    
    coef_str = " + ".join([f"{coef:.2e}x^{i}" if i > 0 else f"{coef:.2f}" 
                         for i, coef in enumerate(reversed(coeficientes))])
    
    ecuaciones.append({
        'archivo': nombre_archivo,
        'tipo_ajuste': 'polinomio grado 2',
        'variable': 'Temperatura(C)',
        'ecuacion': f'y(x) = {coef_str}',
        'r': r2_polinomio,
        'promedio': promedio_temperatura
    })
    
    # Formatear eje x (h:min:s)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Promedio
    ax.plot([tiempo.min(), tiempo.max()], [promedio_temperatura, promedio_temperatura],
            '--', color='green', 
         label=r'$\bar{T}_{SUT} = $' + f'{promedio_temperatura:.2f} ' + r'$\pm$ 0.25 °C')
    
    # Ajustar límites temporales
    ax.set_xlim([hora_inicio, hora_fin])
    ax.set_xlabel('t (h:min:s)', fontsize=16)
    ax.set_ylabel('T (°C)', fontsize=16)
    ax.set_title(f'Temperatura SUT vs. tiempo\n({hora_inicio} - {hora_fin})\n', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=14)
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
# Filtrar por eje
def filtrar_por_eje(df, eje):
    return df[df['eje'] == eje]

def to_sci_notation(x, decimals=3):
    if x == 0:
        return "0"
    exp = floor(log10(abs(x)))
    coeff = x / 10**exp
    return f"{coeff:.{decimals}f}\\times10^{{{exp}}}"

# Función para graficar ajustes
def graficar_ajuste(ax, x, y, color_curva, label_curva, tipo_ajuste='lineal', grado=None, 
                    graficar_puntos=True, color_puntos='black', marker_puntos='.', label_puntos='Datos originales',
                    mostrar_leyenda=True, error_y=None, error_x=None):
    
    if graficar_puntos:
        ax.scatter(x, y, color=color_puntos, marker=marker_puntos, label=label_puntos, alpha=1)
    
    x_nuevo = np.linspace(-30, 44, 500)
    
    if tipo_ajuste == 'lineal':
        # Ajuste lineal
        pendiente_lineal, intercepto_lineal = ajuste_lineal(x, y)
        
        # Calcular coeficiente de Pearson
        y_pred = pendiente_lineal * x + intercepto_lineal
        pearson = pearson_coefficient(y, y_pred)
        
        y_ajuste = pendiente_lineal * x_nuevo + intercepto_lineal
        
        ax.plot(x_nuevo, y_ajuste, color=color_curva, linestyle='--', linewidth=2,
               label=f'{label_curva} (r = {pearson:.4f})')
    
    if mostrar_leyenda:
        ax.legend(fontsize=14)

# Función para graficar ajustes lineales globales
def graficar_ajuste_lineal_global(df_lineal_eje, eje):
    # Primera figura
    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
    fig1.suptitle('Pendiente vs. diferencia de temperatura\n'
                 + f'a = B$_{{SUT}}$ / B$_{{REF}}$, $\Delta$T = T$_{{SUT}}$ - T$_{{REF}}$\nEje {eje}\n', fontsize=18)
    
    # Errores
    error_a_x = [0.02021, 0.01746, 0.01365, 0.01441, 0.01271, 0.01961, 0.01422, 0.02713, 0.01844, 0.01264,
                 0.01815, 0.008857, 0.01976]
    error_a_y = [0.01102, 0.003236, 0.006351, 0.003609, 0.01410, 0.02024, 0.01629, 0.008331, 0.01168, 0.008993,
                 0.006139, 0.003715, 0.006753]
    error_a_z = [0.007941, 0.003098, 0.004696, 0.003330, 0.01029, 0.01937, 0.005320, 0.004295, 0.006514, 0.003762,
                 0.003203, 0.002820, 0.004520]
    
    # Seleccionar el vector de error según el eje
    if eje == 'X':
        error_a = error_a_x
    elif eje == 'Y':
        error_a = error_a_y
    elif eje == 'Z':
        error_a = error_a_z
    
    # Ordenar los datos por temperatura
    df_lineal_eje_ordenado = df_lineal_eje.sort_values(by='promedio')
    
    # Grafico 1: pendiente vs. temperatura
    y1 = df_lineal_eje_ordenado['pendiente']
    y_std1 = np.std(y1)
    
    # Barra de error constante para el eje x
    error_t = 0.25
    
    ax1.scatter(df_lineal_eje_ordenado['promedio'], df_lineal_eje_ordenado['pendiente'], color='red', marker='d',
                s=60, label='Pendiente ' + f'($\sigma$ = {y_std1:.3f})')
    
    # Ajuste lineal
    graficar_ajuste(ax1, df_lineal_eje_ordenado['promedio'], df_lineal_eje_ordenado['pendiente'], 'blue', 'Ajuste lineal',
                    tipo_ajuste='lineal', graficar_puntos=False, error_y=error_a, error_x=error_t)
    print(df_lineal_eje_ordenado['promedio'])
    print(df_lineal_eje_ordenado['pendiente'])
    ax1.set_title('Respuesta', fontsize=16)
    ax1.set_ylabel('a', fontsize=16)
    ax1.set_xlim(-30, 44)
    ax1.set_ylim(0.7, 1.1)
    ax1.tick_params(axis='both', labelsize=14)
    
    # Gráfico 2: error vs. temperatura
    e_r = (df_lineal_eje_ordenado['pendiente'] - 1) * 100
    y_mean2 = np.mean(e_r)
    y_std2 = np.std(e_r)
    print(e_r)
    
    ax2.scatter(df_lineal_eje_ordenado['promedio'], e_r, color='green', marker='^', s=60,
                label='Variación porcentual ' + f'($\sigma = {y_std2:.3f}$$\%$)')
    
    # Línea horizontal del promedio
    ax2.plot([-30, 44], [y_mean2, y_mean2], '--', color='darkorange',
             label=r'$\bar{\epsilon}$' + f' = {y_mean2:.3f}$\%$')
    
    ax2.set_title('Variación porcentual', fontsize=16)
    ax2.set_ylabel('$\epsilon$ (%)', fontsize=16)
    ax2.set_xlim(-30, 44)
    ax2.set_ylim(-35, 10)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_xlabel('$\Delta$T (°C)', fontsize=16)
    
    # Mostrar leyendas
    ax1.legend(fontsize=14, loc='lower right')
    ax2.legend(fontsize=14, loc='lower right')
        
    plt.tight_layout()
    plt.show()
    
    # Segunda figura
    fig2, (ax3, ax4) = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
    fig2.suptitle('Pendiente ponderada vs. diferencia de temperatura\n'
                 + f'a = B$_{{SUT}}$ / B$_{{REF}}$, $\Delta$T = T$_{{SUT}}$ - T$_{{REF}}$\nEje {eje}\n', fontsize=18)
    
    # --- Subplot 1: Producto pendiente y dif. de temperatura ---
    producto = df_lineal_eje_ordenado['pendiente'] * df_lineal_eje_ordenado['promedio']
    ax3.scatter(df_lineal_eje_ordenado['promedio'], producto, color='red', marker='d', s=60,
               label='Pendiente $\\times$ dif. de temperatura')
    
    # Añadir ajuste lineal
    graficar_ajuste(ax3, df_lineal_eje_ordenado['promedio'], producto, 'blue', 'Ajuste lineal',
                    tipo_ajuste='lineal', graficar_puntos=False)
    
    ax3.set_title('Respuesta', fontsize=16)
    ax3.set_ylabel('a$\\times\Delta$T (°C)', fontsize=16)
    ax3.set_xlim(-30, 44)
    ax3.set_ylim(-30, 40)
    ax3.tick_params(axis='both', labelsize=14)
    ax3.legend(fontsize=14)
    
    # --- Subplot 2: Variación relativa del producto ---
    variacion_relativa = (df_lineal_eje_ordenado['pendiente'] - 1) * 100  # En porcentaje
    
    # Estadísticas
    mean_variacion = np.mean(variacion_relativa)
    std_variacion = np.std(variacion_relativa)
    
    ax4.scatter(df_lineal_eje_ordenado['promedio'], variacion_relativa, color='green', marker='^', s=60,
                label=f'Variación porcentual ($\sigma$ = {std_variacion:.3f}%)')
    
    # Línea horizontal del promedio
    ax4.plot([-30, 44], [mean_variacion, mean_variacion], '--', color='darkorange',
             label=r'$\bar{\epsilon}$' + f' = {mean_variacion:.3f}$\%$')
    
    ax4.set_title('Variación porcentual', fontsize=16)
    ax4.set_xlabel('$\Delta$T (°C)', fontsize=16)
    ax4.set_ylabel('$\epsilon$ (%)', fontsize=16)
    ax4.set_xlim(-30, 44)
    ax4.set_ylim(-35, 10)
    ax4.tick_params(axis='both', labelsize=14)
    ax4.legend(fontsize=14, loc='lower right')
    
    plt.tight_layout()
    plt.show()
    
# ---------------------
# Llamadas a funciones
# ---------------------

# Codificación y lectura de archivos
ruta_base = 'C:/Users/gmoca/Documents/python_codigos/Tesis Python/Resultados/iman/data_equitemporal/'

lista_archivos = [
    'datos_03022025_210133_inicio_04022025_102126_fin', # Promedio = 60.38°C
    'datos_04022025_210133_inicio_05022025_102126_fin', # 25.30°C
    'datos_05022025_210133_inicio_06022025_102126_fin', # 46.01°C
    'datos_06022025_210133_inicio_07022025_102126_fin', # 32.44°C
    'datos_12022025_210133_inicio_13022025_102126_fin', # 4.30°C
    'datos_13022025_210133_inicio_14022025_102126_fin', # 1.31°C
    'datos_17012025_210133_inicio_18012025_102126_fin', # 0.69°C
    'datos_21012025_210133_inicio_22012025_102126_fin', # -3.61°C
    'datos_22012025_210133_inicio_23012025_102126_fin', # 3.02°C
    'datos_27012025_210133_inicio_28012025_102126_fin', # -3.90°C
    'datos_28012025_210133_inicio_29012025_102126_fin', # 30.62
    'datos_29012025_210133_inicio_30012025_102126_fin', # 41.49°C
    'datos_30012025_210133_inicio_31012025_102126_fin' # 52.79°C
    ]

temperatura_sensor1 = [
    # Promedio (°C) / Diferencias (°C)
    21.33, # 39.05
    20.14, # 5.16
    20.82, # 25.19
    20.61, # 11.83
    19.16, # -14.86
    18.44, # -17.13
    19.09, # -18.40
    17.28, # -20.90
    16.56, # -13.54
    20.68, # -24.58
    20.28, # 10.33
    18.52, # 22.97
    18.57 # 34.22
    ]

# Lista para almacenar las ecuaciones
ecuaciones = []

# Errores RM3100 (nT)
u_B1=7
u_B2=7

# Recorrer cada archivo en la lista
for nombre_archivo in lista_archivos:
    # Construir la ruta completa del archivo
    archivo = os.path.join(ruta_base, nombre_archivo + '.txt')
    
    # Leer el archivo
    data = pd.read_csv(archivo, sep=' ', header=0, encoding='ISO-8859-1')
    
    # Extraer horas de inicio y fin
    fechas = re.findall(r'\d{8}_\d{6}', nombre_archivo)
    fecha_inicio_str = fechas[0]
    fecha_fin_str = fechas[1]
    hora_inicio = pd.to_datetime(fecha_inicio_str, format='%d%m%Y_%H%M%S')
    hora_inicio_ajustada = hora_inicio
    hora_fin = pd.to_datetime(fecha_fin_str, format='%d%m%Y_%H%M%S')
    
    # Crear dataframe con promedios de 600 datos consecutivos (1 dato cada minuto)
    factor_promedio = 600
    data_promedio = data.groupby(data.index // factor_promedio).mean().reset_index(drop=True)
    
    # Convertir las columnas de microteslas a nanoteslas y renombrarlas
    columnas_muT = ['Bx1(muT)', 'Bx2(muT)', 'By1(muT)', 'By2(muT)', 'Bz1(muT)', 'Bz2(muT)']
    for col in columnas_muT:
        if col in data_promedio.columns:
            # Multiplicar por 1000 para convertir muT a nT
            data_promedio[col] = data_promedio[col] * 1000
            # Renombrar la columna
            nuevo_nombre = col.replace('(muT)', '(nT)')
            data_promedio.rename(columns={col: nuevo_nombre}, inplace=True)
    
    # Filtrar datos para los ejes
    columna_x = ['Bx1(nT)', 'Bx2(nT)']
    columna_y = ['By1(nT)', 'By2(nT)']
    columna_z = ['Bz1(nT)', 'Bz2(nT)']
    columna_temperatura = ['Temperatura(C)']
    
    # Aplicar filtro de outliers y crear columna de tiempo
    data_filtrada = filtrar_outliers(data_promedio, columna_x + columna_y + columna_z +
                                     columna_temperatura).copy()
    data_filtrada['Tiempo'] = pd.date_range(start=hora_inicio, end=hora_fin, periods=len(data_filtrada))
    
    # Rotacion de ejes X y Z del sensor 2
    columnas_rotacion = ["Bx2(nT)", "Bz2(nT)"]
    for columna in columnas_rotacion:
        if columna in data_filtrada.columns:
            data_filtrada[columna] *= -1
    
    # Gráficos de campo 2 vs. campo magnético 1
    graficar_campo_campo(data_filtrada['Bx1(nT)'], data_filtrada['Bx2(nT)'], 'X', nombre_archivo)
    graficar_campo_campo(data_filtrada['By1(nT)'], data_filtrada['By2(nT)'], 'Y', nombre_archivo)
    graficar_campo_campo(data_filtrada['Bz1(nT)'], data_filtrada['Bz2(nT)'], 'Z', nombre_archivo)
    
    # Gráficos de campo 1 y 2 vs. tiempo
    graficar_campo_tiempo(data_filtrada, ['Bx1(nT)', 'Bx2(nT)'], 'X', nombre_archivo)
    graficar_campo_tiempo(data_filtrada, ['By1(nT)', 'By2(nT)'], 'Y', nombre_archivo)
    graficar_campo_tiempo(data_filtrada, ['Bz1(nT)', 'Bz2(nT)'], 'Z', nombre_archivo)
    
    # Gráfico de frecuencia 2 vs. frecuencia 1
    frecuencia1 = calcular_frecuencia(data_filtrada, 'Tiempo1(ms)')
    frecuencia2 = calcular_frecuencia(data_filtrada, 'Tiempo2(ms)')
    df_frecuencias = pd.DataFrame({
        'Frecuencia1': frecuencia1 * factor_promedio,  # Ponderado por el número de filas para promediar
        'Frecuencia2': frecuencia2 * factor_promedio
    })
    
    # Aplicar filtro de outliers a las frecuencias
    df_frecuencias_filtradas = filtrar_outliers(df_frecuencias, ['Frecuencia1', 'Frecuencia2'])
    
    # Extraer las frecuencias filtradas
    frecuencia1_filtrada = df_frecuencias_filtradas['Frecuencia1'].values
    frecuencia2_filtrada = df_frecuencias_filtradas['Frecuencia2'].values
    
    # Graficos de frecuencias filtradas
    graficar_frecuencia_tiempo(data_filtrada, frecuencia1_filtrada, frecuencia2_filtrada, nombre_archivo)
    
    # Gráfico de temperatura vs. tiempo
    graficar_temperatura_tiempo(data_filtrada, nombre_archivo)

# Filtrar ajustes lineales
ajustes_lineales = [ecuacion for ecuacion in ecuaciones if ecuacion['tipo_ajuste'] == 'lineal']

# Convertir a DataFrames para facilitar el manejo
df_lineales = pd.DataFrame(ajustes_lineales)

# Rellenar NaN con 0 en las columnas relevantes de df_lineales
df_lineales['pendiente'].fillna(0, inplace=True)
df_lineales['intercepto'].fillna(0, inplace=True)

# Asignar el promedio de temperatura a los datos de campo magnético (para df_lineales)
for i in range(0, len(df_lineales), 4):
    # 1. Obtener temperatura sensor 2 (que ya está en el DataFrame)
    temperatura_sensor2 = df_lineales.loc[i + 3, 'promedio']
    
    # 2. Asignar temperatura sensor 2 a las primeras 3 filas
    df_lineales.loc[i:i+2, 'promedio'] = temperatura_sensor2
    
    # 3. Calcular y asignar delta temperatura
    archivo_idx = i // 4  # Índice del archivo (0-12)
    temperatura_sensor1_val = temperatura_sensor1[archivo_idx]  # Valor del sensor 1
    delta_temp = temperatura_sensor2 - temperatura_sensor1_val # T_SUT - T_REF
    
    # Asignar a todas las filas del grupo (4 filas)
    df_lineales.loc[i:i+3, 'temp_sensor1'] = temperatura_sensor1_val
    df_lineales.loc[i:i+3, 'temp_sensor2'] = temperatura_sensor2
    df_lineales.loc[i:i+3, 'delta_temp'] = delta_temp

# Bucle principal
ejes = ['X', 'Y', 'Z']

for eje in ejes:
    # Versión temperatura sensor 2
    df_lineal_eje = filtrar_por_eje(df_lineales, eje)

    # Versión delta temperatura
    df_lineal_eje_delta = df_lineal_eje.copy()
    df_lineal_eje_delta['promedio'] = df_lineal_eje_delta['delta_temp']  # Usar delta_temp en lugar del promedio
    
    # Graficar ambas versiones
    graficar_ajuste_lineal_global(df_lineal_eje_delta, f'{eje}')