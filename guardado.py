# Guardado de datos de puerto serial como archivo .txt

import serial
import time
import os

# Configuración del puerto serial
puerto = 'COM6'  # Puerto al que está conectado el Arduino
baud_rate = 9600  # Baudaje configurado en el Arduino

# Crear una conexión serial
ser = serial.Serial(puerto, baud_rate)

try:
    # Generar nombre de archivo único con la hora de inicio
    hora_inicio = time.strftime("%d%m%Y_%H%M%S")
    nombre_archivo = f'datos_{hora_inicio}_inicio.txt'

    # Abrir el archivo en modo escritura
    with open(nombre_archivo, 'w') as archivo:
        print(f"Guardando datos en {nombre_archivo}...")
        while True:
            if ser.in_waiting > 0:
                # Leer una línea de datos desde el puerto serial
                linea = ser.readline().decode('utf-8').strip()
                # Imprimir la línea en la consola
                print(linea)
                # Escribir la línea en el archivo
                archivo.write(linea + '\n')

except KeyboardInterrupt:
    print("Terminando la captura de datos.")

finally:
    # Cerrar el archivo y el puerto serial
    if ser.is_open:
        ser.close()

    # Obtener la hora de fin y renombrar el archivo
    try:
        hora_fin = time.strftime("%d%m%Y_%H%M%S")
        nuevo_nombre_archivo = f'datos_{hora_inicio}_inicio_{hora_fin}_fin.txt'
        os.rename(nombre_archivo, nuevo_nombre_archivo)
        print(f"Los datos han sido guardados en {nuevo_nombre_archivo}.")
    except PermissionError as e:
        print(f"Error al renombrar el archivo: {e}")
        print(f"El archivo se ha guardado con el nombre original: {nombre_archivo}")
