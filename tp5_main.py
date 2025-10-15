import pandas as pd
import matplotlib.pyplot as plt
import os
from tp5_ejercicio1 import generar_informe1
from tp5_ejercicio2 import generar_informe2

def cargar_datos_tp4():
    """Carga y unifica los datos del TP4"""
    print("=== CARGANDO DATOS DEL TP4 ===")

    # Verificar que los archivos existen
    archivos_requeridos = [
        "resultados_completos.xlsx",
        "resultados_completos2.xlsx",
        "resultados_completos3.xlsx"
    ]

    for archivo in archivos_requeridos:
        if not os.path.exists(archivo):
            raise FileNotFoundError(f"No se encuentra: {archivo}")

    # Cargar datos principales
    df_completo = pd.read_excel('resultados_completos.xlsx', sheet_name='Datos Completos')
    df_angulos = pd.read_excel('resultados_completos2.xlsx')
    df_geometricas = pd.read_excel('resultados_completos3.xlsx')

    print(f"Datos cargados:")
    print(f"- Datos completos: {len(df_completo)} registros")
    print(f"- Ángulos: {len(df_angulos)} registros")
    print(f"- Propiedades geométricas: {len(df_geometricas)} registros")

    # Unificar datos
    df_unificado = pd.merge(df_completo, df_angulos, on=['Imagen', 'Tiempo (s)'], how='inner')
    df_unificado = pd.merge(df_unificado, df_geometricas, on=['Imagen', 'Tiempo (s)'], how='inner')

    print(f"\nDataset unificado: {len(df_unificado)} registros")

    return df_unificado

if __name__ == "__main__":
    """Función principal del TP5"""
    print("\n" + "=" * 60)
    print("TRABAJO PRÁCTICO 5 - ANÁLISIS NUMÉRICO")
    print("=" * 60)

    df = cargar_datos_tp4()

    print("\n" + "=" * 60)
    print("EJERCICIO 1.")
    print("=" * 60)

    resultados_ej1 = generar_informe1()

    print("\n" + "=" * 60)
    print("EJERCICIO 2.")
    print("=" * 60)

    resultados_ej2 = generar_informe2(df)
