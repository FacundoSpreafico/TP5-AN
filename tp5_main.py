# tp5_main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def cargar_datos_tp4():
    """Carga y unifica los datos del TP4"""
    print("=== CARGANDO DATOS DEL TP4 ===")

    # Verificar que los archivos existen
    archivos_requeridos = [
        'resultados_completos.xlsx',
        'resultados_completos2.xlsx',
        'resultados_completos3.xlsx'
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
    print("=== TRABAJO PRÁCTICO 5 - ANÁLISIS NUMÉRICO ===")
    print("Análisis de propiedades geométricas y dinámica de gotas")
    print("Basado en datos reales del TP4\n")

    # Cargar datos del TP4
    df = cargar_datos_tp4()

    # EJERCICIO 1: Volumen y Área
    print("\n" + "=" * 60)
    print("EJERCICIO 1: Cálculo de volumen y área de la gota")
    print("=" * 60)

    # Importar y ejecutar ejercicio 1
    from tp5_ejercicio1 import generar_informe1
    resultados_ej1 = generar_informe1()

    # EJERCICIO 2: Dinámica de la gota
    print("\n" + "=" * 60)
    print("EJERCICIO 2: Modelo de la dinámica de la gota")
    print("=" * 60)

    # Importar y ejecutar ejercicio 2
    from tp5_ejercicio2 import generar_informe2
    resultados_ej2 = generar_informe2(df)

    # Generar informe consolidado
    print("\n" + "=" * 60)
    print("INFORME FINAL TP5")
    print("=" * 60)
    print("✓ Ejercicio 1 completado: Volúmenes y áreas calculadas")
    print("✓ Ejercicio 2 completado: Modelo dinámico implementado")
    print("✓ Resultados guardados en:")
    print("  - resultados_tp5_volumen_area.xlsx")
    print("  - resultados_tp5_dinamica.xlsx")
    print("  - graficos_volumen_area_tp5.png")
    print("  - graficos_dinamica_tp5.png")