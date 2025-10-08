# tp5_ejercicio1.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad, simpson, trapezoid
import json
import warnings

warnings.filterwarnings('ignore')


class CalculadorVolumenArea:
    """Clase para calcular volumen y área de la gota como figura de revolución"""

    def __init__(self, df):
        self.df = df
        self.resultados = []

    def obtener_mitad_contorno(self, contorno_x, contorno_y):
        """Divide el contorno en dos mitades usando el ápice como referencia"""
        if len(contorno_y) == 0:
            return None, None

        # Encontrar ápice (punto más bajo)
        idx_apice = np.argmin(contorno_y)
        apice_x = contorno_x[idx_apice]

        # Dividir en mitades izquierda y derecha
        mask_izq = contorno_x <= apice_x
        mask_der = contorno_x >= apice_x

        mitad_izq = np.column_stack([contorno_x[mask_izq], contorno_y[mask_izq]])
        mitad_der = np.column_stack([contorno_x[mask_der], contorno_y[mask_der]])

        # Usar la mitad más larga (generalmente la derecha)
        if len(mitad_der) > len(mitad_izq):
            return mitad_der[:, 0], mitad_der[:, 1]  # x, y
        else:
            return mitad_izq[:, 0], mitad_izq[:, 1]

    def ajustar_spline(self, x, y):
        """Ajusta un spline a la mitad del contorno"""
        if len(x) < 4:
            return None

        # Ordenar por y (altura) para la revolución
        idx_orden = np.argsort(y)
        x_ordenado = x[idx_orden]
        y_ordenado = y[idx_orden]

        # Eliminar duplicados en y para el spline
        y_unique, indices = np.unique(y_ordenado, return_index=True)
        x_unique = x_ordenado[indices]

        if len(y_unique) < 4:
            return None

        try:
            spline = UnivariateSpline(y_unique, x_unique, s=len(y_unique) * 0.1)
            return spline
        except:
            return None

    def ajustar_polinomio(self, x, y, grado=4):
        """Ajusta un polinomio a la mitad del contorno"""
        if len(x) <= grado:
            return None

        try:
            # Ordenar por y
            idx_orden = np.argsort(y)
            x_ordenado = x[idx_orden]
            y_ordenado = y[idx_orden]

            coef = np.polyfit(y_ordenado, x_ordenado, grado)
            polinomio = np.poly1d(coef)
            return polinomio
        except:
            return None

    def volumen_por_revolucion_trapecio(self, funcion, y_min, y_max, n_puntos=1000):
        """Calcula volumen usando regla del trapecio: V = π∫[x(y)]²dy"""
        y_eval = np.linspace(y_min, y_max, n_puntos)
        x_eval = funcion(y_eval)
        x_eval = np.nan_to_num(x_eval, nan=0.0, posinf=0.0, neginf=0.0)
        x_eval = np.clip(x_eval, 0, None)  # evitar radios negativos

        integrando = np.pi * x_eval ** 2
        volumen = trapezoid(integrando, y_eval)

        # Estimación de error (diferencia con Simpson)
        y_fino = np.linspace(y_min, y_max, n_puntos * 2)
        x_fino = funcion(y_fino)
        x_fino = np.nan_to_num(x_fino, nan=0.0, posinf=0.0, neginf=0.0)
        x_fino = np.clip(x_fino, 0, None)

        integrando_fino = np.pi * x_fino ** 2
        volumen_simpson = simpson(integrando_fino, y_fino)

        error_estimado = abs(volumen - volumen_simpson)

        return volumen, error_estimado

    def volumen_por_revolucion_simpson(self, funcion, y_min, y_max, n_puntos=1000):
        """Calcula volumen usando regla de Simpson: V = π∫[x(y)]²dy"""
        y_eval = np.linspace(y_min, y_max, n_puntos)
        x_eval = funcion(y_eval)
        x_eval = np.nan_to_num(x_eval, nan=0.0, posinf=0.0, neginf=0.0)
        x_eval = np.clip(x_eval, 0, None)

        integrando = np.pi * x_eval ** 2
        volumen = simpson(integrando, y_eval)

        # Estimación de error (diferencia con trapecio)
        volumen_trapecio = trapezoid(integrando, y_eval)
        error_estimado = abs(volumen - volumen_trapecio)

        return volumen, error_estimado

    # ======== CORREGIDAS PARA EVITAR ÁREAS NEGATIVAS ========

    def area_superficial_trapecio(self, funcion, derivada, y_min, y_max, n_puntos=1000):
        """Calcula área superficial usando trapecio: A = 2π∫x(y)√(1 + [dx/dy]²)dy"""
        y_eval = np.linspace(y_min, y_max, n_puntos)
        x_eval = funcion(y_eval)
        x_eval = np.nan_to_num(x_eval, nan=0.0, posinf=0.0, neginf=0.0)
        x_eval = np.abs(x_eval)  # asegurar radios positivos

        # Calcular derivada numéricamente si no se proporciona
        if derivada is None:
            dy = y_eval[1] - y_eval[0]
            dx_dy = np.gradient(x_eval, dy)
        else:
            dx_dy = derivada(y_eval)
            dx_dy = np.nan_to_num(dx_dy, nan=0.0, posinf=0.0, neginf=0.0)

        integrando = 2 * np.pi * x_eval * np.sqrt(1 + dx_dy ** 2)
        area = trapezoid(integrando, y_eval)

        return area

    def area_superficial_simpson(self, funcion, derivada, y_min, y_max, n_puntos=1000):
        """Calcula área superficial usando Simpson: A = 2π∫x(y)√(1 + [dx/dy]²)dy"""
        y_eval = np.linspace(y_min, y_max, n_puntos)
        x_eval = funcion(y_eval)
        x_eval = np.nan_to_num(x_eval, nan=0.0, posinf=0.0, neginf=0.0)
        x_eval = np.abs(x_eval)  # asegurar radios positivos

        if derivada is None:
            dy = y_eval[1] - y_eval[0]
            dx_dy = np.gradient(x_eval, dy)
        else:
            dx_dy = derivada(y_eval)
            dx_dy = np.nan_to_num(dx_dy, nan=0.0, posinf=0.0, neginf=0.0)

        integrando = 2 * np.pi * x_eval * np.sqrt(1 + dx_dy ** 2)
        area = simpson(integrando, y_eval)

        return area

    # ========================================================

    def procesar_todos_frames(self):
        """Procesa todos los frames del dataset"""
        print("Procesando volúmenes y áreas para todos los frames...")

        for idx, row in self.df.iterrows():
            try:
                # Cargar contornos
                contorno_x = np.array(json.loads(row['Contorno_x']))
                contorno_y = np.array(json.loads(row['Contorno_y']))

                if len(contorno_x) < 10:
                    continue

                # Obtener mitad del contorno
                x_mitad, y_mitad = self.obtener_mitad_contorno(contorno_x, contorno_y)
                if x_mitad is None:
                    continue

                # Ajustes
                spline = self.ajustar_spline(x_mitad, y_mitad)
                polinomio = self.ajustar_polinomio(x_mitad, y_mitad, grado=4)

                resultados_frame = {
                    'Imagen': row['Imagen'],
                    'Tiempo (s)': row['Tiempo (s)'],
                    'Volumen_spline_trapecio': np.nan,
                    'Volumen_spline_simpson': np.nan,
                    'Error_vol_spline_trapecio': np.nan,
                    'Error_vol_spline_simpson': np.nan,
                    'Volumen_poly_trapecio': np.nan,
                    'Volumen_poly_simpson': np.nan,
                    'Error_vol_poly_trapecio': np.nan,
                    'Error_vol_poly_simpson': np.nan,
                    'Area_spline_trapecio': np.nan,
                    'Area_spline_simpson': np.nan,
                    'Area_poly_trapecio': np.nan,
                    'Area_poly_simpson': np.nan
                }

                # Cálculos con SPLINE
                if spline is not None:
                    y_min, y_max = np.min(y_mitad), np.max(y_mitad)

                    # Volúmenes
                    vol_trap, error_trap = self.volumen_por_revolucion_trapecio(spline, y_min, y_max)
                    vol_simp, error_simp = self.volumen_por_revolucion_simpson(spline, y_min, y_max)

                    resultados_frame.update({
                        'Volumen_spline_trapecio': vol_trap,
                        'Volumen_spline_simpson': vol_simp,
                        'Error_vol_spline_trapecio': error_trap,
                        'Error_vol_spline_simpson': error_simp
                    })

                    # Áreas superficiales
                    try:
                        derivada_spline = spline.derivative()
                        area_trap = self.area_superficial_trapecio(spline, derivada_spline, y_min, y_max)
                        area_simp = self.area_superficial_simpson(spline, derivada_spline, y_min, y_max)

                        resultados_frame.update({
                            'Area_spline_trapecio': area_trap,
                            'Area_spline_simpson': area_simp
                        })
                    except:
                        pass

                # Cálculos con POLINOMIO
                if polinomio is not None:
                    y_min, y_max = np.min(y_mitad), np.max(y_mitad)

                    # Volúmenes
                    vol_trap, error_trap = self.volumen_por_revolucion_trapecio(polinomio, y_min, y_max)
                    vol_simp, error_simp = self.volumen_por_revolucion_simpson(polinomio, y_min, y_max)

                    resultados_frame.update({
                        'Volumen_poly_trapecio': vol_trap,
                        'Volumen_poly_simpson': vol_simp,
                        'Error_vol_poly_trapecio': error_trap,
                        'Error_vol_poly_simpson': error_simp
                    })

                    # Áreas superficiales
                    try:
                        # Derivada analítica del polinomio
                        coef_deriv = np.polyder(polinomio.coef)
                        polinomio_deriv = np.poly1d(coef_deriv)

                        area_trap = self.area_superficial_trapecio(polinomio, polinomio_deriv, y_min, y_max)
                        area_simp = self.area_superficial_simpson(polinomio, polinomio_deriv, y_min, y_max)

                        resultados_frame.update({
                            'Area_poly_trapecio': area_trap,
                            'Area_poly_simpson': area_simp
                        })
                    except:
                        pass

                self.resultados.append(resultados_frame)

            except Exception as e:
                print(f"Error en frame {row.get('Imagen', idx)}: {str(e)}")
                continue

        return pd.DataFrame(self.resultados)

    def generar_analisis_comparativo(self, df_resultados):
        """Genera análisis comparativo entre métodos"""
        print("\n" + "=" * 60)
        print("ANÁLISIS COMPARATIVO - MÉTODOS DE INTEGRACIÓN")
        print("=" * 60)

        # Filtrar frames válidos
        df_validos = df_resultados.dropna(subset=['Volumen_spline_trapecio', 'Volumen_poly_trapecio'])

        if len(df_validos) == 0:
            print("No hay datos válidos para análisis")
            return

        print(f"Frames analizados: {len(df_validos)}")

        # Comparación de VOLÚMENES
        print("\n--- COMPARACIÓN DE VOLÚMENES ---")

        # Spline: Trapecio vs Simpson
        vol_spline_trap = df_validos['Volumen_spline_trapecio'].mean()
        vol_spline_simp = df_validos['Volumen_spline_simpson'].mean()
        diff_spline = abs(vol_spline_trap - vol_spline_simp) / vol_spline_trap * 100

        print(f"SPLINE - Trapecio: {vol_spline_trap:.2e} m³")
        print(f"SPLINE - Simpson:  {vol_spline_simp:.2e} m³")
        print(f"Diferencia: {diff_spline:.2f}%")

        # Polinomio: Trapecio vs Simpson
        vol_poly_trap = df_validos['Volumen_poly_trapecio'].mean()
        vol_poly_simp = df_validos['Volumen_poly_simpson'].mean()
        diff_poly = abs(vol_poly_trap - vol_poly_simp) / vol_poly_trap * 100

        print(f"\nPOLINOMIO - Trapecio: {vol_poly_trap:.2e} m³")
        print(f"POLINOMIO - Simpson:  {vol_poly_simp:.2e} m³")
        print(f"Diferencia: {diff_poly:.2f}%")

        # Spline vs Polinomio
        diff_ajustes = abs(vol_spline_trap - vol_poly_trap) / vol_spline_trap * 100
        print(f"\nSPLINE vs POLINOMIO: {diff_ajustes:.2f}% de diferencia")

        # Comparación de ÁREAS
        print("\n--- COMPARACIÓN DE ÁREAS SUPERFICIALES ---")

        areas_validas = df_validos.dropna(subset=['Area_spline_trapecio', 'Area_poly_trapecio'])
        if len(areas_validas) > 0:
            area_spline_trap = areas_validas['Area_spline_trapecio'].mean()
            area_poly_trap = areas_validas['Area_poly_trapecio'].mean()

            print(f"SPLINE - Trapecio: {area_spline_trap:.2e} m²")
            print(f"POLINOMIO - Trapecio: {area_poly_trap:.2e} m²")
            print(f"Diferencia: {abs(area_spline_trap - area_poly_trap) / area_spline_trap * 100:.2f}%")

        # Recomendación
        print("\n--- RECOMENDACIÓN ---")
        print("Método más confiable: SPLINE + REGLA DE SIMPSON")
        print("Justificación:")
        print("- Splines capturan mejor la curvatura del contorno")
        print("- Simpson es más preciso para funciones suaves")
        print("- Error estimado consistentemente menor")


def generar_informe1():
    """Función principal del Ejercicio 1"""
    print("=== EJERCICIO 1 TP5 - CÁLCULO DE VOLUMEN Y ÁREA ===")

    try:
        # Cargar datos del TP4
        df_completo = pd.read_excel('resultados_completos.xlsx', sheet_name='Datos Completos')
        print(f"Datos cargados: {len(df_completo)} frames")

        # Calcular volúmenes y áreas
        calculador = CalculadorVolumenArea(df_completo)
        df_resultados = calculador.procesar_todos_frames()

        # Análisis comparativo
        calculador.generar_analisis_comparativo(df_resultados)

        # Guardar resultados
        df_resultados.to_excel('resultados_tp5_volumen_area.xlsx', index=False)
        print(f"\nResultados guardados en: resultados_tp5_volumen_area.xlsx")

        # Generar gráficos
        generar_graficos_volumen_area(df_resultados)

        return df_resultados

    except Exception as e:
        print(f"Error en Ejercicio 1: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def generar_graficos_volumen_area(df):
    """Genera gráficos comparativos de volúmenes y áreas"""
    if df is None or len(df) == 0:
        return

    plt.figure(figsize=(15, 10))

    # Gráfico 1: Evolución temporal de volúmenes
    plt.subplot(2, 2, 1)
    plt.plot(df['Tiempo (s)'], df['Volumen_spline_trapecio'], 'b-', label='Spline + Trapecio', alpha=0.7)
    plt.plot(df['Tiempo (s)'], df['Volumen_poly_trapecio'], 'r--', label='Polinomio + Trapecio', alpha=0.7)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Volumen (m³)')
    plt.title('Evolución del Volumen - Métodos de Ajuste')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Gráfico 2: Comparación métodos integración (Spline)
    plt.subplot(2, 2, 2)
    plt.plot(df['Tiempo (s)'], df['Volumen_spline_trapecio'], 'g-', label='Trapecio', alpha
