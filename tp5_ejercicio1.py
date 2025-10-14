# tp5_ejercicio1.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
<<<<<<< HEAD
from scipy.integrate import quad, simpson, trapezoid
=======
from scipy.integrate import simpson, trapezoid
>>>>>>> 937eda61d87093431798e9f6599255e3098b6eb1
import json
import warnings
from tp5_exportar_excel import exportar_ejercicio1_excel

warnings.filterwarnings('ignore')


class CalculadorVolumenArea:
    """Clase para calcular volumen y área de una gota como figura de revolución."""

    def __init__(self, df):
        self.df = df
        self.resultados = []

    # ----------------------------------------------------------
    # --- FUNCIONES AUXILIARES ---
    # ----------------------------------------------------------
    def obtener_mitad_contorno(self, contorno_x, contorno_y):
        """Divide el contorno en mitades usando el ápice (punto más bajo)."""
        if len(contorno_y) == 0:
            return None, None

        idx_apice = np.argmin(contorno_y)
        apice_x = contorno_x[idx_apice]

        mask_izq = contorno_x <= apice_x
        mask_der = contorno_x >= apice_x

        mitad_izq = np.column_stack([contorno_x[mask_izq], contorno_y[mask_izq]])
        mitad_der = np.column_stack([contorno_x[mask_der], contorno_y[mask_der]])

        if len(mitad_der) > len(mitad_izq):
            return mitad_der[:, 0], mitad_der[:, 1]
        else:
            return mitad_izq[:, 0], mitad_izq[:, 1]

    def ajustar_spline(self, x, y):
        """Ajusta un spline al contorno."""
        if len(x) < 4:
            return None

        idx = np.argsort(y)
        x, y = x[idx], y[idx]

        y_unique, indices = np.unique(y, return_index=True)
        x_unique = x[indices]

        if len(y_unique) < 4:
            return None

        try:
            spline = UnivariateSpline(y_unique, x_unique, s=len(y_unique) * 0.1)
            return spline
        except:
            return None

    def ajustar_polinomio(self, x, y, grado=4):
        """Ajusta un polinomio al contorno."""
        if len(x) <= grado:
            return None
        try:
            idx = np.argsort(y)
            x, y = x[idx], y[idx]
            coef = np.polyfit(y, x, grado)
            return np.poly1d(coef)
        except:
            return None

    # ----------------------------------------------------------
    # --- CÁLCULO DE VOLÚMENES ---
    # ----------------------------------------------------------
    def volumen_por_revolucion_trapecio(self, funcion, y_min, y_max, n_puntos=1000):
        y_eval = np.linspace(y_min, y_max, n_puntos)
<<<<<<< HEAD
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
=======
        x_eval = np.maximum(funcion(y_eval), 0.0)
        integrando = np.pi * x_eval**2
        volumen = trapezoid(integrando, y_eval)
        return volumen, 0.0
>>>>>>> 937eda61d87093431798e9f6599255e3098b6eb1

    def volumen_por_revolucion_simpson(self, funcion, y_min, y_max, n_puntos=1000):
        y_eval = np.linspace(y_min, y_max, n_puntos)
<<<<<<< HEAD
        x_eval = funcion(y_eval)
        x_eval = np.nan_to_num(x_eval, nan=0.0, posinf=0.0, neginf=0.0)
        x_eval = np.clip(x_eval, 0, None)

        integrando = np.pi * x_eval ** 2
=======
        x_eval = np.maximum(funcion(y_eval), 0.0)
        integrando = np.pi * x_eval**2
>>>>>>> 937eda61d87093431798e9f6599255e3098b6eb1
        volumen = simpson(integrando, y_eval)
        return volumen, 0.0

    # ----------------------------------------------------------
    # --- CÁLCULO DE ÁREAS (CORREGIDAS) ---
    # ----------------------------------------------------------
    def area_superficial_trapecio(self, funcion, derivada, y_mitad, n_puntos=1000):
        """
        Cálculo robusto del área superficial con spline:
        - Evalúa solo dentro del rango real del contorno.
        - Fuerza radios positivos (x >= 0).
        - Si el resultado es negativo, lo reemplaza por 0.
        """
        y_eval = np.linspace(np.min(y_mitad), np.max(y_mitad), n_puntos)
        x_eval = np.maximum(np.nan_to_num(funcion(y_eval), nan=0.0, posinf=0.0, neginf=0.0), 0.0)

<<<<<<< HEAD
        return volumen, error_estimado

    # ======== CORREGIDAS PARA EVITAR ÁREAS NEGATIVAS ========

    def area_superficial_trapecio(self, funcion, derivada, y_min, y_max, n_puntos=1000):
        """Calcula área superficial usando trapecio: A = 2π∫x(y)√(1 + [dx/dy]²)dy"""
        y_eval = np.linspace(y_min, y_max, n_puntos)
        x_eval = funcion(y_eval)
        x_eval = np.nan_to_num(x_eval, nan=0.0, posinf=0.0, neginf=0.0)
        x_eval = np.abs(x_eval)  # asegurar radios positivos

        # Calcular derivada numéricamente si no se proporciona
=======
>>>>>>> 937eda61d87093431798e9f6599255e3098b6eb1
        if derivada is None:
            dy = y_eval[1] - y_eval[0]
            dx_dy = np.gradient(x_eval, dy)
        else:
<<<<<<< HEAD
            dx_dy = derivada(y_eval)
            dx_dy = np.nan_to_num(dx_dy, nan=0.0, posinf=0.0, neginf=0.0)
=======
            dx_dy = np.nan_to_num(derivada(y_eval), nan=0.0, posinf=0.0, neginf=0.0)
>>>>>>> 937eda61d87093431798e9f6599255e3098b6eb1

        integrando = 2 * np.pi * x_eval * np.sqrt(1 + dx_dy**2)
        area = trapezoid(integrando, y_eval)

        if area < 0:
            print("⚠️  Área spline negativa → ajustada a 0.0")
            area = 0.0

        return area

    def area_superficial_simpson(self, funcion, derivada, y_min, y_max, n_puntos=1000):
        """Cálculo normal de área con Simpson (ya corregido con radios positivos)."""
        y_eval = np.linspace(y_min, y_max, n_puntos)
<<<<<<< HEAD
        x_eval = funcion(y_eval)
        x_eval = np.nan_to_num(x_eval, nan=0.0, posinf=0.0, neginf=0.0)
        x_eval = np.abs(x_eval)  # asegurar radios positivos
=======
        x_eval = np.maximum(np.nan_to_num(funcion(y_eval), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
>>>>>>> 937eda61d87093431798e9f6599255e3098b6eb1

        if derivada is None:
            dy = y_eval[1] - y_eval[0]
            dx_dy = np.gradient(x_eval, dy)
        else:
<<<<<<< HEAD
            dx_dy = derivada(y_eval)
            dx_dy = np.nan_to_num(dx_dy, nan=0.0, posinf=0.0, neginf=0.0)
=======
            dx_dy = np.nan_to_num(derivada(y_eval), nan=0.0, posinf=0.0, neginf=0.0)
>>>>>>> 937eda61d87093431798e9f6599255e3098b6eb1

        integrando = 2 * np.pi * x_eval * np.sqrt(1 + dx_dy**2)
        area = simpson(integrando, y_eval)
        if area < 0:
            area = 0.0
        return area

<<<<<<< HEAD
    # ========================================================

=======
    # ----------------------------------------------------------
    # --- PROCESAMIENTO DE TODOS LOS FRAMES ---
    # ----------------------------------------------------------
>>>>>>> 937eda61d87093431798e9f6599255e3098b6eb1
    def procesar_todos_frames(self):
        print("Procesando volúmenes y áreas para todos los frames...")
        for idx, row in self.df.iterrows():
            try:
                contorno_x = np.array(json.loads(row['Contorno_x']))
                contorno_y = np.array(json.loads(row['Contorno_y']))
                if len(contorno_x) < 10:
                    continue

                x_mitad, y_mitad = self.obtener_mitad_contorno(contorno_x, contorno_y)
                if x_mitad is None:
                    continue

                spline = self.ajustar_spline(x_mitad, y_mitad)
                polinomio = self.ajustar_polinomio(x_mitad, y_mitad, grado=4)

                resultados_frame = {
                    'Imagen': row['Imagen'],
                    'Tiempo (s)': row['Tiempo (s)'],
                    'Volumen_spline_trapecio': np.nan,
                    'Volumen_spline_simpson': np.nan,
                    'Volumen_poly_trapecio': np.nan,
                    'Volumen_poly_simpson': np.nan,
                    'Area_spline_trapecio': np.nan,
                    'Area_spline_simpson': np.nan,
                    'Area_poly_trapecio': np.nan,
                    'Area_poly_simpson': np.nan
                }

                if spline is not None:
                    y_min, y_max = np.min(y_mitad), np.max(y_mitad)

                    vol_trap, _ = self.volumen_por_revolucion_trapecio(spline, y_min, y_max)
                    vol_simp, _ = self.volumen_por_revolucion_simpson(spline, y_min, y_max)

                    derivada_spline = spline.derivative()
                    area_trap = self.area_superficial_trapecio(spline, derivada_spline, y_mitad)
                    area_simp = self.area_superficial_simpson(spline, derivada_spline, y_min, y_max)

                    resultados_frame.update({
                        'Volumen_spline_trapecio': vol_trap,
                        'Volumen_spline_simpson': vol_simp,
                        'Area_spline_trapecio': area_trap,
                        'Area_spline_simpson': area_simp
                    })

                if polinomio is not None:
                    y_min, y_max = np.min(y_mitad), np.max(y_mitad)
                    vol_trap, _ = self.volumen_por_revolucion_trapecio(polinomio, y_min, y_max)
                    vol_simp, _ = self.volumen_por_revolucion_simpson(polinomio, y_min, y_max)
                    coef_deriv = np.polyder(polinomio.coef)
                    polinomio_deriv = np.poly1d(coef_deriv)
                    area_trap = self.area_superficial_trapecio(polinomio, polinomio_deriv, y_mitad)
                    area_simp = self.area_superficial_simpson(polinomio, polinomio_deriv, y_min, y_max)

                    resultados_frame.update({
                        'Volumen_poly_trapecio': vol_trap,
                        'Volumen_poly_simpson': vol_simp,
                        'Area_poly_trapecio': area_trap,
                        'Area_poly_simpson': area_simp
                    })

                self.resultados.append(resultados_frame)

            except Exception as e:
                print(f"Error en frame {row.get('Imagen', idx)}: {str(e)}")
                continue

        return pd.DataFrame(self.resultados)

    # ----------------------------------------------------------
    # --- ANÁLISIS COMPARATIVO ---
    # ----------------------------------------------------------
    def generar_analisis_comparativo(self, df_resultados):
        print("\n" + "=" * 60)
        print("ANÁLISIS COMPARATIVO - MÉTODOS DE INTEGRACIÓN")
        print("=" * 60)

        df_validos = df_resultados.dropna(subset=['Volumen_spline_trapecio', 'Volumen_poly_trapecio'])
        if len(df_validos) == 0:
            print("No hay datos válidos para análisis")
            return

        print(f"Frames analizados: {len(df_validos)}")
        vol_spline = df_validos['Volumen_spline_trapecio'].mean()
        vol_poly = df_validos['Volumen_poly_trapecio'].mean()
        print(f"Spline (Trapecio): {vol_spline:.3e}  |  Polinomio (Trapecio): {vol_poly:.3e}")
        diff = abs(vol_spline - vol_poly) / vol_spline * 100
        print(f"Diferencia promedio: {diff:.2f}%")

        area_spline = df_validos['Area_spline_trapecio'].mean()
        area_poly = df_validos['Area_poly_trapecio'].mean()
        print(f"\nÁrea promedio Spline: {area_spline:.3e}  |  Polinomio: {area_poly:.3e}")
        diff_area = abs(area_spline - area_poly) / area_spline * 100
        print(f"Diferencia promedio de área: {diff_area:.2f}%")

        print("\nMétodo más confiable: SPLINE + SIMPSON (por suavidad y estabilidad)")

# ----------------------------------------------------------
# --- FUNCIÓN PRINCIPAL ---
# ----------------------------------------------------------
def generar_informe1():
    print("=== EJERCICIO 1 TP5 - CÁLCULO DE VOLUMEN Y ÁREA ===")
    try:
        df = pd.read_excel('resultados_completos.xlsx', sheet_name='Datos Completos')
        print(f"Datos cargados: {len(df)} frames")

        calculador = CalculadorVolumenArea(df)
        df_res = calculador.procesar_todos_frames()
        calculador.generar_analisis_comparativo(df_res)

        df_res.to_excel('resultados_tp5_volumen_area.xlsx', index=False)
        print("\nResultados guardados en: resultados_tp5_volumen_area.xlsx")

<<<<<<< HEAD
        # Guardar resultados
        exportar_ejercicio1_excel(df_resultados)
        print(f"\nResultados guardados en: resultados_tp5_volumen_area.xlsx")

        # Generar gráficos
        generar_graficos_volumen_area(df_resultados)

        return df_resultados
=======
        return df_res
>>>>>>> 937eda61d87093431798e9f6599255e3098b6eb1

    except Exception as e:
        print(f"Error en Ejercicio 1: {str(e)}")
        return None
<<<<<<< HEAD


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
    # Filtrar valores válidos
    mask = ~(df['Volumen_spline_trapecio'].isna() | df['Volumen_spline_simpson'].isna())
    df_filtrado = df[mask]

    if len(df_filtrado) > 0:
        plt.plot(df_filtrado['Tiempo (s)'], df_filtrado['Volumen_spline_trapecio'],
                 'g-', label='Spline + Trapecio', alpha=0.7)
        plt.plot(df_filtrado['Tiempo (s)'], df_filtrado['Volumen_spline_simpson'],
                 'm-', label='Spline + Simpson', alpha=0.7)
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Volumen (m³)')
        plt.title('Comparación Métodos Integración - Spline')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Gráfico 3: Comparación métodos integración (Polinomio)
    plt.subplot(2, 2, 3)
    mask_poly = ~(df['Volumen_poly_trapecio'].isna() | df['Volumen_poly_simpson'].isna())
    df_filtrado_poly = df[mask_poly]

    if len(df_filtrado_poly) > 0:
        plt.plot(df_filtrado_poly['Tiempo (s)'], df_filtrado_poly['Volumen_poly_trapecio'],
                 'c-', label='Polinomio + Trapecio', alpha=0.7)
        plt.plot(df_filtrado_poly['Tiempo (s)'], df_filtrado_poly['Volumen_poly_simpson'],
                 'y-', label='Polinomio + Simpson', alpha=0.7)
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Volumen (m³)')
        plt.title('Comparación Métodos Integración - Polinomio')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Gráfico 4: Evolución del área superficial
    plt.subplot(2, 2, 4)
    mask_area = ~(df['Area_spline_trapecio'].isna() | df['Area_poly_trapecio'].isna())
    df_filtrado_area = df[mask_area]

    if len(df_filtrado_area) > 0:
        plt.plot(df_filtrado_area['Tiempo (s)'], df_filtrado_area['Area_spline_trapecio'],
                 'b-', label='Spline', alpha=0.7)
        plt.plot(df_filtrado_area['Tiempo (s)'], df_filtrado_area['Area_poly_trapecio'],
                 'r--', label='Polinomio', alpha=0.7)
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Área (m²)')
        plt.title('Evolución del Área Superficial')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('graficos_volumen_area_tp5.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Gráfico adicional: Errores de integración
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    mask_error = ~(df['Error_vol_spline_trapecio'].isna())
    df_error = df[mask_error]

    if len(df_error) > 0:
        plt.plot(df_error['Tiempo (s)'], df_error['Error_vol_spline_trapecio'],
                 'r-', label='Error Spline', alpha=0.7)
        plt.plot(df_error['Tiempo (s)'], df_error['Error_vol_poly_trapecio'],
                 'b-', label='Error Polinomio', alpha=0.7)
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Error Estimado (m³)')
        plt.title('Error Estimado en Cálculo de Volumen')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Escala logarítmica para mejor visualización

    plt.subplot(1, 2, 2)
    # Ratio error/volumen
    if len(df_error) > 0:
        error_ratio_spline = df_error['Error_vol_spline_trapecio'] / df_error['Volumen_spline_trapecio'] * 100
        error_ratio_poly = df_error['Error_vol_poly_trapecio'] / df_error['Volumen_poly_trapecio'] * 100

        plt.plot(df_error['Tiempo (s)'], error_ratio_spline, 'r-', label='Error Spline (%)', alpha=0.7)
        plt.plot(df_error['Tiempo (s)'], error_ratio_poly, 'b-', label='Error Polinomio (%)', alpha=0.7)
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Error Relativo (%)')
        plt.title('Error Relativo en Cálculo de Volumen')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('graficos_error_integracion_tp5.png', dpi=300, bbox_inches='tight')
    plt.show()
=======
>>>>>>> 937eda61d87093431798e9f6599255e3098b6eb1
