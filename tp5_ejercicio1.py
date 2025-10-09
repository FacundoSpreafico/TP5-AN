# tp5_ejercicio1.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simpson, trapezoid
import json
import warnings

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
        x_eval = np.maximum(funcion(y_eval), 0.0)
        integrando = np.pi * x_eval**2
        volumen = trapezoid(integrando, y_eval)
        return volumen, 0.0

    def volumen_por_revolucion_simpson(self, funcion, y_min, y_max, n_puntos=1000):
        y_eval = np.linspace(y_min, y_max, n_puntos)
        x_eval = np.maximum(funcion(y_eval), 0.0)
        integrando = np.pi * x_eval**2
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

        if derivada is None:
            dy = y_eval[1] - y_eval[0]
            dx_dy = np.gradient(x_eval, dy)
        else:
            dx_dy = np.nan_to_num(derivada(y_eval), nan=0.0, posinf=0.0, neginf=0.0)

        integrando = 2 * np.pi * x_eval * np.sqrt(1 + dx_dy**2)
        area = trapezoid(integrando, y_eval)

        if area < 0:
            print("⚠️  Área spline negativa → ajustada a 0.0")
            area = 0.0

        return area

    def area_superficial_simpson(self, funcion, derivada, y_min, y_max, n_puntos=1000):
        """Cálculo normal de área con Simpson (ya corregido con radios positivos)."""
        y_eval = np.linspace(y_min, y_max, n_puntos)
        x_eval = np.maximum(np.nan_to_num(funcion(y_eval), nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        if derivada is None:
            dy = y_eval[1] - y_eval[0]
            dx_dy = np.gradient(x_eval, dy)
        else:
            dx_dy = np.nan_to_num(derivada(y_eval), nan=0.0, posinf=0.0, neginf=0.0)

        integrando = 2 * np.pi * x_eval * np.sqrt(1 + dx_dy**2)
        area = simpson(integrando, y_eval)
        if area < 0:
            area = 0.0
        return area

    # ----------------------------------------------------------
    # --- PROCESAMIENTO DE TODOS LOS FRAMES ---
    # ----------------------------------------------------------
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

        return df_res

    except Exception as e:
        print(f"Error en Ejercicio 1: {str(e)}")
        return None
