# tp5_ejercicio1.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator  # ✅ spline estable (monótono)
from scipy.integrate import simpson, trapezoid
import json
import warnings

warnings.filterwarnings('ignore')


class CalculadorVolumenArea:
    """Clase para calcular volumen y área de una gota como figura de revolución."""

    # ----------------------------------------------------------
    # --- INICIALIZACIÓN ---
    # ----------------------------------------------------------
    def __init__(self, df, scale=1.0, n_puntos=1000, suavizado=0.1):
        """
        df: DataFrame con columnas ['Imagen', 'Tiempo (s)', 'Contorno_x', 'Contorno_y']
        scale: factor de conversión (mm/px o m/px)
        n_puntos: número de puntos para la integración
        """
        self.df = df
        self.scale = scale
        self.n_puntos = n_puntos
        self.suavizado = suavizado
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

        # ✅ Devuelve la mitad con más puntos (generalmente derecha)
        if len(mitad_der) > len(mitad_izq):
            return mitad_der[:, 0], mitad_der[:, 1]
        else:
            return mitad_izq[:, 0], mitad_izq[:, 1]

    # ✅ spline más estable: PCHIP evita sobreoscilaciones
    def ajustar_spline(self, x, y):
        """Ajuste con interpolación monótona PCHIP (sin sobreoscilaciones)."""
        if len(x) < 4:
            return None

        idx = np.argsort(y)
        y = y[idx]
        x = x[idx]

        y_unique, indices = np.unique(y, return_index=True)
        x_unique = x[indices]

        if len(y_unique) < 4:
            return None

        try:
            spline = PchipInterpolator(y_unique, x_unique, extrapolate=False)
            return spline
        except Exception:
            return None

    def ajustar_polinomio(self, x, y, grado=3):
        """Ajusta un polinomio con y normalizado para evitar mal condicionamiento."""
        if len(x) <= grado + 1:
            return None
        idx = np.argsort(y)
        x, y = x[idx], y[idx]
        y_mean, y_std = np.mean(y), np.std(y)
        if y_std == 0:
            return None
        y_norm = (y - y_mean) / y_std
        try:
            coef = np.polyfit(y_norm, x, grado)
            poly = np.poly1d(coef)
            return lambda yy: poly((yy - y_mean) / y_std)
        except Exception:
            return None

    # ----------------------------------------------------------
    # --- FUNCIONES DE CÁLCULO ---
    # ----------------------------------------------------------
    def _eval_seguro(self, funcion, y_eval):
        """Evalúa la función asegurando valores válidos (sin negativos, NaN, inf)."""
        x = np.nan_to_num(funcion(y_eval), nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(x, 0.0)

    def volumen_por_revolucion_trapecio(self, funcion, y_min, y_max, n_puntos=None):
        n = n_puntos or self.n_puntos
        y_eval = np.linspace(y_min, y_max, n)
        x_eval = self._eval_seguro(funcion, y_eval)
        integrando = np.pi * x_eval**2
        volumen = trapezoid(integrando, y_eval)
        return volumen, 0.0

    def volumen_por_revolucion_simpson(self, funcion, y_min, y_max, n_puntos=None):
        n = n_puntos or self.n_puntos
        y_eval = np.linspace(y_min, y_max, n)
        x_eval = self._eval_seguro(funcion, y_eval)
        integrando = np.pi * x_eval**2
        volumen = simpson(integrando, y_eval)
        return volumen, 0.0

    def area_superficial_trapecio(self, funcion, derivada, y_min, y_max, n_puntos=None):
        n = n_puntos or self.n_puntos
        y_eval = np.linspace(y_min, y_max, n)
        x_eval = self._eval_seguro(funcion, y_eval)

        if derivada is None:
            dy = y_eval[1] - y_eval[0]
            dx_dy = np.gradient(x_eval, dy)
        else:
            dx_dy = np.nan_to_num(derivada(y_eval), nan=0.0)

        integrando = 2 * np.pi * x_eval * np.sqrt(1 + dx_dy**2)
        area = trapezoid(integrando, y_eval)
        return max(area, 0.0)

    def area_superficial_simpson(self, funcion, derivada, y_min, y_max, n_puntos=None):
        n = n_puntos or self.n_puntos
        y_eval = np.linspace(y_min, y_max, n)
        x_eval = self._eval_seguro(funcion, y_eval)

        if derivada is None:
            dy = y_eval[1] - y_eval[0]
            dx_dy = np.gradient(x_eval, dy)
        else:
            dx_dy = np.nan_to_num(derivada(y_eval), nan=0.0)

        integrando = 2 * np.pi * x_eval * np.sqrt(1 + dx_dy**2)
        area = simpson(integrando, y_eval)
        return max(area, 0.0)

    # ----------------------------------------------------------
    # --- PROCESAMIENTO DE TODOS LOS FRAMES ---
    # ----------------------------------------------------------
    def procesar_todos_frames(self):
        print("Procesando volúmenes y áreas para todos los frames...")
        for idx, row in self.df.iterrows():
            try:
                contorno_x = self.scale * np.array(json.loads(row['Contorno_x']))
                contorno_y = self.scale * np.array(json.loads(row['Contorno_y']))
                if len(contorno_x) < 10:
                    continue

                x_mitad, y_mitad = self.obtener_mitad_contorno(contorno_x, contorno_y)
                if x_mitad is None:
                    continue

                spline = self.ajustar_spline(x_mitad, y_mitad)
                polinomio = self.ajustar_polinomio(x_mitad, y_mitad, grado=3)

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

                # --- SPLINE ---
                if spline is not None:
                    y_knots = spline.x  # rango válido del PCHIP
                    y_min, y_max = np.min(y_knots), np.max(y_knots)

                    vol_trap, _ = self.volumen_por_revolucion_trapecio(spline, y_min, y_max)
                    vol_simp, _ = self.volumen_por_revolucion_simpson(spline, y_min, y_max)

                    derivada_spline = spline.derivative()
                    area_trap = self.area_superficial_trapecio(spline, derivada_spline, y_min, y_max)
                    area_simp = self.area_superficial_simpson(spline, derivada_spline, y_min, y_max)

                    resultados_frame.update({
                        'Volumen_spline_trapecio': vol_trap,
                        'Volumen_spline_simpson': vol_simp,
                        'Area_spline_trapecio': area_trap,
                        'Area_spline_simpson': area_simp
                    })

                # --- POLINOMIO ---
                if polinomio is not None:
                    y_min, y_max = np.min(y_mitad), np.max(y_mitad)
                    vol_trap, _ = self.volumen_por_revolucion_trapecio(polinomio, y_min, y_max)
                    vol_simp, _ = self.volumen_por_revolucion_simpson(polinomio, y_min, y_max)

                    dy = y_max - y_min
                    derivada_poly = lambda yy: np.gradient(polinomio(yy), yy)
                    area_trap = self.area_superficial_trapecio(polinomio, derivada_poly, y_min, y_max)
                    area_simp = self.area_superficial_simpson(polinomio, derivada_poly, y_min, y_max)

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

        def rel_diff_sym(a, b):
            denom = (abs(a) + abs(b)) / 2
            return abs(a - b) / denom * 100 if denom > 0 else np.nan

        vol_spline = df_validos['Volumen_spline_trapecio'].mean()
        vol_poly = df_validos['Volumen_poly_trapecio'].mean()
        diff = rel_diff_sym(vol_spline, vol_poly)

        print(f"Spline (Trapecio): {vol_spline:.3e}  |  Polinomio (Trapecio): {vol_poly:.3e}")
        print(f"Diferencia promedio: {diff:.2f}%")

        area_spline = df_validos['Area_spline_trapecio'].mean()
        area_poly = df_validos['Area_poly_trapecio'].mean()
        diff_area = rel_diff_sym(area_spline, area_poly)
        print(f"\nÁrea promedio Spline: {area_spline:.3e}  |  Polinomio: {area_poly:.3e}")
        print(f"Diferencia promedio de área: {diff_area:.2f}%")

        print("\nMétodo más confiable: SPLINE (PCHIP) + SIMPSON (por estabilidad y suavidad)")

# ----------------------------------------------------------
# --- FUNCIÓN PRINCIPAL ---
# ----------------------------------------------------------
def generar_informe1(scale=1.0):
    print("=== EJERCICIO 1 TP5 - CÁLCULO DE VOLUMEN Y ÁREA ===")
    try:
        df = pd.read_excel('resultados_completos.xlsx', sheet_name='Datos Completos')
        print(f"Datos cargados: {len(df)} frames")

        calculador = CalculadorVolumenArea(df, scale=scale)
        df_res = calculador.procesar_todos_frames()
        calculador.generar_analisis_comparativo(df_res)

        df_res.to_excel('resultados_tp5_volumen_area.xlsx', index=False)
        print("\nResultados guardados en: resultados_tp5_volumen_area.xlsx")

        return df_res

    except Exception as e:
        print(f"Error en Ejercicio 1: {str(e)}")
        return None