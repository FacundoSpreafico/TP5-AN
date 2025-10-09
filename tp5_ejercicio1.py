# tp5_ejercicio1.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.integrate import simpson, trapezoid
from scipy.signal import savgol_filter
import json
import warnings

warnings.filterwarnings('ignore')


class CalculadorVolumenArea:
    """Clase para calcular volumen y área de una gota como figura de revolución."""

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

    # ---------------------- AUXILIARES ----------------------
    def obtener_mitad_contorno(self, contorno_x, contorno_y):
        """
        Divide el contorno en mitades usando el ápice (punto más bajo) y
        devuelve (r, y) donde r = |x - x_apice|.
        """
        if len(contorno_y) == 0:
            return None, None

        idx_apice = np.argmin(contorno_y)
        apice_x = contorno_x[idx_apice]

        mask_izq = contorno_x <= apice_x
        mask_der = contorno_x >= apice_x

        mitad_izq = np.column_stack([contorno_x[mask_izq], contorno_y[mask_izq]])
        mitad_der = np.column_stack([contorno_x[mask_der], contorno_y[mask_der]])

        mitad = mitad_der if len(mitad_der) > len(mitad_izq) else mitad_izq
        if len(mitad) == 0:
            return None, None

        x_m, y_m = mitad[:, 0], mitad[:, 1]

        idx = np.argsort(y_m)
        y_ord = y_m[idx]
        r_ord = np.abs(x_m[idx] - apice_x)

        return r_ord, y_ord

    def ajustar_spline(self, r, y):
        """Ajuste PCHIP de r(y)."""
        if r is None or y is None or len(y) < 4:
            return None
        idx = np.argsort(y)
        y = y[idx]; r = r[idx]
        y_unique, indices = np.unique(y, return_index=True)
        r_unique = r[indices]
        if len(y_unique) < 4:
            return None
        try:
            return PchipInterpolator(y_unique, r_unique, extrapolate=False)
        except Exception:
            return None

    def ajustar_polinomio(self, r, y, grado=3):
        """
        Ajusta r(y) con polinomio (sobre y normalizado) y devuelve f(yy)=r.
        NOTA: la derivada se calcula con la misma receta (Savitzky–Golay), no analítica.
        """
        if r is None or y is None or len(y) <= grado + 1:
            return None
        idx = np.argsort(y)
        y, r = y[idx], r[idx]
        y_mean, y_std = np.mean(y), np.std(y)
        if y_std == 0:
            return None
        y_norm = (y - y_mean) / y_std
        try:
            coef = np.polyfit(y_norm, r, grado)
            poly = np.poly1d(coef)
            f = lambda yy: poly((yy - y_mean) / y_std)
            return f
        except Exception:
            return None

    def _eval_seguro(self, funcion, y_eval):
        """Evalúa la función en radios r, clamp a [0,∞) y reemplaza NaN/Inf por 0."""
        r = funcion(y_eval)
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(r, 0.0)

    # ---------------------- VOLUMEN ----------------------
    def volumen_por_revolucion_trapecio(self, funcion, y_min, y_max, n_puntos=None):
        n = n_puntos or self.n_puntos
        y_eval = np.linspace(y_min, y_max, n)
        r_eval = self._eval_seguro(funcion, y_eval)
        integrando = np.pi * r_eval**2
        volumen = trapezoid(integrando, y_eval)
        return volumen, 0.0

    def volumen_por_revolucion_simpson(self, funcion, y_min, y_max, n_puntos=None):
        n = n_puntos or self.n_puntos
        y_eval = np.linspace(y_min, y_max, n)
        r_eval = self._eval_seguro(funcion, y_eval)
        integrando = np.pi * r_eval**2
        volumen = simpson(integrando, y_eval)
        return volumen, 0.0

    # ---------------------- ÁREA: pipeline homogénea ----------------------
    def _area_con_pipeline_unica(self, funcion, y_min, y_max, n_puntos=1000, usar_simpson=False):
        """
        MISMA receta para spline y polinomio:
        1) r = f(y) en grilla uniforme
        2) Suavizado Savitzky–Golay de r
        3) dr/dy vía Savitzky–Golay (derivative=1) sobre la MISMA señal
        4) 2π ∫ r * sqrt(1 + (dr/dy)^2) dy
        """
        y_eval = np.linspace(y_min, y_max, n_puntos)
        r = self._eval_seguro(funcion, y_eval)

        # Ventana Savitzky–Golay: ~2% de la grilla, impar y >= 7
        win = max(7, (n_puntos // 50) | 1)  # fuerza impar
        polyorder = 3 if win > 3 else 2

        r_suav = savgol_filter(r, window_length=win, polyorder=polyorder, mode='interp')
        dr_dy = savgol_filter(r, window_length=win, polyorder=polyorder, deriv=1,
                              delta=(y_eval[1]-y_eval[0]), mode='interp')

        integrando = 2.0 * np.pi * np.maximum(r_suav, 0.0) * np.sqrt(1.0 + dr_dy**2)
        area = simpson(integrando, y_eval) if usar_simpson else trapezoid(integrando, y_eval)
        return max(float(area), 0.0)

    def area_superficial_trapecio(self, funcion, _derivada_no_usada, y_min, y_max, n_puntos=1000):
        return self._area_con_pipeline_unica(funcion, y_min, y_max, n_puntos=n_puntos, usar_simpson=False)

    def area_superficial_simpson(self, funcion, _derivada_no_usada, y_min, y_max, n_puntos=1000):
        return self._area_con_pipeline_unica(funcion, y_min, y_max, n_puntos=n_puntos, usar_simpson=True)

    # ---------------------- RANGO COMÚN ----------------------
    def _rango_comun(self, dom1, dom2, margen=0.0):
        """
        dom1, dom2: tuplas (y_min, y_max)
        Devuelve (ymin_c, ymax_c) = intersección de dominios con un pequeño margen opcional.
        """
        y1_min, y1_max = float(dom1[0]), float(dom1[1])
        y2_min, y2_max = float(dom2[0]), float(dom2[1])
        y_min_c = max(y1_min, y2_min) + margen
        y_max_c = min(y1_max, y2_max) - margen
        if y_max_c <= y_min_c:
            return None, None
        return y_min_c, y_max_c

    # ---------------------- PROCESAMIENTO ----------------------
    def procesar_todos_frames(self):
        print("Procesando volúmenes y áreas para todos los frames...")
        for idx, row in self.df.iterrows():
            try:
                contorno_x = self.scale * np.array(json.loads(row['Contorno_x']))
                contorno_y = self.scale * np.array(json.loads(row['Contorno_y']))
                if len(contorno_x) < 10:
                    continue

                r_mitad, y_mitad = self.obtener_mitad_contorno(contorno_x, contorno_y)
                if r_mitad is None or y_mitad is None or len(r_mitad) < 4:
                    continue

                spline = self.ajustar_spline(r_mitad, y_mitad)
                poly_f = self.ajustar_polinomio(r_mitad, y_mitad, grado=3)

                resultados_frame = {
                    'Imagen': row['Imagen'],
                    'Tiempo (s)': row['Tiempo (s)'],
                    # Totales (cada método en su dominio)
                    'Volumen_spline_trapecio': np.nan,
                    'Volumen_spline_simpson': np.nan,
                    'Volumen_poly_trapecio': np.nan,
                    'Volumen_poly_simpson': np.nan,
                    'Area_spline_trapecio': np.nan,
                    'Area_spline_simpson': np.nan,
                    'Area_poly_trapecio': np.nan,
                    'Area_poly_simpson': np.nan,
                    # Overlap (rango común entre spline y polinomio)
                    'Volumen_spline_trapecio_overlap': np.nan,
                    'Volumen_poly_trapecio_overlap': np.nan,
                    'Area_spline_trapecio_overlap': np.nan,
                    'Area_poly_trapecio_overlap': np.nan
                }

                # --- SPLINE ---
                y_min_s = y_max_s = None
                if spline is not None:
                    y_knots = spline.x
                    y_min_s, y_max_s = float(np.min(y_knots)), float(np.max(y_knots))

                    vol_trap_s, _ = self.volumen_por_revolucion_trapecio(spline, y_min_s, y_max_s)
                    vol_simp_s, _ = self.volumen_por_revolucion_simpson(spline, y_min_s, y_max_s)
                    area_trap_s = self.area_superficial_trapecio(spline, None, y_min_s, y_max_s)
                    area_simp_s = self.area_superficial_simpson(spline, None, y_min_s, y_max_s)

                    resultados_frame.update({
                        'Volumen_spline_trapecio': vol_trap_s,
                        'Volumen_spline_simpson': vol_simp_s,
                        'Area_spline_trapecio': area_trap_s,
                        'Area_spline_simpson': area_simp_s
                    })

                # --- POLINOMIO ---
                y_min_p = y_max_p = None
                if poly_f is not None:
                    y_min_p, y_max_p = float(np.min(y_mitad)), float(np.max(y_mitad))

                    vol_trap_p, _ = self.volumen_por_revolucion_trapecio(poly_f, y_min_p, y_max_p)
                    vol_simp_p, _ = self.volumen_por_revolucion_simpson(poly_f, y_min_p, y_max_p)
                    area_trap_p = self.area_superficial_trapecio(poly_f, None, y_min_p, y_max_p)
                    area_simp_p = self.area_superficial_simpson(poly_f, None, y_min_p, y_max_p)

                    resultados_frame.update({
                        'Volumen_poly_trapecio': vol_trap_p,
                        'Volumen_poly_simpson': vol_simp_p,
                        'Area_poly_trapecio': area_trap_p,
                        'Area_poly_simpson': area_simp_p
                    })

                # --- OVERLAP entre Spline y Polinomio ---
                if (spline is not None) and (poly_f is not None) and (y_min_s is not None) and (y_min_p is not None):
                    y_min_c, y_max_c = self._rango_comun((y_min_s, y_max_s), (y_min_p, y_max_p), margen=0.0)
                    if y_min_c is not None:
                        vol_s_ov, _ = self.volumen_por_revolucion_trapecio(spline, y_min_c, y_max_c)
                        vol_p_ov, _ = self.volumen_por_revolucion_trapecio(poly_f, y_min_c, y_max_c)
                        area_s_ov = self.area_superficial_trapecio(spline, None, y_min_c, y_max_c)
                        area_p_ov = self.area_superficial_trapecio(poly_f, None, y_min_c, y_max_c)

                        resultados_frame.update({
                            'Volumen_spline_trapecio_overlap': vol_s_ov,
                            'Volumen_poly_trapecio_overlap': vol_p_ov,
                            'Area_spline_trapecio_overlap': area_s_ov,
                            'Area_poly_trapecio_overlap': area_p_ov
                        })

                self.resultados.append(resultados_frame)

            except Exception as e:
                print(f"Error en frame {row.get('Imagen', idx)}: {str(e)}")
                continue

        return pd.DataFrame(self.resultados)

    # ---------------------- ANÁLISIS ----------------------
    def generar_analisis_comparativo(self, df_resultados):
        print("\n" + "=" * 60)
        print("ANÁLISIS COMPARATIVO - MÉTODOS DE INTEGRACIÓN (rango común y diff por frame)")
        print("=" * 60)

        def rel_diff_sym(a, b):
            denom = (abs(a) + abs(b)) / 2
            return abs(a - b) / denom * 100 if denom > 0 else np.nan

        # ---- Volumen global (como antes) ----
        df_vol = df_resultados.dropna(subset=['Volumen_spline_trapecio', 'Volumen_poly_trapecio'])
        if len(df_vol) > 0:
            v_s_mean = df_vol['Volumen_spline_trapecio'].mean()
            v_p_mean = df_vol['Volumen_poly_trapecio'].mean()
            diff_v = rel_diff_sym(v_s_mean, v_p_mean)
            print(f"Spline (Trapecio): {v_s_mean:.3e}  |  Polinomio (Trapecio): {v_p_mean:.3e}")
            print(f"Diferencia promedio de volumen (global): {diff_v:.2f}%")
        else:
            print("Sin datos de volumen.")

        # ---- Área: usar rango común y diff por frame ----
        df_area = df_resultados.dropna(subset=['Area_spline_trapecio_overlap', 'Area_poly_trapecio_overlap'])
        if len(df_area) == 0:
            print("\nNo hay suficientes frames con rango común para área.")
            return

        diffs_area = []
        for _, row in df_area.iterrows():
            diffs_area.append(
                rel_diff_sym(row['Area_spline_trapecio_overlap'], row['Area_poly_trapecio_overlap'])
            )

        area_s_mean = df_area['Area_spline_trapecio_overlap'].mean()
        area_p_mean = df_area['Area_poly_trapecio_overlap'].mean()
        diff_area_prom = np.nanmean(diffs_area)

        print(f"\nÁrea promedio Spline (overlap): {area_s_mean:.3e}  |  Polinomio (overlap): {area_p_mean:.3e}")
        print(f"Diferencia promedio de área (promedio de diffs por frame): {diff_area_prom:.2f}%")

        print("\nNotas:")
        print("- Para área comparamos SOLO sobre la intersección de dominios por frame.")
        print("- El diff de área se calcula por frame y luego se promedia (más justo).")
        print("- Método unificado de área: Savitzky–Golay en r(y) y dr/dy para ambos métodos.")
        print("Sugerencia de reporte final: SPLINE (PCHIP) + SIMPSON.")

# ---------------------- MAIN ----------------------
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
