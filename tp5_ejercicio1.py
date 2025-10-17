# tp5_ejercicio1.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.integrate import simpson, trapezoid
from scipy.signal import savgol_filter
import json
import warnings
from tp5_exportar_excel import exportar_ejercicio1_excel

warnings.filterwarnings('ignore')

class CalculadorVolumenArea:
    def __init__(self, df, scale=1.0, n_puntos=50, suavizado=0.5):
        self.df = df
        self.scale = scale
        self.n_puntos = n_puntos
        self.suavizado = suavizado
        self.resultados = []

    def obtener_mitad_contorno(self, contorno_x, contorno_y):
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
        if r is None or y is None or len(y) < 4:
            return None
        idx = np.argsort(y)
        y = y[idx];
        r = r[idx]
        y_unique, indices = np.unique(y, return_index=True)
        r_unique = r[indices]
        if len(y_unique) < 4:
            return None
        try:
            return PchipInterpolator(y_unique, r_unique, extrapolate=False)
        except Exception:
            return None

    def ajustar_polinomio(self, r, y, grado=3):
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
        r = funcion(y_eval)
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(r, 0.0)

    # ---------------------- VOLUMENES ----------------------
    def volumen_por_revolucion_trapecio(self, funcion, y_min, y_max, n_puntos=None):
        n = n_puntos or self.n_puntos
        y_eval = np.linspace(y_min, y_max, n)
        r_eval = self._eval_seguro(funcion, y_eval)
        integrando = np.pi * r_eval ** 2
        volumen = trapezoid(integrando, y_eval)
        return volumen, 0.0

    def volumen_por_revolucion_simpson(self, funcion, y_min, y_max, n_puntos=None):
        n = n_puntos or self.n_puntos
        y_eval = np.linspace(y_min, y_max, n)
        r_eval = self._eval_seguro(funcion, y_eval)
        integrando = np.pi * r_eval ** 2
        volumen = simpson(integrando, y_eval)
        return volumen, 0.0

    # ---------------------- ÁREA ----------------------
    def _area_con_pipeline_unica(self, funcion, y_min, y_max, n_puntos=50, usar_simpson=False):
        y_eval = np.linspace(y_min, y_max, n_puntos)
        r = self._eval_seguro(funcion, y_eval)

        frac = max(0.02, float(self.suavizado))
        win = max(7, int(n_puntos * frac) | 1)
        polyorder = 3 if win > 3 else 2

        r_suav = savgol_filter(r, window_length=win, polyorder=polyorder, mode='interp')
        dr_dy = savgol_filter(r_suav, window_length=win, polyorder=polyorder, deriv=1, delta=(y_eval[1] - y_eval[0]), mode='interp')

        integrando = 2.0 * np.pi * np.maximum(r_suav, 0.0) * np.sqrt(1.0 + dr_dy ** 2)
        area = simpson(integrando, y_eval) if usar_simpson else trapezoid(integrando, y_eval)
        return max(float(area), 0.0)

    def area_superficial_trapecio(self, funcion, _derivada_no_usada, y_min, y_max, n_puntos=50):
        return self._area_con_pipeline_unica(funcion, y_min, y_max, n_puntos=n_puntos, usar_simpson=False)

    def area_superficial_simpson(self, funcion, _derivada_no_usada, y_min, y_max, n_puntos=50):
        return self._area_con_pipeline_unica(funcion, y_min, y_max, n_puntos=n_puntos, usar_simpson=True)

    # ---------------------- CÁLCULO DE ERRORES RELATIVOS ----------------------
    def calcular_errores_relativos(self, df_resultados):
        print("\n" + "=" * 60)
        print("ESTADÍSTICAS DE ERRORES RELATIVOS (%)")
        print("=" * 60)

        def error_relativo(a, b):
            if np.isnan(a) or np.isnan(b) or (a == 0 and b == 0):
                return np.nan
            denom = (abs(a) + abs(b)) / 2.0
            return abs(a - b) / denom * 100.0 if denom > 0 else np.nan

        errores = {}

        # Volumen - Spline: Trapecio vs Simpson (frame por frame)
        if 'Volumen_spline_trapecio' in df_resultados.columns and 'Volumen_spline_simpson' in df_resultados.columns:
            mask = df_resultados['Volumen_spline_trapecio'].notna() & df_resultados['Volumen_spline_simpson'].notna()
            if mask.any():
                errors_frame = []
                for idx in df_resultados[mask].index:
                    a = df_resultados.loc[idx, 'Volumen_spline_trapecio']
                    b = df_resultados.loc[idx, 'Volumen_spline_simpson']
                    err = error_relativo(a, b)
                    if not np.isnan(err):
                        errors_frame.append(err)

                if errors_frame:
                    errores['volumen_spline_trapecio'] = np.mean(errors_frame)
                else:
                    errores['volumen_spline_trapecio'] = 0.0
            else:
                errores['volumen_spline_trapecio'] = 0.0

        # Volumen - Polinomio: Trapecio vs Simpson (frame por frame)
        if 'Volumen_poly_trapecio' in df_resultados.columns and 'Volumen_poly_simpson' in df_resultados.columns:
            mask = df_resultados['Volumen_poly_trapecio'].notna() & df_resultados['Volumen_poly_simpson'].notna()
            if mask.any():
                errors_frame = []
                for idx in df_resultados[mask].index:
                    a = df_resultados.loc[idx, 'Volumen_poly_trapecio']
                    b = df_resultados.loc[idx, 'Volumen_poly_simpson']
                    err = error_relativo(a, b)
                    if not np.isnan(err):
                        errors_frame.append(err)

                if errors_frame:
                    errores['volumen_poly_trapecio'] = np.mean(errors_frame)
                else:
                    errores['volumen_poly_trapecio'] = 0.0
            else:
                errores['volumen_poly_trapecio'] = 0.0

        # Área - Spline: Trapecio vs Simpson (frame por frame)
        if 'Area_spline_trapecio' in df_resultados.columns and 'Area_spline_simpson' in df_resultados.columns:
            mask = df_resultados['Area_spline_trapecio'].notna() & df_resultados['Area_spline_simpson'].notna()
            if mask.any():
                errors_frame = []
                for idx in df_resultados[mask].index:
                    a = df_resultados.loc[idx, 'Area_spline_trapecio']
                    b = df_resultados.loc[idx, 'Area_spline_simpson']
                    err = error_relativo(a, b)
                    if not np.isnan(err):
                        errors_frame.append(err)

                if errors_frame:
                    errores['area_spline_trapecio'] = np.mean(errors_frame)
                else:
                    errores['area_spline_trapecio'] = 0.0
            else:
                errores['area_spline_trapecio'] = 0.0

        # Área - Polinomio: Trapecio vs Simpson (frame por frame)
        if 'Area_poly_trapecio' in df_resultados.columns and 'Area_poly_simpson' in df_resultados.columns:
            mask = df_resultados['Area_poly_trapecio'].notna() & df_resultados['Area_poly_simpson'].notna()
            if mask.any():
                errors_frame = []
                for idx in df_resultados[mask].index:
                    a = df_resultados.loc[idx, 'Area_poly_trapecio']
                    b = df_resultados.loc[idx, 'Area_poly_simpson']
                    err = error_relativo(a, b)
                    if not np.isnan(err):
                        errors_frame.append(err)

                if errors_frame:
                    errores['area_poly_trapecio'] = np.mean(errors_frame)
                else:
                    errores['area_poly_trapecio'] = 0.0
            else:
                errores['area_poly_trapecio'] = 0.0

        print("Volumen Spline Trapecio: {:.4f}%".format(errores.get('volumen_spline_trapecio', 0.0)))
        print("Volumen Spline Simpson: {:.4f}%".format(errores.get('volumen_spline_trapecio', 0.0)))
        print("Volumen Polinomio Trapecio: {:.4f}%".format(errores.get('volumen_poly_trapecio', 0.0)))
        print("Volumen Polinomio Simpson: {:.4f}%".format(errores.get('volumen_poly_trapecio', 0.0)))
        print("Área Spline Trapecio: {:.4f}%".format(errores.get('area_spline_trapecio', 0.0)))
        print("Área Spline Simpson: {:.4f}%".format(errores.get('area_spline_trapecio', 0.0)))
        print("Área Polinomio Trapecio: {:.4f}%".format(errores.get('area_poly_trapecio', 0.0)))
        print("Área Polinomio Simpson: {:.4f}%".format(errores.get('area_poly_trapecio', 0.0)))

        return errores

    # ---------------------- GRÁFICAS DE EVOLUCIÓN ----------------------
    def generar_graficas_evolucion(self, df_resultados):
        print("\nGenerando gráficas de evolución...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        df_valido = df_resultados.dropna(subset=['Volumen_spline_trapecio', 'Volumen_spline_simpson', 'Volumen_poly_trapecio', 'Volumen_poly_simpson',
            'Area_spline_trapecio', 'Area_spline_simpson', 'Area_poly_trapecio', 'Area_poly_simpson'])

        if len(df_valido) == 0:
            print("No hay datos válidos para generar gráficas")
            return

        frames = range(len(df_valido))

        # Gráfica 1: Evolución del Volumen - Spline
        ax1 = axes[0, 0]
        ax1.plot(frames, df_valido['Volumen_spline_trapecio'], 'b-', label='Spline-Trapecio', linewidth=2)
        ax1.plot(frames, df_valido['Volumen_spline_simpson'], 'r--', label='Spline-Simpson', linewidth=2)
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Volumen (mm³)')
        ax1.set_title('Evolución del Volumen - Método Spline')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Gráfica 2: Evolución del Volumen - Polinomio
        ax2 = axes[0, 1]
        ax2.plot(frames, df_valido['Volumen_poly_trapecio'], 'g-', label='Polinomio-Trapecio', linewidth=2)
        ax2.plot(frames, df_valido['Volumen_poly_simpson'], 'm--', label='Polinomio-Simpson', linewidth=2)
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Volumen (mm³)')
        ax2.set_title('Evolución del Volumen - Método Polinomio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Gráfica 3: Evolución del Área - Spline
        ax3 = axes[1, 0]
        ax3.plot(frames, df_valido['Area_spline_trapecio'], 'b-', label='Spline-Trapecio', linewidth=2)
        ax3.plot(frames, df_valido['Area_spline_simpson'], 'r--', label='Spline-Simpson', linewidth=2)
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Área (mm²)')
        ax3.set_title('Evolución del Área - Método Spline')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Gráfica 4: Evolución del Área - Polinomio
        ax4 = axes[1, 1]
        ax4.plot(frames, df_valido['Area_poly_trapecio'], 'g-', label='Polinomio-Trapecio', linewidth=2)
        ax4.plot(frames, df_valido['Area_poly_simpson'], 'm--', label='Polinomio-Simpson', linewidth=2)
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Área (mm²)')
        ax4.set_title('Evolución del Área - Método Polinomio')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('graficos_evolucion_volumen_area_tp5.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Gráficas de evolución guardadas en: graficos_evolucion_volumen_area_tp5.png")

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

                self.resultados.append(resultados_frame)

            except Exception as e:
                print(f"Error en frame {row.get('Imagen', idx)}: {str(e)}")
                continue

        df_resultados = pd.DataFrame(self.resultados)

        self.calcular_errores_relativos(df_resultados)
        self.generar_graficas_evolucion(df_resultados)

        return df_resultados

    def generar_analisis_comparativo(self, df_resultados):
        print("\n" + "=" * 60)
        print("ANÁLISIS COMPARATIVO - MÉTODOS DE INTEGRACIÓN")
        print("=" * 60)

        def rel_diff_sym(a, b):
            denom = (abs(a) + abs(b)) / 2
            return abs(a - b) / denom * 100 if denom > 0 else np.nan

        # ---- Volumen global ----
        df_vol = df_resultados.dropna(subset=['Volumen_spline_trapecio', 'Volumen_poly_trapecio'])
        if len(df_vol) > 0:
            v_s_mean = df_vol['Volumen_spline_trapecio'].mean()
            v_p_mean = df_vol['Volumen_poly_trapecio'].mean()
            diff_v = rel_diff_sym(v_s_mean, v_p_mean)
            print(f"Spline (Trapecio): {v_s_mean:.3e}  |  Polinomio (Trapecio): {v_p_mean:.3e}")
            print(f"Diferencia promedio de volumen (global): {diff_v:.2f}%")
        else:
            print("Sin datos de volumen.")

        # ---- Área global ----
        df_area = df_resultados.dropna(subset=['Area_spline_trapecio', 'Area_poly_trapecio'])
        if len(df_area) > 0:
            area_s_mean = df_area['Area_spline_trapecio'].mean()
            area_p_mean = df_area['Area_poly_trapecio'].mean()
            diff_area = rel_diff_sym(area_s_mean, area_p_mean)
            print(f"Área promedio Spline: {area_s_mean:.3e}  |  Polinomio: {area_p_mean:.3e}")
            print(f"Diferencia promedio de área (global): {diff_area:.2f}%")


def generar_informe1(scale=1.0):
    try:
        df = pd.read_excel('resultados_completos.xlsx', sheet_name='Datos Completos')
        print(f"Datos cargados: {len(df)} frames")

        calculador = CalculadorVolumenArea(df, scale=scale)
        df_res = calculador.procesar_todos_frames()
        calculador.generar_analisis_comparativo(df_res)

        exportar_ejercicio1_excel(df_res, 'resultados_tp5_volumen_area.xlsx')

        print("\nResultados guardados en: resultados_tp5_volumen_area.xlsx")
        print("Gráficas guardadas en: graficos_evolucion_volumen_area_tp5.png")

        return df_res

    except Exception as e:
        print(f"Error en Ejercicio 1: {str(e)}")
        return None