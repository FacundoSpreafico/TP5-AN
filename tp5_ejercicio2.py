# ejercicio2_tp5_corregido.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import warnings

warnings.filterwarnings('ignore')


class SolucionadorEDOGota:
    """Clase para resolver la EDO de la dinámica de la gota"""

    def __init__(self, datos_experimentales):
        self.datos_exp = datos_experimentales
        # Limpiar NaNs y garantizar vectores finitos
        self.tiempos_exp = datos_experimentales['Tiempo (s)'].astype(float).values
        alt_raw = datos_experimentales.get('Centroide_y (µm)', pd.Series(np.nan)).astype(float).values

        # Reemplazar NaNs por el último valor válido o por cero si todos NaN
        if np.all(np.isnan(alt_raw)):
            alt_raw = np.zeros_like(self.tiempos_exp)
        else:
            mask = np.isfinite(alt_raw)
            if not mask.all():
                valid_idx = np.where(mask)[0]
                if valid_idx.size:
                    first, last = valid_idx[0], valid_idx[-1]
                    alt_raw[:first] = alt_raw[first]
                    alt_raw[last + 1:] = alt_raw[last]
                    nans = ~mask
                    alt_raw[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(mask), alt_raw[mask])

        self.alturas_exp = alt_raw * 1e-6  # Convertir µm -> m

        # Parámetros del modelo (colocar estimaciones por defecto)
        self.m = 1e-6  # masa estimada [kg] - se ajustará según el volumen
        self.k = 10.0  # rigidez inicial [N/m]
        self.c = 0.1  # amortiguamiento inicial [Ns/m]

        # altura de equilibrio: usar mediana de la porción final para robustez
        n = len(self.alturas_exp)
        if n > 5:
            tail = max(1, int(0.1 * n))
            try:
                self.yeq = float(np.nanmedian(self.alturas_exp[-tail:]))
            except:
                self.yeq = 0.0
        else:
            self.yeq = float(np.nanmedian(self.alturas_exp)) if np.isfinite(np.nanmedian(self.alturas_exp)) else 0.0

        # Resultados
        self.resultados = {}

    def estimar_parametros_iniciales(self):
        """Estima parámetros iniciales basados en datos experimentales"""
        densidad_agua = 1000  # kg/m³
        try:
            df_volumen = pd.read_excel('resultados_tp5_volumen_area.xlsx')
            volumen_promedio = df_volumen['Volumen_spline_trapecio'].mean()
            if not np.isfinite(volumen_promedio):
                raise ValueError("Volumen promedio no finito")
            if volumen_promedio > 1e-3:
                volumen_m3 = float(volumen_promedio) * 1e-18
            else:
                volumen_m3 = float(volumen_promedio)
            self.m = densidad_agua * volumen_m3
        except Exception:
            # Estimación por defecto
            self.m = 1e-6

        # Estimar rigidez mediante frecuencia observada (simple)
        dt_arr = np.diff(self.tiempos_exp)
        dt = np.mean(dt_arr) if dt_arr.size else 1e-3
        derivada = np.gradient(self.alturas_exp, self.tiempos_exp)
        # Buscar cruces de signo en la derivada (aprox. máximos/minimos => periodos)
        cambios_signo = np.where(np.diff(np.sign(derivada)))[0]
        if len(cambios_signo) > 1:
            periodo_aprox = (self.tiempos_exp[cambios_signo[-1]] - self.tiempos_exp[cambios_signo[0]]) / max(1, (len(cambios_signo) - 1))
            frecuencia = 2 * np.pi / periodo_aprox if periodo_aprox > 0 else 10.0
            self.k = max(1e-6, self.m * frecuencia ** 2)
        else:
            self.k = 5.0

        # Estimar amortiguamiento por decaimiento exponencial aproximado
        if len(self.alturas_exp) > 10:
            tail = 10
            head = min(10, len(self.alturas_exp))
            amplitud_inicial = np.nanmax(self.alturas_exp[:head]) - self.yeq
            amplitud_final = np.nanmax(self.alturas_exp[-tail:]) - self.yeq
            if amplitud_inicial > 0 and amplitud_final > 0:
                # evitar log(0)
                decaimiento = -np.log(amplitud_final / amplitud_inicial) / max(1e-6, (self.tiempos_exp[-1] - self.tiempos_exp[0]))
                self.c = 2 * self.m * decaimiento
            else:
                self.c = 0.1

        print(f"Parámetros estimados:")
        print(f"  Masa (m): {self.m:.2e} kg")
        print(f"  Rigidez (k): {self.k:.2f} N/m")
        print(f"  Amortiguamiento (c): {self.c:.4f} Ns/m")
        print(f"  Altura equilibrio (yeq): {self.yeq:.6e} m")

    def edo_gota(self, t, y):
        """Define la EDO: m·y'' + c·y' + k·(y - yeq) = 0"""
        dydt = np.array([y[1],
                         (-self.c * y[1] - self.k * (y[0] - self.yeq)) / self.m])
        return dydt

    def metodo_taylor_orden3(self, t_span, y0, dt, tol=1e-6, dt_min=1e-9):
        """Método de Taylor de orden 3 con paso adaptativo (implementación robusta)"""
        print("Aplicando método de Taylor orden 3...")

        t = float(t_span[0])
        t_end = float(t_span[1])

        t_list = [t]
        y_list = [np.array(y0, dtype=float)]

        coste_evaluaciones = 0

        while t < t_end - 1e-12:
            # evitar overshoot final
            if t + dt > t_end:
                dt = t_end - t

            y = y_list[-1]

            # Evaluaciones de derivadas
            f1 = self.edo_gota(t, y)
            f2 = self.edo_gota(t + 0.5 * dt, y + 0.5 * dt * f1)
            f3 = self.edo_gota(t + dt, y + dt * f2)

            coste_evaluaciones += 3

            # Taylor orden 3
            y_next = y + dt * f1 + (dt ** 2) / 2 * f2 + (dt ** 3) / 6 * f3

            # Estimación de error local (aprox) usando término de orden 4 negligente:
            # usamos norma del último término como proxy. No es un estimador perfecto,
            # pero útil para adaptar dt en este contexto del TP.
            error_est = np.linalg.norm((dt ** 3) / 6 * f3)
            if error_est > tol and dt > dt_min:
                # reducir dt
                dt_new = dt * 0.8 * (tol / error_est) ** (1 / 3)
                dt = max(dt_new, dt_min)
                # intentamos de nuevo con dt reducido (no avanzamos t)
                continue

            # aceptar paso
            t = t + dt
            t_list.append(t)
            y_list.append(y_next)

            # si el error es demasiado pequeño, aumentar dt ligeramente para eficiencia
            if error_est < tol * 1e-3:
                dt = min(dt * 1.2, (t_end - t)) if t < t_end else dt

        t_values = np.array(t_list)
        y_values = np.vstack(y_list)

        self.resultados['taylor'] = {
            'tiempos': t_values,
            'alturas': y_values[:, 0],
            'velocidades': y_values[:, 1],
            'coste': coste_evaluaciones,
            'dt_promedio': np.mean(np.diff(t_values)) if len(t_values) > 1 else np.nan
        }

        return t_values, y_values

    def metodo_runge_kutta_56(self, t_span, y0, tol=1e-8):
        """Método Runge-Kutta 5-6 (Dormand-Prince / DOP853) usando solve_ivp"""
        print("Aplicando método Runge-Kutta 5-6...")

        start_time = time.time()

        solucion = solve_ivp(self.edo_gota, t_span, y0,
                             method='DOP853',
                             rtol=tol,
                             atol=tol / 100,
                             dense_output=True)

        coste_tiempo = time.time() - start_time

        # Evaluar en los mismos puntos que los datos experimentales para comparación
        t_eval = self.tiempos_exp[self.tiempos_exp <= t_span[1]]
        if t_eval.size == 0:
            t_eval = np.linspace(t_span[0], t_span[1], 200)
        y_eval = solucion.sol(t_eval)

        self.resultados['rk56'] = {
            'tiempos': t_eval,
            'alturas': y_eval[0],
            'velocidades': y_eval[1],
            'coste': int(getattr(solucion, 'nfev', np.nan)),
            'tol_usada': tol,
            'cpu_time': coste_tiempo
        }

        return t_eval, y_eval

    def metodo_adams_bashforth_moulton(self, t_span, y0, dt, tol=1e-6):
        """Método multipaso Adams-Bashforth-Moulton (4º orden)"""
        print("Aplicando método Adams-Bashforth-Moulton...")

        t_values = np.arange(t_span[0], t_span[1] + dt, dt)
        y_values = np.zeros((len(t_values), 2))
        y_values[0] = y0

        # Inicializar con RK4 (primeros 3 pasos)
        for i in range(1, min(4, len(t_values))):
            k1 = dt * self.edo_gota(t_values[i - 1], y_values[i - 1])
            k2 = dt * self.edo_gota(t_values[i - 1] + dt / 2, y_values[i - 1] + k1 / 2)
            k3 = dt * self.edo_gota(t_values[i - 1] + dt / 2, y_values[i - 1] + k2 / 2)
            k4 = dt * self.edo_gota(t_values[i - 1] + dt, y_values[i - 1] + k3)
            y_values[i] = y_values[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        coste_evaluaciones = 4 * min(3, len(t_values) - 1)  # aproximado

        for i in range(4, len(t_values)):
            f0 = self.edo_gota(t_values[i - 1], y_values[i - 1])
            f1 = self.edo_gota(t_values[i - 2], y_values[i - 2])
            f2 = self.edo_gota(t_values[i - 3], y_values[i - 3])
            f3 = self.edo_gota(t_values[i - 4], y_values[i - 4])

            predictor = y_values[i - 1] + dt * (55 / 24 * f0 -
                                                59 / 24 * f1 +
                                                37 / 24 * f2 -
                                                9 / 24 * f3)

            f_pred = self.edo_gota(t_values[i], predictor)
            corrector = y_values[i - 1] + dt * (9 / 24 * f_pred +
                                                19 / 24 * f0 -
                                                5 / 24 * f1 +
                                                1 / 24 * f2)

            coste_evaluaciones += 5

            error = np.linalg.norm(corrector - predictor)
            if error > tol:
                # reducción conservadora de dt (nota: para un ABM adaptativo completo habría que reconstruir la malla)
                dt_new = dt * 0.9 * (tol / error) ** (1 / 4)
                if dt_new < dt * 0.5 and dt_new > 1e-12:
                    print(f"ADVERTENCIA: ABM redujo dt de {dt:.2e} a {dt_new:.2e} en i={i}")
                # aquí simplemente aceptamos el corrector y continuamos; en caso práctico re-evaluar la malla
            y_values[i] = corrector

        self.resultados['adams'] = {
            'tiempos': t_values,
            'alturas': y_values[:, 0],
            'velocidades': y_values[:, 1],
            'coste': coste_evaluaciones,
            'dt_promedio': np.mean(np.diff(t_values))
        }

        return t_values, y_values

    def ajustar_parametros_modelo(self):
        """Ajusta k y c para minimizar error con datos experimentales (búsqueda en grilla simple)"""
        print("\nAjustando parámetros k y c...")

        def error_modelo(params):
            k, c = params
            self.k, self.c = k, c

            t_span = [self.tiempos_exp[0], self.tiempos_exp[-1]]
            # velocidad inicial estimada desde derivada si es posible
            v0 = 0.0
            try:
                v_est = np.gradient(self.alturas_exp, self.tiempos_exp)
                v0 = float(v_est[0])
            except:
                v0 = 0.0

            y0 = [self.alturas_exp[0], v0]

            try:
                t_rk, y_rk = self.metodo_runge_kutta_56(t_span, y0, tol=1e-6)
                alturas_modelo = np.interp(self.tiempos_exp, t_rk, y_rk[0])
                error = np.mean((alturas_modelo - self.alturas_exp) ** 2)
                return float(error)
            except Exception:
                return 1e10

        mejores_params = None
        mejor_error = 1e10

        # rango de búsqueda razonable; pueden ampliarse
        ks = [1, 5, 10, 20, 50, 100]
        cs = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
        for k in ks:
            for c in cs:
                err = error_modelo([k, c])
                if err < mejor_error:
                    mejor_error = err
                    mejores_params = [k, c]

        if mejores_params:
            self.k, self.c = mejores_params
            print(f"Parámetros ajustados: k={self.k:.2f}, c={self.c:.6f}, error={mejor_error:.3e}")

        return mejores_params

    def comparar_metodos(self):
        """Compara los tres métodos numéricos"""
        print("\n" + "=" * 50)
        print("COMPARACIÓN DE MÉTODOS NUMÉRICOS")
        print("=" * 50)

        t_span = [self.tiempos_exp[0], self.tiempos_exp[-1]]

        # velocidad inicial estimada
        v0 = 0.0
        try:
            v_est = np.gradient(self.alturas_exp, self.tiempos_exp)
            v0 = float(v_est[0])
        except:
            v0 = 0.0

        y0 = np.array([self.alturas_exp[0], v0])

        print("\nEjecutando métodos...")

        # Taylor orden 3
        t_taylor, y_taylor = self.metodo_taylor_orden3(t_span, y0, dt=1e-4, tol=1e-6)

        # Runge-Kutta 5-6
        t_rk, y_rk = self.metodo_runge_kutta_56(t_span, y0, tol=1e-8)

        # Adams-Bashforth-Moulton
        t_adams, y_adams = self.metodo_adams_bashforth_moulton(t_span, y0, dt=1e-4, tol=1e-6)

        # Calcular errores vs datos experimentales
        errores = {}
        for metodo, datos in self.resultados.items():
            alturas_interp = np.interp(self.tiempos_exp, datos['tiempos'], datos['alturas'])
            error_rms = np.sqrt(np.mean((alturas_interp - self.alturas_exp) ** 2))
            errores[metodo] = error_rms

        # Reporte de comparación
        print("\n--- COMPARACIÓN DE COSTOS COMPUTACIONALES ---")
        print(f"{'Método':<20} {'Evaluaciones':<15} {'Error RMS (m)':<18} {'dt promedio (s)':<15}")
        print("-" * 80)

        for metodo in ['taylor', 'rk56', 'adams']:
            datos = self.resultados.get(metodo, {})
            coste = datos.get('coste', 'N/A')
            dtprom = datos.get('dt_promedio', 'N/A')
            err = errores.get(metodo, np.nan)
            print(f"{metodo:<20} {str(coste):<15} {err:<18.4e} {str(dtprom):<15}")

        # Análisis de precisión vs costo
        metodo_eficiente = min(errores, key=lambda m: errores[m] * max(1, float(self.resultados[m].get('coste', 1))))
        print(f"\nMétodo más eficiente (error * coste): {metodo_eficiente}")

        return self.resultados, errores

    def generar_graficos_comparativos(self):
        """Genera gráficos comparativos"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        ax = axes[0, 0]
        ax.plot(self.tiempos_exp, self.alturas_exp * 1e6, 'ko-', label='Experimental', alpha=0.7, markersize=3)

        for metodo, style, label in zip(['taylor', 'rk56', 'adams'],
                                        ['r-', 'b--', 'g-.'],
                                        ['Taylor Orden 3', 'RK5-6', 'Adams-B-M']):
            if metodo in self.resultados:
                datos = self.resultados[metodo]
                ax.plot(datos['tiempos'], datos['alturas'] * 1e6, style, label=label, alpha=0.8)

        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Altura (µm)')
        ax.set_title('Comparación: Modelo vs Experimental')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        for metodo, style in zip(['taylor', 'rk56', 'adams'], ['r-', 'b--', 'g-.']):
            if metodo in self.resultados:
                datos = self.resultados[metodo]
                alturas_interp = np.interp(self.tiempos_exp, datos['tiempos'], datos['alturas'])
                denom = np.where(np.abs(self.alturas_exp) < 1e-12, 1e-12, self.alturas_exp)
                error_rel = np.abs((alturas_interp - self.alturas_exp) / denom)
                ax.plot(self.tiempos_exp, error_rel * 100, style, label=f'Error {metodo}', alpha=0.7)

        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Error Relativo (%)')
        ax.set_title('Error Relativo vs Tiempo')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        for metodo, style in zip(['taylor', 'rk56', 'adams'], ['r-', 'b--', 'g-.']):
            if metodo in self.resultados:
                datos = self.resultados[metodo]
                ax.plot(datos['tiempos'], datos['velocidades'] * 1e3, style, label=metodo, alpha=0.7)

        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Velocidad (mm/s)')
        ax.set_title('Velocidad del Centro de Masa')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        parametros_text = f"Parámetros del Modelo:\n" \
                          f"m = {self.m:.2e} kg\n" \
                          f"k = {self.k:.2f} N/m\n" \
                          f"c = {self.c:.4f} Ns/m\n" \
                          f"yeq = {self.yeq * 1e6:.2f} µm\n"
        ax.text(0.05, 0.5, parametros_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='center', bbox=dict(boxstyle="round", facecolor='wheat'))
        ax.axis('off')
        ax.set_title('Parámetros del Modelo Ajustados')

        plt.tight_layout()
        plt.savefig('graficos_dinamica_tp5.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Gráficos guardados en: graficos_dinamica_tp5.png")


def generar_informe2(datos_experimentales):
    """Función principal del Ejercicio 2"""
    print("=== EJERCICIO 2 TP5 - MODELO DE DINÁMICA DE GOTA ===")

    try:
        solucionador = SolucionadorEDOGota(datos_experimentales)
        solucionador.estimar_parametros_iniciales()
        solucionador.ajustar_parametros_modelo()
        resultados, errores = solucionador.comparar_metodos()
        solucionador.generar_graficos_comparativos()
        guardar_resultados_dinamica(resultados, solucionador)
        analizar_desviaciones(solucionador)
        return resultados
    except Exception as e:
        print(f"Error en Ejercicio 2: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def guardar_resultados_dinamica(resultados, solucionador):
    """Guarda resultados en Excel"""
    try:
        with pd.ExcelWriter('resultados_tp5_dinamica.xlsx') as writer:
            if 'taylor' in resultados:
                df_taylor = pd.DataFrame({
                    'Tiempo (s)': resultados['taylor']['tiempos'],
                    'Altura (m)': resultados['taylor']['alturas'],
                    'Velocidad (m/s)': resultados['taylor']['velocidades']
                })
                df_taylor.to_excel(writer, sheet_name='Taylor_Orden3', index=False)

            if 'rk56' in resultados:
                df_rk56 = pd.DataFrame({
                    'Tiempo (s)': resultados['rk56']['tiempos'],
                    'Altura (m)': resultados['rk56']['alturas'],
                    'Velocidad (m/s)': resultados['rk56']['velocidades']
                })
                df_rk56.to_excel(writer, sheet_name='Runge_Kutta_56', index=False)

            if 'adams' in resultados:
                df_adams = pd.DataFrame({
                    'Tiempo (s)': resultados['adams']['tiempos'],
                    'Altura (m)': resultados['adams']['alturas'],
                    'Velocidad (m/s)': resultados['adams']['velocidades']
                })
                df_adams.to_excel(writer, sheet_name='Adams_Bashforth_Moulton', index=False)

            resumen_data = []
            for metodo in ['taylor', 'rk56', 'adams']:
                if metodo in resultados:
                    datos = resultados[metodo]
                    alturas_interp = np.interp(solucionador.tiempos_exp, datos['tiempos'], datos['alturas'])
                    error_rms = np.sqrt(np.mean((alturas_interp - solucionador.alturas_exp) ** 2))

                    resumen_data.append({
                        'Metodo': metodo,
                        'Evaluaciones_Funcion': datos.get('coste', np.nan),
                        'Error_RMS (m)': error_rms,
                        'dt_Promedio (s)': datos.get('dt_promedio', 'N/A'),
                        'Tolerancia': datos.get('tol_usada', 'N/A')
                    })

            df_resumen = pd.DataFrame(resumen_data)
            df_resumen.to_excel(writer, sheet_name='Resumen_Comparativo', index=False)

        print("Resultados guardados en: resultados_tp5_dinamica.xlsx")

    except Exception as e:
        print(f"Error guardando resultados: {e}")


def analizar_desviaciones(solucionador):
    """Analiza posibles causas de desviaciones"""
    print("\n" + "=" * 50)
    print("ANÁLISIS DE DESVIACIONES")
    print("=" * 50)

    print("\nPosibles causas de desviaciones modelo vs experimental:")
    print("1. Simplificación del modelo: EDO lineal vs sistema no lineal real")
    print("2. Parámetros constantes: k y c podrían variar con el tiempo")
    print("3. Efectos de tensión superficial no incluidos en el modelo")
    print("4. Suposición de partícula puntual vs gota deformable")
    print("5. Ruido en mediciones experimentales")
    print("6. Condiciones iniciales aproximadas")

    if 'rk56' in solucionador.resultados:
        datos_rk = solucionador.resultados['rk56']
        alturas_interp = np.interp(solucionador.tiempos_exp, datos_rk['tiempos'], datos_rk['alturas'])
        desviacion_promedio_m = np.mean(np.abs(alturas_interp - solucionador.alturas_exp))
        desviacion_promedio_um = desviacion_promedio_m * 1e6

        mean_alt_m = np.mean(solucionador.alturas_exp) if np.isfinite(np.mean(solucionador.alturas_exp)) and np.mean(solucionador.alturas_exp) != 0 else np.nan
        desviacion_rel_pct = (desviacion_promedio_m / mean_alt_m * 100.0) if np.isfinite(mean_alt_m) else np.nan

        print(f"\nDesviación promedio: {desviacion_promedio_um:.2f} µm")
        if np.isfinite(desviacion_rel_pct):
            print(f"Desviación relativa: {desviacion_rel_pct:.2f}%")
        else:
            print("Desviación relativa: N/A (media de alturas no finita o cero)")


if __name__ == "__main__":
    try:
        datos = pd.read_excel('resultados_completos.xlsx', sheet_name='Datos Completos')
        generar_informe2(datos)
    except Exception as e:
        print(f"Error: {e}")
