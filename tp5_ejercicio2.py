import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import warnings
from tp5_exportar_excel import exportar_ejercicio2_excel

warnings.filterwarnings('ignore')

class SolucionadorEDOGota:
    def __init__(self, datos_experimentales):
        self.datos_exp = datos_experimentales
        self.tiempos_exp = datos_experimentales['Tiempo (s)'].astype(float).values

        col_centroide_y = None
        for col_name in ['Centroide_y (µm)', 'Centroide_y (µm)_x', 'Centroide_y (µm)_y']:
            if col_name in datos_experimentales.columns:
                col_centroide_y = col_name
                break
        
        if col_centroide_y:
            alt_raw = datos_experimentales[col_centroide_y].astype(float).values
        else:
            print(f"  [ADVERTENCIA] No se encontró columna Centroide_y. Columnas disponibles: {datos_experimentales.columns.tolist()}")
            alt_raw = np.full(len(self.tiempos_exp), np.nan)

        if np.all(np.isnan(alt_raw)):
            print(f"  [ADVERTENCIA] Todos los valores de altura son NaN, usando ceros")
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

        self.alturas_exp = alt_raw * 1e-6

        # Parámetros del modelo.
        self.m = 1e-6  # masa estimada [kg].
        self.k = 10.0  # rigidez inicial [N/m].
        self.c = 0.1  # amortiguamiento inicial [Ns/m].

        n = len(self.alturas_exp)
        if n > 5:
            tail = max(5, int(0.2 * n))
            try:
                self.yeq = float(np.nanmedian(self.alturas_exp[-tail:]))
                if not np.isfinite(self.yeq) or abs(self.yeq) < 1e-10:
                    print(f"  [DEBUG] yeq era {self.yeq:.6e}, usando media de todos los datos")
                    self.yeq = float(np.nanmean(self.alturas_exp))
            except Exception as e:
                print(f"  [DEBUG] Excepción calculando yeq: {e}")
                self.yeq = float(np.nanmean(self.alturas_exp))
        else:
            self.yeq = float(np.nanmedian(self.alturas_exp)) if np.isfinite(np.nanmedian(self.alturas_exp)) else float(np.nanmean(self.alturas_exp))

        self.resultados = {}

    def estimar_parametros_iniciales(self):
        densidad_agua = 1000  # [kg/m³]
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
            self.m = 1e-6

        dt_arr = np.diff(self.tiempos_exp)
        dt = np.mean(dt_arr) if dt_arr.size else 1e-3
        derivada = np.gradient(self.alturas_exp, self.tiempos_exp)
        cambios_signo = np.where(np.diff(np.sign(derivada)))[0]
        if len(cambios_signo) > 1:
            periodo_aprox = (self.tiempos_exp[cambios_signo[-1]] - self.tiempos_exp[cambios_signo[0]]) / max(1, (len(cambios_signo) - 1))
            frecuencia = 2 * np.pi / periodo_aprox if periodo_aprox > 0 else 10.0
            self.k = max(1e-6, self.m * frecuencia ** 2)
        else:
            self.k = 5.0

        if len(self.alturas_exp) > 10:
            tail = 10
            head = min(10, len(self.alturas_exp))
            amplitud_inicial = np.nanmax(self.alturas_exp[:head]) - self.yeq
            amplitud_final = np.nanmax(self.alturas_exp[-tail:]) - self.yeq
            if amplitud_inicial > 0 and amplitud_final > 0:
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
        dydt = np.array([y[1],
                         (-self.c * y[1] - self.k * (y[0] - self.yeq)) / self.m])
        return dydt

    def metodo_taylor_orden3(self, t_span, y0, dt_inicial=None, tol=1e-6, dt_min=1e-9):
        print("Aplicando método de Taylor orden 3...")

        if dt_inicial is None:
            omega_n = np.sqrt(self.k / self.m) if self.m > 0 else 100.0
            periodo = 2 * np.pi / omega_n if omega_n > 0 else 1e-3
            dt_inicial = periodo / 20
            dt_inicial = min(dt_inicial, 1e-4)

        t = float(t_span[0])
        t_end = float(t_span[1])
        dt = dt_inicial

        t_list = [t]
        y_list = [np.array(y0, dtype=float)]

        coste_evaluaciones = 0

        while t < t_end - 1e-12:
            if t + dt > t_end:
                dt = t_end - t

            y = y_list[-1]

            f1 = self.edo_gota(t, y)
            f2 = self.edo_gota(t + 0.5 * dt, y + 0.5 * dt * f1)
            f3 = self.edo_gota(t + dt, y + dt * f2)

            coste_evaluaciones += 3

            y_next = y + dt * f1 + (dt ** 2) / 2 * f2 + (dt ** 3) / 6 * f3

            error_est = np.linalg.norm((dt ** 3) / 6 * f3)
            
            if error_est > tol and dt > dt_min:
                dt_new = dt * 0.8 * (tol / error_est) ** (1 / 3)
                dt = max(dt_new, dt_min)
                continue

            t = t + dt
            t_list.append(t)
            y_list.append(y_next)

            if error_est < tol * 0.1 and t < t_end:
                dt = min(dt * 1.2, (t_end - t), dt_inicial * 5)

        t_values = np.array(t_list)
        y_values = np.vstack(y_list)

        self.resultados['taylor'] = {
            'tiempos': t_values,
            'alturas': y_values[:, 0],
            'velocidades': y_values[:, 1],
            'coste': coste_evaluaciones,
            'dt_promedio': np.mean(np.diff(t_values)) if len(t_values) > 1 else np.nan,
            'dt_inicial': dt_inicial
        }

        return t_values, y_values

    def metodo_runge_kutta_56(self, t_span, y0, tol=1e-8):

        start_time = time.time()

        solucion = solve_ivp(self.edo_gota, t_span, y0, method='DOP853', rtol=tol, atol=tol / 100, dense_output=True)

        coste_tiempo = time.time() - start_time

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

    def metodo_adams_bashforth_moulton(self, t_span, y0, dt_inicial=None, tol=1e-6):
        print("Aplicando método Adams-Bashforth-Moulton...")

        if dt_inicial is None:
            omega_n = np.sqrt(self.k / self.m) if self.m > 0 else 100.0
            periodo = 2 * np.pi / omega_n if omega_n > 0 else 1e-3
            dt_inicial = periodo / 100
            dt_inicial = min(dt_inicial, 5e-5)
        
        dt = dt_inicial
        dt_min = dt_inicial / 100
        dt_max = dt_inicial * 5
        t = t_span[0]
        t_end = t_span[1]
        
        t_list = [t]
        y_list = [np.array(y0, dtype=float)]
        
        # Inicializar con RK4
        coste_evaluaciones = 0
        for step in range(3):
            if t >= t_end:
                break
            dt_actual = min(dt, t_end - t)
            k1 = self.edo_gota(t, y_list[-1])
            k2 = self.edo_gota(t + dt_actual / 2, y_list[-1] + dt_actual / 2 * k1)
            k3 = self.edo_gota(t + dt_actual / 2, y_list[-1] + dt_actual / 2 * k2)
            k4 = self.edo_gota(t + dt_actual, y_list[-1] + dt_actual * k3)
            y_new = y_list[-1] + dt_actual * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            
            t += dt_actual
            t_list.append(t)
            y_list.append(y_new)
            coste_evaluaciones += 4

        intentos_reduccion = 0
        max_intentos = 5
        
        while t < t_end - 1e-12:
            if len(t_list) < 4:
                break
                
            dt_actual = min(dt, t_end - t)

            f0 = self.edo_gota(t_list[-1], y_list[-1])
            f1 = self.edo_gota(t_list[-2], y_list[-2])
            f2 = self.edo_gota(t_list[-3], y_list[-3])
            f3 = self.edo_gota(t_list[-4], y_list[-4])
            
            y_pred = y_list[-1] + dt_actual * (55/24 * f0 - 59/24 * f1 + 37/24 * f2 - 9/24 * f3)

            t_new = t + dt_actual
            f_pred = self.edo_gota(t_new, y_pred)
            y_corr = y_list[-1] + dt_actual * (9/24 * f_pred + 19/24 * f0 - 5/24 * f1 + 1/24 * f2)
            
            coste_evaluaciones += 5

            error_local = np.linalg.norm(y_corr - y_pred)
            
            if error_local > tol * 10 and intentos_reduccion < max_intentos:
                dt = max(dt * 0.5, dt_min)
                intentos_reduccion += 1
                continue

            t = t_new
            t_list.append(t)
            y_list.append(y_corr)
            intentos_reduccion = 0

            if error_local > tol:
                factor = 0.95 * (tol / max(error_local, 1e-15)) ** 0.2
            else:
                factor = 1.1
            
            factor = max(0.8, min(1.2, factor))
            dt = min(max(dt * factor, dt_min), dt_max)

        t_values = np.array(t_list)
        y_values = np.vstack(y_list)

        self.resultados['adams'] = {
            'tiempos': t_values,
            'alturas': y_values[:, 0],
            'velocidades': y_values[:, 1],
            'coste': coste_evaluaciones,
            'dt_promedio': np.mean(np.diff(t_values)),
            'dt_inicial': dt_inicial
        }

        return t_values, y_values

    def ajustar_parametros_modelo(self):
        print("\nAjustando parámetros k y c...")

        def error_modelo(params):
            k, c = params
            self.k, self.c = k, c

            t_span = [self.tiempos_exp[0], self.tiempos_exp[-1]]
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
                error = np.sqrt(np.mean((alturas_modelo - self.alturas_exp) ** 2))
                return float(error)
            except Exception:
                return 1e10

        mejores_params = None
        mejor_error = 1e10

        ks = [0.5, 2, 5, 10, 20]
        cs = [0.001, 0.01, 0.05, 0.1]
        
        for k in ks:
            for c in cs:
                err = error_modelo([k, c])
                if err < mejor_error:
                    mejor_error = err
                    mejores_params = [k, c]

        if mejores_params:
            self.k, self.c = mejores_params
            print(f"Parámetros ajustados: k={self.k:.2f}, c={self.c:.6f}, error RMS={mejor_error * 1e6:.2f} µm")

        return mejores_params

    def comparar_metodos(self):
        print("\n" + "=" * 50)
        print("COMPARACIÓN DE MÉTODOS NUMÉRICOS")
        print("=" * 50)

        t_span = [self.tiempos_exp[0], self.tiempos_exp[-1]]

        v0 = 0.0
        try:
            v_est = np.gradient(self.alturas_exp, self.tiempos_exp)
            v0 = float(v_est[0])
        except:
            v0 = 0.0

        y0 = np.array([self.alturas_exp[0], v0])

        print("\nEjecutando métodos...")

        t_taylor, y_taylor = self.metodo_taylor_orden3(t_span, y0, tol=3e-9)
        t_rk, y_rk = self.metodo_runge_kutta_56(t_span, y0, tol=1e-8)
        t_adams, y_adams = self.metodo_adams_bashforth_moulton(t_span, y0, tol=2e-8)

        errores = {}
        errores_relativos = {}
        
        for metodo, datos in self.resultados.items():
            alturas_interp = np.interp(self.tiempos_exp, datos['tiempos'], datos['alturas'])
            
            # Error absoluto RMS
            diferencias = alturas_interp - self.alturas_exp
            error_rms = np.sqrt(np.mean(diferencias ** 2))
            errores[metodo] = error_rms
            
            # Error relativo promedio (%)
            mask_valid = self.alturas_exp != 0
            if mask_valid.any():
                error_rel = np.mean(np.abs(diferencias[mask_valid] / self.alturas_exp[mask_valid])) * 100
                errores_relativos[metodo] = error_rel
            else:
                errores_relativos[metodo] = np.nan

        print("\n--- COMPARACIÓN DE COSTOS COMPUTACIONALES ---")
        print(f"{'Método':<20} {'Evaluaciones':<15} {'Error RMS (µm)':<18} {'Error Rel (%)':<15} {'dt prom (s)':<15}")
        print("-" * 95)

        for metodo in ['taylor', 'rk56', 'adams']:
            datos = self.resultados.get(metodo, {})
            coste = datos.get('coste', 'N/A')
            dtprom = datos.get('dt_promedio', 'N/A')
            err_m = errores.get(metodo, np.nan)
            err_um = err_m * 1e6
            err_rel = errores_relativos.get(metodo, np.nan)

            if isinstance(dtprom, (int, float)) and np.isfinite(dtprom):
                dtprom_str = f"{dtprom:.4e}"
            else:
                dtprom_str = str(dtprom)
            
            print(f"{metodo:<20} {str(coste):<15} {err_um:<18.4f} {err_rel:<15.4f} {dtprom_str:<15}")

        return self.resultados, errores

    def generar_graficos_comparativos(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Gráfico 1: Altura vs Tiempo
        ax = axes[0, 0]
        ax.plot(self.tiempos_exp, self.alturas_exp * 1e6, 'ko-', label='Experimental', alpha=0.7, markersize=3)

        for metodo, style, label in zip(['taylor', 'rk56', 'adams'],
                                        ['r-', 'b--', 'g-.'],
                                        ['Taylor Orden 3', 'RK5-6', 'Adams-B-M']):
            if metodo in self.resultados:
                datos = self.resultados[metodo]
                ax.plot(datos['tiempos'], datos['alturas'] * 1e6, style, label=label, alpha=0.8)

        ax.set_xlabel('Tiempo (s)', fontsize=10)
        ax.set_ylabel('Altura (µm)', fontsize=10)
        ax.set_title('Comparación: Modelo vs Experimental', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Gráfico 2: Error Relativo vs Tiempo
        ax = axes[0, 1]
        for metodo, style in zip(['taylor', 'rk56', 'adams'], ['r-', 'b--', 'g-.']):
            if metodo in self.resultados:
                datos = self.resultados[metodo]
                alturas_interp = np.interp(self.tiempos_exp, datos['tiempos'], datos['alturas'])
                denom = np.where(np.abs(self.alturas_exp) < 1e-12, 1e-12, self.alturas_exp)
                error_rel = np.abs((alturas_interp - self.alturas_exp) / denom)
                ax.plot(self.tiempos_exp, error_rel * 100, style, label=f'{metodo}', alpha=0.7)

        ax.set_xlabel('Tiempo (s)', fontsize=10)
        ax.set_ylabel('Error Relativo (%)', fontsize=10)
        ax.set_title('Error Relativo vs Tiempo', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Gráfico 3: Velocidad vs Tiempo
        ax = axes[1, 0]
        for metodo, style in zip(['taylor', 'rk56', 'adams'], ['r-', 'b--', 'g-.']):
            if metodo in self.resultados:
                datos = self.resultados[metodo]
                ax.plot(datos['tiempos'], datos['velocidades'] * 1e3, style, label=metodo, alpha=0.7)

        ax.set_xlabel('Tiempo (s)', fontsize=10)
        ax.set_ylabel('Velocidad (mm/s)', fontsize=10)
        ax.set_title('Velocidad del Centro de Masa', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Gráfico 4: Evaluaciones acumuladas vs Tiempo (NUEVO)
        ax = axes[1, 1]
        
        for metodo, style, label in zip(['taylor', 'rk56', 'adams'], ['r-', 'b--', 'g-.'], ['Taylor Orden 3', 'RK5-6', 'Adams-B-M']):
            if metodo in self.resultados:
                datos = self.resultados[metodo]
                tiempos = datos['tiempos']
                n_puntos = len(tiempos)
                coste_total = datos.get('coste', 0)

                if n_puntos > 1:
                    tiempo_normalizado = (tiempos - tiempos[0]) / (tiempos[-1] - tiempos[0])
                    eval_acumuladas = coste_total * tiempo_normalizado
                    ax.plot(tiempos, eval_acumuladas, style, label=label, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Tiempo (s)', fontsize=10)
        ax.set_ylabel('Evaluaciones de función acumuladas', fontsize=10)
        ax.set_title('Costo Computacional vs Tiempo', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        for metodo in ['taylor', 'rk56', 'adams']:
            if metodo in self.resultados:
                coste = self.resultados[metodo].get('coste', 0)
                ax.text(0.98, 0.02 + 0.08 * ['taylor', 'rk56', 'adams'].index(metodo), 
                       f'{metodo}: {coste} eval',
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plt.tight_layout()
        plt.savefig('graficos_dinamica_tp5.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nGráficos guardados en: graficos_dinamica_tp5.png")


def generar_informe2(datos_experimentales):
    print("=== EJERCICIO 2 TP5 - MODELO DE DINÁMICA DE GOTA ===")

    try:
        solucionador = SolucionadorEDOGota(datos_experimentales)
        solucionador.estimar_parametros_iniciales()
        solucionador.ajustar_parametros_modelo()
        resultados, errores = solucionador.comparar_metodos()
        solucionador.generar_graficos_comparativos()
        exportar_ejercicio2_excel(resultados, solucionador)
        analizar_desviaciones(solucionador)
        return resultados
    except Exception as e:
        print(f"Error en Ejercicio 2: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def guardar_resultados_dinamica(resultados, solucionador):
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

    print("\n" + "=" * 50)
    print("ANÁLISIS DE DESVIACIONES (Inciso d)")
    print("=" * 50)

    if 'rk56' not in solucionador.resultados:
        print("No hay resultados de RK5-6 para analizar.")
        return

    datos_rk = solucionador.resultados['rk56']
    alturas_modelo = np.interp(solucionador.tiempos_exp, datos_rk['tiempos'], datos_rk['alturas'])
    alturas_exp = solucionador.alturas_exp

    desviaciones = alturas_modelo - alturas_exp
    desviaciones_um = desviaciones * 1e6
    desviaciones_abs = np.abs(desviaciones)

    desv_promedio = np.mean(desviaciones_abs) * 1e6
    desv_std = np.std(desviaciones_abs) * 1e6
    desv_max = np.max(desviaciones_abs) * 1e6
    desv_rms = np.sqrt(np.mean(desviaciones ** 2)) * 1e6
    
    mean_alt = np.mean(alturas_exp) * 1e6
    desv_rel = (np.mean(desviaciones_abs) / np.mean(alturas_exp)) * 100 if np.mean(alturas_exp) > 0 else np.nan
    
    print("\n--- ESTADÍSTICAS DE DESVIACIÓN ---")
    print(f"Desviación promedio:     {desv_promedio:.2f} µm ({desv_rel:.2f}% relativo)")
    print(f"Desviación std:          {desv_std:.2f} µm")
    print(f"Desviación máxima:       {desv_max:.2f} µm")
    print(f"Error RMS:               {desv_rms:.2f} µm")
    print(f"Altura promedio (exp):   {mean_alt:.2f} µm")

    print(f"\nPrecisión actual del modelo: {100 - desv_rel:.1f}% (error relativo {desv_rel:.2f}%)")
    print("Este nivel de precisión es razonable para un modelo simplificado.")

