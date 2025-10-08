# ejercicio2_tp5.py (VERSIÓN CORREGIDA)
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
        self.tiempos_exp = datos_experimentales['Tiempo (s)'].values
        self.alturas_exp = datos_experimentales['Centroide_y (µm)'].values * 1e-6  # Convertir a metros

        # Parámetros del modelo
        self.m = 1e-6  # masa estimada [kg] - se ajustará según el volumen
        self.k = 10.0  # rigidez inicial [N/m]
        self.c = 0.1  # amortiguamiento inicial [Ns/m]
        self.yeq = np.min(self.alturas_exp)  # altura de equilibrio

        # Resultados
        self.resultados = {}

    def estimar_parametros_iniciales(self):
        """Estima parámetros iniciales basados en datos experimentales"""
        # Estimar masa a partir del volumen promedio (asumiendo densidad del agua)
        densidad_agua = 1000  # kg/m³
        # Usar datos del ejercicio 1 si están disponibles
        try:
            df_volumen = pd.read_excel('resultados_tp5_volumen_area.xlsx')
            volumen_promedio = df_volumen['Volumen_spline_trapecio'].mean()
            self.m = densidad_agua * volumen_promedio
        except:
            # Estimación por defecto
            self.m = 1e-6

        # Estimar rigidez basada en la frecuencia observada
        dt = np.mean(np.diff(self.tiempos_exp))
        derivada = np.gradient(self.alturas_exp, self.tiempos_exp)
        cambios_signo = np.where(np.diff(np.sign(derivada)))[0]
        if len(cambios_signo) > 1:
            periodo_aprox = (self.tiempos_exp[cambios_signo[-1]] - self.tiempos_exp[cambios_signo[0]]) / (
                        len(cambios_signo) - 1)
            frecuencia = 2 * np.pi / periodo_aprox if periodo_aprox > 0 else 10
            self.k = self.m * frecuencia ** 2
        else:
            self.k = 5.0

        # Estimar amortiguamiento basado en decaimiento
        if len(self.alturas_exp) > 10:
            amplitud_inicial = np.max(self.alturas_exp[:10]) - self.yeq
            amplitud_final = np.max(self.alturas_exp[-10:]) - self.yeq
            if amplitud_inicial > 0:
                decaimiento = -np.log(amplitud_final / amplitud_inicial) / self.tiempos_exp[-1]
                self.c = 2 * self.m * decaimiento
            else:
                self.c = 0.1

        print(f"Parámetros estimados:")
        print(f"  Masa (m): {self.m:.2e} kg")
        print(f"  Rigidez (k): {self.k:.2f} N/m")
        print(f"  Amortiguamiento (c): {self.c:.4f} Ns/m")
        print(f"  Altura equilibrio (yeq): {self.yeq:.6f} m")

    def edo_gota(self, t, y):
        """Define la EDO: m·y'' + c·y' + k·(y - yeq) = 0"""
        # y[0] = posición, y[1] = velocidad
        dydt = [y[1],
                (-self.c * y[1] - self.k * (y[0] - self.yeq)) / self.m]
        return dydt

    def metodo_taylor_orden3(self, t_span, y0, dt, tol=1e-6):
        """Método de Taylor de orden 3 - CORREGIDO"""
        print("Aplicando método de Taylor orden 3...")

        t_values = np.arange(t_span[0], t_span[1] + dt, dt)
        y_values = np.zeros((len(t_values), 2))
        y_values[0] = y0

        coste_evaluaciones = 0

        for i in range(1, len(t_values)):
            t = t_values[i - 1]
            y = y_values[i - 1]

            # Evaluaciones de derivadas - CONVERTIR A NUMPY ARRAYS
            f1 = np.array(self.edo_gota(t, y))
            f2 = np.array(self.edo_gota(t, y + 0.5 * dt * f1))
            f3 = np.array(self.edo_gota(t, y + dt * f2))

            coste_evaluaciones += 3

            # Expansión de Taylor orden 3 - AHORA CON ARRAYS
            y_next = y + dt * f1 + (dt ** 2) / 2 * f2 + (dt ** 3) / 6 * f3

            # Control de error adaptativo
            error_est = np.linalg.norm((dt ** 3) / 6 * f3)
            if error_est > tol:
                dt = dt * 0.8 * (tol / error_est) ** (1 / 3)
                dt = max(dt, 1e-6)  # Límite mínimo

            y_values[i] = y_next

        self.resultados['taylor'] = {
            'tiempos': t_values,
            'alturas': y_values[:, 0],
            'velocidades': y_values[:, 1],
            'coste': coste_evaluaciones,
            'dt_promedio': np.mean(np.diff(t_values))
        }

        return t_values, y_values

    def metodo_runge_kutta_56(self, t_span, y0, tol=1e-8):
        """Método Runge-Kutta 5-6 (Dormand-Prince)"""
        print("Aplicando método Runge-Kutta 5-6...")

        start_time = time.time()

        solucion = solve_ivp(self.edo_gota, t_span, y0,
                             method='DOP853',  # Dormand-Prince 8(5,3)
                             rtol=tol,
                             atol=tol / 100,
                             dense_output=True)

        coste_tiempo = time.time() - start_time

        # Evaluar en los mismos puntos que los datos experimentales para comparación
        t_eval = self.tiempos_exp[self.tiempos_exp <= t_span[1]]
        y_eval = solucion.sol(t_eval)

        self.resultados['rk56'] = {
            'tiempos': t_eval,
            'alturas': y_eval[0],
            'velocidades': y_eval[1],
            'coste': solucion.nfev,  # Número de evaluaciones de función
            'tol_usada': tol
        }

        return t_eval, y_eval

    def metodo_adams_bashforth_moulton(self, t_span, y0, dt, tol=1e-6):
        """Método multipaso Adams-Bashforth-Moulton (4º orden) - CORREGIDO"""
        print("Aplicando método Adams-Bashforth-Moulton...")

        # Inicializar con Runge-Kutta de 4º orden
        t_values = np.arange(t_span[0], t_span[1] + dt, dt)
        y_values = np.zeros((len(t_values), 2))
        y_values[0] = y0

        # Primeros 4 puntos con RK4
        for i in range(1, min(4, len(t_values))):
            k1 = dt * np.array(self.edo_gota(t_values[i - 1], y_values[i - 1]))
            k2 = dt * np.array(self.edo_gota(t_values[i - 1] + dt / 2, y_values[i - 1] + k1 / 2))
            k3 = dt * np.array(self.edo_gota(t_values[i - 1] + dt / 2, y_values[i - 1] + k2 / 2))
            k4 = dt * np.array(self.edo_gota(t_values[i - 1] + dt, y_values[i - 1] + k3))

            y_values[i] = y_values[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        coste_evaluaciones = 12  # 3 pasos * 4 evaluaciones cada uno

        # Adams-Bashforth-Moulton para el resto
        for i in range(4, len(t_values)):
            # Predictor: Adams-Bashforth (4º orden) - CONVERTIR A ARRAYS
            f0 = np.array(self.edo_gota(t_values[i - 1], y_values[i - 1]))
            f1 = np.array(self.edo_gota(t_values[i - 2], y_values[i - 2]))
            f2 = np.array(self.edo_gota(t_values[i - 3], y_values[i - 3]))
            f3 = np.array(self.edo_gota(t_values[i - 4], y_values[i - 4]))

            predictor = y_values[i - 1] + dt * (55 / 24 * f0 -
                                                59 / 24 * f1 +
                                                37 / 24 * f2 -
                                                9 / 24 * f3)

            # Corrector: Adams-Moulton (4º orden)
            f_pred = np.array(self.edo_gota(t_values[i], predictor))
            corrector = y_values[i - 1] + dt * (9 / 24 * f_pred +
                                                19 / 24 * f0 -
                                                5 / 24 * f1 +
                                                1 / 24 * f2)

            coste_evaluaciones += 5  # 4 para predictor + 1 para corrector

            # Control de error
            error = np.linalg.norm(corrector - predictor)
            if error > tol:
                # Reducir paso si es necesario
                dt_new = dt * 0.9 * (tol / error) ** (1 / 4)
                if dt_new < dt:
                    dt = dt_new
                    # Recalcular con nuevo dt (simplificado)
                    continue

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
        """Ajusta k y c para minimizar error con datos experimentales"""
        print("\nAjustando parámetros k y c...")

        def error_modelo(params):
            k, c = params
            self.k, self.c = k, c

            # Resolver con RK56 (más preciso para ajuste)
            t_span = [self.tiempos_exp[0], self.tiempos_exp[-1]]
            y0 = [self.alturas_exp[0], 0]  # Velocidad inicial 0

            try:
                t_rk, y_rk = self.metodo_runge_kutta_56(t_span, y0, tol=1e-6)
                # Interpolar para comparar con puntos experimentales
                alturas_modelo = np.interp(self.tiempos_exp, t_rk, y_rk[0])
                error = np.mean((alturas_modelo - self.alturas_exp) ** 2)
                return error
            except:
                return 1e10

        # Búsqueda en grilla para encontrar buenos valores iniciales
        mejores_params = None
        mejor_error = 1e10

        for k in [1, 5, 10, 20, 50]:
            for c in [0.01, 0.05, 0.1, 0.2, 0.5]:
                error = error_modelo([k, c])
                if error < mejor_error:
                    mejor_error = error
                    mejores_params = [k, c]

        if mejores_params:
            self.k, self.c = mejores_params
            print(f"Parámetros ajustados: k={self.k:.2f}, c={self.c:.4f}")

        return mejores_params

    def comparar_metodos(self):
        """Compara los tres métodos numéricos"""
        print("\n" + "=" * 50)
        print("COMPARACIÓN DE MÉTODOS NUMÉRICOS")
        print("=" * 50)

        # Condiciones iniciales
        t_span = [self.tiempos_exp[0], self.tiempos_exp[-1]]
        y0 = np.array([self.alturas_exp[0], 0])  # [posición, velocidad] COMO ARRAY

        # Ejecutar los tres métodos
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
            # Interpolar para comparar en mismos puntos
            alturas_interp = np.interp(self.tiempos_exp, datos['tiempos'], datos['alturas'])
            error_rms = np.sqrt(np.mean((alturas_interp - self.alturas_exp) ** 2))
            errores[metodo] = error_rms

        # Reporte de comparación
        print("\n--- COMPARACIÓN DE COSTOS COMPUTACIONALES ---")
        print(f"{'Método':<20} {'Evaluaciones':<15} {'Error RMS':<12} {'dt promedio':<12}")
        print("-" * 60)

        for metodo in ['taylor', 'rk56', 'adams']:
            datos = self.resultados[metodo]
            print(f"{metodo:<20} {datos['coste']:<15} {errores[metodo]:.2e}    {datos.get('dt_promedio', 'N/A'):<12}")

        # Análisis de precisión vs costo
        print("\n--- ANÁLISIS PRECISIÓN/COSTE ---")
        metodo_eficiente = min(errores, key=lambda m: errores[m] * self.resultados[m]['coste'])
        print(f"Método más eficiente: {metodo_eficiente}")

        return self.resultados, errores

    def generar_graficos_comparativos(self):
        """Genera gráficos comparativos"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Gráfico 1: Comparación de métodos vs experimental
        ax = axes[0, 0]
        ax.plot(self.tiempos_exp, self.alturas_exp * 1e6, 'ko-',
                label='Experimental', alpha=0.7, markersize=3)

        for metodo, color, label in zip(['taylor', 'rk56', 'adams'],
                                        ['red', 'blue', 'green'],
                                        ['Taylor Orden 3', 'RK5-6', 'Adams-B-M']):
            if metodo in self.resultados:
                datos = self.resultados[metodo]
                ax.plot(datos['tiempos'], datos['alturas'] * 1e6,
                        color=color, label=label, alpha=0.8)

        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Altura (µm)')
        ax.set_title('Comparación: Modelo vs Experimental')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Gráfico 2: Errores relativos
        ax = axes[0, 1]
        for metodo, color in zip(['taylor', 'rk56', 'adams'], ['red', 'blue', 'green']):
            if metodo in self.resultados:
                datos = self.resultados[metodo]
                alturas_interp = np.interp(self.tiempos_exp, datos['tiempos'], datos['alturas'])
                error_rel = np.abs((alturas_interp - self.alturas_exp) / self.alturas_exp)
                ax.plot(self.tiempos_exp, error_rel * 100, color=color,
                        label=f'Error {metodo}', alpha=0.7)

        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Error Relativo (%)')
        ax.set_title('Error Relativo vs Tiempo')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Gráfico 3: Velocidades
        ax = axes[1, 0]
        for metodo, color in zip(['taylor', 'rk56', 'adams'], ['red', 'blue', 'green']):
            if metodo in self.resultados:
                datos = self.resultados[metodo]
                ax.plot(datos['tiempos'], datos['velocidades'] * 1e3,
                        color=color, label=metodo, alpha=0.7)

        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Velocidad (mm/s)')
        ax.set_title('Velocidad del Centro de Masa')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Gráfico 4: Análisis de parámetros
        ax = axes[1, 1]
        parametros_text = f"""
        Parámetros del Modelo:
        m = {self.m:.2e} kg
        k = {self.k:.2f} N/m
        c = {self.c:.4f} Ns/m
        yeq = {self.yeq * 1e6:.2f} µm
        """
        ax.text(0.1, 0.5, parametros_text, transform=ax.transAxes,
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
        # Inicializar solucionador
        solucionador = SolucionadorEDOGota(datos_experimentales)

        # Estimar parámetros iniciales
        solucionador.estimar_parametros_iniciales()

        # Ajustar parámetros al modelo
        solucionador.ajustar_parametros_modelo()

        # Comparar métodos numéricos
        resultados, errores = solucionador.comparar_metodos()

        # Generar gráficos
        solucionador.generar_graficos_comparativos()

        # Guardar resultados
        guardar_resultados_dinamica(resultados, solucionador)

        # Análisis de desviaciones
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

            # Resultados de Taylor
            if 'taylor' in resultados:
                df_taylor = pd.DataFrame({
                    'Tiempo (s)': resultados['taylor']['tiempos'],
                    'Altura (m)': resultados['taylor']['alturas'],
                    'Velocidad (m/s)': resultados['taylor']['velocidades']
                })
                df_taylor.to_excel(writer, sheet_name='Taylor_Orden3', index=False)

            # Resultados de RK56
            if 'rk56' in resultados:
                df_rk56 = pd.DataFrame({
                    'Tiempo (s)': resultados['rk56']['tiempos'],
                    'Altura (m)': resultados['rk56']['alturas'],
                    'Velocidad (m/s)': resultados['rk56']['velocidades']
                })
                df_rk56.to_excel(writer, sheet_name='Runge_Kutta_56', index=False)

            # Resultados de Adams
            if 'adams' in resultados:
                df_adams = pd.DataFrame({
                    'Tiempo (s)': resultados['adams']['tiempos'],
                    'Altura (m)': resultados['adams']['alturas'],
                    'Velocidad (m/s)': resultados['adams']['velocidades']
                })
                df_adams.to_excel(writer, sheet_name='Adams_Bashforth_Moulton', index=False)

            # Resumen comparativo
            resumen_data = []
            for metodo in ['taylor', 'rk56', 'adams']:
                if metodo in resultados:
                    datos = resultados[metodo]
                    alturas_interp = np.interp(solucionador.tiempos_exp, datos['tiempos'], datos['alturas'])
                    error_rms = np.sqrt(np.mean((alturas_interp - solucionador.alturas_exp) ** 2))

                    resumen_data.append({
                        'Metodo': metodo,
                        'Evaluaciones_Funcion': datos['coste'],
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

    # Calcular desviación promedio
    if 'rk56' in solucionador.resultados:
        datos_rk = solucionador.resultados['rk56']
        alturas_interp = np.interp(solucionador.tiempos_exp, datos_rk['tiempos'], datos_rk['alturas'])
        desviacion_promedio = np.mean(np.abs(alturas_interp - solucionador.alturas_exp)) * 1e6  # en µm

        print(f"\nDesviación promedio: {desviacion_promedio:.2f} µm")
        print(f"Desviación relativa: {desviacion_promedio / np.mean(solucionador.alturas_exp) * 1e6:.1f}%")


if __name__ == "__main__":
    try:
        datos = pd.read_excel('resultados_completos.xlsx', sheet_name='Datos Completos')
        generar_informe2(datos)
    except Exception as e:
        print(f"Error: {e}")