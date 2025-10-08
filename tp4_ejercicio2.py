import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd
import json
import warnings
from scipy.integrate import simpson as simps
import os
from tp4_exportar_excel import exportar_ejercicio2_excel

warnings.filterwarnings("ignore", category=RuntimeWarning)


def ajustar_contornos(df, grado_polinomio=3, suavizado_spline=0.5, bins_y=None):
    resultados = []

    required_columns = ['Contorno_x', 'Contorno_y', 'Imagen', 'Tiempo (s)']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("El DataFrame no contiene las columnas necesarias.")

    for _, row in df.iterrows():
        try:
            x = np.array(json.loads(row['Contorno_x'])) if isinstance(row['Contorno_x'], str) else np.array(row['Contorno_x'])
            y = np.array(json.loads(row['Contorno_y'])) if isinstance(row['Contorno_y'], str) else np.array(row['Contorno_y'])

            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]

            if len(x) < 10:
                continue

            if bins_y is None:
                y_round = np.round(y, decimals=2)
            else:
                y_round = np.digitize(y, bins_y)

            unique_y = np.unique(y_round)
            x_left = []
            y_left = []
            x_right = []
            y_right = []

            for uy in unique_y:
                mask_level = (y_round == uy)
                xs = x[mask_level]
                ys = y[mask_level]
                if len(xs) == 0:
                    continue

                y_mean = float(np.mean(ys))
                x_left.append(np.min(xs))
                y_left.append(y_mean)
                x_right.append(np.max(xs))
                y_right.append(y_mean)

            x_left = np.array(x_left);
            y_left = np.array(y_left)
            x_right = np.array(x_right);
            y_right = np.array(y_right)

            if len(y_left) > 0:
                idx_l = np.argsort(y_left)
                x_left, y_left = x_left[idx_l], y_left[idx_l]
            if len(y_right) > 0:
                idx_r = np.argsort(y_right)
                x_right, y_right = x_right[idx_r], y_right[idx_r]

            spline_izq = None
            spline_der = None
            if len(y_left) > 3:
                s_val = suavizado_spline
                if len(y_left) > 50:
                    s_val *= 10
                spline_izq = UnivariateSpline(y_left, x_left, s=s_val)

            if len(y_right) > 3:
                s_val = suavizado_spline
                if len(y_right) > 50:
                    s_val *= 10
                spline_der = UnivariateSpline(y_right, x_right, s=s_val)

            coef_poly_izq = np.polyfit(y_left, x_left, grado_polinomio) if len(y_left) > grado_polinomio else None
            coef_poly_der = np.polyfit(y_right, x_right, grado_polinomio) if len(y_right) > grado_polinomio else None

            perimetro_izq = np.sum(np.sqrt(np.diff(x_left) ** 2 + np.diff(y_left) ** 2)) if len(x_left) > 1 else np.nan
            perimetro_der = np.sum(np.sqrt(np.diff(x_right) ** 2 + np.diff(y_right) ** 2)) if len(
                x_right) > 1 else np.nan

            diametro_base = np.max(x) - np.min(x) if len(x) > 0 else np.nan
            altura_max = np.max(y) if len(y) > 0 else np.nan
            centro_x = np.mean(x)

            resultados.append({
                'Imagen': row['Imagen'],
                'Tiempo (s)': row['Tiempo (s)'],
                'spline_izq': spline_izq,
                'spline_der': spline_der,
                'coef_poly_izq': coef_poly_izq.tolist() if coef_poly_izq is not None else None,
                'coef_poly_der': coef_poly_der.tolist() if coef_poly_der is not None else None,
                'Perimetro_izq': perimetro_izq,
                'Perimetro_der': perimetro_der,
                'Asimetria_perimetro': abs(perimetro_izq - perimetro_der) / (perimetro_izq + perimetro_der)
                if (perimetro_izq + perimetro_der) > 0 else np.nan,
                'Diametro_base': diametro_base,
                'Altura_max': altura_max,
                'Factor_esparcimiento': diametro_base / altura_max if altura_max > 0 else np.nan,
                'Area': simps(np.abs(x - centro_x), y) if len(x) > 1 else np.nan
            })

        except Exception as e:
            print(f"Error procesando {row.get('Imagen', 'desconocida')}: {str(e)}")
            continue

    return pd.DataFrame(resultados)


def calcular_angulo_contacto(df_ajustes, altura_contacto=50):

    angulos = []
    densidad = 7380  # kg/m³

    if df_ajustes.empty:
        raise ValueError("DataFrame de ajustes está vacío.")

    for idx, (_, row) in enumerate(df_ajustes.iterrows()):
        try:
            imagen_num = int(row['Imagen'].split('_')[-1].split('.')[0])
            if imagen_num < 18:
                angulos.append({
                    'Imagen': row['Imagen'],
                    'Tiempo (s)': row['Tiempo (s)'],
                    'Angulo_izq': np.nan,
                    'Angulo_der': np.nan,
                    'Perimetro_izq': row.get('Perimetro_izq', np.nan),
                    'Perimetro_der': row.get('Perimetro_der', np.nan),
                    'Asimetria_perimetro': row.get('Asimetria_perimetro', np.nan),
                    'Factor_esparcimiento': row.get('Factor_esparcimiento', np.nan),
                    'Tipo_angulo': 'Pre-contacto'
                })
                continue

            angulo_izq = np.nan
            angulo_der = np.nan

            # LADO IZQUIERDO.
            if row['spline_izq'] is not None:
                try:
                    y_min, y_max = row['spline_izq'].get_knots()[0], row['spline_izq'].get_knots()[-1]
                    y_eval = np.linspace(max(0, y_min), min(altura_contacto, y_max), 20)
                    y_eval = y_min + (y_eval - y_min) ** 2 / (y_eval[-1] - y_min)

                    dxdy_izq = row['spline_izq'].derivative()(y_eval)
                    dxdy_izq = np.asarray(dxdy_izq, dtype=float)

                    if np.any(np.isfinite(dxdy_izq)):
                        valid_vals = dxdy_izq[np.isfinite(dxdy_izq)]
                        median = np.median(valid_vals)
                        mad = np.median(np.abs(valid_vals - median))
                        mask = np.abs(valid_vals - median) < 2.0 * mad

                        if np.any(mask):
                            weights = np.exp(-y_eval[np.isfinite(dxdy_izq)][mask] / altura_contacto)

                            theta_left = float(np.average(
                                np.degrees(np.arctan2(1.0, valid_vals[mask])),
                                weights=weights
                            ))

                            angulo_izq = 180 - theta_left
                except Exception as e:
                    angulo_izq = np.nan

            # LADO DERECHO.
            if row['spline_der'] is not None:
                try:
                    y_min, y_max = row['spline_der'].get_knots()[0], row['spline_der'].get_knots()[-1]
                    y_eval = np.linspace(max(0, y_min), min(altura_contacto, y_max), 20)
                    y_eval = y_min + (y_eval - y_min) ** 2 / (y_eval[-1] - y_min)

                    dxdy_der = row['spline_der'].derivative()(y_eval)
                    dxdy_der = np.asarray(dxdy_der, dtype=float)

                    if np.any(np.isfinite(dxdy_der)):
                        valid_vals = dxdy_der[np.isfinite(dxdy_der)]
                        median = np.median(valid_vals)
                        mad = np.median(np.abs(valid_vals - median))
                        mask = np.abs(valid_vals - median) < 2.0 * mad

                        if np.any(mask):
                            weights = np.exp(-y_eval[np.isfinite(dxdy_der)][mask] / altura_contacto)

                            angulo_der = float(np.average(
                                np.degrees(np.arctan2(1.0, valid_vals[mask])),
                                weights=weights
                            ))

                except Exception as e:
                    angulo_der = np.nan

            factor_actual = row.get('Factor_esparcimiento', np.nan)
            tiempo_actual = row['Tiempo (s)']

            umbral_cambio = 0.05
            es_dinamico = True

            if len(angulos) > 0:
                factores_anteriores = [a.get('Factor_esparcimiento', np.nan) for a in angulos
                                       if not np.isnan(a.get('Factor_esparcimiento', np.nan))]
                if factores_anteriores:
                    factor_anterior = factores_anteriores[-1]
                    tiempo_anterior = angulos[-1].get('Tiempo (s)', 0)

                    if not np.isnan(factor_actual) and not np.isnan(
                            factor_anterior) and tiempo_actual > tiempo_anterior:
                        tasa_cambio = abs(
                            (factor_actual - factor_anterior) / (factor_anterior * (tiempo_actual - tiempo_anterior)))
                        es_dinamico = tasa_cambio > umbral_cambio

            angulos.append({
                'Imagen': row['Imagen'],
                'Tiempo (s)': row['Tiempo (s)'],
                'Angulo_izq': angulo_izq,
                'Angulo_der': angulo_der,
                'Perimetro_izq': row.get('Perimetro_izq', np.nan),
                'Perimetro_der': row.get('Perimetro_der', np.nan),
                'Asimetria_perimetro': row.get('Asimetria_perimetro', np.nan),
                'Factor_esparcimiento': factor_actual,
                'Tipo_angulo': 'Dinámico' if es_dinamico else 'Estático'
            })

        except Exception as e:
            print(f"Error calculando ángulos para {row.get('Imagen', 'desconocida')}: {str(e)}")
            angulos.append({
                'Imagen': row.get('Imagen', 'desconocida'),
                'Tiempo (s)': row.get('Tiempo (s)', np.nan),
                'Angulo_izq': np.nan,
                'Angulo_der': np.nan,
                'Perimetro_izq': np.nan,
                'Perimetro_der': np.nan,
                'Asimetria_perimetro': np.nan,
                'Factor_esparcimiento': np.nan,
                'Tipo_angulo': 'Desconocido'
            })
            continue

    return pd.DataFrame(angulos)


def graficar_resultados(df_angulos):
    if df_angulos.empty:
        raise ValueError("No hay datos para graficar.")

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    ax1 = axs[0, 0]
    if 'Angulo_izq' in df_angulos:
        mask_izq = ~df_angulos['Angulo_izq'].isna()
        ax1.plot(df_angulos['Tiempo (s)'][mask_izq], df_angulos['Angulo_izq'][mask_izq], 'b-', label='Izquierdo')
    if 'Angulo_der' in df_angulos:
        mask_der = ~df_angulos['Angulo_der'].isna()
        ax1.plot(df_angulos['Tiempo (s)'][mask_der], df_angulos['Angulo_der'][mask_der], 'r-', label='Derecho',
                 linewidth=2)
    ax1.set_title('Evolución de Ángulos de Contacto', fontsize=12, pad=10)
    ax1.set_xlabel('Tiempo (s)', fontsize=10)
    ax1.set_ylabel('Ángulo (grados)', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    if 'Angulo_izq' in df_angulos and 'Angulo_der' in df_angulos:
        angulos_izq_validos = df_angulos['Angulo_izq'].dropna()
        angulos_der_validos = df_angulos['Angulo_der'].dropna()

        if len(angulos_izq_validos) > 0:
            avg_izq = angulos_izq_validos.mean()
            ax1.axhline(y=avg_izq, color='b', linestyle='--', alpha=0.5)
            ax1.text(0.02, 0.98, f'Prom. Izq: {avg_izq:.1f}°',
                     transform=ax1.transAxes, fontsize=9, verticalalignment='top', color='blue')

        if len(angulos_der_validos) > 0:
            avg_der = angulos_der_validos.mean()
            ax1.axhline(y=avg_der, color='r', linestyle='--', alpha=0.5)
            ax1.text(0.02, 0.93, f'Prom. Der: {avg_der:.1f}°',
                     transform=ax1.transAxes, fontsize=9, verticalalignment='top', color='red')

    ax2 = axs[0, 1]
    if 'Asimetria_perimetro' in df_angulos:
        ax2.plot(df_angulos['Tiempo (s)'], df_angulos['Asimetria_perimetro'], 'g-', linewidth=2)
    ax2.set_title('Asimetría del Perímetro', fontsize=12, pad=10)
    ax2.set_xlabel('Tiempo (s)', fontsize=10)
    ax2.set_ylabel('Asimetría', fontsize=10)
    ax2.grid(True, alpha=0.3)
    if 'Asimetria_perimetro' in df_angulos:
        avg_asim = df_angulos['Asimetria_perimetro'].mean()
        ax2.axhline(y=avg_asim, color='g', linestyle='--', alpha=0.5)
        ax2.text(0.02, 0.98, f'Prom. Asimetría: {avg_asim:.3f}',
                 transform=ax2.transAxes, fontsize=9, verticalalignment='top', color='green')

    ax3 = axs[1, 0]
    if 'Factor_esparcimiento' in df_angulos:
        ax3.plot(df_angulos['Tiempo (s)'], df_angulos['Factor_esparcimiento'], 'm-', linewidth=2)
    ax3.set_title('Factor de Esparcimiento', fontsize=12, pad=10)
    ax3.set_xlabel('Tiempo (s)', fontsize=10)
    ax3.set_ylabel('Factor', fontsize=10)
    ax3.grid(True, alpha=0.3)
    if 'Factor_esparcimiento' in df_angulos:
        avg_factor = df_angulos['Factor_esparcimiento'].mean()
        ax3.axhline(y=avg_factor, color='m', linestyle='--', alpha=0.5)
        ax3.text(0.02, 0.98, f'Prom. Factor: {avg_factor:.3f}',
                 transform=ax3.transAxes, fontsize=9, verticalalignment='top', color='magenta')

    ax4 = axs[1, 1]
    if 'Energia_cinetica' in df_angulos:
        ax4.plot(df_angulos['Tiempo (s)'], df_angulos['Energia_cinetica'], 'c-')
    ax4.grid(True)

    fig.tight_layout()

    fig.suptitle('Análisis de Ángulos de Contacto y Parámetros Relacionados', fontsize=14, y=1.02)

    fig.savefig('resultados_angulos.png', dpi=300, bbox_inches='tight')
    fig.savefig('resultados_ejercicio3.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def generar_informe2():
    print("\n\n=== EJERCICIO 2: Análisis de ángulos de contacto ===")
    try:
        if not os.path.exists('resultados_completos.xlsx'):
            raise FileNotFoundError("No se encontró el archivo 'resultados_completos.xlsx'")

        df = pd.read_excel('resultados_completos.xlsx', sheet_name='Datos Completos')
        required_cols = ['Contorno_x', 'Contorno_y', 'Imagen', 'Tiempo (s)']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"El archivo Excel no contiene las columnas requeridas: {required_cols}")

        df_ajustes = ajustar_contornos(df, grado_polinomio=3, suavizado_spline=1.0)

        df_angulos = calcular_angulo_contacto(df_ajustes, altura_contacto=50)

        exportar_ejercicio2_excel(df_angulos, 'resultados_completos2.xlsx')
        graficar_resultados(df_angulos)

        print("\n====== Resultados ======")
        if not df_angulos.empty:
            angulos_izq_validos = df_angulos['Angulo_izq'].dropna()
            angulos_der_validos = df_angulos['Angulo_der'].dropna()

            if len(angulos_izq_validos) > 0:
                print(f"Ángulos izquierdos: μ = {angulos_izq_validos.mean():.1f}° ± {angulos_izq_validos.std():.1f}°")
            if len(angulos_der_validos) > 0:
                print(f"Ángulos derechos: μ = {angulos_der_validos.mean():.1f}° ± {angulos_der_validos.std():.1f}°")

            if 'Tipo_angulo' in df_angulos:
                counts = df_angulos[df_angulos['Tipo_angulo'].isin(['Dinámico', 'Estático'])][
                    'Tipo_angulo'].value_counts(normalize=True)
                print(f"\n====== Distribución de tipos de ángulo ======")
                for tipo, porcentaje in counts.items():
                    print(f"- {tipo}: {porcentaje:.1%}")

        return df_angulos

    except Exception as e:
        print(f"\nERROR en Ejercicio 2: {str(e)}")
        return None