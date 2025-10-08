import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage import measure
from skimage.filters import threshold_otsu
import pandas as pd
from tqdm import tqdm
import json
from tp4_exportar_excel import exportar_ejercicio1_excel
import tempfile

def procesar_imagen(ruta_imagen, escala=4.13, referencia_pixel=(0, 131)):
    try:
        if escala <= 0:
            raise ValueError("La escala debe ser un valor positivo")

        if not os.path.exists(ruta_imagen):
            raise FileNotFoundError(f"Imagen no encontrada: {ruta_imagen}")

        imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if imagen is None:
            raise ValueError(f"No se pudo cargar la imagen {ruta_imagen}")

        # Suavizado Gaussiano para reducir ruido
        imagen_suavizada = cv2.GaussianBlur(imagen, (5, 5), 0)

        # Binarización (Metodo de Otsu para Umbral Automático)
        thresh = threshold_otsu(imagen_suavizada)
        imagen_binaria = imagen_suavizada > thresh

        # Detección de Contornos - Marching Squares
        contornos = measure.find_contours(imagen_binaria, 0.5)
        if not contornos:
            return None, None, None

        x_ref, y_ref = referencia_pixel
        contornos_arriba = [c for c in contornos if np.mean(c[:, 0]) < (y_ref + 5)]

        if len(contornos_arriba) == 0:
            contorno = min(contornos, key=lambda x: np.mean(x[:, 0]))
        else:
            contorno = max(contornos_arriba, key=lambda x: len(x))

        mask_arriba = contorno[:, 0] < y_ref
        contorno_filtrado = contorno[mask_arriba]

        if contorno_filtrado.shape[0] < 10:
            return None, None, None

        contorno_px = np.zeros_like(contorno_filtrado)
        contorno_px[:, 1] = contorno_filtrado[:, 1] - x_ref 
        contorno_px[:, 0] = y_ref - contorno_filtrado[:, 0]

        contorno_µm = contorno_px * escala

        centro_x = float(np.mean(contorno_µm[:, 1]))
        centro_y = float(np.mean(contorno_µm[:, 0]))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(imagen, cmap='gray')
        ax.plot(contorno_filtrado[:, 1], contorno_filtrado[:, 0], 'r-', linewidth=2, label='Contorno')
        ax.axhline(y=y_ref, color='y', linestyle='--', linewidth=1.0, label='Referencia y=0')
        centro_px_x = centro_x / escala + x_ref
        centro_px_y = y_ref - (centro_y / escala)
        ax.scatter(centro_px_x, centro_px_y, c='blue', marker='x', s=100, label='Centroide')
        ax.axis('off')
        plt.legend(loc='upper right')
        plt.title(f'Procesamiento: {os.path.basename(ruta_imagen)}')
        plt.tight_layout()

        temp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(temp_img.name, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()

        return contorno_µm, (centro_x, centro_y), temp_img.name

    except Exception as e:
        print(f"Error procesando {ruta_imagen}: {str(e)}")
        return None, None, None


def procesar_todas_imagenes(carpeta_imagenes, num_imagenes=126, escala=4.13):
    if not os.path.isdir(carpeta_imagenes):
        raise ValueError(f"La carpeta {carpeta_imagenes} no existe")

    if num_imagenes <= 0:
        raise ValueError("El número de imágenes debe ser positivo")

    datos = []
    fps = 20538  # FPS
    tiempos = np.arange(num_imagenes) / fps

    for i in tqdm(range(1, num_imagenes + 1), desc="Procesando imágenes"):
        num_str = f"{i:04d}"
        nombre_imagen = f"TP4_Gota_{num_str}.jpg"
        ruta_imagen = os.path.join(carpeta_imagenes, nombre_imagen)

        contorno, centro, img_path = procesar_imagen(ruta_imagen, escala)

        dato_imagen = {
            'Imagen': nombre_imagen,
            'Tiempo (s)': tiempos[i - 1],
            'Centroide_x (µm)': None,
            'Centroide_y (µm)': None,
            'N_puntos_contorno': 0,
            'Ruta_imagen_procesada': '',
            'Contorno_x': '[]',
            'Contorno_y': '[]'
        }

        if contorno is not None and centro is not None:
            dato_imagen.update({
                'Centroide_x (µm)': centro[0],
                'Centroide_y (µm)': centro[1],
                'N_puntos_contorno': len(contorno),
                'Ruta_imagen_procesada': img_path,
                'Contorno_x': json.dumps(contorno[:, 1].tolist()),
                'Contorno_y': json.dumps(contorno[:, 0].tolist())
            })

        datos.append(dato_imagen)

    return pd.DataFrame(datos)

def graficar_centroides_vs_tiempo(df, nombre_archivo='centroides_vs_tiempo.png'):
    try:
        plt.figure(figsize=(14, 6))

        plt.style.use('seaborn-v0_8')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

        plt.subplot(1, 2, 1)
        plt.plot(df['Tiempo (s)'], df['Centroide_x (µm)'], 'b-', linewidth=2, label='Posición X')
        plt.xlabel('Tiempo (s)', fontsize=12)
        plt.ylabel('Posición X (µm)', fontsize=12)
        plt.title('Evolución de la posición X del centroide', fontsize=14, pad=20)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=10)

        plt.subplot(1, 2, 2)
        plt.plot(df['Tiempo (s)'], df['Centroide_y (µm)'], 'r-', linewidth=2, label='Posición Y')
        plt.xlabel('Tiempo (s)', fontsize=12)
        plt.ylabel('Posición Y (µm)', fontsize=12)
        plt.title('Evolución de la posición Y del centroide', fontsize=14, pad=20)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nGráfico de centroides guardado como: {nombre_archivo}")

    except Exception as e:
        print(f"\nError al graficar centroides: {str(e)}")


def generar_informe1(carpeta_imagenes, num_imagenes=126):
    print("\n=== EJERCICIO 1: Procesamiento de imágenes ===")

    try:
        if not os.path.exists(carpeta_imagenes):
            raise FileNotFoundError(f"La carpeta '{carpeta_imagenes}' no existe")

        if num_imagenes <= 0:
            raise ValueError("El número de imágenes debe ser positivo")

        print("\nIniciando procesamiento de imágenes...")
        df = procesar_todas_imagenes(carpeta_imagenes, num_imagenes)

        if df.empty:
            raise ValueError("No se pudieron procesar imágenes")

        # Análisis de resultados
        print("\nResumen estadístico:")
        print(f"- Imágenes procesadas: {len(df)}")
        print(f"- Centroide X promedio: {df['Centroide_x (µm)'].mean():.2f} µm")
        print(f"- Centroide Y promedio: {df['Centroide_y (µm)'].mean():.2f} µm")

        if exportar_ejercicio1_excel(df):
            graficar_centroides_vs_tiempo(df)

            plt.figure(figsize=(8, 6))
            plt.plot(df['Centroide_x (µm)'], df['Centroide_y (µm)'], 'b-', linewidth=1.5, label='Trayectoria')
            plt.xlabel('Posición X (µm)', fontsize=12)
            plt.ylabel('Posición Y (µm)', fontsize=12)
            plt.title('Trayectoria del centroide de la gota', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig('trayectoria_centroide.png', dpi=300)
            plt.close()
            return True
        else:
            print("\nProceso completado con errores.")
            return False

    except Exception as e:
        print(f"\nERROR durante el procesamiento: {str(e)}")
        return False