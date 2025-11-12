"""
Ejemplo de uso completo de pySURF.

Este script demuestra cómo usar la implementación de SURF para:
1. Detectar keypoints en una imagen
2. Computar descriptores
3. Matching entre dos imágenes
4. Visualización de resultados
"""

import numpy as np
from pathlib import Path
from pysurf import Surf
from pysurf.match import match_descriptors
from pysurf.viz import draw_keypoints, draw_matches

try:
    from PIL import Image
    import matplotlib.pyplot as plt
except ImportError:
    print("Este ejemplo requiere Pillow y matplotlib")
    print("Instala con: pip install pillow matplotlib")
    exit(1)


def load_image(path):
    """Cargar imagen en escala de grises y normalizar a [0,1]."""
    img = Image.open(path).convert('L')
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array


def main():
    print("=" * 60)
    print("Ejemplo de uso de pySURF")
    print("=" * 60)
    
    # Ruta de la imagen de ejemplo
    image_path = Path(__file__).parent.parent / "images" / "image.png"
    
    if not image_path.exists():
        print(f"ERROR: No se encontró la imagen en {image_path}")
        return
    
    # Cargar imagen
    print(f"\n1. Cargando imagen: {image_path.name}")
    img = load_image(image_path)
    print(f"   Dimensiones: {img.shape}")
    
    # Crear detector SURF
    print("\n2. Inicializando detector SURF")
    surf = Surf(
        hessian_thresh=0.004,  # Threshold para detección
        n_scales=4,            # Número de escalas
        base_filter=9,         # Tamaño de filtro base
        step=6,                # Incremento entre escalas
        upright=False          # False = rotación invariante
    )
    print(f"   Tamaños de filtro: {surf.sizes}")
    
    # Detectar y describir
    print("\n3. Detectando keypoints y computando descriptores...")
    keypoints, descriptors = surf.detect_and_describe(img)
    print(f"   Keypoints detectados: {len(keypoints)}")
    print(f"   Descriptores shape: {descriptors.shape}")
    
    # Mostrar información de algunos keypoints
    if len(keypoints) > 0:
        print("\n4. Información de los primeros 5 keypoints:")
        print("   (x, y, escala, respuesta, ángulo)")
        for i, kp in enumerate(keypoints[:5]):
            x, y, s, val, angle = kp
            print(f"   [{i}] x={x:.1f}, y={y:.1f}, scale={s}, "
                  f"response={val:.6f}, angle={angle:.3f} rad")
    
    # Visualizar keypoints
    print("\n5. Generando visualización...")
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_keypoints(img, keypoints, surf.sizes, ax=ax)
    ax.set_title(f'SURF Keypoints Detectados: {len(keypoints)}')
    
    output_path = Path(__file__).parent.parent / "keypoints_detected.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Visualización guardada en: {output_path}")
    
    # Ejemplo de matching consigo misma (debería tener muchos matches)
    print("\n6. Ejemplo de matching de descriptores...")
    print("   (Matching la imagen consigo misma para demostración)")
    matches = match_descriptors(descriptors, descriptors, ratio=0.8)
    print(f"   Matches encontrados: {len(matches)}")
    
    if len(matches) > 0:
        print("\n   Primeros 5 matches:")
        print("   (idx1, idx2, distancia)")
        for i, (idx1, idx2, dist) in enumerate(matches[:5]):
            print(f"   [{i}] {idx1} <-> {idx2}, dist={dist:.4f}")
    
    # Test con modo U-SURF (sin rotación)
    print("\n7. Comparando con U-SURF (modo upright)...")
    surf_upright = Surf(
        hessian_thresh=0.001,
        n_scales=4,
        upright=True
    )
    kps_upright, desc_upright = surf_upright.detect_and_describe(img)
    print(f"   U-SURF keypoints detectados: {len(kps_upright)}")
    print(f"   Diferencia: {len(keypoints) - len(kps_upright)} keypoints")
    
    # Estadísticas de los descriptores
    print("\n8. Estadísticas de descriptores:")
    if len(descriptors) > 0:
        print(f"   Media: {descriptors.mean():.6f}")
        print(f"   Desv. estándar: {descriptors.std():.6f}")
        print(f"   Min: {descriptors.min():.6f}")
        print(f"   Max: {descriptors.max():.6f}")
        
        # Verificar normalización
        norms = np.linalg.norm(descriptors, axis=1)
        print(f"   Norma promedio: {norms.mean():.6f} (debería ser ~1.0)")
    
    print("\n" + "=" * 60)
    print("Ejemplo completado exitosamente!")
    print("=" * 60)
    
    # Mostrar la imagen (opcional)
    try:
        plt.show()
    except:
        pass  # En caso de que no haya display disponible


if __name__ == "__main__":
    main()
