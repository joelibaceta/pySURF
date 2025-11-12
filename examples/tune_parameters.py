"""
Script para analizar y ajustar los parámetros de SURF.

Prueba diferentes combinaciones de parámetros para ver cómo afectan
la detección de keypoints.
"""

import numpy as np
from pathlib import Path
from pysurf import Surf
import time

try:
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
except ImportError:
    print("Requiere Pillow y matplotlib")
    exit(1)


def load_image(path):
    """Cargar imagen en escala de grises."""
    img = Image.open(path).convert('L')
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array


def test_parameters(img, configs):
    """
    Probar diferentes configuraciones de SURF.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen a procesar
    configs : list of dict
        Lista de configuraciones a probar
    """
    results = []
    
    print("\n" + "="*80)
    print("Probando diferentes configuraciones de SURF")
    print("="*80)
    print(f"Imagen: {img.shape[0]}x{img.shape[1]}")
    print()
    
    for i, config in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] Configuración: {config}")
        
        surf = Surf(**config)
        
        start = time.time()
        keypoints, descriptors = surf.detect_and_describe(img)
        elapsed = time.time() - start
        
        result = {
            'config': config,
            'n_keypoints': len(keypoints),
            'time': elapsed,
            'keypoints': keypoints,
            'descriptors': descriptors
        }
        results.append(result)
        
        print(f"   → Keypoints detectados: {len(keypoints)}")
        print(f"   → Tiempo: {elapsed:.3f}s")
        print()
    
    return results


def visualize_comparison(img, results):
    """
    Visualizar comparación de diferentes configuraciones.
    """
    n_configs = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, result in enumerate(results[:6]):  # Máximo 6 configuraciones
        ax = axes[idx]
        ax.imshow(img, cmap='gray')
        
        keypoints = result['keypoints']
        config = result['config']
        
        # Dibujar keypoints
        for (x, y, s, val, angle) in keypoints:
            # Color según escala
            colors = ['red', 'yellow', 'lime', 'cyan', 'magenta', 'white']
            color = colors[s % len(colors)]
            r = 5
            circle = Circle((x, y), r, color=color, fill=False, lw=1.5, alpha=0.8)
            ax.add_patch(circle)
            
            # Orientación
            ax.plot(
                [x, x + r * np.cos(angle)],
                [y, y + r * np.sin(angle)],
                color=color, lw=1.2, alpha=0.8
            )
        
        # Título con parámetros clave
        title = f"n={result['n_keypoints']}\n"
        title += f"thresh={config['hessian_thresh']}, scales={config['n_scales']}"
        if config.get('upright', False):
            title += " (U-SURF)"
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Ocultar axes sobrantes
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent.parent / "parameter_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparación guardada en: {output_path}")
    
    return fig


def main():
    # Cargar imagen
    image_path = Path(__file__).parent.parent / "images" / "image.png"
    
    if not image_path.exists():
        print(f"ERROR: No se encontró {image_path}")
        return
    
    img = load_image(image_path)
    
    # Definir configuraciones a probar
    configs = [
        # Configuración original
        {
            'hessian_thresh': 0.004,
            'n_scales': 4,
            'base_filter': 9,
            'step': 6,
            'upright': False
        },
        # Threshold más bajo (más keypoints)
        {
            'hessian_thresh': 0.001,
            'n_scales': 4,
            'base_filter': 9,
            'step': 6,
            'upright': False
        },
        # Threshold aún más bajo
        {
            'hessian_thresh': 0.0005,
            'n_scales': 4,
            'base_filter': 9,
            'step': 6,
            'upright': False
        },
        # Más escalas
        {
            'hessian_thresh': 0.004,
            'n_scales': 6,
            'base_filter': 9,
            'step': 6,
            'upright': False
        },
        # Threshold bajo + más escalas
        {
            'hessian_thresh': 0.001,
            'n_scales': 6,
            'base_filter': 9,
            'step': 6,
            'upright': False
        },
        # U-SURF (más rápido)
        {
            'hessian_thresh': 0.001,
            'n_scales': 4,
            'base_filter': 9,
            'step': 6,
            'upright': True
        },
    ]
    
    # Probar configuraciones
    results = test_parameters(img, configs)
    
    # Mostrar resumen
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS")
    print("="*80)
    print(f"{'Config':<8} {'Threshold':<12} {'Scales':<8} {'Upright':<10} {'Keypoints':<12} {'Tiempo (s)'}")
    print("-"*80)
    
    for i, result in enumerate(results):
        config = result['config']
        print(f"{i+1:<8} {config['hessian_thresh']:<12.4f} {config['n_scales']:<8} "
              f"{str(config.get('upright', False)):<10} {result['n_keypoints']:<12} "
              f"{result['time']:.3f}")
    
    print("-"*80)
    
    # Encontrar mejor configuración (más keypoints)
    best = max(results, key=lambda x: x['n_keypoints'])
    print(f"\n✓ Configuración con MÁS keypoints: Config {results.index(best)+1} "
          f"({best['n_keypoints']} keypoints)")
    
    # Encontrar más rápida
    fastest = min(results, key=lambda x: x['time'])
    print(f"✓ Configuración MÁS RÁPIDA: Config {results.index(fastest)+1} "
          f"({fastest['time']:.3f}s)")
    
    # Mejor balance (normalizar y combinar)
    for r in results:
        max_kps = max(res['n_keypoints'] for res in results)
        max_time = max(res['time'] for res in results)
        r['score'] = (r['n_keypoints'] / max_kps) * 0.7 + (1 - r['time'] / max_time) * 0.3
    
    balanced = max(results, key=lambda x: x['score'])
    print(f"✓ Configuración BALANCEADA: Config {results.index(balanced)+1} "
          f"({balanced['n_keypoints']} kps, {balanced['time']:.3f}s)")
    
    print("\n" + "="*80)
    print("RECOMENDACIONES")
    print("="*80)
    
    if best['n_keypoints'] < 50:
        print("\n⚠️  Se detectaron pocos keypoints (< 50)")
        print("   Sugerencias:")
        print("   • Reducir hessian_thresh a 0.001 o menos")
        print("   • Aumentar n_scales a 5 o 6")
        print("   • Verificar que la imagen tenga suficientes características")
    elif best['n_keypoints'] > 500:
        print("\n⚠️  Se detectaron muchos keypoints (> 500)")
        print("   Sugerencias:")
        print("   • Aumentar hessian_thresh a 0.005 o más")
        print("   • Reducir n_scales si el tiempo es crítico")
    else:
        print("\n✓ Cantidad de keypoints en rango óptimo (50-500)")
    
    # Visualizar
    print("\nGenerando visualización comparativa...")
    visualize_comparison(img, results)
    
    print("\n✓ Análisis completado!")
    
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    main()
