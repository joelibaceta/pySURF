"""
Integration tests for the complete SURF pipeline.
"""
import numpy as np
import pytest
from pathlib import Path
from pysurf import PySurf

# Obtener la ruta de la imagen de prueba
IMAGE_PATH = Path(__file__).parent.parent / "images" / "image.png"


@pytest.fixture
def test_image():
    """
    Cargar la imagen de prueba.
    """
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL/Pillow no está instalado")
    
    if not IMAGE_PATH.exists():
        pytest.skip(f"Imagen de prueba no encontrada: {IMAGE_PATH}")
    
    img = Image.open(IMAGE_PATH).convert('L')
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array


@pytest.fixture
def synthetic_image():
    """
    Crear una imagen sintética con patrones detectables.
    """
    img = np.zeros((200, 200), dtype=np.float32)
    
    # Agregar algunos cuadrados y círculos
    img[50:70, 50:70] = 1.0  # Cuadrado
    img[120:150, 120:150] = 0.8  # Otro cuadrado
    
    # Agregar gradientes
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x, y)
    img += 0.2 * (X + Y) / 2
    
    # Normalizar
    img = np.clip(img, 0, 1)
    
    return img


def test_surf_initialization():
    """
    Test que la clase Surf se inicializa correctamente con los parámetros por defecto.
    """
    surf = PySurf()
    assert surf.hessian_thresh == 0.004
    assert surf.n_scales == 4
    assert surf.base_filter == 9
    assert surf.step == 6
    assert surf.upright is False
    assert len(surf.sizes) == 4
    assert surf.sizes == [9, 15, 21, 27]


def test_surf_custom_parameters():
    """
    Test que la clase Surf acepta parámetros personalizados.
    """
    surf = PySurf(
        hessian_thresh=0.001,
        n_scales=5,
        base_filter=12,
        step=8,
        upright=True
    )
    assert surf.hessian_thresh == 0.001
    assert surf.n_scales == 5
    assert surf.base_filter == 12
    assert surf.step == 8
    assert surf.upright is True
    assert surf.sizes == [12, 20, 28, 36, 44]


def test_surf_detect_synthetic(synthetic_image):
    """
    Test que SURF detecta keypoints en una imagen sintética.
    """
    surf = PySurf(hessian_thresh=0.001, n_scales=3)
    keypoints = surf.detect(synthetic_image)
    
    # Verificar que se detectaron algunos keypoints
    assert len(keypoints) > 0
    
    # Verificar estructura de keypoints: (x, y, scale_index, response, angle)
    for kp in keypoints:
        assert len(kp) == 5
        x, y, s, val, angle = kp
        assert 0 <= x < synthetic_image.shape[1]
        assert 0 <= y < synthetic_image.shape[0]
        assert 0 <= s < 3  # n_scales=3
        assert val > 0  # response positivo
        assert -np.pi <= angle <= np.pi


def test_surf_describe_synthetic(synthetic_image):
    """
    Test que SURF genera descriptores válidos.
    """
    surf = PySurf(hessian_thresh=0.001, n_scales=3)
    keypoints = surf.detect(synthetic_image)
    
    if len(keypoints) == 0:
        pytest.skip("No se detectaron keypoints en la imagen sintética")
    
    descriptors = surf.describe(synthetic_image, keypoints)
    
    # Verificar dimensiones
    assert descriptors.shape[0] == len(keypoints)
    assert descriptors.shape[1] == 64  # SURF usa descriptores de 64 dimensiones
    
    # Verificar que están normalizados (norma ~1)
    for desc in descriptors:
        norm = np.linalg.norm(desc)
        assert 0.99 <= norm <= 1.01  # Tolerancia para errores de punto flotante


def test_surf_detect_and_describe_synthetic(synthetic_image):
    """
    Test del pipeline completo con detect_and_describe.
    """
    surf = PySurf(hessian_thresh=0.001, n_scales=3)
    keypoints, descriptors = surf.detect_and_describe(synthetic_image)
    
    # Verificar que se detectaron keypoints
    assert len(keypoints) > 0
    
    # Verificar consistencia entre keypoints y descriptors
    assert len(keypoints) == descriptors.shape[0]
    assert descriptors.shape[1] == 64


def test_surf_upright_mode(synthetic_image):
    """
    Test que el modo U-SURF (upright) funciona correctamente.
    """
    surf = PySurf(hessian_thresh=0.001, n_scales=3, upright=True)
    keypoints, descriptors = surf.detect_and_describe(synthetic_image)
    
    assert len(keypoints) > 0
    assert descriptors.shape[0] == len(keypoints)
    assert descriptors.shape[1] == 64
    
    # En modo upright, el ángulo debería ser 0 o estar definido
    for kp in keypoints:
        angle = kp[4]
        assert isinstance(angle, (int, float))


def test_surf_with_real_image(test_image):
    """
    Test del pipeline completo con la imagen real (image.png).
    """
    surf = PySurf(
        hessian_thresh=0.004,
        n_scales=4,
        base_filter=9,
        step=6,
        upright=False
    )
    
    keypoints, descriptors = surf.detect_and_describe(test_image)
    
    # Verificar que se detectaron keypoints
    assert len(keypoints) > 0, "No se detectaron keypoints en la imagen real"
    
    # Verificar dimensiones
    assert descriptors.shape[0] == len(keypoints)
    assert descriptors.shape[1] == 64
    
    # Verificar que los keypoints están dentro de los límites de la imagen
    for kp in keypoints:
        x, y, s, val, angle = kp
        assert 0 <= x < test_image.shape[1]
        assert 0 <= y < test_image.shape[0]
        assert val > surf.hessian_thresh


def test_surf_empty_image():
    """
    Test que SURF maneja correctamente imágenes sin características.
    """
    # Imagen completamente negra
    empty_img = np.zeros((100, 100), dtype=np.float32)
    surf = PySurf(hessian_thresh=0.001)
    
    keypoints, descriptors = surf.detect_and_describe(empty_img)
    
    # No debería detectar keypoints en una imagen vacía
    assert len(keypoints) == 0
    assert descriptors.shape == (0, 64)


def test_surf_uniform_image():
    """
    Test que SURF maneja correctamente imágenes uniformes.
    """
    # Imagen completamente blanca
    uniform_img = np.ones((100, 100), dtype=np.float32)
    surf = PySurf(hessian_thresh=0.001)
    
    keypoints, descriptors = surf.detect_and_describe(uniform_img)
    
    # Debería detectar muy pocos o ningún keypoint en una imagen uniforme
    # (puede haber algunos artefactos numéricos en los bordes)
    assert len(keypoints) <= 5
    assert descriptors.shape[0] == len(keypoints)


def test_surf_uint8_input():
    """
    Test que SURF acepta imágenes en formato uint8 [0, 255].
    """
    # Crear imagen en uint8
    img_uint8 = (np.random.rand(100, 100) * 255).astype(np.uint8)
    
    surf = PySurf(hessian_thresh=0.001, n_scales=3)
    keypoints, descriptors = surf.detect_and_describe(img_uint8)
    
    # Debería funcionar sin errores
    assert isinstance(keypoints, list)
    assert isinstance(descriptors, np.ndarray)
    assert descriptors.shape[1] == 64


def test_surf_different_thresholds(synthetic_image):
    """
    Test que diferentes thresholds producen diferentes cantidades de keypoints.
    """
    surf_low = PySurf(hessian_thresh=0.0001, n_scales=3)
    surf_high = PySurf(hessian_thresh=0.01, n_scales=3)
    
    kps_low, _ = surf_low.detect_and_describe(synthetic_image)
    kps_high, _ = surf_high.detect_and_describe(synthetic_image)
    
    # Un threshold más bajo debería detectar más keypoints
    assert len(kps_low) >= len(kps_high)


def test_surf_descriptor_uniqueness(synthetic_image):
    """
    Test que los descriptores son razonablemente únicos.
    """
    surf = PySurf(hessian_thresh=0.001, n_scales=3)
    keypoints, descriptors = surf.detect_and_describe(synthetic_image)
    
    if len(keypoints) < 2:
        pytest.skip("Se necesitan al menos 2 keypoints para este test")
    
    # Verificar que no todos los descriptores son idénticos
    for i in range(len(descriptors) - 1):
        diff = np.linalg.norm(descriptors[i] - descriptors[i + 1])
        # Al menos algunos descriptores deberían ser diferentes
        if i == 0:
            assert diff > 1e-6, "Los primeros dos descriptores son idénticos"


def test_surf_reproducibility(synthetic_image):
    """
    Test que SURF produce resultados reproducibles con la misma imagen.
    """
    surf = PySurf(hessian_thresh=0.001, n_scales=3)
    
    kps1, desc1 = surf.detect_and_describe(synthetic_image)
    kps2, desc2 = surf.detect_and_describe(synthetic_image)
    
    # Los resultados deberían ser idénticos
    assert len(kps1) == len(kps2)
    assert desc1.shape == desc2.shape
    assert np.allclose(desc1, desc2)
    
    # Verificar keypoints
    for kp1, kp2 in zip(kps1, kps2):
        for v1, v2 in zip(kp1, kp2):
            assert abs(v1 - v2) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
