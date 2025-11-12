# pySURF

Implementación de SURF (Speeded-Up Robust Features) en Python.

## Instalación

```bash
# Crear y activar entorno virtual
python3 -m venv env
source env/bin/activate 

# Instalar dependencias
make install
# o manualmente:
pip install -r requirements.txt
```

## Tests

El proyecto incluye un Makefile para facilitar la ejecución de tests:

```bash
# Ver todos los comandos disponibles
make help

# Ejecutar tests
make test

# Ejecutar tests con salida detallada
make test-verbose

# Ejecutar tests con cobertura (requiere pytest-cov)
make test-coverage

# Limpiar archivos generados
make clean
```
