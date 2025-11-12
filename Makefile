.PHONY: help install test test-verbose clean lint format

# Variables
PYTHON = env/bin/python
PIP = env/bin/pip
PYTEST = env/bin/pytest

help:
	@echo "Comandos disponibles:"
	@echo "  make install        - Instala las dependencias del proyecto"
	@echo "  make test           - Ejecuta todos los tests"
	@echo "  make test-verbose   - Ejecuta los tests con salida detallada"
	@echo "  make clean          - Limpia archivos generados"
	@echo "  make lint           - Ejecuta el linter (requiere flake8)"
	@echo "  make format         - Formatea el código (requiere black)"

install:
	@echo "Instalando dependencias..."
	$(PIP) install -r requirements.txt

test:
	@echo "Ejecutando tests..."
	PYTHONPATH=. $(PYTEST) tests/ -q

test-verbose:
	@echo "Ejecutando tests con salida detallada..."
	PYTHONPATH=. $(PYTEST) tests/ -v

test-coverage:
	@echo "Ejecutando tests con cobertura..."
	PYTHONPATH=. $(PYTEST) tests/ --cov=pysurf --cov-report=html --cov-report=term

clean:
	@echo "Limpiando archivos generados..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage

lint:
	@echo "Ejecutando linter..."
	$(PYTHON) -m flake8 pysurf/ tests/

format:
	@echo "Formateando código..."
	$(PYTHON) -m black pysurf/ tests/
