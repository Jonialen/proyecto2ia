# Proyecto 2 - CC3045 Inteligencia Artificial

Sistema de IA para ciberseguridad de una red de servidores distribuidos.

## Tasks

| Task | Tema | Algoritmos |
|------|------|------------|
| 1 | Configuracion segura de red (CSP) | Backtracking, Forward Checking, MRV |
| 2 | Defensa adversarial | Minimax, Alpha-Beta |
| 3 | Incertidumbre y latencia | Expectiminimax, MDPs, Bellman |

## Setup

Con uv:
```bash
uv sync
```

Con venv:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
# uv
uv run jupyter notebook proyecto2.ipynb

# venv
jupyter notebook proyecto2.ipynb
```

## Tests

```bash
# uv
uv run pytest

# venv
pytest
```

## Estructura

```
task1_csp.py              # CSP + Factor Graphs
task2_minimax.py           # Minimax + Alpha-Beta
task3_expectiminimax.py    # Expectiminimax + MDPs
tests/                     # 60 tests
proyecto2.ipynb            # Notebook unificado
make_notebook.py           # Genera el notebook desde los .py
```
