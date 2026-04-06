"""
Script para generar el Jupyter notebook proyecto2.ipynb a partir de los tres
archivos fuente task1_csp.py, task2_minimax.py y task3_expectiminimax.py.
"""

import json
import re

BASE = "/home/jonialen/Documents/UVG/s7/ia/proyecto2"

# ─────────────────────────────────────────────────────────────────────────────
# Utilidades para construir celdas con el formato correcto de Jupyter
# ─────────────────────────────────────────────────────────────────────────────

def make_markdown_cell(text: str) -> dict:
    """Crea una celda markdown. 'text' es el contenido completo sin el prefijo # ."""
    lines = text.split("\n")
    # Añadir \n al final de cada línea excepto la última
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def make_code_cell(code: str) -> dict:
    """Crea una celda de código."""
    lines = code.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parser de archivos .py con marcadores # %% y # %% [markdown]
# ─────────────────────────────────────────────────────────────────────────────

def parse_py_cells(filepath: str) -> list[dict]:
    """
    Lee un archivo .py con marcadores de celda y devuelve una lista de
    dicts {'type': 'code'|'markdown', 'content': str}.

    Reglas:
    - Líneas que empiezan con '# %% [markdown]' abren una celda markdown.
    - Líneas que empiezan con '# %%' (sin [markdown]) abren una celda código.
    - Las líneas de marcador en sí mismas NO se incluyen en el contenido.
    - En celdas markdown, las líneas comienzan con '# ' → se elimina ese prefijo.
      Las líneas que son solo '#' se convierten en línea vacía.
    """
    with open(filepath, encoding="utf-8") as f:
        raw_lines = f.readlines()

    cells = []
    current_type = None
    current_lines = []

    def flush():
        if current_type is None:
            return
        content = "".join(current_lines).rstrip("\n")
        cells.append({"type": current_type, "content": content})

    for raw in raw_lines:
        line = raw.rstrip("\n")

        # Detectar marcadores de celda
        if re.match(r"#\s*%%\s*\[markdown\]", line):
            flush()
            current_type = "markdown"
            current_lines = []
            continue

        if re.match(r"#\s*%%(?!\s*\[)", line):
            flush()
            current_type = "code"
            current_lines = []
            continue

        # Acumular línea en la celda actual
        if current_type == "markdown":
            # Quitar el prefijo '# ' o solo '#'
            if line.startswith("# "):
                current_lines.append(line[2:] + "\n")
            elif line == "#":
                current_lines.append("\n")
            else:
                current_lines.append(line + "\n")
        elif current_type == "code":
            current_lines.append(line + "\n")

    flush()
    return cells


# ─────────────────────────────────────────────────────────────────────────────
# Modificaciones específicas para Task 2
# ─────────────────────────────────────────────────────────────────────────────

def patch_task2_cells(cells: list[dict]) -> list[dict]:
    """
    Aplicar modificaciones requeridas para Task 2:
    1. Eliminar la línea  matplotlib.use("Agg")
    2. En visualize_game_snapshots y plot_comparison_chart: reemplazar
       plt.savefig(...) + plt.close() con plt.show()
    3. En el bloque main(): reemplazar las llamadas a guardar snapshots y
       chart con versiones que usen plt.show() en lugar de guardar archivos.
    """
    patched = []
    for cell in cells:
        if cell["type"] != "code":
            patched.append(cell)
            continue

        content = cell["content"]

        # 1. Eliminar matplotlib.use("Agg")
        content = re.sub(r'^\s*matplotlib\.use\("Agg"\).*\n?', '', content, flags=re.MULTILINE)
        # También eliminar el comentario de la misma línea
        content = re.sub(r'^\s*matplotlib\.use\("Agg"\).*', '', content, flags=re.MULTILINE)

        # 2. En funciones de visualización: reemplazar save+close con show
        #    Patrón: plt.savefig(...) seguido (posiblemente con print) de plt.close()
        #    → reemplazar por plt.show()
        content = re.sub(
            r'    plt\.savefig\([^)]*\)[^\n]*\n(?:    print\([^)]*\)\n)?    plt\.close\(\)',
            '    plt.show()',
            content
        )
        # Caso con indent de 4 y path variable
        content = re.sub(
            r'    plt\.savefig\(path[^)]*\)[^\n]*\n(?:    print\([^)]*\)\n)?    plt\.close\(\)',
            '    plt.show()',
            content
        )

        # 3. En main(): las llamadas que guardan imágenes
        #    visualize_game_snapshots(engine, save_prefix="game_state")
        #    → visualize_game_snapshots(engine)  (no cambia nada porque ya usa show)
        #    plot_comparison_chart(engine.metrics, save_path="comparison_chart.png")
        #    → plot_comparison_chart(engine.metrics)
        content = re.sub(
            r'plot_comparison_chart\(engine\.metrics,\s*save_path="[^"]*"\)',
            'plot_comparison_chart(engine.metrics)',
            content
        )

        patched.append({"type": cell["type"], "content": content})

    return patched


# ─────────────────────────────────────────────────────────────────────────────
# Modificaciones para Task 1 (guardar imágenes → plt.show())
# ─────────────────────────────────────────────────────────────────────────────

def patch_task1_cells(cells: list[dict]) -> list[dict]:
    """
    Task 1 guarda imágenes con rutas hardcodeadas.
    Reemplazar plt.savefig + print("guardado") → plt.show()
    """
    patched = []
    for cell in cells:
        if cell["type"] != "code":
            patched.append(cell)
            continue

        content = cell["content"]

        # Reemplazar bloques: plt.savefig(...)  \n  print("...guardado...")
        content = re.sub(
            r'    plt\.savefig\("[^"]*"[^)]*\)\n    print\("[^"]*guardad[^"]*"\)',
            '    plt.show()',
            content
        )
        # Caso sin indent (nivel 0)
        content = re.sub(
            r'^plt\.savefig\("[^"]*"[^)]*\)\nprint\("[^"]*guardad[^"]*"\)',
            'plt.show()',
            content,
            flags=re.MULTILINE
        )

        patched.append({"type": cell["type"], "content": content})

    return patched


# ─────────────────────────────────────────────────────────────────────────────
# Conversión de bloques if __name__ == "__main__"
# ─────────────────────────────────────────────────────────────────────────────

def replace_main_guard(cells: list[dict]) -> list[dict]:
    """
    Reemplaza  if __name__ == "__main__": main()
    (y variantes de una línea)  por  main()
    """
    patched = []
    for cell in cells:
        if cell["type"] != "code":
            patched.append(cell)
            continue

        content = cell["content"]
        # Patrón de una sola línea con main en la misma
        content = re.sub(
            r'if __name__\s*==\s*[\'"]__main__[\'"]\s*:\s*main\(\)',
            'main()',
            content
        )
        # Patrón multilinea
        content = re.sub(
            r'if __name__\s*==\s*[\'"]__main__[\'"]\s*:\s*\n\s*main\(\)',
            'main()',
            content
        )
        patched.append({"type": cell["type"], "content": content})

    return patched


# ─────────────────────────────────────────────────────────────────────────────
# Construcción del notebook
# ─────────────────────────────────────────────────────────────────────────────

def build_notebook() -> dict:
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "version": "3.12.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    cells = notebook["cells"]

    # ── Celda de título / introducción ──────────────────────────────────────
    cells.append(make_markdown_cell(
        "# Proyecto 2 – CC3045 Inteligencia Artificial\n"
        "## Sistema de IA para Ciberseguridad de Red de Servidores Distribuidos\n"
        "**Universidad del Valle de Guatemala**\n"
        "\n"
        "Este notebook integra los tres módulos del Proyecto 2:\n"
        "\n"
        "| Tarea | Tema | Algoritmos |\n"
        "|-------|------|------------|\n"
        "| Task 1 | Configuración Segura de Red (CSP) | Backtracking Básico · Backtracking con FC+MRV |\n"
        "| Task 2 | Defensa Adversarial – Juegos de Suma Cero | Minimax Puro · Minimax con Poda Alfa-Beta |\n"
        "| Task 3 | Incertidumbre y Latencia | Expectiminimax · MDPs · Ecuación de Bellman |\n"
        "\n"
        "**Entorno**: Python 3.12  \n"
        "**Dependencias externas**: `networkx`, `matplotlib`"
    ))

    # ── Separador Task 1 ────────────────────────────────────────────────────
    cells.append(make_markdown_cell(
        "---\n"
        "# Tarea 1: Configuración Segura de la Red usando CSP y Factor Graphs"
    ))

    # Parsear Task 1
    t1_cells = parse_py_cells(f"{BASE}/task1_csp.py")
    t1_cells = patch_task1_cells(t1_cells)
    t1_cells = replace_main_guard(t1_cells)

    # Eliminar la primera celda markdown si repite el título de Task 1
    # (el separador ya lo incluye, pero la mantenemos para el contenido formal)
    for c in t1_cells:
        if c["type"] == "markdown":
            cells.append(make_markdown_cell(c["content"]))
        else:
            # Omitir celdas de código completamente vacías
            if c["content"].strip():
                cells.append(make_code_cell(c["content"]))

    # ── Separador Task 2 ────────────────────────────────────────────────────
    cells.append(make_markdown_cell(
        "---\n"
        "# Tarea 2: Defensa Adversarial – Juegos de Suma Cero (Minimax)"
    ))

    # Parsear Task 2
    t2_cells = parse_py_cells(f"{BASE}/task2_minimax.py")
    t2_cells = patch_task2_cells(t2_cells)
    t2_cells = replace_main_guard(t2_cells)

    for c in t2_cells:
        if c["type"] == "markdown":
            cells.append(make_markdown_cell(c["content"]))
        else:
            if c["content"].strip():
                cells.append(make_code_cell(c["content"]))

    # ── Separador Task 3 ────────────────────────────────────────────────────
    cells.append(make_markdown_cell(
        "---\n"
        "# Tarea 3: Incertidumbre y Latencia (Expectiminimax y MDPs)"
    ))

    # Parsear Task 3
    t3_cells = parse_py_cells(f"{BASE}/task3_expectiminimax.py")
    t3_cells = replace_main_guard(t3_cells)

    for c in t3_cells:
        if c["type"] == "markdown":
            cells.append(make_markdown_cell(c["content"]))
        else:
            if c["content"].strip():
                cells.append(make_code_cell(c["content"]))

    # ── Celda de análisis MDP/Bellman ───────────────────────────────────────
    cells.append(make_markdown_cell(
        "---\n"
        "## Análisis Teórico: MDP y la Ecuación de Bellman\n"
        "\n"
        "### Formulación del MDP (sin oponente)\n"
        "\n"
        "Cuando eliminamos al agente MIN del juego, el problema de captura de nodos\n"
        "se reduce a un **Proceso de Decisión de Markov (MDP)** de un solo agente:\n"
        "\n"
        "| Componente MDP | Definición en el problema |\n"
        "|----------------|---------------------------|\n"
        "| **Estado** s | frozenset de nodos capturados por MAX |\n"
        "| **Acción** a | Intentar capturar un nodo vecino libre |\n"
        "| **Transición** T(s, a, s') | P(s'=s∪{a} \\| s,a) = 0.8;  P(s'=s \\| s,a) = 0.2 |\n"
        "| **Recompensa** R(s, a) | Éxito: valor(a) − 1;  Fallo: −1 |\n"
        "| **Factor de descuento** γ | 0.9 (ejemplo) |\n"
        "\n"
        "### Ecuación de Bellman\n"
        "\n"
        "La **ecuación de Bellman** para el valor óptimo V*(s) expresa que el valor\n"
        "de un estado es el máximo sobre las acciones del valor esperado de aplicar\n"
        "esa acción y seguir la política óptima:\n"
        "\n"
        "$$V(s) = \\max_a \\left[ 0.8 \\cdot \\bigl(\\text{valor}(a) - 1 + \\gamma \\cdot V(s')\\bigr)\n"
        "         + 0.2 \\cdot \\bigl(-1 + \\gamma \\cdot V(s)\\bigr) \\right]$$\n"
        "\n"
        "donde $s' = s \\cup \\{a\\}$.\n"
        "\n"
        "Como $V(s)$ aparece en ambos lados, reordenando:\n"
        "\n"
        "$$V(s) \\cdot (1 - 0.2\\gamma) = \\max_a \\left[ 0.8 \\cdot \\text{valor}(a) - 1\n"
        "         + 0.8\\gamma \\cdot V(s') \\right]$$\n"
        "\n"
        "Esta es la **forma canónica** usada en la iteración de valor implementada en\n"
        "`demostrar_bellman()`.\n"
        "\n"
        "### Relación con Expectiminimax\n"
        "\n"
        "El árbol Expectiminimax es la extensión natural del MDP al entorno de dos\n"
        "agentes.  La diferencia clave:\n"
        "\n"
        "- **MDP**: MAX elige, la naturaleza introduce incertidumbre (nodo de azar).\n"
        "- **Expectiminimax**: MAX elige → nodo de azar → MIN elige → nodo de azar → ...\n"
        "\n"
        "El **valor de un nodo de azar** (tras movimiento de MAX eligiendo acción $a$\n"
        "desde estado $s$) es:\n"
        "\n"
        "$$V_{\\text{chance}}(s, a) = 0.8 \\cdot V_{\\text{MIN}}(s \\cup \\{a\\})\n"
        "                          + 0.2 \\cdot V_{\\text{MIN}}(s)$$\n"
        "\n"
        "MAX entonces elige $a^* = \\arg\\max_a V_{\\text{chance}}(s, a)$.\n"
        "\n"
        "### Diferencia entre Minimax clásico y Expectiminimax en entornos estocásticos\n"
        "\n"
        "| Aspecto | Minimax (Task 2) | Expectiminimax (Task 3) |\n"
        "|---------|-----------------|------------------------|\n"
        "| Modelo del mundo | Determinista (asume éxito 100%) | Estocástico (modela 20% de fallo) |\n"
        "| Árbol de búsqueda | MAX → MIN → MAX → ... | MAX → Chance → MIN → Chance → ... |\n"
        "| Valor de acción $a$ | valor(a) (sin descuento por riesgo) | $0.8 \\cdot$ valor(a) |\n"
        "| Agresividad | Puede sobrestimar nodos de alto valor | Más conservador ante el riesgo |\n"
        "\n"
        "La principal ventaja de Expectiminimax es que **razona correctamente** sobre\n"
        "las consecuencias reales de sus decisiones, mientras que Minimax podría\n"
        "seleccionar estrategias que dependen de éxito garantizado y que en la\n"
        "práctica rinden peor ante la incertidumbre."
    ))

    return notebook


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nb = build_notebook()
    out_path = f"{BASE}/proyecto2.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Notebook creado: {out_path}")
    print(f"Total de celdas: {len(nb['cells'])}")
    # Contar por tipo
    md_count = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    code_count = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"  Markdown: {md_count}  |  Código: {code_count}")
