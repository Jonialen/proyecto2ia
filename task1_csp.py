# %% [markdown]
# # Tarea 1: Configuracion Segura de la Red usando CSP y Factor Graphs
#
# ## Modelado Formal del CSP
#
# **Variables**: X = {x₁, x₂, ..., xₙ} donde cada xᵢ representa un servidor en la red
#
# **Dominios**: D(xᵢ) = {Rojo, Verde, Azul, Amarillo} para todo xᵢ ∈ X
# (4 protocolos de seguridad)
#
# **Restricciones**: C = {(xᵢ, xⱼ) : xᵢ ≠ xⱼ para toda arista (i,j) en el grafo}
# Dos servidores directamente conectados NO pueden tener el mismo protocolo.
#
# **Factor Graph**:
# - Nodos variable: uno por servidor
# - Nodos factor: uno por restriccion de adyacencia (una funcion indicadora fᵢⱼ(xᵢ, xⱼ))
# - Factor fᵢⱼ(xᵢ, xⱼ) = 1 si xᵢ ≠ xⱼ, 0 en caso contrario
# - La asignacion es valida si ∏ fᵢⱼ(xᵢ, xⱼ) = 1 para todos los factores
#
# ## Algoritmos Implementados
# 1. **Backtracking Basico**: busqueda exhaustiva sin optimizaciones
# 2. **Backtracking Optimizado**: con Forward Checking (lookahead) + MRV heuristic

# %%
import random
import time
import copy
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# %% [markdown]
# ## Seccion 1: Generacion del Grafo de la Red

# %%
def _es_4_colorable_rapido(G):
    """
    Verifica si el grafo es 4-colorable via greedy coloring.
    """
    coloreo = nx.coloring.greedy_color(G, strategy="largest_first")
    return max(coloreo.values()) + 1 <= 4

def generar_grafo_red(num_nodos_min=15, num_nodos_max=20, semilla=42):
    """
    Genera un grafo aleatorio conexo (Erdos-Renyi, p=0.25) con 15-20 nodos.
    """
    random.seed(semilla)

    # Elige numero de nodos en el rango dado
    num_nodos = random.randint(num_nodos_min, num_nodos_max)

    # Erdos-Renyi con p=0.25, busca grafo conexo y 4-colorable
    while True:
        G = nx.erdos_renyi_graph(num_nodos, p=0.25, seed=semilla)
        
        if nx.is_connected(G) and _es_4_colorable_rapido(G):
            break
        semilla += 1  

    print(f"Grafo generado: {num_nodos} nodos, {G.number_of_edges()} aristas")
    print(f"Grado promedio: {2 * G.number_of_edges() / num_nodos:.2f}")
    return G

def describir_grafo(G):
    """Imprime estadisticas del grafo."""
    print("Descripcion del Grafo de la Red")
    print(f"  Numero de servidores (nodos): {G.number_of_nodes()}")
    print(f"  Numero de conexiones (aristas): {G.number_of_edges()}")
    print(f"  Grado minimo: {min(dict(G.degree()).values())}")
    print(f"  Grado maximo: {max(dict(G.degree()).values())}")
    print(f"  Grado promedio: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"  ¿Es conexo? {nx.is_connected(G)}")
    print(f"  Coeficiente de clustering promedio: {nx.average_clustering(G):.4f}")

# %% [markdown]
# ## Seccion 2: Modelado del CSP

# %%
# Protocolos de seguridad
PROTOCOLOS = ["Rojo", "Verde", "Azul", "Amarillo"]
COLOR_MAP = {
    "Rojo": "#e74c3c",
    "Verde": "#2ecc71",
    "Azul": "#3498db",
    "Amarillo": "#f1c40f"
}

class CSPRedSegura:
    """
    Modela coloracion de grafos como CSP.
    Variables X = nodos, Dominios D = {Rojo, Verde, Azul, Amarillo},
    Restricciones: xi != xj para cada arista (i,j).
    """

    def __init__(self, grafo):
        self.grafo = grafo
        
        self.variables = list(grafo.nodes())
        
        self.dominios = {v: list(PROTOCOLOS) for v in self.variables}
        
        self.vecinos = {v: list(grafo.neighbors(v)) for v in self.variables}
        
        self.restricciones = list(grafo.edges())

    def es_consistente(self, variable, valor, asignacion):
        """
        Retorna True si asignar valor a variable no viola restricciones con vecinos asignados.
        """
        for vecino in self.vecinos[variable]:
            if vecino in asignacion and asignacion[vecino] == valor:
                return False  # Factor = 0, restriccion violada
        return True  # Todos los factores = 1

    def es_solucion_completa(self, asignacion):
        """Retorna True si la asignacion es completa y valida."""
        if len(asignacion) != len(self.variables):
            return False
        # Verificar todas las restricciones (producto de todos los factores = 1)
        for u, v in self.restricciones:
            if u in asignacion and v in asignacion:
                if asignacion[u] == asignacion[v]:
                    return False
        return True

    def obtener_factores(self):
        """
        Retorna la lista de factores del Factor Graph.
        """
        factores = []
        for u, v in self.restricciones:
            factores.append({
                "nombre": f"f({u},{v})",
                "variables": (u, v),
                "funcion": lambda a, b: 1 if a != b else 0
            })
        return factores

# %% [markdown]
# ## Seccion 3: Backtracking Basico (sin optimizaciones)

# %%
class BacktrackingBasico:
    """
    Backtracking puro sin optimizaciones. Complejidad O(d^n).
    """

    def __init__(self, csp):
        self.csp = csp
        
        self.num_asignaciones = 0
        self.num_backtracks = 0
        self.tiempo_inicio = 0
        self.tiempo_fin = 0

    def reiniciar_metricas(self):
        self.num_asignaciones = 0
        self.num_backtracks = 0

    def seleccionar_variable(self, asignacion):
        """
        Selecciona la primera variable no asignada (orden fijo, sin MRV).
        """
        for v in self.csp.variables:
            if v not in asignacion:
                return v
        return None

    def backtrack(self, asignacion):
        """
        Backtracking recursivo: busca asignacion que satisfaga todos los factores.
        """
        # Caso base
        if len(asignacion) == len(self.csp.variables):
            return asignacion

        variable = self.seleccionar_variable(asignacion)

        for valor in self.csp.dominios[variable]:
            self.num_asignaciones += 1

            if self.csp.es_consistente(variable, valor, asignacion):
                
                asignacion[variable] = valor

                resultado = self.backtrack(asignacion)
                if resultado is not None:
                    return resultado

                # Backtrack
                del asignacion[variable]
                self.num_backtracks += 1

        return None  

    def resolver(self):
        """Resuelve el CSP con backtracking basico."""
        self.reiniciar_metricas()
        self.tiempo_inicio = time.perf_counter()
        solucion = self.backtrack({})
        self.tiempo_fin = time.perf_counter()
        return solucion

    def obtener_metricas(self):
        return {
            "algoritmo": "Backtracking Basico",
            "asignaciones": self.num_asignaciones,
            "backtracks": self.num_backtracks,
            "tiempo_s": self.tiempo_fin - self.tiempo_inicio
        }

# %% [markdown]
# ## Seccion 4: Backtracking Optimizado (Forward Checking + MRV)

# %%
class BacktrackingOptimizado:
    """
    Backtracking con Forward Checking (lookahead) y MRV (fail-first).
    """

    def __init__(self, csp):
        self.csp = csp
        self.num_asignaciones = 0
        self.num_backtracks = 0
        self.tiempo_inicio = 0
        self.tiempo_fin = 0

    def reiniciar_metricas(self):
        self.num_asignaciones = 0
        self.num_backtracks = 0

    def seleccionar_variable_mrv(self, asignacion, dominios_actuales):
        """
        MRV: selecciona variable con dominio mas pequeño (fail-first).
        """
        variables_no_asignadas = [v for v in self.csp.variables if v not in asignacion]
        if not variables_no_asignadas:
            return None

        # Seleccionar variable con dominio mas pequeño (MRV)
        return min(variables_no_asignadas, key=lambda v: len(dominios_actuales[v]))

    def forward_checking(self, variable, valor, asignacion, dominios_actuales):
        """
        Forward checking: elimina valor del dominio de vecinos no asignados.
        Retorna (exito, dominios_podados). Si algun dominio queda vacio, retorna False.
        """
        dominios_podados = {}  # Para poder restaurar al hacer backtrack

        for vecino in self.csp.vecinos[variable]:
            if vecino not in asignacion:
                if valor in dominios_actuales[vecino]:
                    
                    if vecino not in dominios_podados:
                        dominios_podados[vecino] = []
                    dominios_podados[vecino].append(valor)
                    dominios_actuales[vecino].remove(valor)

                    # Dominio vacio: falla
                    if len(dominios_actuales[vecino]) == 0:
                        
                        self._restaurar_dominios(dominios_podados, dominios_actuales)
                        return False, {}

        return True, dominios_podados

    def _restaurar_dominios(self, dominios_podados, dominios_actuales):
        """Restaura dominios podados durante backtrack."""
        for vecino, valores in dominios_podados.items():
            for valor in valores:
                if valor not in dominios_actuales[vecino]:
                    dominios_actuales[vecino].append(valor)

    def backtrack(self, asignacion, dominios_actuales):
        """
        Backtracking recursivo con forward checking y MRV.
        """
        # Caso base
        if len(asignacion) == len(self.csp.variables):
            return asignacion

        # MRV
        variable = self.seleccionar_variable_mrv(asignacion, dominios_actuales)

        for valor in list(dominios_actuales[variable]):
            self.num_asignaciones += 1

            if self.csp.es_consistente(variable, valor, asignacion):
                
                asignacion[variable] = valor

                # Forward checking
                exito, dominios_podados = self.forward_checking(
                    variable, valor, asignacion, dominios_actuales
                )

                if exito:
                    resultado = self.backtrack(asignacion, dominios_actuales)
                    if resultado is not None:
                        return resultado

                # Backtrack
                self._restaurar_dominios(dominios_podados, dominios_actuales)
                del asignacion[variable]
                self.num_backtracks += 1

        return None  # Fallo

    def resolver(self):
        """Resuelve el CSP con backtracking optimizado (FC + MRV)."""
        self.reiniciar_metricas()
        
        dominios_actuales = {v: list(vals) for v, vals in self.csp.dominios.items()}
        self.tiempo_inicio = time.perf_counter()
        solucion = self.backtrack({}, dominios_actuales)
        self.tiempo_fin = time.perf_counter()
        return solucion

    def obtener_metricas(self):
        return {
            "algoritmo": "Backtracking Optimizado (FC + MRV)",
            "asignaciones": self.num_asignaciones,
            "backtracks": self.num_backtracks,
            "tiempo_s": self.tiempo_fin - self.tiempo_inicio
        }

# %% [markdown]
# ## Seccion 5: Factor Graph — Representacion Formal

# %%
class FactorGraph:
    """
    Factor Graph bipartito: nodos variable (servidores) y nodos factor (restricciones).
    f_ij(a,b) = 1 si a != b, 0 en caso contrario.
    """

    def __init__(self, csp):
        self.csp = csp
        self.nodos_variable = csp.variables
        self.nodos_factor = list(csp.grafo.edges())
        
        self.conexiones = self._construir_conexiones()

    def _construir_conexiones(self):
        """
        Construye conexiones del Factor Graph.
        """
        conexiones = defaultdict(list)
        for u, v in self.nodos_factor:
            factor_nombre = f"f({u},{v})"
            conexiones[factor_nombre].extend([u, v])
        return dict(conexiones)

    def evaluar(self, asignacion):
        """
        Evalua el producto de todos los factores. Retorna 1 si es valida, 0 si no.
        """
        for u, v in self.nodos_factor:
            if u in asignacion and v in asignacion:
                if asignacion[u] == asignacion[v]:
                    return 0  # Factor = 0, restriccion violada
        return 1

    def describir(self):
        """Imprime la estructura del Factor Graph."""
        print("Factor Graph")
        print(f"  Nodos variable (servidores): {len(self.nodos_variable)}")
        print(f"  Nodos factor (restricciones): {len(self.nodos_factor)}")
        print(f"  Total aristas del Factor Graph: {2 * len(self.nodos_factor)}")
        print("\n  Factores fᵢⱼ(xᵢ, xⱼ) = 1 si xᵢ ≠ xⱼ, 0 en caso contrario:")
        for i, (u, v) in enumerate(self.nodos_factor[:5]):
            print(f"    f({u},{v}) conecta variable x{u} con variable x{v}")
        if len(self.nodos_factor) > 5:
            print(f"    ... y {len(self.nodos_factor) - 5} factores mas")

    def visualizar(self, ax=None):
        """Dibuja el Factor Graph como grafo bipartito."""
        FG = nx.Graph()

        for v in self.nodos_variable:
            FG.add_node(f"x{v}", tipo="variable")

        for u, v in self.nodos_factor:
            fname = f"f({u},{v})"
            FG.add_node(fname, tipo="factor")
            FG.add_edge(f"x{u}", fname)
            FG.add_edge(f"x{v}", fname)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        var_nodes = [f"x{v}" for v in self.nodos_variable]
        fac_nodes = [f"f({u},{v})" for u, v in self.nodos_factor]

        pos = {}
        for i, n in enumerate(var_nodes):
            pos[n] = (0, i - len(var_nodes) / 2)
        for i, n in enumerate(fac_nodes):
            pos[n] = (2, i - len(fac_nodes) / 2)

        nx.draw_networkx_nodes(FG, pos, nodelist=var_nodes,
                               node_color="#3498db", node_size=300,
                               node_shape='o', ax=ax)
        nx.draw_networkx_nodes(FG, pos, nodelist=fac_nodes,
                               node_color="#e74c3c", node_size=200,
                               node_shape='s', ax=ax)
        nx.draw_networkx_edges(FG, pos, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(FG, pos, font_size=6, ax=ax)

        ax.set_title("Factor Graph (bipartito)\nCirculos=Variables, Cuadros=Factores", fontsize=10)
        ax.axis("off")

# %% [markdown]
# ## Seccion 6: Analisis y Comparacion de Rendimiento

# %%
def comparar_algoritmos(csp, verbose=True):
    """
    Ejecuta ambos algoritmos y compara metricas de rendimiento.
    """
    print("Comparacion de Algoritmos")
    print("Resolviendo CSP con Backtracking Basico...")

    bt_basico = BacktrackingBasico(csp)
    solucion_basica = bt_basico.resolver()
    metricas_basico = bt_basico.obtener_metricas()

    print("Resolviendo CSP con Backtracking Optimizado (FC + MRV)...")

    bt_opt = BacktrackingOptimizado(csp)
    solucion_opt = bt_opt.resolver()
    metricas_opt = bt_opt.obtener_metricas()

    valida_basica = csp.es_solucion_completa(solucion_basica) if solucion_basica else False
    valida_opt = csp.es_solucion_completa(solucion_opt) if solucion_opt else False

    if verbose:
        print(f"{'Metrica':<35} {'Basico':>15} {'Optimizado (FC+MRV)':>17}")
        print(f"{'Solucion encontrada':<35} {'Si' if solucion_basica else 'No':>15} {'Si' if solucion_opt else 'No':>17}")
        print(f"{'Solucion valida':<35} {'Si' if valida_basica else 'No':>15} {'Si' if valida_opt else 'No':>17}")
        print(f"{'Asignaciones intentadas':<35} {metricas_basico['asignaciones']:>15,} {metricas_opt['asignaciones']:>17,}")
        print(f"{'Numero de backtracks':<35} {metricas_basico['backtracks']:>15,} {metricas_opt['backtracks']:>17,}")
        print(f"{'Tiempo de ejecucion (s)':<35} {metricas_basico['tiempo_s']:>15.6f} {metricas_opt['tiempo_s']:>17.6f}")

        if metricas_opt['tiempo_s'] > 0 and metricas_basico['tiempo_s'] > 0:
            speedup = metricas_basico['tiempo_s'] / metricas_opt['tiempo_s']
            reduccion_asig = (1 - metricas_opt['asignaciones'] / max(metricas_basico['asignaciones'], 1)) * 100
            print(f"{'Speedup (basico / optimizado)':<35} {speedup:>15.2f}x")
            print(f"{'Reduccion en asignaciones':<35} {reduccion_asig:>14.1f}%")
        

    return {
        "solucion_basica": solucion_basica,
        "solucion_opt": solucion_opt,
        "metricas_basico": metricas_basico,
        "metricas_opt": metricas_opt
    }

def analisis_detallado(resultados, csp):
    """Imprime analisis detallado del rendimiento."""
    m_b = resultados["metricas_basico"]
    m_o = resultados["metricas_opt"]

    print("Analisis de Rendimiento")

    print("\n  Backtracking Basico:")
    print(f"    - Sin heuristicas: variables elegidas en orden fijo")
    print(f"    - Sin lookahead: inconsistencias detectadas tarde")
    print(f"    - Explora mas del arbol de busqueda de forma innecesaria")

    print("\n  Backtracking Optimizado (FC + MRV):")
    print(f"    - MRV: elige la variable mas restringida primero")
    print(f"      → Detecta fallos antes, reduce ramificacion")
    print(f"    - Forward Checking: elimina valores del dominio de vecinos")
    print(f"      → Poda subarboles imposibles sin explorarlos")

    if m_b["asignaciones"] > 0:
        ratio = m_o["asignaciones"] / m_b["asignaciones"]
        print(f"\n  El algoritmo optimizado exploro solo el {ratio*100:.1f}% de las")
        print(f"  asignaciones del basico ({m_o['asignaciones']:,} vs {m_b['asignaciones']:,})")

    if m_b["tiempo_s"] > 0:
        speedup = m_b["tiempo_s"] / m_o["tiempo_s"]
        print(f"  Aceleracion temporal: {speedup:.2f}x mas rapido")

    print("\n  Relacion con Factor Graphs:")
    print(f"    Forward Checking implementa propagacion local de mensajes")
    print(f"    similar al algoritmo Belief Propagation en Factor Graphs:")
    print(f"    μ_xᵢ→fᵢⱼ(v) = 0 si v fue eliminado del dominio D(xᵢ)")

# %% [markdown]
# ## Seccion 7: Visualizaciones

# %%
def visualizar_grafo_coloreado(G, asignacion, titulo="Red con Protocolos de Seguridad", ax=None):
    """
    Dibuja el grafo con colores segun protocolo asignado.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    colores_nodos = []
    for nodo in G.nodes():
        protocolo = asignacion.get(nodo, "Sin asignar")
        colores_nodos.append(COLOR_MAP.get(protocolo, "#95a5a6"))

    pos = nx.spring_layout(G, seed=42, k=1.5)

    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color="gray", ax=ax)

    nx.draw_networkx_nodes(G, pos, node_color=colores_nodos,
                           node_size=500, ax=ax)

    labels = {n: f"S{n}" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="white",
                            font_weight="bold", ax=ax)

    parches = [mpatches.Patch(color=COLOR_MAP[p], label=f"Protocolo {p}")
               for p in PROTOCOLOS]
    ax.legend(handles=parches, loc="upper left", fontsize=9)
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.axis("off")

def visualizar_metricas_comparacion(resultados):
    """
    Grafico de barras comparando metricas.
    """
    m_b = resultados["metricas_basico"]
    m_o = resultados["metricas_opt"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Comparacion de Rendimiento: Backtracking Basico vs Optimizado",
                 fontsize=13, fontweight="bold")

    algoritmos = ["Basico", "Optimizado\n(FC + MRV)"]
    colores_barras = ["#e74c3c", "#2ecc71"]

    # Grafico 1: Asignaciones
    vals_asig = [m_b["asignaciones"], m_o["asignaciones"]]
    bars = axes[0].bar(algoritmos, vals_asig, color=colores_barras, edgecolor="black", width=0.5)
    axes[0].set_title("Asignaciones Intentadas", fontweight="bold")
    axes[0].set_ylabel("Numero de asignaciones")
    for bar, val in zip(bars, vals_asig):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                     f"{val:,}", ha="center", va="bottom", fontsize=9)

    # Grafico 2: Backtracks
    vals_bt = [m_b["backtracks"], m_o["backtracks"]]
    bars = axes[1].bar(algoritmos, vals_bt, color=colores_barras, edgecolor="black", width=0.5)
    axes[1].set_title("Numero de Backtracks", fontweight="bold")
    axes[1].set_ylabel("Numero de backtracks")
    for bar, val in zip(bars, vals_bt):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                     f"{val:,}", ha="center", va="bottom", fontsize=9)

    # Grafico 3: Tiempo
    vals_t = [m_b["tiempo_s"] * 1000, m_o["tiempo_s"] * 1000]  # en ms
    bars = axes[2].bar(algoritmos, vals_t, color=colores_barras, edgecolor="black", width=0.5)
    axes[2].set_title("Tiempo de Ejecucion", fontweight="bold")
    axes[2].set_ylabel("Tiempo (ms)")
    for bar, val in zip(bars, vals_t):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("/home/jonialen/Documents/UVG/s7/ia/proyecto2/task1_metricas.png",
                dpi=150, bbox_inches="tight")
    print("\n  Grafico de metricas guardado en: task1_metricas.png")
    return fig

def visualizar_todo(G, resultados, factor_graph):
    """
    Genera figura compuesta con todas las visualizaciones.
    """
    fig = plt.figure(figsize=(18, 14))

    ax1 = fig.add_subplot(2, 3, 1)  # Grafo original
    ax2 = fig.add_subplot(2, 3, 2)  # Grafo con solucion basica
    ax3 = fig.add_subplot(2, 3, 3)  # Grafo con solucion optimizada
    ax4 = fig.add_subplot(2, 3, 4)  # Factor graph (parcial)
    ax5 = fig.add_subplot(2, 3, 5)  # Barras asignaciones
    ax6 = fig.add_subplot(2, 3, 6)  # Barras tiempo

    pos = nx.spring_layout(G, seed=42, k=1.5)
    nx.draw_networkx(G, pos, ax=ax1, node_color="#95a5a6", node_size=400,
                     font_size=7, font_color="white", font_weight="bold",
                     edge_color="gray", alpha=0.8,
                     labels={n: f"S{n}" for n in G.nodes()})
    ax1.set_title("Red de Servidores\n(sin protocolo asignado)", fontsize=10)
    ax1.axis("off")

    if resultados["solucion_basica"]:
        visualizar_grafo_coloreado(G, resultados["solucion_basica"],
                                   "Solucion: Backtracking Basico", ax=ax2)
    else:
        ax2.text(0.5, 0.5, "Sin solucion encontrada\n(grafo no 4-colorable)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=11)
        ax2.set_title("Solucion: Backtracking Basico", fontsize=10)
        ax2.axis("off")

    if resultados["solucion_opt"]:
        visualizar_grafo_coloreado(G, resultados["solucion_opt"],
                                   "Solucion: Backtracking Optimizado\n(FC + MRV)", ax=ax3)
    else:
        ax3.text(0.5, 0.5, "Sin solucion encontrada\n(grafo no 4-colorable)",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=11)
        ax3.set_title("Solucion: Backtracking Optimizado\n(FC + MRV)", fontsize=10)
        ax3.axis("off")

    _visualizar_factor_graph_parcial(factor_graph, ax4)

    m_b = resultados["metricas_basico"]
    m_o = resultados["metricas_opt"]
    categorias = ["Asignaciones", "Backtracks"]
    vals_b = [m_b["asignaciones"], m_b["backtracks"]]
    vals_o = [m_o["asignaciones"], m_o["backtracks"]]
    x = range(len(categorias))
    width = 0.35
    bars1 = ax5.bar([xi - width/2 for xi in x], vals_b, width,
                    label="Basico", color="#e74c3c", edgecolor="black")
    bars2 = ax5.bar([xi + width/2 for xi in x], vals_o, width,
                    label="Optimizado", color="#2ecc71", edgecolor="black")
    ax5.set_xticks(list(x))
    ax5.set_xticklabels(categorias)
    ax5.set_title("Asignaciones y Backtracks", fontsize=10, fontweight="bold")
    ax5.legend(fontsize=8)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                 f"{h:,}", ha="center", va="bottom", fontsize=7)

    tiempos = [m_b["tiempo_s"] * 1000, m_o["tiempo_s"] * 1000]
    colores = ["#e74c3c", "#2ecc71"]
    algs = ["Basico", "Optimizado\n(FC+MRV)"]
    bars = ax6.bar(algs, tiempos, color=colores, edgecolor="black", width=0.5)
    ax6.set_title("Tiempo de Ejecucion (ms)", fontsize=10, fontweight="bold")
    ax6.set_ylabel("ms")
    for bar, val in zip(bars, tiempos):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                 f"{val:.3f}ms", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Tarea 1: CSP — Configuracion Segura de Red de Servidores",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("/home/jonialen/Documents/UVG/s7/ia/proyecto2/task1_completo.png",
                dpi=150, bbox_inches="tight")
    print("  Figura completa guardada en: task1_completo.png")
    return fig

def _visualizar_factor_graph_parcial(factor_graph, ax):
    """Dibuja un subconjunto del Factor Graph."""
    
    vars_subset = list(factor_graph.nodos_variable)[:6]
    vars_set = set(vars_subset)

    FG = nx.Graph()

    for v in vars_subset:
        FG.add_node(f"x{v}", tipo="variable")

    factores_subset = []
    for u, v in factor_graph.nodos_factor:
        if u in vars_set and v in vars_set:
            fname = f"f({u},{v})"
            FG.add_node(fname, tipo="factor")
            FG.add_edge(f"x{u}", fname)
            FG.add_edge(f"x{v}", fname)
            factores_subset.append(fname)

    if len(FG.nodes()) < 2:
        ax.text(0.5, 0.5, "Factor Graph\n(sin factores en subconjunto)",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    pos = nx.spring_layout(FG, seed=10)
    var_nodes = [f"x{v}" for v in vars_subset if f"x{v}" in FG]
    fac_nodes = [n for n in FG.nodes() if n not in var_nodes]

    nx.draw_networkx_nodes(FG, pos, nodelist=var_nodes, node_color="#3498db",
                           node_size=400, ax=ax)
    nx.draw_networkx_nodes(FG, pos, nodelist=fac_nodes, node_color="#e74c3c",
                           node_size=200, node_shape="s", ax=ax)
    nx.draw_networkx_edges(FG, pos, alpha=0.6, ax=ax)
    nx.draw_networkx_labels(FG, pos, font_size=6, font_color="white",
                            font_weight="bold", ax=ax)
    ax.set_title("Factor Graph (subconjunto)\n● Variables  ■ Factores", fontsize=9)
    ax.axis("off")

# %% [markdown]
# ## Seccion 8: Benchmark con Grafos de Dificultad Variable

# %%
def benchmark_dificultad_variable():
    """
    Compara algoritmos en grafos de distinta densidad.
    """
    print("Benchmark: Dificultad Variable")
    print("(Comparacion en multiples instancias para ilustrar escalabilidad)")

    configuraciones = [
        {"p": 0.15, "semilla": 100, "label": "Baja densidad (p=0.15)"},
        {"p": 0.22, "semilla": 200, "label": "Media densidad (p=0.22)"},
        {"p": 0.28, "semilla": 300, "label": "Alta densidad (p=0.28)"},
    ]

    resultados_benchmark = []

    for cfg in configuraciones:
        
        semilla = cfg["semilla"]
        for intento in range(50):
            G_test = nx.erdos_renyi_graph(18, p=cfg["p"], seed=semilla + intento)
            if nx.is_connected(G_test) and _es_4_colorable_rapido(G_test):
                break

        csp_test = CSPRedSegura(G_test)

        bt_b = BacktrackingBasico(csp_test)
        bt_b.resolver()
        m_b = bt_b.obtener_metricas()

        bt_o = BacktrackingOptimizado(csp_test)
        bt_o.resolver()
        m_o = bt_o.obtener_metricas()

        speedup = m_b["tiempo_s"] / m_o["tiempo_s"] if m_o["tiempo_s"] > 0 else float("inf")
        reduccion = (1 - m_o["asignaciones"] / max(m_b["asignaciones"], 1)) * 100

        resultados_benchmark.append({
            "label": cfg["label"],
            "aristas": G_test.number_of_edges(),
            "asig_basico": m_b["asignaciones"],
            "asig_opt": m_o["asignaciones"],
            "bt_basico": m_b["backtracks"],
            "bt_opt": m_o["backtracks"],
            "speedup": speedup,
            "reduccion": reduccion
        })

    print(f"\n{'Instancia':<28} {'Aristas':>7} {'Asig.Bas':>9} {'Asig.Opt':>9} "
          f"{'BT.Bas':>7} {'BT.Opt':>7} {'Speedup':>8} {'Reduc.%':>8}")
    
    for r in resultados_benchmark:
        print(f"  {r['label']:<26} {r['aristas']:>7} {r['asig_basico']:>9,} {r['asig_opt']:>9,} "
              f"{r['bt_basico']:>7,} {r['bt_opt']:>7,} {r['speedup']:>7.2f}x {r['reduccion']:>7.1f}%")
    

    return resultados_benchmark

# %% [markdown]
# ## Seccion 9: Funcion Principal

# %%
def main():
    """Ejecuta el pipeline completo: genera grafo, resuelve CSP, compara y visualiza."""
    print("Task 1: Configuracion segura de red (CSP + Factor Graphs)\n")

    G = generar_grafo_red(num_nodos_min=15, num_nodos_max=20, semilla=42)
    describir_grafo(G)

    csp = CSPRedSegura(G)
    factor_graph = FactorGraph(csp)

    print(f"\nCSP: {len(csp.variables)} variables, dominio={PROTOCOLOS}, {len(csp.restricciones)} restricciones")
    factor_graph.describir()

    print("\nResolviendo...")
    resultados = comparar_algoritmos(csp, verbose=True)

    analisis_detallado(resultados, csp)

    if resultados["solucion_basica"]:
        val_fg = factor_graph.evaluar(resultados["solucion_basica"])
        print(f"\nFactor Graph basica: prod(fij) = {val_fg} ({'valida' if val_fg == 1 else 'invalida'})")
    if resultados["solucion_opt"]:
        val_fg = factor_graph.evaluar(resultados["solucion_opt"])
        print(f"Factor Graph optimizada: prod(fij) = {val_fg} ({'valida' if val_fg == 1 else 'invalida'})")

    if resultados["solucion_opt"]:
        print("\nProtocolos asignados:")
        sol = resultados["solucion_opt"]
        for protocolo in PROTOCOLOS:
            servidores = [f"S{v}" for v, p in sorted(sol.items()) if p == protocolo]
            print(f"  {protocolo:8s}: {', '.join(servidores) if servidores else '(ninguno)'}")

    benchmark_dificultad_variable()

    try:
        visualizar_todo(G, resultados, factor_graph)
        plt.show()
    except Exception as e:
        print(f"Visualizacion omitida: {e}")

    return resultados

# %%
if __name__ == "__main__":
    main()
