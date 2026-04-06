# %% [markdown]
# # Task 2 - Defensa Adversarial (Juegos de Suma Cero)
# ## CC3045 - Inteligencia Artificial, UVG
#
# Este modulo implementa un juego de suma cero sobre una red de ciberseguridad.
# - **MAX (Defensa)**: intenta maximizar la suma de valores de nodos capturados.
# - **MIN (Atacante/Hacker)**: intenta minimizar dicha suma (para la defensa).
#
# Se implementan dos algoritmos de busqueda adversarial:
# 1. **Minimax puro** – recorre el arbol de juego completo hasta profundidad d_max.
# 2. **Minimax con Poda Alfa-Beta** – optimiza el arbol eliminando ramas que no
#    pueden influir en la decision optima (α ≥ β → poda).

# %%
import random
import copy
import math
import time
from typing import Optional, Tuple, List, Dict

import networkx as nx
import matplotlib
matplotlib.use("Agg")  # modo no interactivo; quitar si se ejecuta en Jupyter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# %% [markdown]
# ## 1. Generacion del Grafo de Red
# %%

SEED = 42          # reproducibilidad
NUM_NODES = 17     # entre 15 y 20 nodos
MIN_VALUE = 1      # valor minimo de informacion
MAX_VALUE = 20     # valor maximo de informacion
D_MAX = 4          # profundidad maxima de busqueda (heuristica)
TURN_LIMIT = 30    # limite de turnos para terminar el juego

def build_network(n: int = NUM_NODES, seed: int = SEED) -> nx.Graph:
    """
    Construye un grafo de red de ciberseguridad con n nodos.
    Usa Barabasi-Albert para simular topologia realista (hub-and-spoke),
    garantizando que el grafo sea conexo.
    A cada nodo se le asigna un 'info_value' entero aleatorio en [MIN_VALUE, MAX_VALUE].
    """
    rng = random.Random(seed)
    # Barabasi-Albert: m=2, grafo conexo
    G = nx.barabasi_albert_graph(n, m=2, seed=seed)
    
    for node in G.nodes():
        G.nodes[node]["info_value"] = rng.randint(MIN_VALUE, MAX_VALUE)
    return G

# %% [markdown]
# ## 2. Estado del Juego (GameState)
# %%

class GameState:
    """
    Estado inmutable del juego: nodos de cada jugador, turno actual.
    """

    def __init__(
        self,
        graph: nx.Graph,
        defender_nodes: set,
        attacker_nodes: set,
        is_max_turn: bool = True,
        turn: int = 0,
    ):
        self.graph = graph
        self.defender_nodes = set(defender_nodes)
        self.attacker_nodes = set(attacker_nodes)
        self.is_max_turn = is_max_turn
        self.turn = turn

    # ------------------------------------------------------------------
    def captured_nodes(self) -> set:
        """Nodos capturados por ambos jugadores."""
        return self.defender_nodes | self.attacker_nodes

    def free_nodes(self) -> set:
        """Nodos libres."""
        return set(self.graph.nodes()) - self.captured_nodes()

    # ------------------------------------------------------------------
    def available_moves(self, for_defender: bool) -> List[int]:
        """
        Nodos libres adyacentes a los controlados por el jugador.
        """
        controlled = self.defender_nodes if for_defender else self.attacker_nodes
        candidates = set()
        for node in controlled:
            for neighbor in self.graph.neighbors(node):
                if neighbor not in self.captured_nodes():
                    candidates.add(neighbor)
        return sorted(candidates)  

    # ------------------------------------------------------------------
    def apply_move(self, node: int) -> "GameState":
        """
        Retorna nuevo GameState tras capturar node. No muta el estado actual.
        """
        new_defender = set(self.defender_nodes)
        new_attacker = set(self.attacker_nodes)
        if self.is_max_turn:
            new_defender.add(node)
        else:
            new_attacker.add(node)
        return GameState(
            self.graph,
            new_defender,
            new_attacker,
            is_max_turn=not self.is_max_turn,
            turn=self.turn + 1,
        )

    # ------------------------------------------------------------------
    def is_terminal(self) -> bool:
        """
        True si el juego termino (todos capturados, sin movimientos, o limite de turnos).
        """
        if self.turn >= TURN_LIMIT:
            return True
        if not self.free_nodes():
            return True
        
        current_is_max = self.is_max_turn
        moves = self.available_moves(for_defender=current_is_max)
        return len(moves) == 0

    # ------------------------------------------------------------------
    def terminal_score(self) -> int:
        """
        Score terminal: sum(defensor) - sum(atacante).
        """
        def_score = sum(self.graph.nodes[n]["info_value"] for n in self.defender_nodes)
        att_score = sum(self.graph.nodes[n]["info_value"] for n in self.attacker_nodes)
        return def_score - att_score

    # ------------------------------------------------------------------
    def __repr__(self):
        return (
            f"GameState(turn={self.turn}, max_turn={self.is_max_turn}, "
            f"defender={sorted(self.defender_nodes)}, "
            f"attacker={sorted(self.attacker_nodes)})"
        )

# %% [markdown]
# ## 3. Funcion de Evaluacion Heuristica — Eval(s)
#
# Como alcanzar nodos terminales es inviable para grafos de 17+ nodos con
# branching factor alto, limitamos la busqueda a d_max = 4 capas y aplicamos
# una funcion heuristica Eval(s) en los nodos hoja del arbol truncado.
#
# **Componentes de Eval(s)**:
# 1. **Diferencia de valores controlados** – ventaja material directa.
# 2. **Diferencia de cantidad de nodos** – ventaja territorial.
# 3. **Potencial de expansion del defensor** – suma de valores de nodos libres
#    adyacentes al defensor (oportunidades futuras de MAX).
# 4. **Potencial de expansion del atacante** – idem para MIN (restado).
# 5. **Ventaja de conectividad** – grado promedio de los nodos controlados
#    (mas conexiones → mas opciones de expansion).
# %%

def evaluate(state: GameState) -> float:
    """
    Eval(s) heuristica: estima utilidad del estado.
    Eval = w1*material + w2*territorio + w3*frontera + w4*conectividad.
    """
    G = state.graph

    # Componente 1: ventaja material
    def_material = sum(G.nodes[n]["info_value"] for n in state.defender_nodes)
    att_material = sum(G.nodes[n]["info_value"] for n in state.attacker_nodes)
    delta_material = def_material - att_material   # positivo → ventaja defensor

    # Componente 2: ventaja territorial
    delta_territory = len(state.defender_nodes) - len(state.attacker_nodes)

    # Componente 3-4: potencial de expansion (frontera)
    def frontier_value(controlled: set) -> float:
        seen = set()
        total = 0.0
        for node in controlled:
            for nb in G.neighbors(node):
                if nb not in state.captured_nodes() and nb not in seen:
                    seen.add(nb)
                    total += G.nodes[nb]["info_value"]
        return total

    def_frontier = frontier_value(state.defender_nodes)
    att_frontier = frontier_value(state.attacker_nodes)
    delta_frontier = def_frontier - att_frontier

    # Componente 5: ventaja de conectividad
    def avg_degree(nodes: set) -> float:
        if not nodes:
            return 0.0
        return sum(G.degree(n) for n in nodes) / len(nodes)

    delta_connectivity = avg_degree(state.defender_nodes) - avg_degree(state.attacker_nodes)

    # Pesos
    w1, w2, w3, w4 = 3.0, 1.0, 1.5, 0.5

    return (
        w1 * delta_material
        + w2 * delta_territory
        + w3 * delta_frontier
        + w4 * delta_connectivity
    )

# %% [markdown]
# ## 4. Minimax Puro
#
# Implementacion clasica del algoritmo Minimax (Russell & Norvig, cap. 5):
#
# ```
# MINIMAX-VALUE(s):
#   if TERMINAL(s) or depth == 0: return EVAL(s)
#   if MAX-turn:
#       return max over a in ACTIONS(s): MINIMAX-VALUE(RESULT(s,a))
#   else:
#       return min over a in ACTIONS(s): MINIMAX-VALUE(RESULT(s,a))
# ```
#
# Se añade un contador de nodos expandidos para analisis comparativo.
# %%

def minimax(
    state: GameState,
    depth: int,
    counter: Dict[str, int],
) -> Tuple[float, Optional[int]]:
    """
    Minimax puro. Retorna (valor, mejor_movimiento).
    """
    counter["nodes"] += 1  

    # Caso base
    if state.is_terminal() or depth == 0:
        if state.is_terminal():
            return state.terminal_score(), None
        return evaluate(state), None

    current_is_max = state.is_max_turn
    moves = state.available_moves(for_defender=current_is_max)

    
    if not moves:
        return evaluate(state), None

    best_move = None

    if current_is_max:
        # MAX maximiza
        best_val = -math.inf
        for move in moves:
            new_state = state.apply_move(move)
            val, _ = minimax(new_state, depth - 1, counter)
            if val > best_val:
                best_val = val
                best_move = move
        return best_val, best_move

    else:
        # MIN minimiza
        best_val = math.inf
        for move in moves:
            new_state = state.apply_move(move)
            val, _ = minimax(new_state, depth - 1, counter)
            if val < best_val:
                best_val = val
                best_move = move
        return best_val, best_move

# %% [markdown]
# ## 5. Minimax con Poda Alfa-Beta
#
# La poda alfa-beta mantiene dos valores:
# - **α (alpha)**: mejor valor que MAX puede garantizar hasta ahora (inicia en -∞)
# - **β (beta)**: mejor valor que MIN puede garantizar hasta ahora (inicia en +∞)
#
# Regla de poda:
# - En nodo MAX: si v ≥ β → **poda beta** (MIN nunca elegira este camino)
# - En nodo MIN: si v ≤ α → **poda alfa** (MAX nunca elegira este camino)
#
# La decision optima es IDENTICA a Minimax; solo se evitan ramas innecesarias.
#
# ```
# ALPHA-BETA(s, α, β):
#   if TERMINAL(s) or depth == 0: return EVAL(s)
#   if MAX-turn:
#       v = -∞
#       for a in ACTIONS(s):
#           v = max(v, ALPHA-BETA(RESULT(s,a), α, β))
#           if v ≥ β: return v  ← poda beta
#           α = max(α, v)
#   else:
#       v = +∞
#       for a in ACTIONS(s):
#           v = min(v, ALPHA-BETA(RESULT(s,a), α, β))
#           if v ≤ α: return v  ← poda alfa
#           β = min(β, v)
#   return v
# ```
# %%

def alpha_beta(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    counter: Dict[str, int],
) -> Tuple[float, Optional[int]]:
    """
    Minimax con poda alfa-beta. Retorna (valor, mejor_movimiento).
    """
    counter["nodes"] += 1

    # Caso base
    if state.is_terminal() or depth == 0:
        if state.is_terminal():
            return state.terminal_score(), None
        return evaluate(state), None

    current_is_max = state.is_max_turn
    moves = state.available_moves(for_defender=current_is_max)

    if not moves:
        return evaluate(state), None

    best_move = None

    if current_is_max:
        # MAX maximiza
        v = -math.inf
        for move in moves:
            new_state = state.apply_move(move)
            child_val, _ = alpha_beta(new_state, depth - 1, alpha, beta, counter)
            if child_val > v:
                v = child_val
                best_move = move
            alpha = max(alpha, v)
            # Poda beta
            if v >= beta:
                break  # ← poda beta
        return v, best_move

    else:
        # MIN minimiza
        v = math.inf
        for move in moves:
            new_state = state.apply_move(move)
            child_val, _ = alpha_beta(new_state, depth - 1, alpha, beta, counter)
            if child_val < v:
                v = child_val
                best_move = move
            beta = min(beta, v)
            # Poda alfa
            if v <= alpha:
                break  # ← poda alfa
        return v, best_move

# %% [markdown]
# ## 6. Motor del Juego
# %%

class GameEngine:
    """
    Motor del juego: turnos, metricas y visualizacion.
    """

    def __init__(self, graph: nx.Graph, use_alpha_beta_for_display: bool = True):
        self.graph = graph
        self.use_alpha_beta_for_display = use_alpha_beta_for_display
        self.history: List[GameState] = []
        self.metrics: List[Dict] = []  # por turno: {turn, mm_nodes, ab_nodes, move}
        self.pos = nx.spring_layout(graph, seed=SEED)  # layout fijo

    # ------------------------------------------------------------------
    def _choose_start_nodes(self) -> Tuple[int, int]:
        """
        Elige nodos iniciales: defensor toma el de mayor valor, atacante el segundo.
        """
        sorted_nodes = sorted(
            self.graph.nodes(),
            key=lambda n: (self.graph.nodes[n]["info_value"], self.graph.degree(n)),
            reverse=True,
        )
        defender_start = sorted_nodes[0]
        # Atacante: mejor nodo restante con cierta distancia al defensor
        for candidate in sorted_nodes[1:]:
            if candidate != defender_start:
                attacker_start = candidate
                break
        return defender_start, attacker_start

    # ------------------------------------------------------------------
    def initialize(self) -> GameState:
        """Crea estado inicial."""
        d_start, a_start = self._choose_start_nodes()
        state = GameState(
            self.graph,
            defender_nodes={d_start},
            attacker_nodes={a_start},
            is_max_turn=True,
            turn=0,
        )
        self.history.append(state)
        print(f"Defensor inicia en nodo {d_start} (valor={self.graph.nodes[d_start]['info_value']})")
        print(f"Atacante inicia en nodo {a_start} (valor={self.graph.nodes[a_start]['info_value']})")
        return state

    # ------------------------------------------------------------------
    def play_turn(self, state: GameState) -> Tuple[GameState, Dict]:
        """
        Ejecuta un turno: compara Minimax puro vs Alfa-Beta y aplica el movimiento.
        """
        role = "DEFENSOR (MAX)" if state.is_max_turn else "ATACANTE (MIN)"

        # --- Minimax puro ---
        mm_counter = {"nodes": 0}
        t0 = time.perf_counter()
        mm_val, mm_move = minimax(state, D_MAX, mm_counter)
        mm_time = time.perf_counter() - t0

        # --- Alfa-Beta ---
        ab_counter = {"nodes": 0}
        t0 = time.perf_counter()
        ab_val, ab_move = alpha_beta(state, D_MAX, -math.inf, math.inf, ab_counter)
        ab_time = time.perf_counter() - t0

        # Verificacion: ambos deben dar el mismo valor
        assert abs(mm_val - ab_val) < 1e-9, (
            f"Inconsistencia: Minimax={mm_val}, AlphaBeta={ab_val}"
        )

        
        chosen_move = ab_move if self.use_alpha_beta_for_display else mm_move
        chosen_move = chosen_move if chosen_move is not None else mm_move

        metric = {
            "turn": state.turn,
            "role": role,
            "move": chosen_move,
            "eval_value": ab_val,
            "mm_nodes": mm_counter["nodes"],
            "ab_nodes": ab_counter["nodes"],
            "mm_time_ms": mm_time * 1000,
            "ab_time_ms": ab_time * 1000,
            "pruning_reduction_pct": (
                100.0 * (1 - ab_counter["nodes"] / mm_counter["nodes"])
                if mm_counter["nodes"] > 0 else 0.0
            ),
        }
        self.metrics.append(metric)

        if chosen_move is not None:
            new_state = state.apply_move(chosen_move)
        else:
            
            new_state = GameState(
                state.graph,
                state.defender_nodes,
                state.attacker_nodes,
                is_max_turn=not state.is_max_turn,
                turn=state.turn + 1,
            )

        self.history.append(new_state)
        return new_state, metric

    # ------------------------------------------------------------------
    def run(self) -> GameState:
        """Ejecuta el juego completo."""
        state = self.initialize()
        print(f"\nd_max={D_MAX}, limite={TURN_LIMIT} turnos\n")

        while not state.is_terminal():
            state, metric = self.play_turn(state)
            move_str = (
                f"nodo {metric['move']} (v={self.graph.nodes[metric['move']]['info_value']})"
                if metric["move"] is not None
                else "sin movimiento"
            )
            print(
                f"T{metric['turn']:2d} {metric['role']:14s} | {move_str:25s} | "
                f"eval={metric['eval_value']:+.1f} | "
                f"MM={metric['mm_nodes']:5d} AB={metric['ab_nodes']:5d} (-{metric['pruning_reduction_pct']:.0f}%)"
            )

        self._print_final_score(state)
        return state

    # ------------------------------------------------------------------
    def _print_final_score(self, state: GameState):
        """Imprime marcador final."""
        G = self.graph
        def_score = sum(G.nodes[n]["info_value"] for n in state.defender_nodes)
        att_score = sum(G.nodes[n]["info_value"] for n in state.attacker_nodes)
        free = state.free_nodes()
        free_score = sum(G.nodes[n]["info_value"] for n in free)

        winner = "DEFENSOR" if def_score > att_score else (
            "ATACANTE" if att_score > def_score else "EMPATE"
        )
        print(f"\nResultado: Defensor={def_score} ({len(state.defender_nodes)} nodos), "
              f"Atacante={att_score} ({len(state.attacker_nodes)} nodos) -> {winner} ({def_score - att_score:+d})")

# %% [markdown]
# ## 7. Visualizacion
# %%

def visualize_state(
    state: GameState,
    pos: Dict,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
):
    """
    Dibuja el grafo: azul=defensor, rojo=atacante, gris=libre.
    """
    G = state.graph
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    node_colors = []
    for n in G.nodes():
        if n in state.defender_nodes:
            node_colors.append("#4472C4")   # azul
        elif n in state.attacker_nodes:
            node_colors.append("#FF4444")   # rojo
        else:
            node_colors.append("#AAAAAA")   # gris libre

    labels = {n: f"{n}\n({G.nodes[n]['info_value']})" for n in G.nodes()}

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color="#555555")
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=800,
        edgecolors="black",
        linewidths=1.5,
    )
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7, font_color="white")

    legend_handles = [
        mpatches.Patch(color="#4472C4", label="Defensor (MAX)"),
        mpatches.Patch(color="#FF4444", label="Atacante (MIN)"),
        mpatches.Patch(color="#AAAAAA", label="Libre"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        print(f"  Guardado: {save_path}")
    if show:
        plt.show()

def visualize_game_snapshots(engine: GameEngine, save_prefix: str = "game_state"):
    """
    Visualiza snapshots: inicio, mitad y fin del juego.
    """
    snapshots = []
    n = len(engine.history)
    if n >= 1:
        snapshots.append((0, "Estado Inicial"))
    if n >= 3:
        mid = n // 2
        snapshots.append((mid, f"Estado Intermedio (turno {mid})"))
    if n >= 2:
        snapshots.append((n - 1, f"Estado Final (turno {n - 1})"))

    ncols = len(snapshots)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7))
    if ncols == 1:
        axes = [axes]

    for ax, (idx, title) in zip(axes, snapshots):
        visualize_state(engine.history[idx], engine.pos, title=title, ax=ax)

    plt.tight_layout()
    path = f"{save_prefix}_snapshots.png"
    plt.savefig(path, bbox_inches="tight", dpi=120)
    print(f"Snapshots guardados en: {path}")
    plt.close()

# %% [markdown]
# ## 8. Tabla Comparativa: Minimax vs Alpha-Beta
# %%

def print_comparison_table(metrics: List[Dict]):
    """
    Tabla comparativa de nodos expandidos: Minimax vs Alfa-Beta por turno.
    """
    print(
        f"{'Turno':>5} | {'Jugador':14s} | {'Nodos MM':>9} | {'Nodos AB':>9} | "
        f"{'Reduccion':>10} | {'Tiempo MM (ms)':>14} | {'Tiempo AB (ms)':>14}"
    )
    
    total_mm = total_ab = 0
    for m in metrics:
        print(
            f"{m['turn']:>5} | {m['role']:14s} | {m['mm_nodes']:>9,} | "
            f"{m['ab_nodes']:>9,} | {m['pruning_reduction_pct']:>9.1f}% | "
            f"{m['mm_time_ms']:>14.2f} | {m['ab_time_ms']:>14.2f}"
        )
        total_mm += m["mm_nodes"]
        total_ab += m["ab_nodes"]
    
    overall_reduction = (
        100.0 * (1 - total_ab / total_mm) if total_mm > 0 else 0.0
    )
    print(
        f"{'TOTAL':>5} | {'':14s} | {total_mm:>9,} | {total_ab:>9,} | "
        f"{overall_reduction:>9.1f}% | {'':>14} | {'':>14}"
    )
    
    print(f"\nResumen: Alpha-Beta expandio {overall_reduction:.1f}% menos nodos que Minimax puro.")
    print(f"En el mejor caso teorico, la poda alfa-beta alcanza O(b^(d/2)) vs O(b^d) de Minimax.")

def plot_comparison_chart(metrics: List[Dict], save_path: str = "comparison_chart.png"):
    """
    Grafico comparativo de nodos expandidos por turno.
    """
    turns = [m["turn"] for m in metrics]
    mm_nodes = [m["mm_nodes"] for m in metrics]
    ab_nodes = [m["ab_nodes"] for m in metrics]

    x = range(len(turns))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Grafico de nodos expandidos por turno
    ax1 = axes[0]
    bars1 = ax1.bar([xi - width / 2 for xi in x], mm_nodes, width,
                    label="Minimax puro", color="#4472C4", alpha=0.85)
    bars2 = ax1.bar([xi + width / 2 for xi in x], ab_nodes, width,
                    label="Alpha-Beta", color="#FF4444", alpha=0.85)
    ax1.set_xlabel("Turno")
    ax1.set_ylabel("Nodos Expandidos")
    ax1.set_title("Nodos Expandidos: Minimax vs Alpha-Beta por Turno")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([str(t) for t in turns])
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Grafico de porcentaje de reduccion
    ax2 = axes[1]
    reductions = [m["pruning_reduction_pct"] for m in metrics]
    ax2.bar(list(x), reductions, color="#70AD47", alpha=0.85)
    ax2.set_xlabel("Turno")
    ax2.set_ylabel("Reduccion (%)")
    ax2.set_title("Porcentaje de Reduccion de Nodos (Poda Alfa-Beta)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([str(t) for t in turns])
    ax2.set_ylim(0, 100)
    ax2.axhline(y=50, color="red", linestyle="--", alpha=0.5,
                label="50% (umbral teorico minimo con orden ideal)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    print(f"Grafico comparativo guardado en: {save_path}")
    plt.close()

# %% [markdown]
# ## 9. Funcion Principal
# %%

def main():
    """Ejecuta simulacion completa: red, juego, metricas y visualizaciones."""
    print("Task 2: Defensa adversarial (Minimax / Alpha-Beta)\n")

    G = build_network(NUM_NODES, SEED)
    print(f"Red: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas, grado promedio {sum(d for _, d in G.degree()) / G.number_of_nodes():.2f}")

    engine = GameEngine(G, use_alpha_beta_for_display=True)
    final_state = engine.run()

    print_comparison_table(engine.metrics)

    visualize_game_snapshots(engine, save_prefix="game_state")
    plot_comparison_chart(engine.metrics, save_path="comparison_chart.png")

    if engine.metrics:
        avg_mm = sum(m["mm_nodes"] for m in engine.metrics) / len(engine.metrics)
        avg_ab = sum(m["ab_nodes"] for m in engine.metrics) / len(engine.metrics)
        avg_red = sum(m["pruning_reduction_pct"] for m in engine.metrics) / len(engine.metrics)
        print(f"\nResumen: d_max={D_MAX}, {len(engine.metrics)} turnos")
        print(f"  Nodos promedio - Minimax: {avg_mm:.1f}, Alpha-Beta: {avg_ab:.1f}")
        print(f"  Reduccion promedio por poda: {avg_red:.1f}%")

    return final_state, engine

if __name__ == "__main__":
    main()
