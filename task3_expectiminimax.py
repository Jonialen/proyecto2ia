# %% [markdown]
# # Task 3 – Incertidumbre y Latencia (Expectiminimax y MDPs)
#
# ## Contexto
# El juego adversarial de captura de nodos en un grafo ahora opera en un entorno
# estocastico: cuando MAX o MIN intentan capturar un nodo, existe un **20% de
# probabilidad de que la accion falle** y el jugador pierda su turno sin capturar nada.
#
# ## Estructura del arbol Expectiminimax
# La diferencia fundamental con Minimax clasico es la insercion de **nodos de azar**
# (chance nodes) entre cada capa de decision:
#
#   MAX → Chance(p=0.8 exito, p=0.2 fallo) → MIN → Chance → MAX → ...
#
# En cada nodo de azar, el valor esperado es:
#   V(chance) = 0.8 * V(hijo_exito) + 0.2 * V(hijo_fallo)
#
# ## Reflexion MDP (escenario sin oponente)
# Si no hubiera MIN, el problema se reduce a un MDP de un solo agente:
#
#   - Estado s: frozenset de nodos capturados por MAX
#   - Accion a: intentar capturar nodo vecino a
#   - Transicion: con prob 0.8 → s' = s ∪ {a}; con prob 0.2 → s' = s (fallo)
#   - Recompensa: exito → valor(a) − 1; fallo → −1 (costo de latencia por turno)
#
# La Ecuacion de Bellman para V(s) es:
#
#   V(s) = max_a [ 0.8 * (valor(a) − 1 + γ·V(s')) + 0.2 * (−1 + γ·V(s)) ]
#
# Donde s' = s ∪ {a}.  Como V(s) aparece en ambos lados:
#   V(s)·(1 − 0.2·γ) = max_a [ 0.8·valor(a) − 1 + 0.8·γ·V(s') ]
#
# Esta es la forma canonica que usamos en value iteration mas abajo.

# %%
import random
import math
import statistics
from collections import defaultdict
from typing import Optional

PROB_EXITO   = 0.8   # probabilidad de que una accion tenga exito
PROB_FALLO   = 0.2   # probabilidad de que la accion falle (turno perdido)
D_MAX        = 4     # profundidad maxima del arbol de busqueda
NUM_NODOS    = 16    # nodos en el grafo del juego
NUM_JUEGOS   = 50    # partidas por matchup para analisis estadistico
LIMITE_TURNOS = 60   # limite de turnos para evitar juegos infinitos
SEMILLA_BASE = 42    # semilla para reproducibilidad

# %% [markdown]
# ## 1. Grafo del juego (implementado sin dependencias externas)
#
# El grafo se representa con un diccionario de adyacencia (dict[int, set])
# y un diccionario de valores de nodo (dict[int, int]).

# %%
class Grafo:
    """
    Grafo no dirigido como lista de adyacencia.
    """

    def __init__(self, num_nodos: int):
        self.num_nodos = num_nodos
        self.nodos = list(range(num_nodos))
        self.adyacencia: dict[int, set] = {n: set() for n in self.nodos}
        self.valores: dict[int, int] = {}

    def agregar_arista(self, u: int, v: int) -> None:
        self.adyacencia[u].add(v)
        self.adyacencia[v].add(u)

    def vecinos(self, nodo: int) -> set:
        return self.adyacencia[nodo]

    def valor(self, nodo: int) -> int:
        return self.valores[nodo]

    def es_conectado(self) -> bool:
        """BFS para verificar conectividad."""
        visitados = set()
        cola = [0]
        while cola:
            nodo = cola.pop()
            if nodo in visitados:
                continue
            visitados.add(nodo)
            cola.extend(self.adyacencia[nodo] - visitados)
        return len(visitados) == self.num_nodos

def crear_grafo(num_nodos: int = NUM_NODOS, semilla: int = SEMILLA_BASE) -> Grafo:
    """
    Crea grafo aleatorio conectado (Erdos-Renyi, p=0.35). Valores de nodo en [1,10].
    """
    rng = random.Random(semilla)
    intentos = 0
    while True:
        G = Grafo(num_nodos)
        
        for u in range(num_nodos):
            for v in range(u + 1, num_nodos):
                if rng.random() < 0.35:
                    G.agregar_arista(u, v)
        if G.es_conectado():
            break
        intentos += 1
        rng = random.Random(semilla + intentos)

    
    rng2 = random.Random(semilla + 100)
    for nodo in G.nodos:
        G.valores[nodo] = rng2.randint(1, 10)
    return G

# %% [markdown]
# ## 2. Estado del juego

# %%
class EstadoJuego:
    """
    Estado inmutable del juego: nodos capturados por cada jugador, turno.
    """

    __slots__ = ('grafo', 'capturados_max', 'capturados_min', 'turno', 'turno_num')

    def __init__(self, grafo: Grafo,
                 capturados_max: frozenset,
                 capturados_min: frozenset,
                 turno: str,
                 turno_num: int = 0):
        self.grafo = grafo
        self.capturados_max = capturados_max
        self.capturados_min = capturados_min
        self.turno = turno
        self.turno_num = turno_num

    def nodos_libres(self) -> frozenset:
        todos = frozenset(self.grafo.nodos)
        return todos - self.capturados_max - self.capturados_min

    def movimientos_validos(self, jugador: str) -> list:
        """
        Movimientos validos: nodos libres adyacentes a los capturados.
        """
        capturados = (self.capturados_max if jugador == 'MAX'
                      else self.capturados_min)
        libres = self.nodos_libres()
        if not capturados:
            return sorted(libres)
        vecinos_libres = set()
        for nodo in capturados:
            vecinos_libres |= (self.grafo.vecinos(nodo) & libres)
        return sorted(vecinos_libres)

    def es_terminal(self) -> bool:
        return len(self.nodos_libres()) == 0 or self.turno_num >= LIMITE_TURNOS

    def aplicar_accion(self, nodo: Optional[int], jugador: str) -> 'EstadoJuego':
        """
        Retorna nuevo estado tras la accion. None = turno perdido.
        """
        nuevo_max = self.capturados_max
        nuevo_min = self.capturados_min
        if nodo is not None:
            if jugador == 'MAX':
                nuevo_max = self.capturados_max | frozenset([nodo])
            else:
                nuevo_min = self.capturados_min | frozenset([nodo])
        siguiente = 'MIN' if jugador == 'MAX' else 'MAX'
        return EstadoJuego(
            grafo=self.grafo,
            capturados_max=nuevo_max,
            capturados_min=nuevo_min,
            turno=siguiente,
            turno_num=self.turno_num + 1,
        )

    def score_max(self) -> int:
        return sum(self.grafo.valor(n) for n in self.capturados_max)

    def score_min(self) -> int:
        return sum(self.grafo.valor(n) for n in self.capturados_min)

def estado_inicial(grafo: Grafo) -> EstadoJuego:
    return EstadoJuego(
        grafo=grafo,
        capturados_max=frozenset(),
        capturados_min=frozenset(),
        turno='MAX',
        turno_num=0,
    )

# %% [markdown]
# ## 3. Funcion de Evaluacion Heuristica (compartida con Task 2)
#
# Eval(s) aproxima el valor Minimax real combinando tres señales:
#   1. Diferencia de puntaje acumulado (mayor peso, refleja el marcador actual).
#   2. Potencial de frontera: sum valores alcanzables MAX − sum valores alcanzables MIN.
#      Indica quien tiene acceso a nodos mas valiosos en el siguiente movimiento.
#   3. Control territorial: nodos capturados MAX − nodos capturados MIN.
#
# Estos tres factores capturan las dimensiones mas relevantes del estado
# sin explorar el arbol hasta las hojas terminales.

# %%
def evaluar(estado: EstadoJuego) -> float:
    """Eval(s) heuristica para nodos no terminales."""
    # Puntaje actual
    diff_score = estado.score_max() - estado.score_min()

    # Potencial de frontera
    frontera_max = estado.movimientos_validos('MAX')
    frontera_min = estado.movimientos_validos('MIN')
    pot_max = sum(estado.grafo.valor(n) for n in frontera_max)
    pot_min = sum(estado.grafo.valor(n) for n in frontera_min)
    diff_pot = pot_max - pot_min

    # Territorio
    control = len(estado.capturados_max) - len(estado.capturados_min)

    return diff_score * 2.0 + diff_pot * 0.8 + control * 0.5

# %% [markdown]
# ## 4. Minimax con Poda Alfa-Beta (Task 2)
#
# Agente determinista que **asume exito garantizado** en cada accion.
# No modela la probabilidad de fallo; opera como si el mundo fuera perfecto.
# Se usa como baseline en el analisis comparativo.

# %%
class AgenteMinimaxAlfaBeta:
    """
    Minimax alfa-beta (mundo determinista, sin nodos de azar).
    """

    def __init__(self, d_max: int = D_MAX):
        self.d_max = d_max
        self.nodos_expandidos = 0

    def elegir_accion(self, estado: EstadoJuego) -> Optional[int]:
        self.nodos_expandidos = 0
        movimientos = estado.movimientos_validos('MAX')
        if not movimientos:
            return None
        mejor_val = -math.inf
        mejor_mov = movimientos[0]
        alfa, beta = -math.inf, math.inf
        for mov in movimientos:
            nuevo = estado.aplicar_accion(mov, 'MAX')
            val = self._min_valor(nuevo, 1, alfa, beta)
            if val > mejor_val:
                mejor_val = val
                mejor_mov = mov
            alfa = max(alfa, mejor_val)
        return mejor_mov

    def _max_valor(self, estado: EstadoJuego, prof: int,
                   alfa: float, beta: float) -> float:
        self.nodos_expandidos += 1
        if estado.es_terminal() or prof >= self.d_max:
            return evaluar(estado)
        movimientos = estado.movimientos_validos('MAX')
        if not movimientos:
            return evaluar(estado)
        val = -math.inf
        for mov in movimientos:
            val = max(val, self._min_valor(estado.aplicar_accion(mov, 'MAX'),
                                           prof + 1, alfa, beta))
            if val >= beta:
                return val  # poda beta
            alfa = max(alfa, val)
        return val

    def _min_valor(self, estado: EstadoJuego, prof: int,
                   alfa: float, beta: float) -> float:
        self.nodos_expandidos += 1
        if estado.es_terminal() or prof >= self.d_max:
            return evaluar(estado)
        movimientos = estado.movimientos_validos('MIN')
        if not movimientos:
            return evaluar(estado)
        val = math.inf
        for mov in movimientos:
            val = min(val, self._max_valor(estado.aplicar_accion(mov, 'MIN'),
                                           prof + 1, alfa, beta))
            if val <= alfa:
                return val  # poda alfa
            beta = min(beta, val)
        return val

# %% [markdown]
# ## 5. Expectiminimax (Task 3)
#
# ### Relacion con la teoria
# El arbol Expectiminimax extiende Minimax insertando **nodos de azar** despues
# de cada capa de decision.  La estructura es:
#
#   MAX → ChanceNode → MIN → ChanceNode → MAX → ...
#
# **Valor de un nodo de azar** (para accion a desde estado s, jugador MAX):
#
#   V_chance(s, a) = 0.8 · V_MIN(s ∪ {a})   <- exito: a se captura
#                  + 0.2 · V_MIN(s)           <- fallo: turno perdido, sin captura
#
# El agente MAX elige a* = argmax_a V_chance(s, a), y el agente MIN
# minimiza simetricamente.
#
# La clave es que Expectiminimax conoce la probabilidad de fallo y la
# incorpora en su evaluacion.  Minimax clasico ignora el fallo y sobrestima
# el valor de sus acciones.

# %%
class AgenteExpectiminimax:
    """
    Expectiminimax con nodos de azar (0.8 exito / 0.2 fallo).
    """

    def __init__(self, d_max: int = D_MAX):
        self.d_max = d_max
        self.nodos_expandidos = 0

    def elegir_accion(self, estado: EstadoJuego) -> Optional[int]:
        self.nodos_expandidos = 0
        movimientos = estado.movimientos_validos('MAX')
        if not movimientos:
            return None
        mejor_val = -math.inf
        mejor_mov = movimientos[0]
        for mov in movimientos:
            val = self._chance_max(estado, mov, 1)
            if val > mejor_val:
                mejor_val = val
                mejor_mov = mov
        return mejor_mov

    def _chance_max(self, estado: EstadoJuego, mov: int, prof: int) -> float:
        """
        Chance node MAX: V = 0.8*V_MIN(exito) + 0.2*V_MIN(fallo)
        """
        self.nodos_expandidos += 1
        exito = estado.aplicar_accion(mov, 'MAX')   # turno ganado
        fallo = estado.aplicar_accion(None, 'MAX')  # turno perdido
        return (PROB_EXITO * self._min_valor(exito, prof)
                + PROB_FALLO * self._min_valor(fallo, prof))

    def _chance_min(self, estado: EstadoJuego, mov: int, prof: int) -> float:
        """
        Chance node MIN: V = 0.8*V_MAX(exito) + 0.2*V_MAX(fallo)
        """
        self.nodos_expandidos += 1
        exito = estado.aplicar_accion(mov, 'MIN')
        fallo = estado.aplicar_accion(None, 'MIN')
        return (PROB_EXITO * self._max_valor(exito, prof)
                + PROB_FALLO * self._max_valor(fallo, prof))

    def _max_valor(self, estado: EstadoJuego, prof: int) -> float:
        """MAX maximiza el valor esperado."""
        self.nodos_expandidos += 1
        if estado.es_terminal() or prof >= self.d_max:
            return evaluar(estado)
        movimientos = estado.movimientos_validos('MAX')
        if not movimientos:
            return evaluar(estado)
        return max(self._chance_max(estado, mov, prof + 1) for mov in movimientos)

    def _min_valor(self, estado: EstadoJuego, prof: int) -> float:
        """MIN minimiza el valor esperado."""
        self.nodos_expandidos += 1
        if estado.es_terminal() or prof >= self.d_max:
            return evaluar(estado)
        movimientos = estado.movimientos_validos('MIN')
        if not movimientos:
            return evaluar(estado)
        return min(self._chance_min(estado, mov, prof + 1) for mov in movimientos)

# %% [markdown]
# ## 6. Agente Aleatorio

# %%
class AgenteAleatorio:
    """
    Agente aleatorio: elige movimiento uniforme al azar.
    """

    def __init__(self, rng: random.Random):
        self.rng = rng

    def elegir_accion(self, estado: EstadoJuego, jugador: str) -> Optional[int]:
        movimientos = estado.movimientos_validos(jugador)
        if not movimientos:
            return None
        return self.rng.choice(movimientos)

# %% [markdown]
# ## 7. Simulador estocastico
#
# El **entorno** aplica la incertidumbre al ejecutar cada accion decidida.
# Tanto el agente inteligente como el aleatorio son afectados por el mismo 20%
# de probabilidad de fallo — esto modela la latencia de red del enunciado.

# %%
def simular_accion(estado: EstadoJuego, nodo: Optional[int],
                   jugador: str, rng: random.Random) -> EstadoJuego:
    """
    Ejecuta accion con 80% exito / 20% fallo.
    """
    if nodo is None:
        return estado.aplicar_accion(None, jugador)
    if rng.random() < PROB_FALLO:
        return estado.aplicar_accion(None, jugador)   # fallo
    return estado.aplicar_accion(nodo, jugador)        # exito

class ResultadoPartida:
    """Resultados de una partida."""

    def __init__(self, ganador, score_max, score_min,
                 num_turnos, nodos_exp, vals_max_intentados):
        self.ganador = ganador
        self.score_max = score_max
        self.score_min = score_min
        self.diferencia = score_max - score_min
        self.num_turnos = num_turnos
        self.nodos_exp = nodos_exp
        self.vals_max_intentados = vals_max_intentados  # para analisis agresividad

def jugar_partida(agente_max, agente_min: AgenteAleatorio,
                  grafo: Grafo, rng_env: random.Random) -> ResultadoPartida:
    """
    Juega una partida completa en entorno estocastico.
    """
    estado = estado_inicial(grafo)
    nodos_exp_total = 0
    vals_intentados = []   # valores de nodos que MAX intento capturar

    while not estado.es_terminal():
        if estado.turno == 'MAX':
            nodo = agente_max.elegir_accion(estado)
            nodos_exp_total += agente_max.nodos_expandidos
            if nodo is not None:
                vals_intentados.append(grafo.valor(nodo))
            estado = simular_accion(estado, nodo, 'MAX', rng_env)
        else:
            nodo = agente_min.elegir_accion(estado, 'MIN')
            estado = simular_accion(estado, nodo, 'MIN', rng_env)

    sc_max = estado.score_max()
    sc_min = estado.score_min()
    if sc_max > sc_min:
        ganador = 'MAX'
    elif sc_min > sc_max:
        ganador = 'MIN'
    else:
        ganador = 'EMPATE'

    return ResultadoPartida(ganador, sc_max, sc_min,
                            estado.turno_num, nodos_exp_total, vals_intentados)

# %% [markdown]
# ## 8. Analisis comparativo: Minimax vs Expectiminimax

# %%
def ejecutar_analisis(num_juegos: int = NUM_JUEGOS,
                      semilla_base: int = SEMILLA_BASE) -> None:
    """
    Compara Minimax vs Expectiminimax contra agente aleatorio (50 partidas).
    """
    print(f"Analisis comparativo: {num_juegos} partidas, 20% prob. fallo\n")

    grafo = crear_grafo(NUM_NODOS, semilla=semilla_base)
    valores_grafo = [grafo.valor(n) for n in grafo.nodos]
    print(f"Grafo: {grafo.num_nodos} nodos, sum={sum(valores_grafo)}, prom={sum(valores_grafo)/len(valores_grafo):.2f}")

    res_mm: list[ResultadoPartida] = []
    res_ex: list[ResultadoPartida] = []

    for i in range(num_juegos):
        semilla_i = semilla_base + i * 31

        # Misma semilla para ambos matchups
        rng_env_mm  = random.Random(semilla_i)
        rng_min_mm  = random.Random(semilla_i + 1)
        rng_env_ex  = random.Random(semilla_i)
        rng_min_ex  = random.Random(semilla_i + 1)

        agente_mm = AgenteMinimaxAlfaBeta(d_max=D_MAX)
        agente_ex = AgenteExpectiminimax(d_max=D_MAX)

        r_mm = jugar_partida(agente_mm, AgenteAleatorio(rng_min_mm), grafo, rng_env_mm)
        r_ex = jugar_partida(agente_ex, AgenteAleatorio(rng_min_ex), grafo, rng_env_ex)

        res_mm.append(r_mm)
        res_ex.append(r_ex)

    _imprimir_resultados("MATCHUP A: Minimax alfa-beta vs Aleatorio", res_mm)
    _imprimir_resultados("MATCHUP B: Expectiminimax vs Aleatorio", res_ex)
    _analisis_agresividad(res_mm, res_ex)

def _imprimir_resultados(titulo: str, resultados: list[ResultadoPartida]) -> None:
    n = len(resultados)
    victorias = sum(1 for r in resultados if r.ganador == 'MAX')
    derrotas  = sum(1 for r in resultados if r.ganador == 'MIN')
    empates   = sum(1 for r in resultados if r.ganador == 'EMPATE')
    diffs  = [r.diferencia for r in resultados]
    turnos = [r.num_turnos for r in resultados]
    vals   = [v for r in resultados for v in r.vals_max_intentados]

    print(f"\n{titulo}")
    print(f"  Victorias: {victorias}/{n} ({100*victorias/n:.1f}%), Derrotas: {derrotas}/{n}, Empates: {empates}/{n}")
    print(f"  Dif. puntaje: prom={statistics.mean(diffs):+.2f}, mediana={statistics.median(diffs):+.2f}"
          + (f", sigma={statistics.stdev(diffs):.2f}" if n > 1 else ""))
    print(f"  Turnos por partida       – prom: {statistics.mean(turnos):.1f}")
    if vals:
        print(f"  Valor promedio de nodo intentado por MAX: {statistics.mean(vals):.2f}"
              f"  [proxy de agresividad]")

def _analisis_agresividad(res_mm: list[ResultadoPartida],
                           res_ex: list[ResultadoPartida]) -> None:
    """
    Compara agresividad: valor promedio de nodo elegido por cada agente.
    """
    vals_mm = [v for r in res_mm for v in r.vals_max_intentados]
    vals_ex = [v for r in res_ex for v in r.vals_max_intentados]
    tasa_mm = sum(1 for r in res_mm if r.ganador == 'MAX') / len(res_mm)
    tasa_ex = sum(1 for r in res_ex if r.ganador == 'MAX') / len(res_ex)

    prom_mm = statistics.mean(vals_mm) if vals_mm else 0
    prom_ex = statistics.mean(vals_ex) if vals_ex else 0

    print(f"\nAgresividad: valor promedio nodo elegido - MM={prom_mm:.2f}, EMM={prom_ex:.2f}")
    print(f"Victoria: MM={100*tasa_mm:.1f}%, EMM={100*tasa_ex:.1f}%")

    diff_vals = prom_ex - prom_mm
    if diff_vals < -0.25:
        print("Expectiminimax mas conservador: descuenta valor esperado por riesgo (0.8*v)")
    elif diff_vals > 0.25:
        print("Expectiminimax mas agresivo: busca nodos de mayor valor para compensar fallos")
    else:
        print("Agresividad similar: EMM ajusta valores (x0.8) pero mantiene la estrategia")

    delta_victoria = tasa_ex - tasa_mm
    if delta_victoria > 0.04:
        print(f"EMM gana {100*delta_victoria:.1f}% mas: modelar incertidumbre da ventaja real")
    elif delta_victoria < -0.04:
        print(f"MM gana {100*abs(delta_victoria):.1f}% mas: contra oponente debil, mundo perfecto basta")
    else:
        print("Tasas similares: la diferencia es en el razonamiento, no en victorias vs aleatorio")

# %% [markdown]
# ## 9. Demostracion de la Ecuacion de Bellman (MDP sin oponente)
#
# Resolvemos el MDP con value iteration para mostrar la politica optima
# en un subgrafo pequeño (primeros 5 nodos).
#
# Recordamos la ecuacion a resolver:
#
#   V(s) = max_a [ 0.8·(valor(a) − 1 + γ·V(s ∪ {a})) + 0.2·(−1 + γ·V(s)) ]
#
# Esta es equivalente a la forma canonica:
#   V(s)·(1 − 0.2·γ) = max_a [ 0.8·valor(a) − 1 + 0.8·γ·V(s') ]
#
# Usamos iteracion sincronica estandar hasta convergencia.

# %%
def demostrar_bellman(grafo: Grafo, gamma: float = 0.9,
                      num_iter: int = 300,
                      subgrafo_nodos: int = 5) -> None:
    """
    Value iteration en subgrafo. Muestra politica optima y valores convergidos.
    """
    print(f"\nValue iteration MDP (sin oponente): {subgrafo_nodos} nodos, gamma={gamma}")

    nodos = list(grafo.nodos)[:subgrafo_nodos]
    nodos_set = set(nodos)
    adj: dict[int, set] = {n: (grafo.vecinos(n) & nodos_set) for n in nodos}

    for n in nodos:
        print(f"  Nodo {n}: valor={grafo.valor(n)}, vecinos={sorted(adj[n])}")

    todos = frozenset(nodos)
    V: dict[frozenset, float] = defaultdict(float)

    for it in range(num_iter):
        V_nuevo: dict[frozenset, float] = {}
        max_delta = 0.0
        for bits in range(1 << len(nodos)):
            capturados = frozenset(
                nodos[i] for i in range(len(nodos)) if bits & (1 << i)
            )
            if capturados == todos:
                V_nuevo[capturados] = 0.0
                continue
            
            if not capturados:
                acciones = list(nodos)  # inicio: cualquier nodo
            else:
                acciones = sorted({v for n in capturados
                                   for v in adj[n] if v not in capturados})
            if not acciones:
                V_nuevo[capturados] = 0.0
                continue
            # V(s) = max_a [ 0.8*(R(a)-1 + γ*V(s')) + 0.2*(-1 + γ*V(s)) ]
            
            mejor = -math.inf
            for a in acciones:
                s_prima = capturados | frozenset([a])
                r_a = grafo.valor(a)
                val = (PROB_EXITO * (r_a - 1 + gamma * V[s_prima])
                       + PROB_FALLO * (-1 + gamma * V[capturados]))
                mejor = max(mejor, val)
            V_nuevo[capturados] = mejor
            max_delta = max(max_delta, abs(mejor - V.get(capturados, 0.0)))
        V = V_nuevo
        if max_delta < 1e-6 and it > 10:
            print(f"  Convergio en {it+1} iteraciones (Δ < 1e-6)")
            break

    
    vacio = frozenset()
    acciones_ini = list(nodos)
    print(f"\n  V(∅) = {V[vacio]:.4f}  (valor esperado desde el inicio)")
    print("\n  Politica optima desde ∅:")
    mejor_a = None
    mejor_v = -math.inf
    for a in acciones_ini:
        s_prima = frozenset([a])
        r_a = grafo.valor(a)
        val = (PROB_EXITO * (r_a - 1 + gamma * V[s_prima])
               + PROB_FALLO * (-1 + gamma * V[vacio]))
        marca = ""
        if val > mejor_v:
            mejor_v = val
            mejor_a = a
        print(f"    Capturar nodo {a} (valor={r_a:2d}): V_esperado = {val:.4f}")
    print(f"\nPolitica optima: capturar nodo {mejor_a} (valor={grafo.valor(mejor_a)})")

# %% [markdown]
# ## 10. Funcion Principal

# %%
def main() -> None:
    print("Task 3: Expectiminimax y MDPs\n")

    ejecutar_analisis(num_juegos=NUM_JUEGOS, semilla_base=SEMILLA_BASE)

    grafo_demo = crear_grafo(NUM_NODOS, semilla=SEMILLA_BASE)
    demostrar_bellman(grafo_demo, gamma=0.9, num_iter=300, subgrafo_nodos=5)

    print("\nResumen:")
    print("  Minimax: MAX->MIN->... asume exito garantizado")
    print("  Expectiminimax: MAX->Chance(0.8/0.2)->MIN->Chance->... pondera valor esperado")
    print("  V_chance(s,a) = 0.8*V_MIN(s+{a}) + 0.2*V_MIN(s)")
    print("  Bellman: V(s) = max_a [0.8*(val(a)-1+g*V(s')) + 0.2*(-1+g*V(s))]")

if __name__ == '__main__':
    main()
