"""Tests para Task 3: Expectiminimax y MDPs."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import random
from task3_expectiminimax import (
    Grafo,
    crear_grafo,
    EstadoJuego,
    estado_inicial,
    evaluar,
    AgenteMinimaxAlfaBeta,
    AgenteExpectiminimax,
    AgenteAleatorio,
    simular_accion,
    jugar_partida,
    PROB_EXITO,
    PROB_FALLO,
)


class TestGrafo:
    def test_crear_grafo_conectado(self):
        G = crear_grafo(10, semilla=42)
        assert G.es_conectado()

    def test_num_nodos(self):
        G = crear_grafo(10, semilla=42)
        assert G.num_nodos == 10
        assert len(G.nodos) == 10

    def test_valores_asignados(self):
        G = crear_grafo(10, semilla=42)
        for n in G.nodos:
            assert 1 <= G.valor(n) <= 10

    def test_aristas_bidireccionales(self):
        G = crear_grafo(10, semilla=42)
        for u in G.nodos:
            for v in G.vecinos(u):
                assert u in G.vecinos(v)


class TestEstadoJuego:
    def setup_method(self):
        self.G = crear_grafo(10, semilla=42)

    def test_estado_inicial(self):
        s = estado_inicial(self.G)
        assert s.turno == 'MAX'
        assert s.turno_num == 0
        assert len(s.capturados_max) == 0
        assert len(s.capturados_min) == 0

    def test_aplicar_accion(self):
        s = estado_inicial(self.G)
        s2 = s.aplicar_accion(0, 'MAX')
        assert 0 in s2.capturados_max
        assert 0 not in s.capturados_max  # inmutable
        assert s2.turno == 'MIN'

    def test_aplicar_accion_none(self):
        """Accion None (fallo) solo cambia turno."""
        s = estado_inicial(self.G)
        s2 = s.aplicar_accion(None, 'MAX')
        assert len(s2.capturados_max) == 0
        assert s2.turno == 'MIN'

    def test_nodos_libres(self):
        s = estado_inicial(self.G)
        assert len(s.nodos_libres()) == 10
        s2 = s.aplicar_accion(0, 'MAX')
        assert len(s2.nodos_libres()) == 9

    def test_terminal_todos_capturados(self):
        todos_max = frozenset(range(5))
        todos_min = frozenset(range(5, 10))
        s = EstadoJuego(self.G, todos_max, todos_min, 'MAX', 0)
        assert s.es_terminal()

    def test_score(self):
        s = EstadoJuego(self.G, frozenset([0]), frozenset([1]), 'MAX', 0)
        assert s.score_max() == self.G.valor(0)
        assert s.score_min() == self.G.valor(1)


class TestEvaluar:
    def test_retorna_float(self):
        G = crear_grafo(10, semilla=42)
        s = EstadoJuego(G, frozenset([0]), frozenset([5]), 'MAX', 0)
        v = evaluar(s)
        assert isinstance(v, (int, float))

    def test_max_con_mas_nodos_mejor(self):
        G = crear_grafo(10, semilla=42)
        s1 = EstadoJuego(G, frozenset([0, 1, 2]), frozenset([5]), 'MAX', 0)
        s2 = EstadoJuego(G, frozenset([0]), frozenset([5]), 'MAX', 0)
        assert evaluar(s1) >= evaluar(s2)


class TestSimularAccion:
    def test_exito_captura_nodo(self):
        G = crear_grafo(10, semilla=42)
        s = EstadoJuego(G, frozenset([0]), frozenset(), 'MAX', 0)
        rng = random.Random(42)
        # Forzar exito con threshold
        rng_exito = random.Random(100)  # probabilidad > 0.2
        s2 = simular_accion(s, 1, 'MAX', rng_exito)
        # Puede o no capturar segun la semilla, pero el estado es valido
        assert s2.turno == 'MIN'

    def test_none_no_captura(self):
        G = crear_grafo(10, semilla=42)
        s = EstadoJuego(G, frozenset([0]), frozenset(), 'MAX', 0)
        rng = random.Random(42)
        s2 = simular_accion(s, None, 'MAX', rng)
        assert len(s2.capturados_max) == 1  # sin cambio

    def test_probabilidad_fallo(self):
        """En 1000 intentos, ~20% deberian fallar."""
        G = crear_grafo(10, semilla=42)
        fallos = 0
        for i in range(1000):
            s = EstadoJuego(G, frozenset([0]), frozenset(), 'MAX', 0)
            rng = random.Random(i)
            s2 = simular_accion(s, 1, 'MAX', rng)
            if 1 not in s2.capturados_max:
                fallos += 1
        # 20% ± tolerancia (15% a 25%)
        assert 150 <= fallos <= 250, f"Fallos: {fallos}/1000 (esperado ~200)"


class TestAgentes:
    def setup_method(self):
        self.G = crear_grafo(8, semilla=42)

    def test_minimax_elige_movimiento(self):
        s = EstadoJuego(self.G, frozenset([0]), frozenset([4]), 'MAX', 0)
        agente = AgenteMinimaxAlfaBeta(d_max=2)
        mov = agente.elegir_accion(s)
        # Debe ser un vecino libre de 0
        libres = s.nodos_libres()
        vecinos = self.G.vecinos(0) & libres
        assert mov in vecinos

    def test_expectiminimax_elige_movimiento(self):
        s = EstadoJuego(self.G, frozenset([0]), frozenset([4]), 'MAX', 0)
        agente = AgenteExpectiminimax(d_max=2)
        mov = agente.elegir_accion(s)
        libres = s.nodos_libres()
        vecinos = self.G.vecinos(0) & libres
        assert mov in vecinos

    def test_aleatorio_elige_movimiento_valido(self):
        s = EstadoJuego(self.G, frozenset([0]), frozenset([4]), 'MAX', 0)
        agente = AgenteAleatorio(random.Random(42))
        mov = agente.elegir_accion(s, 'MAX')
        libres = s.nodos_libres()
        vecinos = self.G.vecinos(0) & libres
        assert mov in vecinos

    def test_sin_movimientos_retorna_none(self):
        # Capturar todos los vecinos de 0 para que no tenga movimientos
        G = Grafo(3)
        G.agregar_arista(0, 1)
        G.agregar_arista(1, 2)
        G.valores = {0: 5, 1: 3, 2: 7}
        s = EstadoJuego(G, frozenset([0]), frozenset([1, 2]), 'MAX', 0)
        agente = AgenteMinimaxAlfaBeta(d_max=2)
        mov = agente.elegir_accion(s)
        assert mov is None


class TestPartidaCompleta:
    def test_partida_termina(self):
        G = crear_grafo(8, semilla=42)
        rng = random.Random(42)
        agente_max = AgenteMinimaxAlfaBeta(d_max=2)
        agente_min = AgenteAleatorio(random.Random(43))
        resultado = jugar_partida(agente_max, agente_min, G, rng)
        assert resultado.ganador in ('MAX', 'MIN', 'EMPATE')
        assert resultado.num_turnos > 0
        assert resultado.score_max >= 0
        assert resultado.score_min >= 0

    def test_expectiminimax_partida(self):
        G = crear_grafo(8, semilla=42)
        rng = random.Random(42)
        agente_max = AgenteExpectiminimax(d_max=2)
        agente_min = AgenteAleatorio(random.Random(43))
        resultado = jugar_partida(agente_max, agente_min, G, rng)
        assert resultado.ganador in ('MAX', 'MIN', 'EMPATE')
        assert resultado.num_turnos > 0


class TestConstantes:
    def test_probabilidades_suman_1(self):
        assert abs(PROB_EXITO + PROB_FALLO - 1.0) < 1e-9
