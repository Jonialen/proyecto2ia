"""Tests para Task 2: Minimax y Alpha-Beta."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
from task2_minimax import (
    build_network,
    GameState,
    evaluate,
    minimax,
    alpha_beta,
    GameEngine,
    NUM_NODES,
    SEED,
    D_MAX,
)


class TestBuildNetwork:
    def test_nodos_correctos(self):
        G = build_network(17, 42)
        assert G.number_of_nodes() == 17

    def test_valores_info(self):
        G = build_network(17, 42)
        for n in G.nodes():
            v = G.nodes[n]["info_value"]
            assert 1 <= v <= 20

    def test_grafo_conexo(self):
        import networkx as nx
        G = build_network(17, 42)
        assert nx.is_connected(G)


class TestGameState:
    def setup_method(self):
        self.G = build_network(17, 42)

    def test_estado_inicial(self):
        s = GameState(self.G, {0}, {1}, is_max_turn=True, turn=0)
        assert 0 in s.defender_nodes
        assert 1 in s.attacker_nodes
        assert len(s.free_nodes()) == 15

    def test_apply_move_inmutable(self):
        s = GameState(self.G, {0}, {1}, is_max_turn=True, turn=0)
        s2 = s.apply_move(2)
        # Estado original no cambia
        assert 2 not in s.defender_nodes
        assert 2 in s2.defender_nodes
        assert s2.is_max_turn is False
        assert s2.turn == 1

    def test_captured_nodes(self):
        s = GameState(self.G, {0, 2}, {1, 3}, is_max_turn=True)
        assert s.captured_nodes() == {0, 1, 2, 3}

    def test_terminal_todos_capturados(self):
        todos = set(self.G.nodes())
        mitad = set(list(todos)[:len(todos)//2])
        otra = todos - mitad
        s = GameState(self.G, mitad, otra, is_max_turn=True)
        assert s.is_terminal()

    def test_available_moves_adyacentes(self):
        s = GameState(self.G, {0}, {5}, is_max_turn=True)
        moves = s.available_moves(for_defender=True)
        vecinos_0 = set(self.G.neighbors(0)) - {5}
        assert set(moves) == vecinos_0

    def test_terminal_score(self):
        s = GameState(self.G, {0}, {1}, is_max_turn=True)
        expected = self.G.nodes[0]["info_value"] - self.G.nodes[1]["info_value"]
        assert s.terminal_score() == expected


class TestEvaluate:
    def test_simetria(self):
        """Si ambos jugadores tienen los mismos nodos espejados, eval debe reflejar diferencia."""
        G = build_network(17, 42)
        s1 = GameState(G, {0}, set(), is_max_turn=True)
        s2 = GameState(G, set(), {0}, is_max_turn=True)
        # MAX con nodo 0 deberia tener eval mas alto que MIN con nodo 0
        assert evaluate(s1) > evaluate(s2)

    def test_mas_nodos_mejor(self):
        G = build_network(17, 42)
        s1 = GameState(G, {0, 1, 2}, {5}, is_max_turn=True)
        s2 = GameState(G, {0}, {5}, is_max_turn=True)
        assert evaluate(s1) >= evaluate(s2)


class TestMinimaxVsAlphaBeta:
    """Verifica que Minimax puro y Alfa-Beta dan el mismo resultado."""

    def setup_method(self):
        self.G = build_network(10, 42)  # Grafo pequeno para velocidad

    def test_mismo_valor(self):
        s = GameState(self.G, {0}, {5}, is_max_turn=True, turn=0)
        mm_counter = {"nodes": 0}
        ab_counter = {"nodes": 0}
        mm_val, mm_move = minimax(s, 3, mm_counter)
        ab_val, ab_move = alpha_beta(s, 3, -math.inf, math.inf, ab_counter)
        assert abs(mm_val - ab_val) < 1e-9

    def test_alpha_beta_menos_nodos(self):
        s = GameState(self.G, {0}, {5}, is_max_turn=True, turn=0)
        mm_counter = {"nodes": 0}
        ab_counter = {"nodes": 0}
        minimax(s, 3, mm_counter)
        alpha_beta(s, 3, -math.inf, math.inf, ab_counter)
        assert ab_counter["nodes"] <= mm_counter["nodes"]

    def test_profundidad_0_retorna_eval(self):
        s = GameState(self.G, {0}, {5}, is_max_turn=True, turn=0)
        counter = {"nodes": 0}
        val, move = minimax(s, 0, counter)
        assert move is None
        assert abs(val - evaluate(s)) < 1e-9

    def test_multiple_turnos_consistencia(self):
        """Varios turnos seguidos: alfa-beta siempre da mismo valor que minimax."""
        s = GameState(self.G, {0}, {5}, is_max_turn=True, turn=0)
        for _ in range(4):
            if s.is_terminal():
                break
            mm_c = {"nodes": 0}
            ab_c = {"nodes": 0}
            mm_val, mm_move = minimax(s, 2, mm_c)
            ab_val, ab_move = alpha_beta(s, 2, -math.inf, math.inf, ab_c)
            assert abs(mm_val - ab_val) < 1e-9
            move = ab_move if ab_move is not None else mm_move
            if move is not None:
                s = s.apply_move(move)
            else:
                break


class TestGameEngine:
    def test_juego_completo(self):
        G = build_network(10, 42)
        engine = GameEngine(G)
        final = engine.run()
        assert final.is_terminal()
        assert len(engine.metrics) > 0

    def test_metricas_registradas(self):
        G = build_network(10, 42)
        engine = GameEngine(G)
        engine.run()
        for m in engine.metrics:
            assert m["mm_nodes"] > 0
            assert m["ab_nodes"] > 0
            assert m["ab_nodes"] <= m["mm_nodes"]
