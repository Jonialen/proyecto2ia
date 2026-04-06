"""Tests para Task 1: CSP con Factor Graphs."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import networkx as nx
from task1_csp import (
    generar_grafo_red,
    CSPRedSegura,
    BacktrackingBasico,
    BacktrackingOptimizado,
    FactorGraph,
    PROTOCOLOS,
)


class TestGrafoGeneracion:
    def test_grafo_conexo(self):
        G = generar_grafo_red(semilla=42)
        assert nx.is_connected(G)

    def test_grafo_rango_nodos(self):
        G = generar_grafo_red(num_nodos_min=15, num_nodos_max=20, semilla=42)
        assert 15 <= G.number_of_nodes() <= 20

    def test_grafo_diferentes_semillas(self):
        G1 = generar_grafo_red(semilla=1)
        G2 = generar_grafo_red(semilla=99)
        assert G1.number_of_nodes() > 0
        assert G2.number_of_nodes() > 0


class TestCSP:
    def setup_method(self):
        self.G = generar_grafo_red(semilla=42)
        self.csp = CSPRedSegura(self.G)

    def test_variables_son_nodos(self):
        assert set(self.csp.variables) == set(self.G.nodes())

    def test_dominios_tienen_4_colores(self):
        for v in self.csp.variables:
            assert len(self.csp.dominios[v]) == 4
            assert set(self.csp.dominios[v]) == set(PROTOCOLOS)

    def test_consistencia_misma_color_vecinos(self):
        """Asignar el mismo color a dos vecinos debe ser inconsistente."""
        u, v = list(self.G.edges())[0]
        asignacion = {u: "Rojo"}
        assert not self.csp.es_consistente(v, "Rojo", asignacion)

    def test_consistencia_diferente_color_vecinos(self):
        """Colores distintos en vecinos deben ser consistentes."""
        u, v = list(self.G.edges())[0]
        asignacion = {u: "Rojo"}
        assert self.csp.es_consistente(v, "Azul", asignacion)

    def test_solucion_incompleta_no_valida(self):
        asignacion = {0: "Rojo"}
        assert not self.csp.es_solucion_completa(asignacion)


class TestBacktrackingBasico:
    def setup_method(self):
        self.G = generar_grafo_red(semilla=42)
        self.csp = CSPRedSegura(self.G)

    def test_encuentra_solucion(self):
        bt = BacktrackingBasico(self.csp)
        sol = bt.resolver()
        assert sol is not None

    def test_solucion_valida(self):
        bt = BacktrackingBasico(self.csp)
        sol = bt.resolver()
        assert self.csp.es_solucion_completa(sol)

    def test_solucion_usa_solo_protocolos_validos(self):
        bt = BacktrackingBasico(self.csp)
        sol = bt.resolver()
        for v, color in sol.items():
            assert color in PROTOCOLOS

    def test_metricas_positivas(self):
        bt = BacktrackingBasico(self.csp)
        bt.resolver()
        m = bt.obtener_metricas()
        assert m["asignaciones"] > 0
        assert m["tiempo_s"] >= 0


class TestBacktrackingOptimizado:
    def setup_method(self):
        self.G = generar_grafo_red(semilla=42)
        self.csp = CSPRedSegura(self.G)

    def test_encuentra_solucion(self):
        bt = BacktrackingOptimizado(self.csp)
        sol = bt.resolver()
        assert sol is not None

    def test_solucion_valida(self):
        bt = BacktrackingOptimizado(self.csp)
        sol = bt.resolver()
        assert self.csp.es_solucion_completa(sol)

    def test_menos_asignaciones_que_basico(self):
        """El optimizado debe hacer menos asignaciones que el basico."""
        bt_b = BacktrackingBasico(self.csp)
        bt_b.resolver()
        bt_o = BacktrackingOptimizado(self.csp)
        bt_o.resolver()
        assert bt_o.obtener_metricas()["asignaciones"] <= bt_b.obtener_metricas()["asignaciones"]

    def test_misma_validez_que_basico(self):
        """Ambos algoritmos deben producir soluciones validas."""
        sol_b = BacktrackingBasico(self.csp).resolver()
        sol_o = BacktrackingOptimizado(self.csp).resolver()
        assert self.csp.es_solucion_completa(sol_b)
        assert self.csp.es_solucion_completa(sol_o)


class TestFactorGraph:
    def setup_method(self):
        self.G = generar_grafo_red(semilla=42)
        self.csp = CSPRedSegura(self.G)
        self.fg = FactorGraph(self.csp)

    def test_factores_por_arista(self):
        assert len(self.fg.nodos_factor) == self.G.number_of_edges()

    def test_solucion_valida_evalua_1(self):
        sol = BacktrackingOptimizado(self.csp).resolver()
        assert self.fg.evaluar(sol) == 1

    def test_solucion_invalida_evalua_0(self):
        """Asignar el mismo color a todos los nodos debe dar 0."""
        asignacion = {v: "Rojo" for v in self.csp.variables}
        if self.G.number_of_edges() > 0:
            assert self.fg.evaluar(asignacion) == 0


class TestGrafoPequeno:
    """Tests con grafo trivial para verificar correctitud basica."""

    def test_grafo_triangulo(self):
        """Triangulo: 3 nodos, 3 aristas. Requiere 3 colores minimo."""
        G = nx.cycle_graph(3)
        csp = CSPRedSegura(G)
        sol = BacktrackingOptimizado(csp).resolver()
        assert sol is not None
        assert csp.es_solucion_completa(sol)
        # Todos los colores deben ser distintos en un triangulo
        colores = set(sol.values())
        assert len(colores) == 3

    def test_grafo_bipartito(self):
        """Grafo bipartito completo K3,3: 2-colorable."""
        G = nx.complete_bipartite_graph(3, 3)
        csp = CSPRedSegura(G)
        sol = BacktrackingOptimizado(csp).resolver()
        assert sol is not None
        assert csp.es_solucion_completa(sol)
