import networkx as nx
import numpy as np
from qiskit.quantum_info import SparsePauliOp

class InvalidGraphError(Exception):
    pass

class InvalidSetError(Exception):
    pass

class InvalidProblemError(Exception):
    pass

class InvalidParameterError(Exception):
    pass

class SpinGlass:

    def __init__(self):
        self.hamiltonian = None
        self.num_qubits = None
        self.problem_info = {}

    def _build_z_term(self, qubit_indices, num_qubits):
        term = ['I'] * num_qubits
        for idx in qubit_indices:
            term[idx] = 'Z'
        return ''.join(term)

    def _add_one_body_term(self, terms, idx, coeff, num_qubits):
        z_term = self._build_z_term([idx], num_qubits)
        terms.append((z_term, coeff))

    def _add_two_body_term(self, terms, idx1, idx2, coeff, num_qubits):
        z_term = self._build_z_term([idx1, idx2], num_qubits)
        terms.append((z_term, coeff))

    def _add_equality_constraint(self, terms, var_indices, target, penalty, num_qubits):
        for idx in var_indices:
            linear_coeff = penalty * (1 - 2 * target) * (-0.5)
            self._add_one_body_term(terms, idx, linear_coeff, num_qubits)
        for i_idx, i in enumerate(var_indices):
            for j in var_indices[i_idx + 1:]:
                quad_coeff = penalty * 0.25
                self._add_two_body_term(terms, i, j, quad_coeff, num_qubits)

    def fromGraph(self, G, problem, **kwargs):
        if problem == 'isomorphism':
            if not isinstance(G, tuple) or len(G) != 2:
                raise InvalidGraphError("Isomorphism requires tuple of two graphs (G1, G2).")
            G1, G2 = G
            if not isinstance(G1, nx.Graph) or not isinstance(G2, nx.Graph):
                raise InvalidGraphError("Both inputs must be valid NetworkX graphs.")
        else:
            if not isinstance(G, (nx.Graph, nx.DiGraph)):
                raise InvalidGraphError("Input is not a valid NetworkX graph.")

        problems = ['clique', 'coloring', 'covering', 'fes', 'hamiltonian', 'partition',
                    'tsp', 'clique_cover', 'minimal_maximal_matching', 'mst',
                    'steiner_tree', 'dfvs', 'ufvs', 'isomorphism']
        if problem not in problems:
            raise InvalidProblemError(f"Problem '{problem}' not supported.")

        hamiltonian_terms = []
        nodes = list(G.nodes()) if problem != 'isomorphism' else list(G[0].nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        if problem == 'clique':
            num_qubits = n
            for i in range(n):
                z_term = 'I' * i + 'Z' + 'I' * (n - i - 1)
                hamiltonian_terms.append((z_term, -1))
            for i in range(n):
                for j in range(i + 1, n):
                    if not G.has_edge(nodes[i], nodes[j]):
                        z_term = 'I' * i + 'Z' + 'I' * (j - i - 1) + 'Z' + 'I' * (n - j - 1)
                        hamiltonian_terms.append((z_term, 1))

        elif problem == 'coloring':
            k = kwargs.get('k', 3)
            num_qubits = n * k
            A = kwargs.get('penalty_A', 2.0)

            def qubit_idx(v, c):
                return v * k + c

            for v in range(n):
                color_vars = [qubit_idx(v, c) for c in range(k)]
                self._add_equality_constraint(hamiltonian_terms, color_vars, 1, A, num_qubits)
            
            for u, v in G.edges():
                u_idx, v_idx = node_to_idx[u], node_to_idx[v]
                for c in range(k):
                    qu = qubit_idx(u_idx, c)
                    qv = qubit_idx(v_idx, c)
                    self._add_one_body_term(hamiltonian_terms, qu, -A * 0.25, num_qubits)
                    self._add_one_body_term(hamiltonian_terms, qv, -A * 0.25, num_qubits)
                    self._add_two_body_term(hamiltonian_terms, qu, qv, A * 0.25, num_qubits)

        elif problem == 'covering':
            num_qubits = n
            A = kwargs.get('penalty_A', 2.0)
            B = 1.0
            for i in range(n):
                self._add_one_body_term(hamiltonian_terms, i, -B * 0.5, n)
            for u, v in G.edges():
                u_idx, v_idx = node_to_idx[u], node_to_idx[v]
                self._add_one_body_term(hamiltonian_terms, u_idx, A * 0.25, n)
                self._add_one_body_term(hamiltonian_terms, v_idx, A * 0.25, n)
                self._add_two_body_term(hamiltonian_terms, u_idx, v_idx, A * 0.25, n)

        elif problem == 'fes':
            edges = list(G.edges())
            m = len(edges)
            edge_to_idx = {(min(u, v), max(u, v)): i for i, (u, v) in enumerate(edges)}
            A = kwargs.get('penalty_A', 2.0)
            B = 1.0

            cycles = nx.cycle_basis(G)

            slack_bits_per_cycle = []
            for cycle in cycles:
                cycle_len = len(cycle)
                max_excess = max(0, cycle_len - 1)
                bits = int(np.ceil(np.log2(max_excess + 1))) if max_excess > 0 else 0
                slack_bits_per_cycle.append(bits)

            total_slack = sum(slack_bits_per_cycle)
            num_qubits = m + total_slack

            for i in range(m):
                self._add_one_body_term(hamiltonian_terms, i, -B * 0.5, num_qubits)

            slack_start = m
            for cycle_idx, cycle in enumerate(cycles):
                cycle_edges = []
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    edge_key = (min(u, v), max(u, v))
                    if edge_key in edge_to_idx:
                        cycle_edges.append(edge_to_idx[edge_key])

                if not cycle_edges:
                    continue

                bits = slack_bits_per_cycle[cycle_idx]
                slack_indices = list(range(slack_start, slack_start + bits))
                slack_coeffs = [2 ** k for k in range(bits)]
                target = 1

                for e in cycle_edges:
                    self._add_one_body_term(hamiltonian_terms, e, A * (1 - 2 * target) * (-0.5), num_qubits)
                
                for k, slack_idx in enumerate(slack_indices):
                    coeff = -slack_coeffs[k]
                    linear_coeff = A * coeff * (coeff - 2 * target) * 0.25
                    self._add_one_body_term(hamiltonian_terms, slack_idx, linear_coeff, num_qubits)

                for i_idx, e1 in enumerate(cycle_edges):
                    for e2 in cycle_edges[i_idx + 1:]:
                        self._add_two_body_term(hamiltonian_terms, e1, e2, A * 0.25, num_qubits)

                for e in cycle_edges:
                    for k, slack_idx in enumerate(slack_indices):
                        coeff = 1 * (-slack_coeffs[k])
                        self._add_two_body_term(hamiltonian_terms, e, slack_idx, A * coeff * 0.25, num_qubits)

                for k1_idx, slack_idx1 in enumerate(slack_indices):
                    for k2_idx in range(k1_idx + 1, len(slack_indices)):
                        slack_idx2 = slack_indices[k2_idx]
                        coeff = slack_coeffs[k1_idx] * slack_coeffs[k2_idx]
                        self._add_two_body_term(hamiltonian_terms, slack_idx1, slack_idx2, A * coeff * 0.25, num_qubits)

                slack_start += bits

        elif problem == 'hamiltonian':
            num_qubits = n * n
            A = kwargs.get('penalty_A', n + 1)
            def qubit_idx(v, p):
                return v * n + p
            for v in range(n):
                for p in range(n):
                    idx = qubit_idx(v, p)
                    self._add_one_body_term(hamiltonian_terms, idx, A * 0.5, num_qubits)
                for p1 in range(n):
                    for p2 in range(p1 + 1, n):
                        self._add_two_body_term(hamiltonian_terms, qubit_idx(v, p1),
                                               qubit_idx(v, p2), A * 0.25, num_qubits)
            for p in range(n):
                for v in range(n):
                    idx = qubit_idx(v, p)
                    self._add_one_body_term(hamiltonian_terms, idx, A * 0.5, num_qubits)
                for v1 in range(n):
                    for v2 in range(v1 + 1, n):
                        self._add_two_body_term(hamiltonian_terms, qubit_idx(v1, p),
                                               qubit_idx(v2, p), A * 0.25, num_qubits)
            for p in range(n):
                p_next = (p + 1) % n
                for u in range(n):
                    for v in range(n):
                        if u != v and not G.has_edge(nodes[u], nodes[v]):
                            self._add_two_body_term(hamiltonian_terms, qubit_idx(u, p),
                                                   qubit_idx(v, p_next), A * 0.25, num_qubits)

        elif problem == 'partition':
            num_qubits = n
            A = kwargs.get('penalty_A', 1.0)
            for i in range(n):
                self._add_one_body_term(hamiltonian_terms, i, A * 0.5, n)
            for i in range(n):
                for j in range(i + 1, n):
                    self._add_two_body_term(hamiltonian_terms, i, j, A * 0.25, n)
            for u, v in G.edges():
                u_idx, v_idx = node_to_idx[u], node_to_idx[v]
                self._add_one_body_term(hamiltonian_terms, u_idx, -0.5, n)
                self._add_one_body_term(hamiltonian_terms, v_idx, -0.5, n)
                self._add_two_body_term(hamiltonian_terms, u_idx, v_idx, 0.5, n)

        elif problem == 'tsp':
            num_qubits = n * n
            max_weight = max([G[u][v].get('weight', 1) for u, v in G.edges()], default=1)
            A = kwargs.get('penalty_A', max_weight * n + 1)
            def qubit_idx(v, p):
                return v * n + p
            for v in range(n):
                for p in range(n):
                    idx = qubit_idx(v, p)
                    self._add_one_body_term(hamiltonian_terms, idx, A * 0.5, num_qubits)
                for p1 in range(n):
                    for p2 in range(p1 + 1, n):
                        self._add_two_body_term(hamiltonian_terms, qubit_idx(v, p1),
                                               qubit_idx(v, p2), A * 0.25, num_qubits)
            for p in range(n):
                for v in range(n):
                    idx = qubit_idx(v, p)
                    self._add_one_body_term(hamiltonian_terms, idx, A * 0.5, num_qubits)
                for v1 in range(n):
                    for v2 in range(v1 + 1, n):
                        self._add_two_body_term(hamiltonian_terms, qubit_idx(v1, p),
                                               qubit_idx(v2, p), A * 0.25, num_qubits)
            for p in range(n):
                p_next = (p + 1) % n
                for u in range(n):
                    for v in range(n):
                        if u != v and not G.has_edge(nodes[u], nodes[v]):
                            self._add_two_body_term(hamiltonian_terms, qubit_idx(u, p),
                                                   qubit_idx(v, p_next), A * 0.25, num_qubits)
            for u, v, data in G.edges(data=True):
                weight = data.get('weight', 1)
                u_idx, v_idx = node_to_idx[u], node_to_idx[v]
                for p in range(n):
                    p_next = (p + 1) % n
                    self._add_one_body_term(hamiltonian_terms, qubit_idx(u_idx, p),
                                           -weight * 0.25, num_qubits)
                    self._add_one_body_term(hamiltonian_terms, qubit_idx(v_idx, p_next),
                                           -weight * 0.25, num_qubits)
                    self._add_two_body_term(hamiltonian_terms, qubit_idx(u_idx, p),
                                           qubit_idx(v_idx, p_next), weight * 0.25, num_qubits)
                    self._add_one_body_term(hamiltonian_terms, qubit_idx(v_idx, p),
                                           -weight * 0.25, num_qubits)
                    self._add_one_body_term(hamiltonian_terms, qubit_idx(u_idx, p_next),
                                           -weight * 0.25, num_qubits)
                    self._add_two_body_term(hamiltonian_terms, qubit_idx(v_idx, p),
                                           qubit_idx(u_idx, p_next), weight * 0.25, num_qubits)

        elif problem == 'clique_cover':
            k = kwargs.get('k', 2)
            num_qubits = n * k
            A = kwargs.get('penalty_A', 2.0)
            def qubit_idx(v, c):
                return v * k + c
            for v in range(n):
                for c in range(k):
                    idx = qubit_idx(v, c)
                    self._add_one_body_term(hamiltonian_terms, idx, A * 0.5, num_qubits)
                for c1 in range(k):
                    for c2 in range(c1 + 1, k):
                        self._add_two_body_term(hamiltonian_terms, qubit_idx(v, c1),
                                               qubit_idx(v, c2), A * 0.25, num_qubits)
            for i in range(n):
                for j in range(i + 1, n):
                    if not G.has_edge(nodes[i], nodes[j]):
                        for c in range(k):
                            self._add_two_body_term(hamiltonian_terms, qubit_idx(i, c),
                                                   qubit_idx(j, c), A * 0.25, num_qubits)

        elif problem == 'minimal_maximal_matching':
            edges = list(G.edges())
            m = len(edges)
            num_qubits = m
            edge_to_idx = {(min(u, v), max(u, v)): i for i, (u, v) in enumerate(edges)}
            A = kwargs.get('penalty_A', 2.0)
            B = 1.0
            for i in range(m):
                self._add_one_body_term(hamiltonian_terms, i, -B * 0.5, m)
            for v in nodes:
                incident_edges = []
                for u, w in edges:
                    if v == u or v == w:
                        edge_key = (min(u, w), max(u, w))
                        incident_edges.append(edge_to_idx[edge_key])
                for i_idx, e1 in enumerate(incident_edges):
                    for e2 in incident_edges[i_idx + 1:]:
                        self._add_two_body_term(hamiltonian_terms, e1, e2, A * 0.25, m)
            for v in nodes:
                incident_edges = []
                for u, w in edges:
                    if v == u or v == w:
                        edge_key = (min(u, w), max(u, w))
                        incident_edges.append(edge_to_idx[edge_key])
                if incident_edges:
                    for e in incident_edges:
                        self._add_one_body_term(hamiltonian_terms, e, A * 0.5, m)

        elif problem == 'mst':
            edges = list(G.edges(data=True))
            m = len(edges)
            edge_to_idx = {(min(u, v), max(u, v)): i for i, (u, v, _) in enumerate(edges)}
            total_weight = sum([e[2].get('weight', 1) for e in edges])
            A = kwargs.get('penalty_A', total_weight + 1)
            B = 1.0

            cycles = nx.cycle_basis(G)

            slack_bits_per_cycle = []
            for cycle in cycles:
                cycle_len = len(cycle)
                max_slack = cycle_len - 1
                bits = int(np.ceil(np.log2(max_slack + 1))) if max_slack > 0 else 0
                slack_bits_per_cycle.append(bits)

            total_slack = sum(slack_bits_per_cycle)
            num_qubits = m + total_slack

            for i, (u, v, data) in enumerate(edges):
                weight = data.get('weight', 1)
                self._add_one_body_term(hamiltonian_terms, i, -B * weight * 0.5, num_qubits)

            target = n - 1
            for i in range(m):
                self._add_one_body_term(hamiltonian_terms, i, A * (1 - 2 * target) * (-0.5), num_qubits)
            for i in range(m):
                for j in range(i + 1, m):
                    self._add_two_body_term(hamiltonian_terms, i, j, A * 0.25, num_qubits)

            slack_start = m
            for cycle_idx, cycle in enumerate(cycles):
                cycle_edges = []
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    edge_key = (min(u, v), max(u, v))
                    if edge_key in edge_to_idx:
                        cycle_edges.append(edge_to_idx[edge_key])

                if not cycle_edges:
                    continue

                cycle_target = len(cycle) - 1
                bits = slack_bits_per_cycle[cycle_idx]
                slack_indices = list(range(slack_start, slack_start + bits))
                slack_coeffs = [2 ** k for k in range(bits)]

                for e in cycle_edges:
                    self._add_one_body_term(hamiltonian_terms, e, A * (1 - 2 * cycle_target) * (-0.5), num_qubits)

                for k, slack_idx in enumerate(slack_indices):
                    coeff = slack_coeffs[k]
                    linear_coeff = A * coeff * (coeff - 2 * cycle_target) * 0.25
                    self._add_one_body_term(hamiltonian_terms, slack_idx, linear_coeff, num_qubits)

                for i_idx, e1 in enumerate(cycle_edges):
                    for e2 in cycle_edges[i_idx + 1:]:
                        self._add_two_body_term(hamiltonian_terms, e1, e2, A * 0.25, num_qubits)

                for e in cycle_edges:
                    for k, slack_idx in enumerate(slack_indices):
                        coeff = 1 * slack_coeffs[k]
                        self._add_two_body_term(hamiltonian_terms, e, slack_idx, A * coeff * 0.25, num_qubits)

                for k1_idx, slack_idx1 in enumerate(slack_indices):
                    for k2_idx in range(k1_idx + 1, len(slack_indices)):
                        slack_idx2 = slack_indices[k2_idx]
                        coeff = slack_coeffs[k1_idx] * slack_coeffs[k2_idx]
                        self._add_two_body_term(hamiltonian_terms, slack_idx1, slack_idx2, A * coeff * 0.25, num_qubits)

                slack_start += bits

        elif problem == 'steiner_tree':
            terminals = kwargs.get('terminals', list(G.nodes()))
            terminal_set = set(terminals)
            edges = list(G.edges(data=True))
            m = len(edges)
            edge_to_idx = {(min(u, v), max(u, v)): i for i, (u, v, _) in enumerate(edges)}
            total_weight = sum([e[2].get('weight', 1) for e in edges])
            A = kwargs.get('penalty_A', total_weight + 1)
            B = 1.0

            cycles = nx.cycle_basis(G)

            terminal_incident = {}
            slack_bits_terminal = []
            for t in terminals:
                incident = []
                for u, v, _ in edges:
                    if t == u or t == v:
                        edge_key = (min(u, v), max(u, v))
                        incident.append(edge_to_idx[edge_key])
                terminal_incident[t] = incident
                max_excess = max(0, len(incident) - 1)
                bits = int(np.ceil(np.log2(max_excess + 1))) if max_excess > 0 else 0
                slack_bits_terminal.append(bits)

            slack_bits_cycle = []
            for cycle in cycles:
                cycle_len = len(cycle)
                max_slack = cycle_len - 1
                bits = int(np.ceil(np.log2(max_slack + 1))) if max_slack > 0 else 0
                slack_bits_cycle.append(bits)

            total_slack = sum(slack_bits_terminal) + sum(slack_bits_cycle)
            num_qubits = m + total_slack

            for i, (u, v, data) in enumerate(edges):
                weight = data.get('weight', 1)
                self._add_one_body_term(hamiltonian_terms, i, -B * weight * 0.5, num_qubits)

            slack_start = m
            for t_idx, t in enumerate(terminals):
                incident = terminal_incident[t]
                if not incident:
                    continue

                bits = slack_bits_terminal[t_idx]
                slack_indices = list(range(slack_start, slack_start + bits))
                slack_coeffs = [2 ** k for k in range(bits)]
                target = 1

                for e in incident:
                    self._add_one_body_term(hamiltonian_terms, e, A * (1 - 2 * target) * (-0.5), num_qubits)

                for k, slack_idx in enumerate(slack_indices):
                    coeff = -slack_coeffs[k]
                    linear_coeff = A * coeff * (coeff - 2 * target) * 0.25
                    self._add_one_body_term(hamiltonian_terms, slack_idx, linear_coeff, num_qubits)

                for i_idx, e1 in enumerate(incident):
                    for e2 in incident[i_idx + 1:]:
                        self._add_two_body_term(hamiltonian_terms, e1, e2, A * 0.25, num_qubits)

                for e in incident:
                    for k, slack_idx in enumerate(slack_indices):
                        coeff = 1 * (-slack_coeffs[k])
                        self._add_two_body_term(hamiltonian_terms, e, slack_idx, A * coeff * 0.25, num_qubits)

                for k1_idx, slack_idx1 in enumerate(slack_indices):
                    for k2_idx in range(k1_idx + 1, len(slack_indices)):
                        slack_idx2 = slack_indices[k2_idx]
                        coeff = slack_coeffs[k1_idx] * slack_coeffs[k2_idx]
                        self._add_two_body_term(hamiltonian_terms, slack_idx1, slack_idx2, A * coeff * 0.25, num_qubits)

                slack_start += bits

            for cycle_idx, cycle in enumerate(cycles):
                cycle_edges = []
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    edge_key = (min(u, v), max(u, v))
                    if edge_key in edge_to_idx:
                        cycle_edges.append(edge_to_idx[edge_key])

                if not cycle_edges:
                    continue

                cycle_target = len(cycle) - 1
                bits = slack_bits_cycle[cycle_idx]
                slack_indices = list(range(slack_start, slack_start + bits))
                slack_coeffs = [2 ** k for k in range(bits)]

                for e in cycle_edges:
                    self._add_one_body_term(hamiltonian_terms, e, A * (1 - 2 * cycle_target) * (-0.5), num_qubits)

                for k, slack_idx in enumerate(slack_indices):
                    coeff = slack_coeffs[k]
                    linear_coeff = A * coeff * (coeff - 2 * cycle_target) * 0.25
                    self._add_one_body_term(hamiltonian_terms, slack_idx, linear_coeff, num_qubits)

                for i_idx, e1 in enumerate(cycle_edges):
                    for e2 in cycle_edges[i_idx + 1:]:
                        self._add_two_body_term(hamiltonian_terms, e1, e2, A * 0.25, num_qubits)

                for e in cycle_edges:
                    for k, slack_idx in enumerate(slack_indices):
                        coeff = 1 * slack_coeffs[k]
                        self._add_two_body_term(hamiltonian_terms, e, slack_idx, A * coeff * 0.25, num_qubits)

                for k1_idx, slack_idx1 in enumerate(slack_indices):
                    for k2_idx in range(k1_idx + 1, len(slack_indices)):
                        slack_idx2 = slack_indices[k2_idx]
                        coeff = slack_coeffs[k1_idx] * slack_coeffs[k2_idx]
                        self._add_two_body_term(hamiltonian_terms, slack_idx1, slack_idx2, A * coeff * 0.25, num_qubits)

                slack_start += bits

        elif problem == 'dfvs':
            if not isinstance(G, nx.DiGraph):
                raise InvalidGraphError("DFVS requires a directed graph (DiGraph).")

            A = kwargs.get('penalty_A', n + 1)
            B = 1.0

            try:
                cycles = list(nx.simple_cycles(G))
            except:
                cycles = []

            slack_bits_per_cycle = []
            for cycle in cycles:
                cycle_len = len(cycle)
                max_excess = max(0, cycle_len - 1)
                bits = int(np.ceil(np.log2(max_excess + 1))) if max_excess > 0 else 0
                slack_bits_per_cycle.append(bits)

            total_slack = sum(slack_bits_per_cycle)
            num_qubits = n + total_slack

            for i in range(n):
                self._add_one_body_term(hamiltonian_terms, i, -B * 0.5, num_qubits)

            slack_start = n
            for cycle_idx, cycle in enumerate(cycles):
                cycle_indices = [node_to_idx[v] for v in cycle]
                if not cycle_indices:
                    continue

                bits = slack_bits_per_cycle[cycle_idx]
                slack_indices = list(range(slack_start, slack_start + bits))
                slack_coeffs = [2 ** k for k in range(bits)]
                target = 1

                for v in cycle_indices:
                    self._add_one_body_term(hamiltonian_terms, v, A * (1 - 2 * target) * (-0.5), num_qubits)

                for k, slack_idx in enumerate(slack_indices):
                    coeff = -slack_coeffs[k]
                    linear_coeff = A * coeff * (coeff - 2 * target) * 0.25
                    self._add_one_body_term(hamiltonian_terms, slack_idx, linear_coeff, num_qubits)

                for i_idx, v1 in enumerate(cycle_indices):
                    for v2 in cycle_indices[i_idx + 1:]:
                        self._add_two_body_term(hamiltonian_terms, v1, v2, A * 0.25, num_qubits)

                for v in cycle_indices:
                    for k, slack_idx in enumerate(slack_indices):
                        coeff = 1 * (-slack_coeffs[k])
                        self._add_two_body_term(hamiltonian_terms, v, slack_idx, A * coeff * 0.25, num_qubits)

                for k1_idx, slack_idx1 in enumerate(slack_indices):
                    for k2_idx in range(k1_idx + 1, len(slack_indices)):
                        slack_idx2 = slack_indices[k2_idx]
                        coeff = slack_coeffs[k1_idx] * slack_coeffs[k2_idx]
                        self._add_two_body_term(hamiltonian_terms, slack_idx1, slack_idx2, A * coeff * 0.25, num_qubits)

                slack_start += bits

        elif problem == 'ufvs':
            A = kwargs.get('penalty_A', n + 1)
            B = 1.0

            cycles = nx.cycle_basis(G)

            slack_bits_per_cycle = []
            for cycle in cycles:
                cycle_len = len(cycle)
                max_excess = max(0, cycle_len - 1)
                bits = int(np.ceil(np.log2(max_excess + 1))) if max_excess > 0 else 0
                slack_bits_per_cycle.append(bits)

            total_slack = sum(slack_bits_per_cycle)
            num_qubits = n + total_slack

            for i in range(n):
                self._add_one_body_term(hamiltonian_terms, i, -B * 0.5, num_qubits)

            slack_start = n
            for cycle_idx, cycle in enumerate(cycles):
                cycle_indices = [node_to_idx[v] for v in cycle]
                if not cycle_indices:
                    continue

                bits = slack_bits_per_cycle[cycle_idx]
                slack_indices = list(range(slack_start, slack_start + bits))
                slack_coeffs = [2 ** k for k in range(bits)]
                target = 1

                for v in cycle_indices:
                    self._add_one_body_term(hamiltonian_terms, v, A * (1 - 2 * target) * (-0.5), num_qubits)

                for k, slack_idx in enumerate(slack_indices):
                    coeff = -slack_coeffs[k]
                    linear_coeff = A * coeff * (coeff - 2 * target) * 0.25
                    self._add_one_body_term(hamiltonian_terms, slack_idx, linear_coeff, num_qubits)

                for i_idx, v1 in enumerate(cycle_indices):
                    for v2 in cycle_indices[i_idx + 1:]:
                        self._add_two_body_term(hamiltonian_terms, v1, v2, A * 0.25, num_qubits)

                for v in cycle_indices:
                    for k, slack_idx in enumerate(slack_indices):
                        coeff = 1 * (-slack_coeffs[k])
                        self._add_two_body_term(hamiltonian_terms, v, slack_idx, A * coeff * 0.25, num_qubits)
                
                for k1_idx, slack_idx1 in enumerate(slack_indices):
                    for k2_idx in range(k1_idx + 1, len(slack_indices)):
                        slack_idx2 = slack_indices[k2_idx]
                        coeff = slack_coeffs[k1_idx] * slack_coeffs[k2_idx]
                        self._add_two_body_term(hamiltonian_terms, slack_idx1, slack_idx2, A * coeff * 0.25, num_qubits)

                slack_start += bits

        elif problem == 'isomorphism':
            G1, G2 = G
            n1 = G1.number_of_nodes()
            n2 = G2.number_of_nodes()
            if n1 != n2:
                raise InvalidGraphError("Graphs must have same number of vertices.")
            nodes1 = list(G1.nodes())
            nodes2 = list(G2.nodes())
            num_qubits = n1 * n1
            A = kwargs.get('penalty_A', 2.0)
            def qubit_idx(u, v):
                return u * n1 + v
            for u in range(n1):
                for v in range(n1):
                    idx = qubit_idx(u, v)
                    self._add_one_body_term(hamiltonian_terms, idx, A * 0.5, num_qubits)
                for v1 in range(n1):
                    for v2 in range(v1 + 1, n1):
                        self._add_two_body_term(hamiltonian_terms, qubit_idx(u, v1),
                                               qubit_idx(u, v2), A * 0.25, num_qubits)
            for v in range(n1):
                for u in range(n1):
                    idx = qubit_idx(u, v)
                    self._add_one_body_term(hamiltonian_terms, idx, A * 0.5, num_qubits)
                for u1 in range(n1):
                    for u2 in range(u1 + 1, n1):
                        self._add_two_body_term(hamiltonian_terms, qubit_idx(u1, v),
                                               qubit_idx(u2, v), A * 0.25, num_qubits)
            for u in range(n1):
                for u_prime in range(u + 1, n1):
                    edge_in_G1 = G1.has_edge(nodes1[u], nodes1[u_prime])
                    for v in range(n1):
                        for v_prime in range(n1):
                            if v != v_prime:
                                edge_in_G2 = G2.has_edge(nodes2[v], nodes2[v_prime])
                                if edge_in_G1 != edge_in_G2:
                                    self._add_two_body_term(hamiltonian_terms,
                                                          qubit_idx(u, v),
                                                          qubit_idx(u_prime, v_prime),
                                                          A * 0.25, num_qubits)

        else:
            raise InvalidProblemError(f"Problem {problem} is not yet implemented.")

        self.num_qubits = num_qubits
        if hamiltonian_terms:
            pauli_terms, coeffs = zip(*hamiltonian_terms)
            self.hamiltonian = SparsePauliOp.from_list(
                list(zip(pauli_terms, coeffs))
            )

    def fromSet(self, S, problem, **kwargs):
        problems = ['partition', 'exact_cover', 'set_packing', 'set_cover', 'sat', 'knapsack']
        if problem not in problems:
            raise InvalidProblemError(f"Problem '{problem}' not supported.")

        hamiltonian_terms = []

        if problem == 'partition':
            if isinstance(S, list):
                if len(set(S)) != len(S):
                    raise InvalidSetError("Set contains duplicate elements.")
            elif isinstance(S, dict):
                S = S.get('elements', [])
            elif not isinstance(S, (set, list)):
                raise InvalidSetError("Input is not a valid set or list.")
            N = len(S)
            S_list = list(S)
            num_qubits = N
            for i in range(N):
                for j in range(i + 1, N):
                    coeff = 2 * S_list[i] * S_list[j]
                    z_term = 'I' * i + 'Z' + 'I' * (j - i - 1) + 'Z' + 'I' * (N - j - 1)
                    hamiltonian_terms.append((z_term, coeff))

        elif problem == 'exact_cover':
            if not isinstance(S, dict):
                raise InvalidSetError("Exact cover requires dict with 'universe' and 'subsets'.")
            universe = S.get('universe', [])
            subsets = S.get('subsets', [])
            m = len(subsets)
            num_qubits = m
            A = kwargs.get('penalty_A', 1.0)
            for alpha in universe:
                containing = [i for i, subset in enumerate(subsets) if alpha in subset]
                for i in containing:
                    self._add_one_body_term(hamiltonian_terms, i, A * 0.5, m)
                for i_idx, i in enumerate(containing):
                    for j in containing[i_idx + 1:]:
                        self._add_two_body_term(hamiltonian_terms, i, j, A * 0.25, m)

        elif problem == 'set_packing':
            if not isinstance(S, dict):
                raise InvalidSetError("Set packing requires dict with 'universe' and 'subsets'.")
            universe = S.get('universe', [])
            subsets = S.get('subsets', [])
            m = len(subsets)
            num_qubits = m
            A = kwargs.get('penalty_A', 2.0)
            B = 1.0
            for i in range(m):
                self._add_one_body_term(hamiltonian_terms, i, B * 0.5, m)
            for alpha in universe:
                containing = [i for i, subset in enumerate(subsets) if alpha in subset]
                for i_idx, i in enumerate(containing):
                    for j in containing[i_idx + 1:]:
                        self._add_two_body_term(hamiltonian_terms, i, j, A * 0.25, m)

        elif problem == 'set_cover':
            if not isinstance(S, dict):
                raise InvalidSetError("Set cover requires dict with 'universe' and 'subsets'.")
            universe = S.get('universe', [])
            subsets = S.get('subsets', [])
            m = len(subsets)
            A = kwargs.get('penalty_A', 2.0)
            B = 1.0

            slack_bits_per_element = []
            for alpha in universe:
                containing = [i for i, subset in enumerate(subsets) if alpha in subset]
                max_excess = max(0, len(containing) - 1)
                bits = int(np.ceil(np.log2(max_excess + 1))) if max_excess > 0 else 0
                slack_bits_per_element.append(bits)

            total_slack_bits = sum(slack_bits_per_element)
            num_qubits = m + total_slack_bits

            for i in range(m):
                self._add_one_body_term(hamiltonian_terms, i, -B * 0.5, num_qubits)

            slack_start = m
            for elem_idx, alpha in enumerate(universe):
                containing = [i for i, subset in enumerate(subsets) if alpha in subset]
                if not containing:
                    continue

                bits = slack_bits_per_element[elem_idx]
                target = 1

                slack_indices = list(range(slack_start, slack_start + bits))
                slack_coeffs = [2 ** k for k in range(bits)]

                for i in containing:
                    self._add_one_body_term(hamiltonian_terms, i, A * (1 - 2 * target) * (-0.5), num_qubits)

                for k, slack_idx in enumerate(slack_indices):
                    coeff = -slack_coeffs[k]
                    linear_coeff = A * coeff * (coeff - 2 * target) * 0.25
                    self._add_one_body_term(hamiltonian_terms, slack_idx, linear_coeff, num_qubits)

                for i_idx, i in enumerate(containing):
                    for j in containing[i_idx + 1:]:
                        self._add_two_body_term(hamiltonian_terms, i, j, A * 0.25, num_qubits)

                for i in containing:
                    for k, slack_idx in enumerate(slack_indices):
                        coeff = 1 * (-slack_coeffs[k])
                        self._add_two_body_term(hamiltonian_terms, i, slack_idx, A * coeff * 0.25, num_qubits)

                for k1_idx, slack_idx1 in enumerate(slack_indices):
                    for k2_idx in range(k1_idx + 1, len(slack_indices)):
                        slack_idx2 = slack_indices[k2_idx]
                        coeff = slack_coeffs[k1_idx] * slack_coeffs[k2_idx]
                        self._add_two_body_term(hamiltonian_terms, slack_idx1, slack_idx2, A * coeff * 0.25, num_qubits)

                slack_start += bits

        elif problem == 'sat':
            if not isinstance(S, dict):
                raise InvalidSetError("SAT requires dict with 'num_variables' and 'clauses'.")
            n = S.get('num_variables', 0)
            clauses = S.get('clauses', [])
            A = kwargs.get('penalty_A', 1.0)

            num_3_clauses = sum(1 for c in clauses if len(c) == 3)
            num_qubits = n + num_3_clauses
            ancilla_idx = n

            for clause in clauses:
                literals = []
                for var, negated in clause:
                    sign = -1 if negated else 1
                    literals.append((var, sign))

                if len(literals) == 1:
                    var, sign = literals[0]
                    self._add_one_body_term(hamiltonian_terms, var, A * sign * 0.5, num_qubits)

                elif len(literals) == 2:
                    v1, s1 = literals[0]
                    v2, s2 = literals[1]
                    self._add_one_body_term(hamiltonian_terms, v1, A * s1 * 0.25, num_qubits)
                    self._add_one_body_term(hamiltonian_terms, v2, A * s2 * 0.25, num_qubits)
                    self._add_two_body_term(hamiltonian_terms, v1, v2, A * s1 * s2 * 0.25, num_qubits)

                elif len(literals) == 3:
                    v1, s1 = literals[0]
                    v2, s2 = literals[1]
                    v3, s3 = literals[2]
                    anc = ancilla_idx
                    ancilla_idx += 1

                    P = A * 3

                    self._add_one_body_term(hamiltonian_terms, v1, P * s1 * 0.25, num_qubits)
                    self._add_one_body_term(hamiltonian_terms, v2, P * s2 * 0.25, num_qubits)
                    self._add_two_body_term(hamiltonian_terms, v1, v2, P * s1 * s2 * 0.25, num_qubits)

                    self._add_one_body_term(hamiltonian_terms, v1, -P * s1 * 0.5, num_qubits)
                    self._add_one_body_term(hamiltonian_terms, anc, P * 0.5, num_qubits)
                    self._add_two_body_term(hamiltonian_terms, v1, anc, P * s1 * 0.5, num_qubits)

                    self._add_one_body_term(hamiltonian_terms, v2, -P * s2 * 0.5, num_qubits)
                    self._add_one_body_term(hamiltonian_terms, anc, P * 0.5, num_qubits)
                    self._add_two_body_term(hamiltonian_terms, v2, anc, P * s2 * 0.5, num_qubits)

                    self._add_one_body_term(hamiltonian_terms, anc, -P * 1.5, num_qubits)

                    self._add_one_body_term(hamiltonian_terms, anc, -A * 0.25, num_qubits)
                    self._add_one_body_term(hamiltonian_terms, v3, A * s3 * 0.25, num_qubits)
                    self._add_two_body_term(hamiltonian_terms, anc, v3, -A * s3 * 0.25, num_qubits)

        elif problem == 'knapsack':
            if not isinstance(S, dict):
                raise InvalidSetError("Knapsack requires dict with 'weights', 'values', 'capacity'.")
            weights = S.get('weights', [])
            values = S.get('values', [])
            capacity = S.get('capacity', 0)
            n = len(weights)
            K = int(np.ceil(np.log2(capacity + 1))) if capacity > 0 else 1
            num_qubits = n + K
            A = kwargs.get('penalty_A', max(values) + 1 if values else 1)
            B = 1.0
            for i in range(n):
                self._add_one_body_term(hamiltonian_terms, i, B * values[i] * 0.5, num_qubits)
            for i in range(n):
                self._add_one_body_term(hamiltonian_terms, i, -A * capacity * weights[i] * 0.5, num_qubits)
            for i in range(n):
                for j in range(i + 1, n):
                    self._add_two_body_term(hamiltonian_terms, i, j, A * weights[i] * weights[j] * 0.25, num_qubits)
            for k in range(K):
                slack_idx = n + k
                coeff = 2 ** k
                self._add_one_body_term(hamiltonian_terms, slack_idx, -A * capacity * coeff * 0.5, num_qubits)
                for i in range(n):
                    self._add_two_body_term(hamiltonian_terms, i, slack_idx, A * weights[i] * coeff * 0.25, num_qubits)
            for k1 in range(K):
                for k2 in range(k1 + 1, K):
                    coeff = A * (2 ** k1) * (2 ** k2) * 0.25
                    self._add_two_body_term(hamiltonian_terms, n + k1, n + k2, coeff, num_qubits)

        self.num_qubits = num_qubits
        if hamiltonian_terms:
            pauli_terms, coeffs = zip(*hamiltonian_terms)
            self.hamiltonian = SparsePauliOp.from_list(
                list(zip(pauli_terms, coeffs))
            )

    def fromBILP(self, c, A_eq=None, b_eq=None, A_ineq=None, b_ineq=None, penalty_A=None):
        c = np.array(c)
        n = len(c)
        num_slack = 0
        slack_bits = []
        if A_ineq is not None and b_ineq is not None:
            A_ineq = np.array(A_ineq)
            b_ineq = np.array(b_ineq)
            for i in range(len(b_ineq)):
                max_slack = int(b_ineq[i])
                bits = int(np.ceil(np.log2(max_slack + 1))) if max_slack > 0 else 1
                slack_bits.append(bits)
                num_slack += bits
        num_qubits = n + num_slack
        self.num_qubits = num_qubits
        if penalty_A is None:
            penalty_A = np.abs(c).sum() + 1
        hamiltonian_terms = []
        for i in range(n):
            self._add_one_body_term(hamiltonian_terms, i, -c[i] * 0.5, num_qubits)
        if A_eq is not None and b_eq is not None:
            A_eq = np.array(A_eq)
            b_eq = np.array(b_eq)
            for j in range(len(b_eq)):
                for i in range(n):
                    coeff = penalty_A * A_eq[j, i] * (A_eq[j, i] - 2 * b_eq[j]) * 0.25
                    self._add_one_body_term(hamiltonian_terms, i, coeff, num_qubits)
                for i in range(n):
                    for k in range(i + 1, n):
                        coeff = penalty_A * A_eq[j, i] * A_eq[j, k] * 0.25
                        self._add_two_body_term(hamiltonian_terms, i, k, coeff, num_qubits)
        if A_ineq is not None and b_ineq is not None:
            slack_start = n
            for j in range(len(b_ineq)):
                bits = slack_bits[j]
                for i in range(n):
                    coeff = penalty_A * A_ineq[j, i] * (A_ineq[j, i] - 2 * b_ineq[j]) * 0.25
                    self._add_one_body_term(hamiltonian_terms, i, coeff, num_qubits)
                for k in range(bits):
                    slack_idx = slack_start + k
                    slack_coeff = 2 ** k
                    coeff = penalty_A * slack_coeff * (slack_coeff - 2 * b_ineq[j]) * 0.25
                    self._add_one_body_term(hamiltonian_terms, slack_idx, coeff, num_qubits)
                for i in range(n):
                    for k in range(i + 1, n):
                        coeff = penalty_A * A_ineq[j, i] * A_ineq[j, k] * 0.25
                        self._add_two_body_term(hamiltonian_terms, i, k, coeff, num_qubits)
                for i in range(n):
                    for k in range(bits):
                        slack_idx = slack_start + k
                        slack_coeff = 2 ** k
                        coeff = penalty_A * A_ineq[j, i] * slack_coeff * 0.25
                        self._add_two_body_term(hamiltonian_terms, i, slack_idx, coeff, num_qubits)
                for k1 in range(bits):
                    for k2 in range(k1 + 1, bits):
                        coeff = penalty_A * (2 ** k1) * (2 ** k2) * 0.25
                        self._add_two_body_term(hamiltonian_terms, slack_start + k1,
                                               slack_start + k2, coeff, num_qubits)
                slack_start += bits
        if hamiltonian_terms:
            pauli_terms, coeffs = zip(*hamiltonian_terms)
            self.hamiltonian = SparsePauliOp.from_list(
                list(zip(pauli_terms, coeffs))
            )

    def fromSchedule(self, jobs, deadlines, weights, penalty_A=None):
        n = len(jobs)
        T = sum(jobs)
        num_qubits = n * T
        self.num_qubits = num_qubits
        if penalty_A is None:
            penalty_A = sum(weights) + 1
        hamiltonian_terms = []
        def qubit_idx(j, t):
            return j * T + t
        for j in range(n):
            valid_starts = range(T - jobs[j] + 1)
            for t in valid_starts:
                idx = qubit_idx(j, t)
                self._add_one_body_term(hamiltonian_terms, idx, penalty_A * 0.5, num_qubits)
            for t1 in valid_starts:
                for t2 in list(valid_starts)[list(valid_starts).index(t1) + 1:]:
                    self._add_two_body_term(hamiltonian_terms, qubit_idx(j, t1),
                                           qubit_idx(j, t2), penalty_A * 0.25, num_qubits)
        for t in range(T):
            running_vars = []
            for j in range(n):
                for start_t in range(max(0, t - jobs[j] + 1), min(t + 1, T - jobs[j] + 1)):
                    running_vars.append(qubit_idx(j, start_t))
            for i_idx, v1 in enumerate(running_vars):
                for v2 in running_vars[i_idx + 1:]:
                    if v1 != v2:
                        self._add_two_body_term(hamiltonian_terms, v1, v2, penalty_A * 0.25, num_qubits)
        for j in range(n):
            for t in range(T - jobs[j] + 1):
                completion_time = t + jobs[j]
                tardiness = max(0, completion_time - deadlines[j])
                if tardiness > 0:
                    idx = qubit_idx(j, t)
                    self._add_one_body_term(hamiltonian_terms, idx, weights[j] * tardiness * 0.5, num_qubits)
        if hamiltonian_terms:
            pauli_terms, coeffs = zip(*hamiltonian_terms)
            self.hamiltonian = SparsePauliOp.from_list(
                list(zip(pauli_terms, coeffs))
            )
