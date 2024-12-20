# This script contains functions for converting
# an NP-complete problem into its corresponding
# Ising spin glass model
import networkx as nx
from qiskit.quantum_info import SparsePauliOp

class InvalidGraphError(Exception):
    # Raised when the input graph is invalid
    print(Exception)
    pass

class InvalidSetError(Exception):
    # Raised when the input set is invalid
    print(Exception)
    pass

class InvalidProblemError(Exception):
    # Raised when the specified problem is not supported
    print(Exception)
    pass

class SpinGlass:

    def __init__(self):
        self.hamiltonian = None

    def fromGraph(self, G, problem):
        if not isinstance(G, nx.Graph):
            raise InvalidGraphError("Input is not a valid NetworkX graph.")

        problems = ['clique', 'coloring', 'covering', 'fes', 'hamiltonian', 'partition']
        if problem not in problems:
            raise InvalidProblemError("Problem not supported.")

        hamiltonian_terms = []
        n = G.number_of_nodes()

        if problem == 'clique':
            for i in range(n):
                z_term = 'I' * i + 'Z' + 'I' * (n - i - 1)
                hamiltonian_terms.append((z_term, -1))

            for i in range(n):
                for j in range(i + 1, n):
                    if not G.has_edge(i, j):
                        z_term = 'I' * i + 'Z' + 'I' * (j - i - 1) + 'Z' + 'I' * (n - j - 1)
                        hamiltonian_terms.append((z_term, 1))

        elif problem == 'coloring':
            for u, v in G.edges():
                for color in range(4):  # Four-color theorem
                    z_term_u = 'I' * u + 'Z' + 'I' * (n - u - 1)
                    z_term_v = 'I' * v + 'Z' + 'I' * (n - v - 1)
                    combined_term = ''.join(['I' if i != u and i != v else 'Z' for i in range(n)])
                    hamiltonian_terms.append((combined_term, 1))
            for u in G.nodes():
                z_term = 'I' * u + 'Z' + 'I' * (n - u - 1)
                hamiltonian_terms.append((z_term, -1))

        elif problem == 'covering':
            for u in G.nodes():
                adjacent_sum_term = ''.join(['I' if i != u else 'Z' for i in range(n)])
                hamiltonian_terms.append((adjacent_sum_term, 1))
            for u in G.nodes():
                z_term = 'I' * u + 'Z' + 'I' * (n - u - 1)
                hamiltonian_terms.append((z_term, -1))

        elif problem == 'fes':
            for u, v in G.edges():
                z_term = 'I' * u + 'Z' + 'I' * (v - u - 1) + 'Z' + 'I' * (n - v - 1)
                hamiltonian_terms.append((z_term, 1))
            for u, v in G.edges():
                z_term = 'I' * u + 'Z' + 'I' * (v - u - 1) + 'Z' + 'I' * (n - v - 1)
                hamiltonian_terms.append((z_term, -1))

        elif problem == 'hamiltonian':
            n = G.number_of_nodes()
            for u in G.nodes():
                for v in G.nodes():
                    if u != v:
                        z_term = 'I' * u + 'Z' + 'I' * (v - u - 1) + 'Z' + 'I' * (n - v - 1)
                        hamiltonian_terms.append((z_term, -1))
            for u in G.nodes():
                z_term = 'I' * u + 'Z' + 'I' * (n - u - 1)
                hamiltonian_terms.append((z_term, 1))
            for u in G.nodes():
                for v in G.nodes():
                    if not G.has_edge(u, v):
                        z_term = 'I' * u + 'Z' + 'I' * (v - u - 1) + 'Z' + 'I' * (n - v - 1)
                        hamiltonian_terms.append((z_term, 2))

        elif problem == 'partition':
            for u in G.nodes():
                z_term = 'I' * u + 'Z' + 'I' * (n - u - 1)
                hamiltonian_terms.append((z_term, 1))
            for u, v in G.edges():
                z_term = 'I' * u + 'Z' + 'I' * (v - u - 1) + 'Z' + 'I' * (n - v - 1)
                hamiltonian_terms.append((z_term, 2))

        else:
            raise InvalidProblemError(f"Problem {problem} is not yet implemented.")

        # Assemble Hamiltonian
        if hamiltonian_terms:
            pauli_terms, coeffs = zip(*hamiltonian_terms)
            self.hamiltonian = SparsePauliOp.from_list(
                list(zip(pauli_terms, coeffs))
            )

    def fromSet(self, S, problem):
        # Input validation
        if isinstance(S, list):
            if len(set(S)) != len(S):
                raise InvalidSetError("Set contains duplicate elements.")
        elif not isinstance(S, set):
            raise InvalidSetError("Input is not a valid set or list.")

        problems = ['partition']
        if problem not in problems:
            raise InvalidProblemError("Problem not supported.")

        N = len(S)
        hamiltonian_terms = []

        if problem == 'partition':
            # Dividing set into two disjoint subsets of equal sum
            for i in range(N):
                # Self-interaction terms
                coeff = S[i] ** 2
                z_term = 'I' * i + 'Z' + 'I' * (N - i - 1)
                hamiltonian_terms.append((z_term, coeff))
            for i in range(N):
                for j in range(i + 1, N):
                    # Cross-interaction terms
                    coeff = 2 * S[i] * S[j]
                    z_term = 'I' * i + 'Z' + 'I' * (j - i - 1) + 'Z' + 'I' * (N - j - 1)
                    hamiltonian_terms.append((z_term, coeff))

        # Assemble Hamiltonian
        if hamiltonian_terms:
            pauli_terms, coeffs = zip(*hamiltonian_terms)
            self.hamiltonian = SparsePauliOp.from_list(
                list(zip(pauli_terms, coeffs))
            )

