# simulation.py
# This script contains functions for simulation of
# a properly converted Ising spin glass

import numpy as np
from qiskit.quantum_info import SparsePauliOp

class MissingHamiltonianError(Exception):
    """Raised when a required Hamiltonian is missing."""
    pass

class NotExecutedError(Exception):
    """Raised when results are requested before execution."""
    pass

class Simulation:

    def __init__(self):
        self.num_qubits = None
        self.num_steps = None
        self.target_hamiltonian = None
        self.transverse_hamiltonian = None
        self.spectral_gaps = []
        self.simulation_results = None

    def initialize(self, spinglass, num_steps=100):
        if spinglass.hamiltonian is None:
            raise MissingHamiltonianError("SpinGlass object does not contain a valid Hamiltonian.")

        self.target_hamiltonian = spinglass.hamiltonian
        self.num_qubits = spinglass.hamiltonian.num_qubits
        self.num_steps = num_steps

        transverse_terms = []
        for i in range(self.num_qubits):
            x_term = 'I' * i + 'X' + 'I' * (self.num_qubits - i - 1)
            transverse_terms.append((x_term, -1))
        pauli_terms, coeffs = zip(*transverse_terms)
        self.transverse_hamiltonian = SparsePauliOp.from_list(
            list(zip(pauli_terms, coeffs))
        )

    def execute(self):
        if self.target_hamiltonian is None or self.transverse_hamiltonian is None:
            raise MissingHamiltonianError("Hamiltonians are not properly initialized.")

        simulation_data = []
        spectral_gaps = []

        delta_t = 1 / self.num_steps
        current_state = np.ones(2 ** self.num_qubits) / np.sqrt(2 ** self.num_qubits)

        for t in range(self.num_steps):
            s = t / (self.num_steps - 1)

            hamiltonian_t = (1 - s) * self.transverse_hamiltonian + s * self.target_hamiltonian

            evolution_operator = np.linalg.matrix_power(
                np.eye(2 ** self.num_qubits) - 1j * delta_t * hamiltonian_t.to_matrix(), 1
            )

            current_state = evolution_operator @ current_state
            simulation_data.append(current_state)

            eigenvalues, eigenvectors = np.linalg.eig(hamiltonian_t.to_matrix())
            eigenvalues = sorted(np.absolute(eigenvalues))
            spectral_gap = (eigenvalues[1] - eigenvalues[0]).item()
            spectral_gaps.append(spectral_gap)


        self.spectral_gaps = spectral_gaps
        self.simulation_results = simulation_data

    def results(self):
        if self.simulation_results is None:
            raise NotExecutedError("Simulation has not been executed yet.")
        return {
            "spectral_gaps": self.spectral_gaps,
            "simulation_data": self.simulation_results,
        }

