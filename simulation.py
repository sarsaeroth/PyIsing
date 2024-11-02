# This script contains functions for simulation of
# a properly converted Ising spin glass

class Simulation:

    def __init__(self):
        self.num_qubits = False
        self.num_steps = 
        self.target_hamiltonian = False
        self.transverse_hamiltonian = False
        self.spectral_gap = False

    def initialize(self, spinglass, num_steps=100):
        self.target_hamiltonian = spinglass.hamiltonian
        self.num_qubits = self.target_hamiltonian.num_qubits

    def execute(self):
        if not self.target_hamiltonian or not self.transverse_hamiltonian:
            raise MissingHamiltonianError

    def results(self):
        if not self.spectral_gap:
            raise NotExecutedError
