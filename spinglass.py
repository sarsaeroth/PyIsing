# This script contains functions for converting
# an NP-complete problem into its corresponding
# Ising spin glass model
from qiskit.quantum_info import SparsePauliOp

class SpinGlass:
    
    def __init__(self):
        self.hamiltonian = False

    def fromGraph(self, G, problem):
        #Input validation
        if not type(G) is not networkx.classes.graph.Graph:
            #Only supports Networkx for now
            return InvalidGraphError
        problems = ['hamiltonian']
        if problem not in problems:
            raise InvalidProblemError

        if problem == 'hamiltonian':
            #Finding a Hamiltonian path in a graph
            pass
    
    def fromSet(self, S, problem):
        #Input validation
        if type(S) is list:
            if (len(set(S)) != len(S)):
                raise InvalidSetError
        if type(S) is not set and type(S) is not list:
            raise InvalidSetError
        problems = ['partition']
        if problem not in problems:
            raise InvalidProblemError
        N = len(S)
        
        if problem == 'partition':
            #Dividing set into two disjoint subsets of equal sum 
            hamiltonian_terms = []
            for i in range(N):
                #Self-interaction terms
                coeff = S[i] ** 2
                z_term = 'I'*i + 'Z' + 'I'*(N-i-1)
                hamiltonian_terms.append((z_term, coeff))
            for i in range(N):
                for j in range(i+1,N):
                    #Cross-interaction terms
                    coeff = 2*S[i]*S[j]
                    z_term = 'I'*i + 'Z' + 'I'*(j-i-1)
                    z_term = z_term + 'Z' + 'I'*(N-j-1)
                    hamiltonian_terms.append((z_term, coeff))
            pauli_terms, coeffs = zip(*hamiltonian_terms)
            hamiltonian = SparsePauliOp.from_list(
                    list(zip(pauli_terms, coeffs))
            )
            return hamiltonian
