import pdb; pdb.set_trace()

import networkx as nx
from networkx.algorithms.approximation.clique import max_clique
from spinglass import SpinGlass
from simulation import Simulation
from qiskit.quantum_info import SparsePauliOp

G = nx.complete_graph(5)
G.add_edge(4, 5)
    
spinglass = SpinGlass()

spinglass.fromGraph(G, "clique")

simulation = Simulation()
simulation.initialize(spinglass, num_steps=100)
simulation.execute()
results = simulation.results()

print("\nSimulation Results:")
print(f"Spectral Gaps: {results['spectral_gaps']}")
print(f"Final State: {results['simulation_data'][-1]}")

'''
is_k5_minor = max_clique(G) == 5  # Compare to classical approximation algorithm
print("\nVerification:")
if is_k5_minor:
    print("The graph contains the K5 minor.")
else:
    print("The graph does not contain the K5 minor.")
'''
