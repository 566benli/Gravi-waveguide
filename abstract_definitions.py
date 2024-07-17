from abc import ABC, abstractmethod
import numpy as np

class Hamiltonian(ABC):
    def __init__(self):
        self.H = None
    def eigen(self):
        es, vs = np.linalg.eig(self.H)
        # We need to implement sorting from small to large values~
        sorted_indices = np.argsort(es)
        es_new = es[sorted_indices]
        vs_new = vs[:, sorted_indices]
        idxs = np.linspace(1, len(vs[0]), len(vs[0]))
        return es_new, vs_new, idxs


