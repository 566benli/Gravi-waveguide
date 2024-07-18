from abc import ABC, abstractmethod
import numpy as np
import torch

class Hamiltonian(ABC):
    def __init__(self):
        self.H = None
        self.eigen = False
        self.num_es, self.es, self.vs = None, None, None

    # Finds all eigenvalues & eigenvectors and sort them
    # self.vs dimension: [vector_dim, num_eigenvalues]
    def setup_eigen(self):
        if self.eigen:
            return
        if self.H is None:
            ValueError('Hamiltonian is not defined')
        es, vs = torch.linalg.eig(self.H)
        # We need to implement sorting from small to large values~ (add normalization
        sorted_indices = torch.argsort(es)
        self.es = es[sorted_indices]
        self.vs = vs[:, sorted_indices]
        self.num_es = len(self.es)
        # Normalize
        norm = torch.linalg.norm(self.vs, dim=0)
        self.vs = (self.vs / norm)

    # make this function output the history of state evolution up to time T, in energy eigenbasis
    def eigen_evolve(self, psi, dt):  #make dt T instead:
        if not self.eigen:
            self.setup_eigen()

        # Compute the corresponding coefficients.
        expanded_psi = psi.unsqueeze(1).expand(-1, self.num_es)
        coeffs = torch.linalg.vecdot(self.vs, expanded_psi, dim=0)

        # Implement this!
        psi_evolved = None

        # Delete this
        # A list for final state.
        #final_list = [coeffs[i] * self.vs[i] * np.exp(-1j * self.es[i] * dt) for i in range(len(self.vs))]

        # Combine into the final vector.
        # psi_evolved = np.sum(final_list, axis=0)

        return psi_evolved

    # do the job as the normal run does, utilizing the new eigen_evolve
    def run(self, psi, T):

    # have a function to calculate partial innter product between two-level atom basis and the state