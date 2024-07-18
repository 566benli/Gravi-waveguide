from abc import ABC, abstractmethod
import numpy as np

class Hamiltonian(ABC):
    def __init__(self):
        self.H = None
        self.eigen = False
        self.es, self.vs = None

    # finds all eigenvalues & eigenvectors and sort them
    def setup_eigen(self):
        if self.eigen:
            return
        if self.H is None:
            ValueError('Hamiltonian is not defined')
        es, vs = np.linalg.eig(self.H)
        # We need to implement sorting from small to large values~
        sorted_indices = np.argsort(es)
        self.es = es[sorted_indices]
        self.vs = vs[:, sorted_indices]

    # make this function output the history of state evolution up to time T, in energy eigenbasis
    def eigen_evolve(self, psi, dt):  #make dt T instead:
        if not self.eigen:
            self.setup_eigen()

        # Compute the basis vectors (normalized).
        basis_vectors = [self.vs[:, j] / np.linalg.norm(self.vs[:, j]) for j in range(len(self.vs[0]))]

        # Compute the corresponding coefficients.
        coeffs = [np.vdot(basis_vectors[i], psi) for i in range(len(basis_vectors))]

        # A list for final state.
        final_list = [coeffs[i] * basis_vectors[i] * np.exp(-1j * self.es[i] * dt) for i in range(len(basis_vectors))]

        # Combine into the final vector.
        psi_evolved = np.sum(final_list, axis=0)

        return psi_evolved

    # do the job as the normal run does, utilizing the new eigen_evolve
    def run(self, psi, T):

    # have a function to calculate partial innter product between two-level atom basis and the state