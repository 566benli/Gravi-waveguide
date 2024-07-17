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





# %%
# Total length of the chain: L
# Maximum photon number: N

# Consider (single) non-interacting electron case!
def hop(n, d, L):
    ans = np.zeros((L + 1, L + 1), dtype=complex)
    ans[n + d][n] = 1
    return ans


def site(n, L):
    ans = np.zeros((L + 1, L + 1), dtype=complex)
    ans[n][n] = 1
    return ans


# def sigma_up():
#     ans = np.zeros((2,2), dtype = complex)
#     ans[0][1] = 1
#     return ans

# def sigma_down():
#     ans = np.zeros((2,2), dtype = complex)
#     ans[1][0] = 1
#     return ans

# Consider at most one photonic mode.
def A_dag(N):
    ans = np.zeros((N + 1, N + 1), dtype=complex)
    for n in range(N):
        ans[n + 1][n] = np.sqrt(n + 1)
    return ans


def A(N):
    ans = np.zeros((N + 1, N + 1), dtype=complex)
    for n in range(1, N + 1):
        ans[n - 1][n] = np.sqrt(n)
    return ans


def A_phi(N, dphi):
    return A(N) * ((1j + 1) + (1j - 1) * np.exp(1j * dphi)) / 2
    # return A(N) * ((1 + 1j)* np.exp(1j*dphi) + (1j - 1))/2


def A_dag_phi(N, dphi):
    return A_dag(N) * np.conj(((1j + 1) + (1j - 1) * np.exp(1j * dphi)) / 2)
    # return A_dag(N) * np.conj(((1 + 1j)* np.exp(1j*dphi) + (1j - 1))/2)


# J_i = J(1 - (-1)**i * \delta)
# Hamiltonian for the chain only
def H_ssh(L, J, delta, Delta, Omega, g0):
    # Initialize Hamiltonian matrix
    H = np.zeros((L + 1, L + 1), dtype=complex)

    # Set energy levels of the two-level atom and coupling strengths
    H[0, 0] = Omega
    H[0, 1] = g0
    H[1, 0] = np.conj(g0)  # Hermitian conjugate

    # Set hopping terms and on-site energies for the SSH chain
    for n in range(1, L):
        J_n = J * (1 + (-1) ** n * delta)
        H += -J_n * hop(n, 1, L) + np.conj(-J_n * hop(n, 1, L)).T
        H += Delta * site(n, L)
    H += Delta * site(L, L)
    return H


# Hilbert Space: H_{ph} \otimes H_{e} \otimes H_{a}
def H_couple(L, J, delta, Delta, g, N, A0, dphi, Omega, g0):
    dim = (N + 1) * (L + 1)
    ans = np.zeros((dim, dim), dtype=complex)
    mat = np.zeros((L + 1, L + 1), dtype=complex)
    mat[0][0] = Omega
    mat[0][1] = g0
    mat[1][0] = np.conj(g0)
    ans += np.kron(np.identity(N + 1), mat)
    for n in range(1, L):
        J_n = J * (1 + (-1) ** n * delta)
        mat = -J_n * expm(-1j * g / np.sqrt(L) * A0 * (
                    np.kron(A_phi(N, dphi), np.identity(L + 1)) + np.kron(A_dag_phi(N, dphi),
                                                                          np.identity(L + 1)))) @ np.kron(
            np.identity(N + 1), hop(n, 1, L))
        ans += mat + np.conj(mat.T)

    for n in range(1, L + 1):
        ans += Delta * np.kron(np.identity(N + 1), site(n, L))

    return ans


def H_photon(N, dphi, omega):
    ans = np.zeros((N + 1, N + 1), dtype=complex)
    ans += omega * (1 / 2 * np.identity(N + 1) + A_dag_phi(N, dphi) @ A_phi(N, dphi))
    final = np.kron(ans, np.identity(L + 1))
    return final


def H_e(n, L, J, delta, Delta, g, N, A0, dphi, Omega, g0):
    mat = np.zeros((N + 1, N + 1), dtype=complex)
    mat[n][n] = 1
    H = H_couple(L, J, delta, Delta, g, N, A0, dphi, Omega, g0) @ np.kron(mat, np.identity(L + 1))
    H_reshape = H.reshape((N + 1, L + 1, N + 1, L + 1))
    ans = np.tensordot(H_reshape, np.identity(N + 1), axes=([0, 2], [0, 1]))
    # print(H.shape)
    # print(H_reshape.shape)
    return ans


# Trace out the degrees of freedom of the first and the third dimension of H_reshape, and integrate it with the first and the second dimensions of the identity~


def H_t(omega, N, L, dphi, Delta, g0, g, J, A0, delta, Omega):
    ans = np.zeros(((N + 1) * (L + 1), (N + 1) * (L + 1)), dtype=complex)
    ans += H_couple(L, J, delta, Delta, g, N, A0, dphi, Omega, g0) + H_photon(N, dphi, omega)
    return ans
