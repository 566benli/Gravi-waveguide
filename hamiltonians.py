from abc import ABC, abstractmethod
from abstract_definitions import Hamiltonian
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# Total length of the chain: L
# Maximum photon number: N

# Consider (single) non-interacting electron case!
def hop(n, d, L):
    ans = torch.zeros((L + 1, L + 1), dtype=torch.cfloat)
    ans[n + d][n] = 1
    return ans


def site(n, L):
    ans = torch.zeros((L + 1, L + 1), dtype=torch.cfloat)
    ans[n][n] = 1
    return ans


# def sigma_up():
#     ans = torch.zeros((2,2), dtype = complex)
#     ans[0][1] = 1
#     return ans

# def sigma_down():
#     ans = torch.zeros((2,2), dtype = complex)
#     ans[1][0] = 1
#     return ans

# Consider at most one photonic mode.
def A_dag(N):
    ans = torch.zeros((N + 1, N + 1), dtype=torch.cfloat)
    for n in range(N):
        ans[n + 1][n] = torch.sqrt(n + 1)
    return ans


def A(N):
    ans = torch.zeros((N + 1, N + 1), dtype=torch.cfloat)
    for n in range(1, N + 1):
        ans[n - 1][n] = torch.sqrt(n)
    return ans


def A_phi(N, dphi):
    return A(N) * ((1j + 1) + (1j - 1) * torch.exp(1j * dphi)) / 2
    # return A(N) * ((1 + 1j)* torch.exp(1j*dphi) + (1j - 1))/2


def A_dag_phi(N, dphi):
    return A_dag(N) * torch.conj(((1j + 1) + (1j - 1) * torch.exp(1j * dphi)) / 2)
    # return A_dag(N) * torch.conj(((1 + 1j)* torch.exp(1j*dphi) + (1j - 1))/2)


# J_i = J(1 - (-1)**i * \delta)
# Hamiltonian for the chain only
def H_ssh(L, J, delta, Delta, Omega, g0):
    # Initialize Hamiltonian matrix
    H = torch.zeros((L + 1, L + 1), dtype=torch.cfloat)

    # Set energy levels of the two-level atom and coupling strengths
    H[0, 0] = Omega
    H[0, 1] = g0
    H[1, 0] = torch.conj(g0)  # Hermitian conjugate

    # Set hopping terms and on-site energies for the SSH chain
    for n in range(1, L):
        J_n = J * (1 + (-1) ** n * delta)
        H += -J_n * hop(n, 1, L) + torch.conj(-J_n * hop(n, 1, L)).T
        H += Delta * site(n, L)
    H += Delta * site(L, L)
    return H


# Hilbert Space: H_{ph} \otimes H_{e} \otimes H_{a}
def H_couple(L, J, delta, Delta, g, N, A0, dphi, Omega, g0):
    dim = (N + 1) * (L + 1)
    ans = torch.zeros((dim, dim), dtype=torch.cfloat)
    mat = torch.zeros((L + 1, L + 1), dtype=torch.cfloat)
    mat[0][0] = Omega
    mat[0][1] = g0
    mat[1][0] = torch.conj(g0)
    ans += torch.kron(torch.identity(N + 1), mat)
    for n in range(1, L):
        J_n = J * (1 + (-1) ** n * delta)
        mat = -J_n * expm(-1j * g / torch.sqrt(L) * A0 * (
                    torch.kron(A_phi(N, dphi), torch.identity(L + 1)) + torch.kron(A_dag_phi(N, dphi),
                                                                          torch.identity(L + 1)))) @ torch.kron(
            torch.identity(N + 1), hop(n, 1, L))
        ans += mat + torch.conj(mat.T)

    for n in range(1, L + 1):
        ans += Delta * torch.kron(torch.identity(N + 1), site(n, L))

    return ans


def H_photon(N, dphi, omega):
    ans = torch.zeros((N + 1, N + 1), dtype=torch.cfloat)
    ans += omega * (1 / 2 * torch.identity(N + 1) + A_dag_phi(N, dphi) @ A_phi(N, dphi))
    final = torch.kron(ans, torch.identity(L + 1))
    return final


def H_e(n, L, J, delta, Delta, g, N, A0, dphi, Omega, g0):
    mat = torch.zeros((N + 1, N + 1), dtype=torch.cfloat)
    mat[n][n] = 1
    H = H_couple(L, J, delta, Delta, g, N, A0, dphi, Omega, g0) @ torch.kron(mat, torch.identity(L + 1))
    H_reshape = H.view(N + 1, L + 1, N + 1, L + 1)
    ans = torch.tensordot(H_reshape, torch.identity(N + 1), dims=([0, 2], [0, 1]))
    # print(H.shape)
    # print(H_reshape.shape)
    return ans


# Trace out the degrees of freedom of the first and the third dimension of H_reshape, and integrate it with the first and the second dimensions of the identity~


def H_t(omega, N, L, dphi, Delta, g0, g, J, A0, delta, Omega):
    ans = torch.zeros(((N + 1) * (L + 1), (N + 1) * (L + 1)), dtype=torch.cfloat)
    ans += H_couple(L, J, delta, Delta, g, N, A0, dphi, Omega, g0) + H_photon(N, dphi, omega)
    return ans

class SSH_photon(Hamiltonian):
    def __init__(self, omega, N, L, dphi, Delta, g0, g, J, A0, delta, Omega):
        super().__init__()
        self.H = H_t(omega, N, L, dphi, Delta, g0, g, J, A0, delta, Omega).to(device)