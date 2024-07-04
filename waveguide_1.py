# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 00:24:17 2024

@author: PC
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve
from scipy.linalg import expm, solve_lyapunov
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import linregress



#%%

# H_c = \hbar \omega (a_{c}^{\dag}a_{c} + 1/2)
# a_{c} = \frac{1-e^{i \Delta\phi}}{2} a
# H_a = \Delta(\sigma_{+}\sigma_{-})
# H_int = g_{0}(a\sigma_{+} + a^{\dag}\sigma_{-})
# H_ssh = \sum_{n} - J_{n} e^{i/\hbar g/\sqrt{L} (a_{c} + a_{c}^{\dag}) (R_{n} - R_{n+1}) } c_{n+1}^{\dag}c_{n} + h.c.
# J_{n} = J(1 + (-1)^{n}\delta)



def A(N):
    ans = np.zeros((N+1,N+1), dtype = complex)
    for n in range(1,N+1):
        ans[n-1][n] = np.sqrt(n)
    return ans 


def A_dag(N):
    ans = np.zeros((N+1,N+1), dtype = complex)
    for n in range(N):
        ans[n+1][n] = np.sqrt(n+1)
    return ans

def A_c(N,dphi):
    ans = (1 - np.exp(1j * dphi))/2 * A(N)
    return ans

def A_c_dag(N,dphi):
    ans = (1 - np.exp(-1j * dphi))/2 * A_dag(N)
    return ans

# The function for computing the direct sum of a list of matrices!
def direct_sum(matrices):
    """Computes the direct sum of a list of matrices using numpy.block."""
    # Create a block diagonal matrix from the list of matrices
    # The matrix list is converted into a list of lists where each inner list
    # contains one matrix and the rest as empty lists to form a block diagonal structure.
    block_diagonal = [[np.zeros_like(m) if i != j else m for i, m in enumerate(matrices)] for j in range(len(matrices))]
    return np.block(block_diagonal)


def sigma_z():
    ans = np.array([[1,0],[0,-1]], dtype = complex)
    return ans

def C(i,L):
    c_hat = np.array([[0,0],[1,0]], dtype = complex)
    ide = np.identity(2, dtype = complex)
    matrices = [ide] * L
    matrices[i] = c_hat
    for j in range(i):
        matrices[j] = -1* sigma_z()
    ans = matrices[0]
    for i in range(1,L):
        ans = np.kron(ans, matrices[i])
    return ans

def C_dag(i,L):
    c_hat_dag = np.array([[0,1],[0,0]], dtype = complex)
    ide = np.identity(2, dtype = complex)
    matrices = [ide] * L
    matrices[i] = c_hat_dag
    for j in range(i):
        matrices[j] = -1* sigma_z()
    ans = matrices[0]
    for i in range(1,L):
       ans = np.kron(ans, matrices[i])
    return ans

def H_c(omega, N, L, dphi):
    hc = omega * (A_c_dag(N,dphi) @ A_c(N,dphi) + 1/2*np.identity(N+1))
    ans = np.kron(hc, np.kron(np.identity(2**L), np.identity(2)))
    return ans

def H_a(Delta, N, L):
    ha = Delta * (C_dag(0,1) @ C(0,1))
    ans = np.kron(np.identity(N+1), np.kron(np.identity(2**L), ha))
    return ans
# Here, \sigma_{+} reduces to C_dag(0,1)!

def H_int(g0, N, L, dphi):
    c = np.kron(np.identity(N+1), np.kron(C(0,L), np.identity(2)))
    c_dag = np.kron(np.identity(N+1), np.kron(C_dag(0,L), np.identity(2)))
    sigma_u = np.kron(np.identity(N+1),np.kron(np.identity(2**L), C_dag(0,1)))
    sigma_d = np.kron(np.identity(N+1),np.kron(np.identity(2**L), C(0,1)))
    
    ans = g0 * (c @ sigma_u + c_dag @ sigma_d)
    
    return ans


def H_ssh(N, L, dphi, g, J, delta):
    ans = np.zeros(((N+1)*2**L*2, (N+1)*2**L*2), dtype = complex)
    a = np.kron(A_c(N, dphi), np.kron(np.identity(2**L), np.identity(2)))
    a_dag = np.kron(A_c_dag(N, dphi), np.kron(np.identity(2**L), np.identity(2)))
    for n in range(L-1):
        J_n = J * (1 - (-1)**n * delta)
        mat = -J_n * expm(1j * g/(np.sqrt(L)) * (a + a_dag)) @ (np.kron(np.identity(N+1), np.kron(C_dag(n+1,L) @ C(n,L), np.identity(2))))
        ans += mat + np.conj(mat).T
    return ans

def H_t(omega, N, L, dphi, Delta, g0, g, J, delta):
    ans = np.zeros(((N+1)*2**L*2, (N+1)*2**L*2), dtype = complex)
    ans += H_ssh(N, L, dphi, g, J, delta) + H_int(g0, N, L, dphi) + H_a(Delta, N, L) + H_c(omega, N, L, dphi)
    return ans

def H_e(omega, N, L, dphi, Delta, g0, g, J, delta, n):
    H = H_t(omega, N, L, dphi, Delta, g0, g, J, delta)
    mat = np.zeros((N+1,N+1), dtype = complex)
    mat[n][n] = 1
    proj = np.kron(mat, np.kron(np.identity(2**L), np.identity(2)))
    H_reshaped = (H @ proj).reshape((N+1, 2**L, 2, N+1, 2**L, 2))
    ans = np.tensordot(H_reshaped, np.identity(N+1), axes = ([0,3],[0,1]))
    ans = ans.reshape((2**L*2, 2**L*2))
    print(H.shape)
    print(proj.shape)
    print((H @ proj).shape)
    print(H_reshaped.shape)
    print(ans.shape)

    return ans

def eigen(H):
    es, vs = np.linalg.eig(H)
    # We need to implement sorting from small to large values~
    sorted_indices = np.argsort(es)
    es_new = es[sorted_indices]
    vs_new = vs[:,sorted_indices]
    idxs = np.linspace(1,len(vs[0]),len(vs[0]))
    return es_new, vs_new, idxs


# A good thing is that here the Hamiltonian is time independent. 
def crank_nicolson(psi, dt, omega, N, L, dphi, Delta, g0, g, J, delta):
    """Perform a single Crank-Nicolson step."""
    # Identity matrix
    I = np.identity(len(psi))
    
    # Compute the Hamiltonian matrix at time t
    H = H_t(omega, N, L, dphi, Delta, g0, g, J, delta)
    print(H_t)
    
    # Construct the matrices for the linear system
    A = (I - 1j * dt/2 * H)
    B = (I + 1j * dt/2 * H)
    
    # Solve the linear system
    psi_next = solve(A, B @ psi)
    return psi_next

def run(T, omega, N, L, dphi, Delta, g0, g, J, delta):
    
    psi_times = []
    rho_times = []
    dt = 0.1 
    times = np.arange(0, T, dt)
    
   
    
    # Initial state (should be normalized)
    ket_ph = np.zeros(N+1, dtype = complex)
    ket_ph[N] = 1
    #print(ket_ph)
    up = np.array([1,0], dtype = complex)
    down = np.array([0,1], dtype = complex)
    #matrices = [down]*L
    #print(matrices)
    ket_e = np.zeros(2**L, dtype = complex)
    ket_e[-1] = 1
    #print(ket_e)
    ket_a = up
    #print(ket_a)
    psi_0 = np.kron(ket_ph, np.kron(ket_e,ket_a))
    
    psi_0 = psi_0 / np.linalg.norm(psi_0)
    rho_0 = np.outer(psi_0, np.conj(psi_0))
    #print(np.inner(np.conj(psi_0),psi_0))
    
    # Time evolution
    psi_t = psi_0
    #print(len(psi_t))
    for t in times:
        psi_times.append(psi_t)
        psi_t = crank_nicolson(psi_t, dt, omega, N, L, dphi, Delta, g0, g, J, delta)
        rho_t = np.outer(psi_t, np.conj(psi_t))
        rho_times.append(rho_t)
        print(np.inner(np.conj(psi_t),psi_t))
    #psi_t now holds the state at time T
    #print("State at time T:", psi_t)
    
    return psi_t, psi_0, psi_times, times, rho_0, rho_t, rho_times
# psi_t: the final state; psi_0: the initial state; psi_times: the list for the states at each time increment; times: the array for time increments. 

def prob_up(rho, rho_0, N, L):
    mat = rho @ rho_0
    prob = np.trace(mat)
    return prob
        
        
def fisher_info(T, omega, N, L, dphi, Delta, g0, g, J, delta, diff = 1e-6):
    psi_t_small, psi_0_small, psi_times_small, times_small, rho_0_small, rho_t_small, rho_times_small = run(T, omega, N, L, dphi - diff, Delta, g0, g, J, delta)
    psi_t, psi_0, psi_times, times, rho_0, rho_t, rho_times = run(T, omega, N, L, dphi, Delta, g0, g, J, delta)
    psi_t_big, psi_0_big, psi_times_big, times_big, rho_0_big, rho_t_big, rho_times_big = run(T, omega, N, L, dphi + diff, Delta, g0, g, J, delta)
    d_rho = (rho_t_big - rho_t_small)/(2*diff)
    L = solve_lyapunov(rho_t, 2*d_rho)
    F = np.trace(rho_t @ L @ L)
    
    return F

# \hat{gamma} = \sum_{n} (-1)^{n} \hat{c}_{n}^{\dag} \hat{c}_{n}
def gamma_hat(L):
    ans = np.zeros((2*L,2*L), dtype = complex)
    for n in range(L):
        ans += (-1)**n * C_dag(n,L) @  C(n,L)
    return ans

def F_norm(A, L):
    val = 0
    for i in range(2*L):
        for j in range(2*L):
            val += (np.abs(A[i][j]))**2
    ans = np.sqrt(val)
    return ans

def check_chiral(H):
    gamma = gamma_hat(L)
    mat = H @ gamma + gamma @ H
    ans = F_norm(mat, L)
    return ans


#%%
L = 5
J = 1
delta = 0.5
Delta = 0
N = 5
n = 5
g = 1
g0 = 0
A0 = 1
dphi = np.pi/4

T = 50
omega = 1

#H = H_t(omega, N, L, dphi, Delta, g0, g, J, delta)
#print(H)
#diff = H - np.conj(H).T
#print(diff)


def plot_energy():
    #H = H_ssh(L, J, delta, Delta)
    #H = H_couple(L, J, delta, Delta, g, N, A0)
    H = H_e(omega, N, L, dphi, Delta, g0, g, J, delta, n)
    print(H)
    es, vs, idxs = eigen(H)
    plt.figure()
    plt.title(f'Energy Spectrum: delta = {delta:.2f}, g = {g:.2f} ')
    plt.xlabel('idx')
    plt.ylabel('E')
    plt.scatter(idxs, es, marker = 'o', s=8)
    plt.plot(idxs, np.zeros(len(idxs)) + Delta, color = 'r', linestyle = '--')
    plt.plot(idxs, np.ones(len(idxs))*2*delta + Delta, color = 'm', linestyle = '--')
    plt.plot(idxs, np.ones(len(idxs))*(-2)*delta + Delta, color = 'g', linestyle = '--')
    #plt.legend()
    plt.show()
    return
plot_energy()


def plot_state():
    #H = H_ssh(L, J, delta, Delta)
    #H = H_couple(L, J, delta, Delta, g, N, A0)
    H = H_e(omega, N, L, dphi, Delta, g0, g, J, delta, n)
    print(H)
    es, vs, idxs = eigen(H)
    for i in range(len(vs[1])):
        plt.figure()
        plt.title(f'State Configuration: delta = {delta:.2f}, g = {g:.2f}')
        plt.xlabel('idx')
        plt.ylabel(r'|$\psi$|')
        plt.bar(idxs, np.abs(vs[:,i]))
        plt.show()
    return
plot_state()

def plot_energy_para():
    deltas = np.linspace(-1,1,20)
    gs = np.linspace(0,20,3)
    colors = ['r', 'g', 'b']
    plt.figure()
    plt.xlabel(r'$\delta$')
    plt.ylabel('E')
    plt.title(r'Energy spectrum with $\delta$')
    for k in range(3):
        g = gs[k]
        c = colors[k]
        Es = np.zeros((L,len(deltas)), dtype = complex)
        for j in range(len(deltas)):
            delta = deltas[j]
            H = H_e(n, L, J, delta, Delta, g, N, A0, dphi)
            es, vs, idxs = eigen(H)
            for i in range(len(es)):
                Es[i][j] = es[i]
        for i in range(len(es)):
            plt.scatter(deltas, Es[i,:], color = c, marker = 'o')
    plt.show()   
    return
#plot_energy_para()





def plot_ex_prob():
    gs = np.linspace(0,10,3)
    #g = 5
    dphis = [0, np.pi/4, np.pi/3, np.pi/2, np.pi]
    labels = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{3}$', r'$\frac{\pi}{2}$', r'$\pi$' ]
    #dphi = np.pi/8
    plt.figure(figsize = (12,8))
    plt.title(r'Excited state prob,  $g = {g:.2f}$')
    plt.xlabel('t')
    plt.ylabel('P(t)')
    
    
    for j in range(len(dphis)):
        dphi = dphis[j]
        psi_t, psi_0, psi_times, times, rho_0, rho_t, rho_times = run(T, omega, N, L, dphi, Delta, g0, g, J, delta)
        probs = np.zeros(len(times))
        for i in range(len(times)):
            rho = rho_times[i]
            probs[i] = prob_up(rho, rho_0, N, L)
        plt.plot(times, probs, linestyle = '-', label = labels[j])
        
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    return
plot_ex_prob()

def plot_fisher():
    Ts = np.linspace(1,200,25)
    Fs = np.zeros(len(Ts))
    dphis = [0, np.pi/4, np.pi/3, np.pi/2, np.pi]
    labels = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{3}$', r'$\frac{\pi}{2}$', r'$\pi$' ]
    fig, ax = plt.subplots(1,2, figsize = (12,8))
    ax[0].set_title(r'Fisher Information')
    ax[0].set_xlabel('T')
    ax[0].set_ylabel(r'$F_{\Delta \phi} (T)$')
    
    # Logarithm plot~
    ax[1].set_title(r'Fisher Information logarithm')
    ax[1].set_xlabel(r'$\log{T}$')
    ax[1].set_ylabel(r'$\log{F_{\Delta \phi} (T)}$')
    
    for j in range(len(dphis)):
        dphi = dphis[j]
        label_now = labels[j]
        for i in range(len(Ts)):
            T = Ts[i]
            F = fisher_info(T, omega, N, L, dphi, Delta, g0, g, J, delta, diff=1e-6)
            Fs[i] = F
        ax[0].plot(Ts, Fs, linestyle = '--', label = label_now)
        ax[1].plot(np.log(Ts), np.log(Fs), linestyle = '--', label = label_now)
        
        result = linregress(np.log(Ts), np.log(Fs))
        slope = result.slope
        intercept = result.intercept
        ax[1].plot(np.log(Ts), slope * np.log(Ts) + intercept, linestyle = '-', label = f'{slope:.2f} x + {intercept:.2f}')
    
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.show()
    return
#plot_fisher()