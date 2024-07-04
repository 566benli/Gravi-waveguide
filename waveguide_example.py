# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:05:23 2024

@author: PC
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import solve
from scipy.linalg import expm, solve_lyapunov
from scipy.stats import linregress
from scipy.special import comb
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline
import seaborn as sns


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
# Total length of the chain: L
# Maximum photon number: N

# Consider (single) non-interacting electron case!
def hop(n, d, L):
    ans = np.zeros((L+1,L+1), dtype = complex)
    ans[n+d][n] = 1
    return ans


def site(n, L):
    ans = np.zeros((L+1,L+1), dtype = complex)
    ans[n][n] = 1
    return ans

def sigma_up():
    ans = np.zeros((2,2), dtype = complex)
    ans[0][1] = 1
    return ans

def sigma_down():
    ans = np.zeros((2,2), dtype = complex)
    ans[1][0] = 1
    return ans

# Consider at most one photonic mode. 
def A_dag(N):
    ans = np.zeros((N+1,N+1), dtype = complex)
    for n in range(N):
        ans[n+1][n] = np.sqrt(n+1)
    return ans

def A(N):
    ans = np.zeros((N+1,N+1), dtype = complex)
    for n in range(1,N+1):
        ans[n-1][n] = np.sqrt(n)
    return ans

def A_phi(N, dphi):
    return A(N) * ((1j + 1) + (1j - 1) * np.exp(1j * dphi))/2
    #return A(N) * ((1 + 1j)* np.exp(1j*dphi) + (1j - 1))/2

def A_dag_phi(N, dphi):
    return A_dag(N) * np.conj(((1j + 1) + (1j - 1) * np.exp(1j * dphi))/2)
    #return A_dag(N) * np.conj(((1 + 1j)* np.exp(1j*dphi) + (1j - 1))/2)
    

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
        J_n = J*(1 + (-1)**n * delta)
        H += -J_n * hop(n, 1, L) + np.conj(-J_n * hop(n, 1, L)).T
        H += Delta * site(n, L)
    H += Delta * site(L, L)
    return H

# Hilbert Space: H_{ph} \otimes H_{e} \otimes H_{a}
def H_couple(L, J, delta, Delta, g, N, A0, dphi, Omega, g0):
    dim = (N+1) * (L+1)
    ans = np.zeros((dim,dim), dtype = complex)
    mat = np.zeros((L+1, L+1), dtype = complex)
    mat[0][0] = Omega
    mat[0][1] = g0
    mat[1][0] = np.conj(g0)
    ans += np.kron(np.identity(N+1), mat)
    for n in range (1,L):
        J_n = J * (1 + (-1)**n * delta)
        mat = -J_n * expm(-1j*g/np.sqrt(L) * A0*(np.kron(A_phi(N, dphi), np.identity(L+1)) + np.kron(A_dag_phi(N, dphi), np.identity(L+1)) )) @ np.kron(np.identity(N+1), hop(n, 1, L))
        ans +=  mat + np.conj(mat.T)
    
    for n in range(1,L+1):
        ans += Delta * np.kron(np.identity(N+1), site(n,L))

    return ans


def H_photon(N, dphi, omega):
    ans = np.zeros((N+1,N+1), dtype = complex)
    ans += omega * (1/2 * np.identity(N+1) + A_dag_phi(N, dphi) @ A_phi(N, dphi))
    final = np.kron(ans, np.identity(L+1))
    return final




def H_e(n, L, J, delta, Delta, g, N, A0, dphi, Omega, g0):
    mat = np.zeros((N+1,N+1), dtype = complex)
    mat[n][n] = 1
    H = H_couple(L, J, delta, Delta, g, N, A0, dphi, Omega, g0) @ np.kron(mat, np.identity(L+1))
    H_reshape = H.reshape((N+1, L+1, N+1, L+1))
    ans = np.tensordot(H_reshape, np.identity(N+1), axes=([0,2], [0,1]))
    #print(H.shape)
    #print(H_reshape.shape)
    return ans
# Trace out the degrees of freedom of the first and the third dimension of H_reshape, and integrate it with the first and the second dimensions of the identity~


def H_t(omega, N, L, dphi, Delta, g0, g, J, A0, delta, Omega):
    ans = np.zeros(((N+1)*(L+1), (N+1)*(L+1)), dtype = complex)
    ans += H_couple(L, J, delta, Delta, g, N, A0, dphi, Omega, g0) + H_photon(N, dphi, omega)
    return ans



def eigen(H):
    es, vs = np.linalg.eig(H)
    # We need to implement sorting from small to large values~
    sorted_indices = np.argsort(es)
    es_new = es[sorted_indices]
    vs_new = vs[:,sorted_indices]
    idxs = np.linspace(1,len(vs[0]),len(vs[0]))
    return es_new, vs_new, idxs

def Rho_e(rho, N, L):
    rho_reshape = rho.reshape((N+1, L+1, N+1, L+1))
    ans = np.tensordot(rho_reshape, np.identity(N+1), axes=([0,2], [0,1]))
    return ans


def Rho_ph(rho, N, L):
    rho_reshape = rho.reshape((N+1, L+1, N+1, L+1))
    ans = np.tensordot(rho_reshape, np.identity(L+1), axes=([1,3], [0,1]))
    return ans

# A good thing is that here the Hamiltonian is time independent. 
def crank_nicolson(psi, dt, omega, N, L, dphi, Delta, g0, g, J, A0, delta, Omega):
    """Perform a single Crank-Nicolson step."""
    # Identity matrix
    I = np.identity(len(psi))
    
    # Compute the Hamiltonian matrix at time t
    H = H_t(omega, N, L, dphi, Delta, g0, g, J, A0, delta, Omega)
    
    
    # Construct the matrices for the linear system
    A = (I - 1j * dt/2 * H)
    B = (I + 1j * dt/2 * H)
    
    # Solve the linear system
    psi_next = solve(A, B @ psi)
    return psi_next

def run(T, omega, N, L, dphi, Delta, g0, g, J, A0, delta, Omega):
    
    psi_times = []
    rho_times = []
    dt = 0.5 
    times = np.arange(0, T, dt)
    
   
    
    # Initial state (should be normalized)
    ket_ph = np.zeros(N+1, dtype = complex)
    ket_ph[N] = 1
    #print(ket_ph)
    ket_e = np.zeros(L+1, dtype = complex)
    ket_e[0] = 1
    psi_0 = np.kron(ket_ph, ket_e)
    
    psi_0 = psi_0 / np.linalg.norm(psi_0)
    rho_0 = np.outer(psi_0, np.conj(psi_0))
    #print(np.inner(np.conj(psi_0),psi_0))
    
    # Time evolution
    psi_t = psi_0
    #print(len(psi_t))
    for t in times:
        psi_times.append(psi_t)
        psi_t = crank_nicolson(psi_t, dt, omega, N, L, dphi, Delta, g0, g, J, A0, delta, Omega)
        rho_t = np.outer(psi_t, np.conj(psi_t))
        rho_times.append(rho_t)
        #print(np.inner(np.conj(psi_t),psi_t))
    #psi_t now holds the state at time T
    #print("State at time T:", psi_t)
    
    return psi_t, psi_0, psi_times, times, rho_0, rho_t, rho_times
# psi_t: the final state; psi_0: the initial state; psi_times: the list for the states at each time increment; times: the array for time increments. 

def prob_up(rho, rho_0, N, L):
    mat = rho @ rho_0
    prob = np.real(np.trace(mat))
    return prob
# Notice that the initial density matrix is exactly the projector onto the excited state of the two levle atom!      
        
def fisher_info(T, omega, N, L, dphi, Delta, g0, g, J, A0, delta, Omega, diff=1e-6):
    psi_t_small, psi_0_small, psi_times_small, times_small, rho_0_small, rho_t_small, rho_times_small = run(T, omega, N, L, dphi - diff, Delta, g0, g, J, A0, delta, Omega)
    psi_t, psi_0, psi_times, times, rho_0, rho_t, rho_times = run(T, omega, N, L, dphi, Delta, g0, g, J, A0, delta, Omega)
    psi_t_big, psi_0_big, psi_times_big, times_big, rho_0_big, rho_t_big, rho_times_big = run(T, omega, N, L, dphi + diff, Delta, g0, g, J, A0, delta, Omega)
    d_rho = (rho_t_big - rho_t_small)/(2*diff)
    L = solve_lyapunov(rho_t, 2*d_rho)
    F = np.real(np.trace(rho_t @ L @ L))
    
    return F

def generate_data(prob, M):
    data = np.random.binomial(n = 1, p = prob, size = M)
    return data

# Uniform probability distribution
def prior_func(theta, a, b):
    return 1 / (b - a) if a <= theta <= b else 0

def likelihood_func(m, M, prob):
    # psi_t, psi_0, psi_times, times, rho_0, rho_t, rho_times = run(T, omega, N, L, theta, Delta, g0, g, J, A0, delta, Omega)
    # prob = prob_up(rho_t, rho_0, N, L)
    con_prob = comb(M, m, exact = True) * prob**m *(1 - prob)**(M - m)
    return con_prob

def unormalized_posterior_func(a, b, m, M, prob, theta):
    ans = prior_func(theta, a, b) * likelihood_func(m, M, prob)
    return ans

def Bayes_estimation(a, b, M, T, omega, N, L, dphi, Delta, g0, g, J, A0, delta, Omega):
    num = 50
    ite = 50
    thetas = np.linspace(a, b, num)
    # Initialize the posterior values. 
    posterior_values = np.zeros(num)
    # Also generate prior values as a reference.
    prior_values = np.array([prior_func(theta, a, b) for theta in thetas])
    # Generate the conditional probability.
    psi_t, psi_0, psi_times, times, rho_0, rho_t, rho_times = run(T, omega, N, L, dphi, Delta, g0, g, J, A0, delta, Omega) # dphi encodes the information of the real parameter!!!
    prob = prob_up(rho_t, rho_0, N, L)
    print(f'prob = {prob:.2f}')
    # dphi is the real value. 
    
    for j in range(num):
        theta = thetas[j]
        values = np.zeros(ite)
        for i in range(ite):
            print(f'Iteration num = {i}')
            data = generate_data(prob, M)
            m = np.sum(data)
            
            psi_t, psi_0, psi_times, times, rho_0, rho_t, rho_times = run(T, omega, N, L, theta, Delta, g0, g, J, A0, delta, Omega) # dphi encodes the information of the real parameter!!!
            prob_theta = prob_up(rho_t, rho_0, N, L)
            post_value = unormalized_posterior_func(a, b, m, M, prob_theta, theta)
            values[i] = post_value
            print(f'post_value = {post_value}')
        real_value = np.mean(values)
        posterior_values[j] = real_value
        print(f'real_value = {real_value}')
    
   
    
    # Normalize the posterior values.
    posterior_values /= np.sum(posterior_values) * (thetas[1] - thetas[0])
    print(f'posterior_values = {posterior_values}')
    
    print(f'prior_values = {prior_values}')
    return thetas, prior_values, posterior_values


def Bayes_estimation_delta(a, b, M, T, omega, N, L, dphi, Delta, g0, g, J, A0, delta_real, Omega):
    num = 50
    ite = 10
    deltas = np.linspace(a, b, num)
    # Initialize the posterior values.
    posterior_values = np.zeros(num)
    # Initialize the conditional values.
    #condition_values = np.zeros(num)
    # Generate the prior values.
    prior_values = np.array([prior_func(delta, a, b) for delta in deltas])
    
    # Generate the conditional probability.
    psi_t, psi_0, psi_times, times, rho_0, rho_t, rho_times = run(T, omega, N, L, dphi, Delta, g0, g, J, A0, delta_real, Omega) # delta_real encodes the information of the real parameter!!!
    prob = prob_up(rho_t, rho_0, N, L)
    print(f'prob = {prob:.2f}')
    
    
    for j in range(num):
        delta = deltas[j]
        values = np.zeros(ite)
        #c_values = np.zeros(ite)
        print(f'point num = {j}')
        for i in range(ite):
            print(f'Iteration num = {i}')
            data = generate_data(prob, M)
            m = np.sum(data)
            
            psi_t, psi_0, psi_times, times, rho_0, rho_t, rho_times = run(T, omega, N, L, dphi, Delta, g0, g, J, A0, delta, Omega) 
            prob_delta = prob_up(rho_t, rho_0, N, L)
            print(f'prob_delta = {prob_delta}')
            con_value = likelihood_func(m, M, prob_delta)
            #c_values[i] = con_value
            print(f'con_value = {con_value}')
            post_value = unormalized_posterior_func(a, b, m, M, prob_delta, delta)
            values[i] = post_value
            print(f'post_value = {post_value}')
        real_value = np.mean(values)
        posterior_values[j] = real_value
        print(f'real_value = {real_value}')
        #real_c_value = np.mean(c_values)
        #condition_values[j] = real_c_value
       # print(f'real_c_value = {real_c_value}')
    
   
    
    # Normalize the posterior values.
    posterior_values /= np.sum(posterior_values) * (deltas[1] - deltas[0])
    print(f'posterior_values = {posterior_values}')
    
    
    
    
    # Check if the posterior distribution is normalized
    integral = simps(posterior_values, deltas)
    print(f"Integral of the normalized posterior distribution: {integral}")
    return deltas, prior_values, posterior_values

# Iterating over ite samples!
# def Bayes_estimation(rho, rho_0, theta, N, L, M, ite = 100):
#     low = 0
#     high = np.pi
#     prior_prob = prior_func(theta, low, high)
#     prob = prob_up(rho, rho_0, N, L)
#     data = generate_data(prob, M)
#     m = 0
#     for i in range(M):
#         if (data[i] == 1):
#             m += 1
#     con_prob = comb(M, m, exact = True) * prob**m * (1 - prob)**(M - m)
#     post_prob = prior_prob * con_prob
    
#     #Normalization
    
#     return post_prob



# \hat{gamma} = \sum_{n} (-1)^{n} \hat{c}_{n}^{\dag} \hat{c}_{n}
def gamma_hat(L):
    ans = np.zeros((L+1,L+1), dtype = complex)
    for n in range(1, L+1):
        ans += -(-1)**n * site(n,L)
    return ans

def F_norm(A, L):
    val = 0
    for i in range(L+1):
        for j in range(L+1):
            val += (np.abs(A[i][j]))**2
    ans = np.sqrt(val)
    return ans

def check_chiral(H):
    gamma = gamma_hat(L)
    mat = H @ gamma + gamma @ H
    ans = F_norm(mat, L)
    return ans



#%%
L = 16
J = 1
delta = 0.5
Delta = 0 # Chemical potential
N = 10
n = 5
g = 1
g0 = 0.25
A0 = 1
dphi = np.pi/4
Omega = 1 # Omega is real Delta, the detuning of the two level system!

T = 20
T_max = 50
omega = 0.15    


a = 0
#b = np.pi
b = 1
delta_real = 0.5
M = 50


#start = time.time()
#print(H_ssh(L,J,delta,Delta))
#print(H_couple(L,J,delta,Delta,g,N,A0))
#end = time.time()
#print(end - start)

def plot_energy():
    #H = H_ssh(L, J, delta, Delta, Omega, g0)
    #H = H_couple(L, J, delta, Delta, g, N, A0, dphi, Omega, g0)
    for n in range(N+1):
        H = H_e(n, L, J, delta, Delta, g, N, A0, dphi, Omega, g0)
        #H = H_ssh(L, J, delta, Delta, Omega, g0)
        #H = H_couple(L, J, delta, Delta, g, N, A0, dphi, Omega, g0)
        plt.figure(figsize=(10, 8))
        sns.heatmap(np.real(H), annot=True, cmap='coolwarm')
        plt.title('Electronic Hamiltonian')
        plt.show()
        
        es, vs, idxs = eigen(H)
        plt.figure()
        plt.title(f'Energy Spectrum: delta = {delta:.2f}, g = {g:.2f}, g0 = {g0:.2f}, n = {n:.2f} ')
        plt.xlabel('idx')
        plt.ylabel('E')
        plt.scatter(idxs, es, marker = 'o', s=8)
        plt.plot(idxs, np.zeros(len(idxs)) + Delta, color = 'r', linestyle = '--')
        #plt.plot(idxs, np.ones(len(idxs))*2*delta + Delta, color = 'm', linestyle = '--')
        #plt.plot(idxs, np.ones(len(idxs))*(-2)*delta + Delta, color = 'g', linestyle = '--')
        #plt.legend()
        plt.show()
    
    return
#plot_energy()


def plot_state():
    #H = H_ssh(L, J, delta, Delta, Omega, g0)
    #H = H_couple(L, J, delta, Delta, g, N, A0, dphi, Omega, g0)
    H = H_e(n, L, J, delta, Delta, g, N, A0, dphi, Omega, g0)
    #plt.figure(figsize=(10, 8))
    #sns.heatmap(np.real(H), annot=True, cmap='coolwarm')
    #plt.title('Electronic Hamiltonian')
    #plt.show()
    
    es, vs, idxs = eigen(H)
    for i in range(len(vs[1])):
        plt.figure()
        plt.title(f'State Configuration: delta = {delta:.2f}, g = {g:.2f}, g0 = {g0:.2f}')
        plt.xlabel('idx')
        plt.ylabel(r'|$\psi$|')
        plt.bar(idxs, np.abs(vs[:,i]))
        plt.show()
    return
#plot_state()

# def plot_energy_para():
#     deltas = np.linspace(0,1,20)
#     gs = np.linspace(0,20,3)
#     colors = ['r', 'g', 'b']
#     plt.figure()
#     plt.xlabel(r'$\delta$')
#     plt.ylabel('E')
#     plt.title(r'Energy spectrum with $\delta$')
#     for k in range(3):
#         g = gs[k]
#         c = colors[k]
#         Es = np.zeros((L,len(deltas)), dtype = complex)
#         for j in range(len(deltas)):
#             delta = deltas[j]
#             H = H_e(n, L, J, delta, Delta, g, N, A0, dphi, Omega, g0)
#             es, vs, idxs = eigen(H)
#             for i in range(len(es)):
#                 Es[i][j] = es[i]
#         for i in range(len(es)):
#             plt.scatter(deltas, Es[i,:], color = c, marker = 'o')
#     plt.show()   
#     return
# #plot_energy_para()


def plot_ex_prob():
    global probs
    gs = np.linspace(0,10,3)
    #g = 0.1
    #dphis = [0, np.pi/4, np.pi/3, np.pi/2, np.pi]
    #labels = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{3}$', r'$\frac{\pi}{2}$', r'$\pi$' ]
    #dphi = np.pi/8
    plt.figure(figsize = (12,8))
    plt.title(r'Excited state prob,  $dphi = \frac{\pi}{4}$')
    plt.xlabel('t')
    plt.ylabel('P(t)')
    
    
    for j in range(len(gs)):
        g = gs[j]
        psi_t, psi_0, psi_times, times, rho_0, rho_t, rho_times = run(T, omega, N, L, dphi, Delta, g0, g, J, A0, delta, Omega)
        probs = np.zeros(len(times))
        for i in range(len(times)):
            rho = rho_times[i]
            probs[i] = prob_up(rho, rho_0, N, L)
        plt.plot(times, probs, linestyle = '-', label = f'g = {g:.2f}')
        
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    return
#plot_ex_prob()

def plot_density_mat():
    psi_t, psi_0, psi_times, times, rho_0, rho_t, rho_times = run(T, omega, N, L, dphi, Delta, g0, g, J, A0, delta, Omega)
    for i in range(len(times)):
        rho = rho_times[i]
        #print(np.trace(rho))
        rho_e = Rho_e(rho, N, L)
        rho_ph = Rho_ph(rho, N, L)
        #print(np.trace(rho_e), np.trace(rho_ph))
        
        fig, axs = plt.subplots(1,2, figsize = (14,7))
        axs[0].set_title('Electronic Density Matrix')
        sns.heatmap(np.real(rho_e), ax = axs[0], annot = True, fmt = '.2f', cmap = 'coolwarm' )
        
        axs[1].set_title('Photonic Density Matrix')
        sns.heatmap(np.real(rho_ph), ax = axs[1], annot = True, fmt = '.2f', cmap = 'viridis')
        
        plt.tight_layout()
        plt.show()
        
    return
#plot_density_mat()

def plot_fisher():
    Ts = np.linspace(1,T_max,20)
    Fs = np.zeros(len(Ts))
    #dphis = [0, np.pi]
    #labels = ['0', r'$\pi$']
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
            F = fisher_info(T, omega, N, L, dphi, Delta, g0, g, J, A0, delta, Omega, diff=1e-6)
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

def plot_Bayesian():
    thetas, prior_values, posterior_values = Bayes_estimation(a, b, M, T, omega, N, L, dphi, Delta, g0, g, J, A0, delta, Omega)
    plt.figure()
    plt.title(r'$\Delta\phi$ after Bayes estimation')
    plt.xlabel(r'$\Delta\phi$')
    plt.ylabel(r'$\rho (\Delta\phi)$')
    plt.plot(thetas, prior_values, linestyle = '--', label = 'prior')
    plt.plot(thetas, posterior_values, linestyle = '-', label = 'post')
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.show()
    return
#plot_Bayesian()

def plot_Bayesian_delta():
    deltas, prior_values, posterior_values = Bayes_estimation_delta(a, b, M, T, omega, N, L, dphi, Delta, g0, g, J, A0, delta_real, Omega)
    plt.figure()
    plt.title(r'$\delta$ after Bayes estimation')
    plt.xlabel(r'$\delta$')
    plt.ylabel(r'$\rho (\delta)$')
    plt.plot(deltas, prior_values, linestyle = '--', label = 'prior')
    plt.scatter(deltas, posterior_values, marker = 'x', label = 'post')
    #plt.plot(deltas, condition_values, linestyle = '--', label = 'condition')
    
    # Cubic spline interpolation
    spline_interp = make_interp_spline(deltas, posterior_values, k=3)  # k=3 for cubic spline
    x_spline = np.linspace(min(deltas), max(deltas), 500)
    y_spline = spline_interp(x_spline)
    plt.plot(x_spline, y_spline, linestyle = '-', label = 'post fit')
    
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.show()
    return
plot_Bayesian_delta()