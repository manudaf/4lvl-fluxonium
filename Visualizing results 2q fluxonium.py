'''
Minimal code to import a json file and visualizing the results that we've obtained.
'''
#%% Imports

import json
# Load json file
with open("fluxonium_2q.json", "r", encoding="utf-8") as file:
    sol_dict_json = json.load(file)

import numpy as np
import jax as jx
import jax.numpy as jnp
jx.config.update("jax_enable_x64", True)
import pandas as pd 
import matplotlib.pyplot as plt
import scipy as sp
import scipy.constants as const
from itertools import combinations


plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,      # Use LaTeX for text rendering
    "figure.dpi": 300,        # High resolution for figures

    "font.size": 13,          # Base font size
    "axes.titlesize": 15,     # Title font
    "axes.labelsize": 13,     # X/Y label size
    "xtick.labelsize": 12,    # X tick labels
    "ytick.labelsize": 12,    # Y tick labels
    "legend.fontsize": 12,    # Legend text
    "figure.titlesize":15#,
    #"text.latex.preamble": r"\boldmath"   # Figure title
    # "axes.labelweight": "bold",   # X/Y label weight
    # "font.weight": "bold",        # Base font weight
    # "xtick.major.width": 1.5,     # Thicker X ticks
    # "ytick.major.width": 1.5,     # Thicker Y ticks
    # "axes.linewidth": 1.5         # Thicker axis spines
})
import scienceplots

plt.style.use("science")
 
import seaborn as sns
from cycler import cycler
import matplotlib as mpl
 
# Colormap for heatmaps/images
mpl.rcParams["image.cmap"] = "crest_r"
 
# Colors for line plots
N_COLORS = 10
MAKO_COLORS = sns.color_palette("crest_r", n_colors=N_COLORS)
 
mpl.rcParams["axes.prop_cycle"] = cycler(color=MAKO_COLORS)


_2pi = 2.0 * jx.numpy.pi
kb=const.Boltzmann
hbar=const.hbar




#%% Functions


N_grid=sol_dict_json['Setup']['N_grid']
N_grid=2**8
phi_max=sol_dict_json['Setup']['phi_max']
N_trunc=sol_dict_json['Setup']['N_trunc']
N_plot_levels=sol_dict_json['Setup']['N_plot_levels']

# =============================================================================
# Phase grid and FFT helpers
# =============================================================================
def phase_grid(phi_max: float, N: int):
    dphi = 2.0 * phi_max / N
    phi  = -phi_max + dphi * np.arange(N)
    return phi, dphi
 
 
def spectral_kvec(phi_max: float, N: int): # M: I assume this is the conjugate momenta of phi (aka n)
    _, dphi = phase_grid(phi_max, N)
    return 2.0 * np.pi * np.fft.fftfreq(N, d=dphi)
 
 
phi_vec_np, dphi = phase_grid(phi_max, N_grid)
k_vec_np         = spectral_kvec(phi_max, N_grid)
 
phi_vec     = jx.numpy.array(phi_vec_np, dtype=jx.numpy.float64) #M: phi op?
k_vec       = jx.numpy.array(k_vec_np,   dtype=jx.numpy.float64) #M: n op?
 
Phi2_diag   = phi_vec ** 2
CosPhi_diag = jx.numpy.cos(phi_vec)
SinPhi_diag = jx.numpy.sin(phi_vec) ## Why you need this?

# =============================================================================
# Basic helpers
# =============================================================================
def dag(O):
    return jx.numpy.conj(O.T)


# =============================================================================
# FFT-based operators (charge Q̂) and diagonal operator (phase φ̂)
# =============================================================================

@jx.jit
def Q_apply(psi):
    return jx.numpy.fft.ifft(k_vec * jx.numpy.fft.fft(psi))
 
 
@jx.jit
def Q2_apply(psi):
    return jx.numpy.fft.ifft((k_vec ** 2) * jx.numpy.fft.fft(psi))
 
 
@jx.jit
def Q_apply_batch(V):
    return jx.vmap(Q_apply, in_axes=1, out_axes=1)(V)
 
 
@jx.jit
def Q2_apply_batch(V):
    return jx.vmap(Q2_apply, in_axes=1, out_axes=1)(V)
 
 
# φ̂ is diagonal in the phase basis, so applying it is just an
# elementwise multiplication by the phi grid — no FFT required.
@jx.jit
def Phi_apply(psi):
    return phi_vec * psi
 
 
@jx.jit
def Phi_apply_batch(V):
    return jx.vmap(Phi_apply, in_axes=1, out_axes=1)(V)
 
 
def _build_Q2_matrix():
    eye    = jx.numpy.eye(N_grid, dtype=jx.numpy.complex128)
    Q2_mat = Q2_apply_batch(eye)
    residual = float(jx.numpy.max(jx.numpy.abs(Q2_mat - dag(Q2_mat))))
    assert residual < 1e-8, f"Q² not Hermitian; residual = {residual:.2e}"
    return Q2_mat
 
 
Q2_grid = _build_Q2_matrix()
 
 
Q2_grid = _build_Q2_matrix()

# =============================================================================
# Fluxonium Hamiltonian — returns 4 lowest levels, Q_trunc, and Phi_trunc
# =============================================================================
@jx.jit
def Fluxonium_fft(EC, EL, EJ, fx):
 
    phi_ext = 2.0 * jx.numpy.pi * fx
 
    H = (
        4.0 * EC * Q2_grid
        + 0.5 * EL * jx.numpy.diag((phi_vec - phi_ext) ** 2)
 
        - EJ * jx.numpy.diag(CosPhi_diag)
    )
 
    evals, evecs = jx.scipy.linalg.eigh(H)
 
    evals = evals - evals[0]
 
    E_low   = evals[:N_trunc]
 
    V_low   = evecs[:, :N_trunc]
 
    Q_trunc   = dag(V_low) @ Q_apply_batch(V_low) ## Funcion Q_apply(psi) hace \hat n |psi>. Esta linea calcula <psi_i|n|psi_j>
    Phi_trunc = dag(V_low) @ Phi_apply_batch(V_low)
 
    return E_low, Q_trunc, Phi_trunc


# =============================================================================
# Coherence time functions
# =============================================================================
@jx.jit
def T1_fun(i,j,E,Q,EC):

    # Qc=2e6
    w_ij=jnp.abs(E[i]-E[j])*1e9 #rad/s
    Qc=1e6*(_2pi*6e9/jnp.abs(w_ij))**(0.7)
    T= 20e-3 # mK

    def coth(x):
        return 1.0 / jnp.tanh(x)

    return (8*EC*1e9/Qc)*(jnp.abs(Q[i,j])**2)*(1+coth(hbar*w_ij/(2*kb*T)))

@jx.jit
def _wi_of_fx(EL,fx, Phi_i):
    '''
    We are computing T_phi as: d(w_i-w_j)/dfx. Since the hamiltonian has an analytical derivative we can use the Hellmann-Feynman:

    dE_i/dfx=<phi_i|dH/dfx|phi_i>=<phi_i|(-2pi*EL(phi-2pi*fx))|phi_i> = -2pi*EL(<phi>-2pi*fx)
    '''

    return (-1)*_2pi*EL*(Phi_i-_2pi*fx)


@jx.jit
def Tdeph(i,j,Phi,EL,fx):
    '''
    i,j are the levels
    fx is the external flux

    We are computing T_phi as: d(w_i-w_j)/dfx. Since the hamiltonian has an analytical derivative we can use the Hellmann-Feynman:

    dE_i/dfx=<phi_i|dH/dfx|phi_i>=<phi_i|(-2pi*EL(phi-2pi*fx))|phi_i>=2pi*EL(<phi>-2pi*fx)
    '''

    dE_i=_wi_of_fx(EL,fx,Phi[i,i])
    dE_j=_wi_of_fx(EL,fx,Phi[j,j])

    _dwij_dfx=jx.numpy.abs(dE_i-dE_j)*1e9 # rad/s


    A = 1e-6
    wlow = 1e-9*_2pi
    texp = 1e4

    gamma_deph = (jx.numpy.sqrt(2)*A*_dwij_dfx*jx.numpy.sqrt(jx.numpy.abs(jx.numpy.log(wlow*texp))))

    return 1e6/gamma_deph

# =============================================================================
# NumPy Hamiltonian builder (for flux sweeps)
# =============================================================================
def _build_H_numpy(EC, EL, EJ, fx):
    cosf = np.cos(2.0 * np.pi * fx)
    sinf = np.sin(2.0 * np.pi * fx)
    return (
        4.0 * EC * np.asarray(Q2_grid, dtype=np.complex128)
        + 0.5 * EL * np.diag(phi_vec_np ** 2)
        - EJ * np.diag(cosf * np.cos(phi_vec_np) + sinf * np.sin(phi_vec_np))
    )
 
 
def Fluxonium_fft_full(EC, EL, EJ, fx):
    H = _build_H_numpy(EC, EL, EJ, fx)
    evals, evecs = np.linalg.eigh(H)
    evals = evals - evals[0]


     
    E_low   = evals[:N_trunc]
    
    V_low   = evecs[:, :N_trunc]
    
    Q_trunc   = dag(V_low) @ Q_apply_batch(V_low) ## Funcion Q_apply(psi) hace \hat n |psi>. Esta linea calcula <psi_i|n|psi_j>
    Phi_trunc = dag(V_low) @ Phi_apply_batch(V_low)
    return evals, evecs

#%% Visualizing pareto results

F=np.array(sol_dict_json['Solutions']['Costs'])
X=np.array(sol_dict_json['Solutions']['Parameters']) # Solutions (params)


# We flip the signs since we were maximizing the costs, now we want to see the magnitudes right

F_scale0=-1e3*F[:,0] ## In MHz. Min Anharm
F_scale1=-F[:,1] ## T_1
F_scale2=-F[:,2] ## T_phi



fig = plt.figure()


max_matrix_element=sol_dict_json['Restrictions']['max_matrix_element']
min_matrix_element=sol_dict_json['Restrictions']['min_matrix_element']


plt.scatter(F_scale1,F_scale2,c=(F_scale0),cmap='viridis', marker='.',s=0.7)
plt.colorbar(label=r"$\alpha_{min} \ [\textrm{MHz}]$")
plt.xlabel(r'$T_1 \ [\mu s]$')
plt.ylabel(r'$T_\varphi \ [\mu s]$')
plt.title(rf'Pareto for ${min_matrix_element}<|\langle \hat n\rangle|<{max_matrix_element}$')
plt.savefig(f'pareto for n_in_{min_matrix_element}_{max_matrix_element} low EL.png')

plt.show()

plt.figure(1)
plt.scatter(F_scale0, F_scale1,marker='.',s=0.7)
plt.xlabel(r"$\alpha_{min} \ [\textrm{MHz}]$")
plt.ylabel(r"$T_1 [\mu s]$")
plt.show()


plt.figure(2)
plt.scatter(X[:,0], X[:,2],marker='.',s=0.7)
plt.xlabel(r"$E_C\ [\textrm{GHz}]$")
plt.ylabel(r"$E_J\ [\textrm{GHz}]$")
plt.show()

plt.figure(3)
plt.scatter(X[:,0], X[:,1],marker='.',s=0.7)
plt.xlabel(r"$E_C\ [\textrm{GHz}]$")
plt.ylabel(r"$E_L\ [\textrm{GHz}]$")
plt.show()


print('-'*80)

'''
This will compute a table that tells you the relation that each parameter has with each other.
The values span from [-1,1].
When it's near 1 it means that when value1 increases, value2 will also increase
When it's near -1 it means that when value 1 increases, value2 decreases.

This is useful to see paterns and relations between the parameters.
'''

df = pd.DataFrame(np.hstack([X,F]), columns=[ "EC", "EL", "EJ", "fx", "Alpha_min", "T_1", 'T_phi'] ) 

print(df.corr())

print('-'*80)



# Analysing how much the parameters are changing to see if we are exploring the whole landscape 

print('Range of parameters:')
for i, name in enumerate(["EC", "EL", "EJ", "fx"]):
    print(f'{name} goes from {X[:, i].min()} to {X[:, i].max()}')
print('-'*50)

 





#%% Visualizing ONE solution of the pareto

#Printing the solutions
keys=['min_Ec','max_alpha','max_T1','max_Tphi','max_T2','max_w_q']
units=['GHz', 'MHz', 'µs','µs','µs', 'GHz']

for i,key in enumerate(keys):
    print('-'*125)
    EC_sol, EL_sol, EJ_sol, fx_sol=sol_dict_json["Solutions"][key]
    label=sol_dict_json["Solutions"]['Labels'][key]
    print(f'Solution {key} ({label:.3f} {units[i]}): Ec={EC_sol}, EL={EL_sol}, EJ={EJ_sol}, fx={fx_sol}')
print('-'*125)

## Choose an optimal solution from keys
## For example, we choose the solution with the lowest Ec

EC_opt, EL_opt, EJ_opt, fx_opt=sol_dict_json["Solutions"][keys[0]]

print(f'Considering the solution {key}: Ec={EC_opt}, EL={EL_opt}, EJ={EJ_opt}, fx={fx_opt}')

## Transition frequencies
E, Q, Phi = Fluxonium_fft(EC_opt,EL_opt,EJ_opt,fx_opt)


trans_freq =np.zeros((N_trunc,N_trunc), dtype=float)
for i in range(N_trunc):
    for j in range(N_trunc):
            trans_freq[i,j]= np.abs(E[j]-E[i])# Convert to GHz

trans_freq=trans_freq/_2pi

trans_freq_trunc = trans_freq#[:5,:5]
N = trans_freq_trunc.shape[0]

# Plot

fig, ax = plt.subplots(figsize=(6,5))

vmax = np.max(np.abs(trans_freq_trunc))

im = ax.imshow(
    trans_freq_trunc,
    cmap="viridis",      # diverging
    vmin=0,
    vmax=vmax
)

labels = [rf"$|{i}\rangle$" for i in range(N)]
ax.set_xticks(np.arange(N))
ax.set_yticks(np.arange(N))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

for i in range(N):
    for j in range(N):
        ax.text(j, i,
                f"{trans_freq_trunc[i,j]:.2f}",
                ha="center", va="center",
                color="white", size=15)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r"$\omega_{ij}/(2\pi)$ [GHz]")

ax.set_aspect("equal")
plt.tight_layout()
# plt.savefig("transition_frequencies_2q_fluxonium.png", dpi=300)
plt.show()

## Finding anharm min:

upper_triangle=trans_freq[np.triu_indices(6, k=1)][:-1]

anharms=[]
indices=[]
for comb in combinations(range(len(upper_triangle)),2):
    anharms.append(np.abs(upper_triangle[comb[0]]-upper_triangle[comb[1]]))
    indices.append(comb)

idx=indices[np.argmin(anharms)]
rows, collums = np.triu_indices(trans_freq.shape[0], k=1)


print(f"Minimum anharmonicity = {np.min(anharms)*1e3} MHz. Between w{rows[idx[0]]}{collums[idx[0]]} and w{rows[idx[1]]}{collums[idx[1]]}")



## Transition Matrix elements
for i, comb in enumerate(combinations(range(N_trunc),2)):
    print('-'*50)
    print(f'Transition: |<{comb[0]}|n|{comb[1]}>|: {jnp.abs(Q[comb[0],comb[1]]):.4f}')

## Coherence times

fx_scan = np.linspace(0.01, 0.49, 201)
Tdeph_l = np.zeros((201, 4, 4))
Tde1_l = np.zeros((201, 4, 4))
for k in range(201):
    for i in range(4):
        for j in range(4):
            E_t, Q_t, Phi_t = Fluxonium_fft(EC_opt,EL_opt,EJ_opt,fx_scan[k])

            Tdeph_l[k,i,j] = Tdeph(i,j,Phi_t,EL_opt, fx_scan[k])
            Tde1_l[k,i,j]=T1_fun(i,j,E_t,Q_t,EC_opt)

######## TPhi
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.set_ylabel(r'$T_\varphi[\mu s]$')
ax.set_xlabel(r'$\varphi_e$')
ax.set_ylim(1e0,1e3)

ax.set_yscale('log')
for i in range(4):
    for j in range(i+1,4):
        ax.plot(fx_scan, Tdeph_l[:,i,j], label=f'{i,j}')

Tdeph_total=1/(1/Tdeph_l[:,0,1] + 1/Tdeph_l[:,0,2] + 1/Tdeph_l[:,1,3] + 1/Tdeph_l[:,2,3] + 1/Tdeph_l[:,1,2]) #maybe this is wrong (lack of states)

ax.plot(fx_scan, Tdeph_total, label=f'Total', color='r')
ax.axvline(fx_opt, color="k",    ls="--", lw=1.2,
           label=rf"$f_{{x}}={fx_opt:.3f}$")
ax.legend(loc = 'upper right')


######## T1
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.set_ylabel(r'$T_1[\mu s]$')
ax.set_xlabel(r'$\varphi_e$')
ax.set_ylim(1e0,1e3)

ax.set_yscale('log')
for i in range(4):
    for j in range(i+1,4):
        ax.plot(fx_scan, Tde1_l[:,i,j], label=f'{i,j}')

Tde1_total=1/(1/Tde1_l[:,0,1] + 1/Tde1_l[:,0,2] + 1/Tde1_l[:,1,3] + 1/Tde1_l[:,2,3] + 1/Tde1_l[:,1,2])#maybe this is wrong (lack of states)

ax.plot(fx_scan, Tde1_total, label=f'Total', color='r')
ax.axvline(fx_opt, color="k",    ls="--", lw=1.2,
           label=rf"$f_{{x}}={fx_opt:.3f}$")
ax.legend(loc = 'upper right')


######## T2
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.set_ylabel(r'$T_2[\mu s]$')
ax.set_xlabel(r'$\varphi_e$')
ax.set_ylim(1e0,1e3)

ax.set_yscale('log')
t2_list=[]
for i in range(4):
    for j in range(i+1,4):
        t2=1/((1/(2*np.array(Tde1_l[:,i,j])))+(1/(np.array(Tdeph_l[:,i,j]))))
        t2_list.append(1/t2)
        ax.plot(fx_scan, t2, label=f'{i,j}')

T2_tot=1/sum(np.array(t2_list))
ax.plot(fx_scan, T2_tot, label=f'Total', color='r')
ax.axvline(fx_opt, color="k",    ls="--", lw=1.2,
           label=rf"$f_{{x}}={fx_opt:.3f}$")
ax.legend(loc = 'upper right')
# %%
#check