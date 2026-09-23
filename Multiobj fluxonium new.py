# -*- coding: utf-8 -*-
"""
Fluxonium single-device two-qubit optimization.
 
Find one set of circuit parameters (EC, EL, EJ) such that:
 
    Anharmonicity min > 250 MHz
    T2 > 5 µs
    0.01 < |<n>| < 0.8
 
where  ωij = Ei - Ej,  nij = |⟨i|Q̂|j⟩| (charge matrix elements),
and    phi_ij = |⟨i|φ̂|j⟩| (flux/phase matrix elements).
 
The joint cost is minimised over the 4-parameter vector (EC, EL, EJ, fx).
After optimisation, a full flux sweep visualises both crossing conditions,
now including the φ̂ (phase) operator matrix elements alongside Q̂.
"""



# %% Imports
# =============================================================================
# Imports and JAX configuration
# =============================================================================
import numpy as np
import scipy as sp
import scipy.constants as const
import jax as jx
import jax.numpy as jnp
from itertools import combinations
import pandas as pd 
import json

sol_dict={} # Dictionary where we are going to store the solutions


## Optimizer
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

 
jx.config.update("jax_enable_x64", True)
jx.config.update("jax_platform_name", "cpu")
 
try:
    import optax
    _OPTAX_AVAILABLE = False
except ImportError:
    _OPTAX_AVAILABLE = True
    print("optax not found – skipping Adam warm-start.")
 

import matplotlib.pyplot as plt
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
 
# %% Helpers
# =============================================================================
# Basic helpers
# =============================================================================
def dag(O):
    return jx.numpy.conj(O.T)
 
 
def dag_np(O):
    return np.conj(O.T)
 

'''
M: This function is useful for optimization when your params are varying several orders of magnitude.
What it does is to let the variable x (that will be optimized) to be unbounded and returning a value
between 0 and 1. So x\in(-infty,infty) and sigmoid\in[0,1]. That means that x can be what ever.
'''

def sigmoid(x):
    return 1.0 / (1.0 + jx.numpy.exp(-x))
 
 
def inv_sigmoid(y):
    eps = 1e-14
    y = np.clip(y, eps, 1.0 - eps)
    return np.log(y / (1.0 - y))
 
 
# %% Bounds
# =============================================================================
# Parameter ranges
# =============================================================================
_2pi = 2.0 * jx.numpy.pi
kb=const.Boltzmann
hbar=const.hbar

# Bound for driving amplitude
max_driving_amplitude=0.5 # In GHz. This is the max Omega/2 in order to not heat the system significantly with the driving.


 
## Bounds in GHz
EC_range = (0.2, 3.8)
EL_range = (0.5, 5)
EJ_range = (2.5, 20)
# fx_range  = (0.0, 0.1)
fx_range  = (0.49, 0.5)

sol_dict["Bounds"]={}
sol_dict["Bounds"]["Ec"]=EC_range
sol_dict['Bounds']['EL']=EL_range
sol_dict['Bounds']['EJ']=EJ_range
sol_dict['Bounds']['fx']=fx_range


## Bounds cost functions
sol_dict["Restrictions"]={}

min_anharm=0.25 # GHz
min_qubit_freq=1.0 # GHz. This is only for the transition w01, see the cost function 1.
sol_dict['Restrictions']['min_anharm']=min_anharm
sol_dict['Restrictions']['min_qubit_freq']=min_qubit_freq


max_matrix_element=0.8
min_matrix_element=0.01

sol_dict['Restrictions']['max_matrix_element']=max_matrix_element
sol_dict['Restrictions']['min_matrix_element']=min_matrix_element


min_coherence_time_t1=5 #\mu s
min_coherence_time_tphi=5 #\mu s

sol_dict['Restrictions']['min_coherence_time_t1']=min_coherence_time_t1
sol_dict['Restrictions']['min_coherence_time_tphi']=min_coherence_time_tphi


kill_transition_bool=False # With this bolean you can kill some transitions, doesn't behave very well
kill_transitions_list=[(3,4)] # Tuple. Can be several
max_element_to_kill=0.01 ## Max value of the matrix element transition that we want to kill

sol_dict['Restrictions']['kill_transition_bool']=kill_transition_bool
sol_dict['Restrictions']['kill_transitions_list']=kill_transitions_list
sol_dict['Restrictions']['max_element_to_kill']=max_element_to_kill


 
# Grid / truncation — 4 levels needed (indices 0–3)
N_grid        = 128
phi_max       = 12.0
N_trunc       = 6 # Number of energy levels that we consider in a truncated fluxonium (does not compute every level)
N_plot_levels = 10 # Number of energy levels that we consider in a full fluxonium diagonalization (computes every level but only shows N)
 
assert N_trunc >= 4, "N_trunc must be >= 4." # This is bc we want to do a 4 level syst.

sol_dict['Setup']={}
sol_dict['Setup']['N_grid']=N_grid
sol_dict['Setup']['phi_max']=phi_max
sol_dict['Setup']['N_trunc']=N_trunc
sol_dict['Setup']['N_plot_levels']=N_plot_levels


 
# Cost weights. M: we will try to avoid this by using multiobj
_scale_w   = _2pi * 1.0   # normalise ω differences by 1 GHz
_eps       = 1e-12
 
w_crossing = 1.0    # (ωij - ωkl)² / scale_w²
w_matrix   = 1.0    # (nij - nkl)²
w_amp      = 0.01   # 1/(nij+nkl) regulariser against trivial zeros
w_flat     = 1.0    # (dω/dfx)² / scale_w² sweet-spot regulariser
 
 
# %% Functions
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

# T1

@jx.jit
def Gamma_1_fun(i,j,E,Q,EC):

    w_ij=(E[i]-E[j])*1e9 # to Hz
    # Qc=1e6*(6e9/jnp.abs(w_ij))**(0.7) ## This is more for the transmon

    def tan_dc(w):
        tan_0=4e-6
        epsilon=0.26
        w_ref=6e9 # Hz
        tan_c=tan_0*(jnp.abs(w)/w_ref)**epsilon
        return tan_c

    Qc=1/tan_dc(w_ij)
    T= 20e-3 # K

    def coth(x):
        return 1.0 / jnp.tanh(x)

 
    EC=EC*1e9 # To Hz
    gamma_t1=(8*EC/Qc)*(jnp.abs(Q[i,j])**2)*(1+coth(hbar*w_ij*_2pi/(2*kb*T)))*jnp.sign(w_ij)


    return gamma_t1 # s^-1


@jx.jit
def T1_fun(E,Q,EC):

    gamma_1_list=[]
    for i in range(4): # logical levels
        for j in range(N_trunc): # logical levels + some leakage
            if i!=j:
                gamma_1_list.append(Gamma_1_fun(i, j, E, Q, EC))

    gamma_1=jnp.sum(jnp.array(gamma_1_list))

    T1_tot=1e6/gamma_1 # Total T1 time, in \mu s


    return T1_tot





# T_phi

@jx.jit
def _wi_of_phi_ext(EL,fx, Phi_i):
    '''
    We are computing T_phi as: d(w_i-w_j)/dphi_ext. Since the hamiltonian has an analytical derivative we can use the Hellmann-Feynman:

    dE_i/dphi_ext=<phi_i|dH/dphi_ext|phi_i>=<phi_i|(-1*EL(phi-2pi*fx))|phi_i> = -EL(<phi>-2pi*fx)
    '''

    return (-1)*EL*(Phi_i-_2pi*fx) #GHz


@jx.jit
def Gamma_phi_fun(i,j,Phi,EL,fx):
    '''
    i,j are the levels
    fx is the external flux

    We are computing T_phi as: d(w_i-w_j)/dfx. Since the hamiltonian has an analytical derivative we can use the Hellmann-Feynman:

    dE_i/dfx=<phi_i|dH/dfx|phi_i>=<phi_i|(-2pi*EL(phi-2pi*fx))|phi_i>=2pi*EL(<phi>-2pi*fx)
    '''

    dE_i=_wi_of_phi_ext(EL,fx,Phi[i,i])
    dE_j=_wi_of_phi_ext(EL,fx,Phi[j,j])

    _dwij_dfx=jx.numpy.abs(dE_i-dE_j)*1e9 # Hz


    A = 1e-6 # adimensional, en scqubits ponen A*Phi_0 pero tb derivan dw_ij/dPhi así que se irán los Phi_0.
    wlow = 1e-9
    texp = 1e4

    gamma_deph = (jx.numpy.sqrt(2)*A*_dwij_dfx*jx.numpy.sqrt(jx.numpy.abs(jx.numpy.log(wlow*texp))))

    return gamma_deph #s^-1


@jx.jit
def Tdeph(Phi,EL,fx):

    gamma_phi_l=[]
    for comb in combinations(range(4),2): # Only logical lvls. Symmetric: (0,1) = (1,0)
        gamma_phi_l.append(Gamma_phi_fun(comb[0],comb[1],Phi,EL,fx))

    gamma_tot=jnp.sum(jnp.array(gamma_phi_l))
    Tphi_tot=1e6/gamma_tot # to \mu s

    return Tphi_tot


# T2

@jx.jit
def T2_fun(EC,EL,EJ,fx):
    E_t, Q_t, Phi_t = Fluxonium_fft(EC,EL,EJ,fx)

    Tphi_tot = Tdeph(Phi_t,EL,fx)
    T1_tot=T1_fun(E_t,Q_t,EC)

    T2_tot=1/(1/(2*T1_tot)+1/Tphi_tot)

    return T2_tot, T1_tot, Tphi_tot #\mu s


# Single qubit gate time tg:



@jx.jit(static_argnames="states")
def tg_fun(EC,EL,EJ,fx,theta=jnp.pi,states=(0,1)):
    E, Q, _ = Fluxonium_fft(EC,EL,EJ,fx)

    # Compute the transition frequecies between levels:
    
    trans_freq =jnp.zeros((N_trunc,N_trunc), dtype=float)
    for i in range(N_trunc):
        for j in range(N_trunc):
                trans_freq = trans_freq.at[i, j].set(jnp.abs(E[j]-E[i]))


    wd= trans_freq[states[0],states[1]] #transition w_ij, in GHz

    sum_list=[] #Approx 1
    subs_list=[] #Approx 2


    for i in states:
        for j in range(N_trunc):
            if i!=j:
                omega_sum_t=jnp.abs((trans_freq[i,j]+wd)/Q[i,j])
                sum_list.append(omega_sum_t)

                if ((i,j) != states) and ((j,i) != states):
                    omega_subs_t=jnp.abs((trans_freq[i,j]-wd)/Q[i,j])
                    subs_list.append(omega_subs_t)

    # Approx 1: RWA
    gamma_rwa=5e-2
    RWA=gamma_rwa*jnp.min(jnp.array(sum_list))

    # Approx 2: Secular
    gamma_sec=5e-2
    sec=gamma_sec*jnp.min(jnp.array(subs_list))

    # Gate time:
    Omega2_max=jnp.min(jnp.array([RWA,sec,max_driving_amplitude]))*1e9*_2pi #convert to rad/s

    tg=jnp.abs((theta/(Q[states[0],states[1]]*Omega2_max))*1e9) #convert to ns

    return tg, (RWA,sec)
 

# %% Cost Funtions
# =============================================================================
# Per-flux-point cost terms
# =============================================================================


## =============================================================================
## Manu's work space
## =============================================================================
'''
The plan here is to build 3 cost functions:
    1. Anharmonicity: Look for anharmonicity in all energy transitions. The lowest one is the most problematic.
        Should be above \alpha>300 MHz
    2. Matrix elements of n: In order to do the driving. Ideal: |<n>|~0.2. But look for 0.01<|<n>|<0.5
    3. T_phi: Dephasing time, depends on the derivative. Look for T_phi>5 \mu s.

To achieve this it would be faster and more accurate to compute it symbolically since the Hamiltonian is exactly derivable.
'''




@jx.jit
def _cost1_anharm(E):
    '''
    Anharmonicity: Look for anharmonicity in all energy transitions. The lowest one is the most problematic.
    Should be above \alpha>300 MHz
    '''
   
    omegas =jnp.zeros((N_trunc,N_trunc), dtype=float)
    for i in range(N_trunc):
        for j in range(N_trunc):
                omegas = omegas.at[i,j].set(jnp.abs(E[j]-E[i])) # GHz

    upper_triangle=omegas[jnp.triu_indices(N_trunc, k=1)][:-1] # Dont consider the w45, k=1 removes the diagonal

    anharms=[]
    for comb in combinations(range(len(upper_triangle)),2):
        anharms.append(jnp.abs(upper_triangle[comb[0]]-upper_triangle[comb[1]]))

    anharms = jnp.array(anharms)  # GHz

    ## Objective:
    cost_obj=(-1)*jnp.min(anharms)

    ## Restriction:
    restrictions=[]

    # Min anharmonicity
    restriction1=[min_anharm-jx.numpy.min(anharms)] # My anharmocity has to be at least min_anharm
    restrictions+=restriction1

    # Min qubit frequency
    restriction2=[] # only take into account the logical levels

    # Modify this to include more levels
    restriction2.append(min_qubit_freq-jnp.abs(omegas[0,1]))

    restrictions+=restriction2

    return cost_obj, restrictions







@jx.jit
def _cost2_matrix_elements(E,Q,EC):
    '''
    Here we code the T1 objective.

    Matrix elements of n: 0.01<|<n>|<0.8.

    We also can put the restriction of killing the leakage channel.
    '''

    ## Objective:
    T1_tot=T1_fun(E,Q,EC)

    cost=(-1)*jx.numpy.abs(T1_tot)
    
    ## Restrictions
    retrictions=[]

    # Min T1
    restriction_t1=min_coherence_time_t1-T1_tot
    retrictions.append(restriction_t1)


    # kill the leakage
    if kill_transition_bool:
        kill_leakage_rest=[]
        for i in range(len(kill_transitions_list)):
            a,b=kill_transitions_list[i]
            kill_leakage_rest.append(jx.numpy.abs(Q[a, b])-max_element_to_kill)
        
        retrictions+=kill_leakage_rest
    
    
    # Max matrix elements
    n_max=[]

    for comb in combinations(range(6), 2): #every combination of 2 elements within 6 levels
        if comb == (4, 5): # the transition between the leakages is irrelevant
            continue
        n_max.append(jx.numpy.abs(Q[comb[0], comb[1]]) - max_matrix_element)

    

    


    retrictions+=n_max

    # Min matrix elements
    n_min=[]

    for comb in combinations(range(4),2):## every combination of 2 elements within 4 levels
        n_min.append(min_matrix_element-jx.numpy.abs(Q[comb[0], comb[1]])) 



    retrictions+=n_min


    return cost, retrictions




@jx.jit
def _cost3_coherence_time(Phi, EL, fx):
    '''
    Phi: output of Fluxonium_fft, is a matrix
    EL, fx: scalars
    '''

    
    Tphi_tot=Tdeph(Phi,EL,fx)

    cost=(-1)*jx.numpy.abs(Tphi_tot)

    restriction1=min_coherence_time_tphi-Tphi_tot

    restrictions=[restriction1]


    return cost, restrictions


#%% Optimization

# --------------------------------------------------------------------------
# 1. Search space
# --------------------------------------------------------------------------
BOUNDS = {
    "EC": (EC_range[0], EC_range[1]),
    "EL": (EL_range[0], EL_range[1]),
    "EJ": (EJ_range[0], EJ_range[1]),
    "fx": (fx_range[0], fx_range[1]),
}
PARAM_NAMES = list(BOUNDS.keys())  # ["EC", "EL", "EJ", "fx"]
XL = np.array([BOUNDS[p][0] for p in PARAM_NAMES])
XU = np.array([BOUNDS[p][1] for p in PARAM_NAMES])
 
# NSGA-II run settings
POP_SIZE = 400
N_GEN = 200
CROSSOVER_PROB = 0.9
CROSSOVER_ETA = 15
MUTATION_ETA = 20
seed=2

sol_dict['Optimizer']={}
sol_dict['Optimizer']['POP_SIZE']=POP_SIZE
sol_dict['Optimizer']['N_GEN']=N_GEN
sol_dict['Optimizer']['CROSSOVER_PROB']=CROSSOVER_PROB
sol_dict['Optimizer']['CROSSOVER_ETA']=CROSSOVER_ETA
sol_dict['Optimizer']['MUTATION_ETA']=MUTATION_ETA
sol_dict['Optimizer']['seed']=seed


 
 
# --------------------------------------------------------------------------
# 2. Combine your three cost functions into a single JAX evaluation
# --------------------------------------------------------------------------
@jx.jit
def evaluate_single(params):
    EC, EL, EJ, fx = params
 
    E, Q, Phi = Fluxonium_fft(EC, EL, EJ, fx)
 
    cost1, r1 = _cost1_anharm(E)
    cost2, r2 = _cost2_matrix_elements(E,Q,EC)
    cost3, r3 = _cost3_coherence_time(Phi, EL, fx)
 
    objectives = jx.numpy.stack([cost1, cost2, cost3])
    constraints = jx.numpy.concatenate([jx.numpy.stack(r1), jx.numpy.stack(r2), jx.numpy.stack(r3)])
 
    return objectives, constraints
 
 
# Vectorize over the whole population in one shot (M: I guess this is what paralellizes things)
evaluate_batch = jx.jit(jx.vmap(evaluate_single))
 
 
def _infer_dims():
    """Run once on the bounds midpoint to learn n_obj / n_constr automatically."""
    x0 = jx.numpy.asarray((XL + XU) / 2.0)
    objs, cons = evaluate_single(x0)
    return int(objs.shape[0]), int(cons.shape[0])
 
 
N_OBJ, N_CONSTR = _infer_dims()
# print(f"Detected {N_OBJ} objectives and {N_CONSTR} constraints.")
 
 
# --------------------------------------------------------------------------
# 3. pymoo Problem wrapper
# --------------------------------------------------------------------------
class FluxoniumProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=len(PARAM_NAMES),
            n_obj=N_OBJ,
            n_constr=N_CONSTR,
            xl=XL,
            xu=XU,
        )
 
    def _evaluate(self, X, out, *args, **kwargs):
        X_jax = jx.numpy.asarray(X)
        objs, cons = evaluate_batch(X_jax)
        out["F"] = np.asarray(objs)
        out["G"] = np.asarray(cons)
 
 
# --------------------------------------------------------------------------
# 4. Algorithm setup and run
# --------------------------------------------------------------------------
problem = FluxoniumProblem()
 
algorithm = NSGA2(
    pop_size=POP_SIZE,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
    mutation=PM(eta=MUTATION_ETA),
    eliminate_duplicates=True,
)

termination = get_termination("n_gen", N_GEN)

res = minimize(
    problem,
    algorithm,
    termination,
    seed=seed,
    save_history=False,
    verbose=True
)


## This is for the result to return something of it does not converge

#     ,
#     return_least_infeasible=True  
# )

# print("Least infeasible X:", res.X)
# print("Least infeasible CV:", res.CV)
# print("Per-restriction G:", res.G)

# Store the solutions

F=res.F # Costs
X=res.X # Solutions (params)
sol_dict['Solutions']={}
sol_dict['Solutions']['Costs']=F
sol_dict['Solutions']['Parameters']=X

# We flip the signs since we were maximizing the costs, now we want to see the magnitudes right

F_scale0=-1e3*F[:,0] ## In MHz. Min Anharm
F_scale1=-F[:,1] ## T_1
F_scale2=-F[:,2] ## T_phi

## Characteristic points

sol_dict["Solutions"]['Labels']={}


## 1. Minimum Ec (max C):
index=np.argmin(X[:,0]) # X[all_solutions,Ec]
EC_opt, EL_opt, EJ_opt, fx_opt= X[index,:]
sol=(EC_opt,EL_opt,EJ_opt,fx_opt)
sol_dict["Solutions"]['min_Ec']=sol
sol_dict["Solutions"]['Labels']['min_Ec']=EC_opt


## 2. Max alpha
index=np.argmax(F_scale0)
EC_opt, EL_opt, EJ_opt, fx_opt= X[index,:]
sol=(EC_opt,EL_opt,EJ_opt,fx_opt)
sol_dict["Solutions"]['max_alpha']=sol
sol_dict["Solutions"]['Labels']['max_alpha']=F_scale0[index]


## 3. Max T1
index=np.argmax(F_scale1)
EC_opt, EL_opt, EJ_opt, fx_opt= X[index,:]
sol=(EC_opt,EL_opt,EJ_opt,fx_opt)
sol_dict["Solutions"]['max_T1']=sol
sol_dict["Solutions"]['Labels']['max_T1']=F_scale1[index]


## 4. Max Tphi
index=np.argmax(F_scale2)
EC_opt, EL_opt, EJ_opt, fx_opt= X[index,:]
sol=(EC_opt,EL_opt,EJ_opt,fx_opt)
sol_dict["Solutions"]['max_Tphi']=sol
sol_dict["Solutions"]['Labels']['max_Tphi']=F_scale2[index]

## 5. Max T2

T2_list=1/((1/(2*np.array(F_scale1)))+(1/(np.array(F_scale2))))
index=np.argmax(T2_list)
EC_opt, EL_opt, EJ_opt, fx_opt= X[index,:]
sol=(EC_opt,EL_opt,EJ_opt,fx_opt)
sol_dict["Solutions"]['max_T2']=sol
sol_dict["Solutions"]['Labels']['max_T2']=T2_list[index]



## 6. Max w_q
def omega_matrix_min(E):
    omegas =jnp.zeros((N_trunc,N_trunc), dtype=float)
    for i in range(N_trunc):
        for j in range(N_trunc):
                omegas = omegas.at[i,j].set(jnp.abs(E[j]-E[i])) # GHz

    upper_triangle=omegas[jnp.triu_indices(N_trunc, k=1)][:-1] # Dont consider the w45

    return min(upper_triangle)


min_omega_list=[] 
for i in range(len(X)):
    EC_t, EL_t, EJ_t, fx_t= X[i,:]
    E,_,_=Fluxonium_fft(EC_t,EL_t,EJ_t,fx_t)
    min_omega_list.append(omega_matrix_min(E))

index=np.argmax(min_omega_list)
EC_opt, EL_opt, EJ_opt, fx_opt= X[index,:]
sol=(EC_opt,EL_opt,EJ_opt,fx_opt)
sol_dict["Solutions"]['max_w_q']=sol
sol_dict["Solutions"]['Labels']['max_w_q']=min_omega_list[index]



#%% Save json

def to_serializable(obj):
    """Recursively convert numpy/JAX arrays and other non-JSON types to serializable ones."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.complexfloating, complex)):
        return {"real": obj.real.item() if hasattr(obj.real, "item") else obj.real,
                "imag": obj.imag.item() if hasattr(obj.imag, "item") else obj.imag}
    elif hasattr(obj, "tolist"):  # catches JAX arrays too
        return obj.tolist()
    else:
        return obj

with open("fluxonium_2q.json", "w", encoding="utf-8") as file:
    json.dump(to_serializable(sol_dict), file, indent=4, ensure_ascii=False)



# %% Visualizing pareto results

# We flip the signs since we were maximizing the costs, now we want to see the magnitudes right

F_scale0=-1e3*F[:,0] ## In MHz. Min Anharm
F_scale1=-F[:,1] ## T_1
F_scale2=-F[:,2] ## T_phi



fig = plt.figure()



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




# Computing gate times for pareto front

tg_list=np.zeros((len(X),4,4))
max_tg=[]
for k in range(len(X)):
    EC,EL,EJ,fx=X[k]
    for i in range(4):
        for j in range(4):
            if (i!=j) and (i<j):
                tg_list[k,i,j]=tg_list[k,j,i]=tg_fun(EC,EL,EJ,fx,theta=jnp.pi,states=(i,j))[0]

    max_tg.append(np.max(tg_list[k,:,:]))

plt.figure(4)
plt.scatter(T2_list, max_tg,marker='.',s=0.7)
plt.xlabel(r"$T_2\ [\mu s]$")
plt.ylabel(r"$\max(t_g)\ [ns]$")
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
    EC_sol, EL_sol, EJ_sol, fx_sol=sol_dict["Solutions"][key]
    label=sol_dict["Solutions"]['Labels'][key]
    print(f'Solution {key} ({label:.3f} {units[i]}): Ec={EC_sol}, EL={EL_sol}, EJ={EJ_sol}, fx={fx_sol}')
print('-'*125)



## Choose an optimal solution
## For example, we choose the solution with the lowest Ec

sol_key=keys[0]

EC_opt, EL_opt, EJ_opt, fx_opt=sol_dict["Solutions"][sol_key]

print(f'Considering the solution {sol_key}: Ec={EC_opt}, EL={EL_opt}, EJ={EJ_opt}, fx={fx_opt}')

## Transition frequencies
E, Q, Phi = Fluxonium_fft(EC_opt,EL_opt,EJ_opt,fx_opt)





trans_freq =np.zeros((N_trunc,N_trunc), dtype=float)
for i in range(N_trunc):
    for j in range(N_trunc):
            trans_freq[i,j]= np.abs(E[j]-E[i])# Convert to GHz

trans_freq=trans_freq

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
fx_sweep=201
fx_scan = np.linspace(0.01, 0.49, fx_sweep)

Tdeph_l = np.zeros((fx_sweep, 4, 4))
Tphi_total_fx=np.zeros((fx_sweep))

Tde1_l = np.zeros((fx_sweep, 4, N_trunc))
T1_total_fx=np.zeros((fx_sweep))

for k in range(fx_sweep):
    E_t, Q_t, Phi_t = Fluxonium_fft(EC_opt,EL_opt,EJ_opt,fx_scan[k])

    for comb in combinations(range(4),2): # Only logical lvls. Symmetric: (0,1) = (1,0)
        Tdeph_l[k,comb[0],comb[1]] = 1e6/Gamma_phi_fun(comb[0],comb[1],Phi_t,EL_opt,fx_scan[k]) # In \mu s
    Tphi_total_fx[k]=Tdeph(Phi_t,EL_opt,fx_scan[k])

    for i in range(4): # logical levels
        for j in range(N_trunc): # logical levels + some leakage
            if i!=j:
                Tde1_l[k,i,j] = 1e6/Gamma_1_fun(i, j, E_t, Q_t, EC_opt)
    T1_total_fx[k] = T1_fun(E_t,Q_t,EC_opt)



######## TPhi
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.set_ylabel(r'$T_\varphi[\mu s]$')
ax.set_xlabel(r'$\varphi_e$')
ax.set_ylim(1e0,1e3)

ax.set_yscale('log')
for i in range(4):
    for j in range(i+1,4):
        ax.plot(fx_scan, Tdeph_l[:,i,j], label=f'{i,j}')


ax.plot(fx_scan, Tphi_total_fx, label=f'Total', color='r')
ax.axvline(fx_opt, color="k",    ls="--", lw=1.2,
           label=rf"$f_{{x}}={fx_opt:.3f}$")


index_opt=np.argmin(np.abs(fx_scan-fx_opt))
ax.plot(fx_scan[index_opt],Tphi_total_fx[index_opt],ls='',color='k',label=fr'$T_\varphi^*={Tphi_total_fx[index_opt]:.3f}$')
ax.legend(ncols=2,loc = 'upper right')


######## T1
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.set_ylabel(r'$T_1[\mu s]$')
ax.set_xlabel(r'$\varphi_e$')
ax.set_ylim(1e0,1e3)

ax.set_yscale('log')
for i in range(4):
    for j in range(4):
        if i!=j:
            ax.plot(fx_scan, Tde1_l[:,i,j], label=f'{i,j}')


ax.plot(fx_scan, T1_total_fx, label=f'Total', color='r')
ax.axvline(fx_opt, color="k",    ls="--", lw=1.2,
           label=rf"$f_{{x}}={fx_opt:.3f}$")


ax.plot(fx_scan[index_opt],T1_total_fx[index_opt],ls='',color='k',label=fr'$T_1^*={T1_total_fx[index_opt]:.3f}$')
ax.legend(ncols=2,loc = 'upper right')



######## T2
T2_sweep=1/(1/(2*T1_total_fx)+1/Tphi_total_fx)
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.set_ylabel(r'$T_2[\mu s]$')
ax.set_xlabel(r'$\varphi_e$')
ax.set_ylim(1e0,1e3)

ax.set_yscale('log')

ax.plot(fx_scan, T2_sweep, label=f'Total', color='r')
ax.axvline(fx_opt, color="k",    ls="--", lw=1.2,
           label=rf"$f_{{x}}={fx_opt:.3f}$")

ax.plot(fx_scan[index_opt],T2_sweep[index_opt],ls='',color='k',label=fr'$T_2^*={T2_sweep[index_opt]:.3f}$')
ax.legend(ncols=2,loc = 'upper right')












 
# %% Analysing results (Fran)
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
    return evals, evecs
 
 
# %% Functions
# =============================================================================
# Gauge-fixed flux sweep
# =============================================================================
def gauge_fix_to_previous(prev_vecs, new_vecs):
    fixed = new_vecs.copy()
    for k in range(fixed.shape[1]):
        ov = np.vdot(prev_vecs[:, k], fixed[:, k])
        if np.abs(ov) > 1e-6:
            fixed[:, k] *= np.conj(ov) / np.abs(ov)
    return fixed
 
 
def reorder_and_gauge_fix(prev_vecs, new_vecs, n_keep):
    overlap = np.abs(prev_vecs.conj().T @ new_vecs[:, :n_keep]) ** 2
    _, col_ind = sp.optimize.linear_sum_assignment(-overlap)
    reordered = new_vecs[:, :n_keep][:, col_ind]
    reordered = gauge_fix_to_previous(prev_vecs, reordered)
    return reordered, col_ind
 
 
def tracked_flux_sweep(EC, EL, EJ, fx_array, n_keep=N_plot_levels):
    """
    Sweep flux with gauge-fixed eigenstates.
 
    Returns
    -------
    E_tracked   : (n_flux, n_keep)
    N_tracked   : (n_flux, 6)   charge matrix elements
                  columns: n02, n13, n10, n32, n01, n12
    Phi_tracked : (n_flux, 6)   phase (φ̂) matrix elements
                  columns: phi02, phi13, phi10, phi32, phi01, phi12
    W_tracked   : (n_flux, 4)   columns: ω10, ω20, ω31, ω32
    """
    n_flux = len(fx_array)
 
    E_tracked   = np.zeros((n_flux, n_keep))
    N_tracked   = np.zeros((n_flux, 6))
    Phi_tracked = np.zeros((n_flux, 6))
    W_tracked   = np.zeros((n_flux, 4))
 
    evals0, vecs0 = Fluxonium_fft_full(EC, EL, EJ, fx_array[0])
    prev_vecs     = vecs0[:, :n_keep].copy()
    E_tracked[0]  = evals0[:n_keep]
 
    def _fill_row(i, vecs):
        # Charge operator Q̂ (spectral derivative via FFT)
        QV  = np.asarray(Q_apply_batch(jx.numpy.array(vecs, dtype=jx.numpy.complex128)))
        Qp  = dag_np(vecs) @ QV
        N_tracked[i, 0] = np.abs(Qp[0, 2])   # n02
        N_tracked[i, 1] = np.abs(Qp[1, 3])   # n13
        N_tracked[i, 2] = np.abs(Qp[1, 0])   # n10
        N_tracked[i, 3] = np.abs(Qp[3, 2])   # n32
        N_tracked[i, 4] = np.abs(Qp[0, 1])   # n01
        N_tracked[i, 5] = np.abs(Qp[1, 2])   # n12
 
        # Phase operator φ̂ (diagonal in this basis)
        PhiV = phi_vec_np[:, None] * vecs
        Phip = dag_np(vecs) @ PhiV
        Phi_tracked[i, 0] = np.abs(Phip[0, 2])   # phi02
        Phi_tracked[i, 1] = np.abs(Phip[1, 3])   # phi13
        Phi_tracked[i, 2] = np.abs(Phip[1, 0])   # phi10
        Phi_tracked[i, 3] = np.abs(Phip[3, 2])   # phi32
        Phi_tracked[i, 4] = np.abs(Phip[0, 1])   # phi01
        Phi_tracked[i, 5] = np.abs(Phip[1, 2])   # phi12
 
        W_tracked[i, 0] = E_tracked[i, 1] - E_tracked[i, 0]   # ω10
        W_tracked[i, 1] = E_tracked[i, 2] - E_tracked[i, 0]   # ω20
        W_tracked[i, 2] = E_tracked[i, 3] - E_tracked[i, 1]   # ω31
        W_tracked[i, 3] = E_tracked[i, 3] - E_tracked[i, 2]   # ω32
 
    _fill_row(0, prev_vecs)
 
    for i in range(1, n_flux):
        evals, vecs           = Fluxonium_fft_full(EC, EL, EJ, fx_array[i])
        tracked_vecs, col_ind = reorder_and_gauge_fix(prev_vecs, vecs, n_keep)
        E_tracked[i]          = evals[col_ind]
        _fill_row(i, tracked_vecs)
        prev_vecs = tracked_vecs.copy()
 
    return E_tracked, N_tracked, Phi_tracked, W_tracked
 
 
# %% Flux sweep
# =============================================================================
# Flux sweep
# =============================================================================
fx_scan = np.linspace(0.01, 0.49, fx_sweep)
 
print("\nRunning flux sweep...")
E_scan, N_scan, Phi_scan, W_scan = tracked_flux_sweep(EC_opt, EL_opt, EJ_opt, fx_scan)
print("Done.")
 
 
# %%
# =============================================================================
# Plot 1: transition frequencies vs flux
# W_scan cols: 0=ω10, 1=ω20, 2=ω31, 3=ω32
# =============================================================================
w_labels = [r"$\omega_{10}$", r"$\omega_{20}$", r"$\omega_{31}$", r"$\omega_{32}$"]
colors   = ["C0", "C1", "C2", "C3"]
 
fig, ax = plt.subplots(figsize=(8.0, 4.0))
 
for k, (label, color) in enumerate(zip(w_labels, colors)):
    ax.plot(fx_scan, W_scan[:, k] / _2pi, lw=1.8, color=color, label=label)
 
ax.axvline(fx_opt, color="k",    ls="--", lw=1.2,
           label=rf"$f_{{x}}={fx_opt:.3f}$ ($\omega_{{20}}=\omega_{{31}}$)")

 
ax.set_xlabel(r"$\varphi_x$", fontsize=14)
ax.set_ylabel(r"Transition frequency / $2\pi$ (GHz)", fontsize=12)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
# fig.savefig("CrossingFrequencies.pdf")
plt.show()
 
 
# %%
# =============================================================================
# Plot 2: charge (Q̂) matrix elements vs flux
# N_scan cols: 0=n02, 1=n13, 2=n10, 3=n32, 4=n01, 5=n12
# =============================================================================
n_labels = [r"$|n_{02}|$", r"$|n_{13}|$", r"$|n_{10}|$",
            r"$|n_{32}|$", r"$|n_{01}|$", r"$|n_{12}|$"]
styles = [("-", 2.5, "C0"), ("-", 2.5, "C1"),
          ("--", 2.5, "C0"), ("--", 2.5, "C1")]
 
fig, ax = plt.subplots(figsize=(8.0, 4.0))
 
for k, (label, (ls, lw, color)) in enumerate(zip(n_labels, styles)):
    ax.plot(fx_scan, N_scan[:, k], ls, lw=lw, color=color, label=label)
 
ax.axvline(fx_opt, color="k",    ls="--", lw=1.2,
           label=rf"$f_{{x}}={fx_opt:.3f}$")

 
# ax.set_yscale("log")
ax.set_xlabel(r"$\varphi_x$", fontsize=14)
ax.set_ylabel(r"$|\langle i|\hat{Q}|j\rangle|$", fontsize=14)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
# fig.savefig("CrossingMatrixElements.pdf")
plt.show()
 
 
# %%
# =============================================================================
# Plot 2b: phase (φ̂) matrix elements vs flux — NEW
# Phi_scan cols: 0=phi02, 1=phi13, 2=phi10, 3=phi32, 4=phi01, 5=phi12
# =============================================================================
phi_labels = [r"$|\varphi_{02}|$", r"$|\varphi_{13}|$", r"$|\varphi_{10}|$",
              r"$|\varphi_{32}|$", r"$|\varphi_{01}|$", r"$|\varphi_{12}|$"]
 
fig, ax = plt.subplots(figsize=(8.0, 4.0))
 
for k, (label, (ls, lw, color)) in enumerate(zip(phi_labels, styles)):
    ax.plot(fx_scan, Phi_scan[:, k], ls, lw=lw, color=color, label=label)
 
ax.axvline(fx_opt, color="k",    ls="--", lw=1.2,
           label=rf"$f_{{x}}={fx_opt:.3f}$")

 
ax.set_xlabel(r"$\varphi_x$", fontsize=14)
ax.set_ylabel(r"$|\langle i|\hat{\varphi}|j\rangle|$", fontsize=14)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
# fig.savefig("CrossingPhiMatrixElements.pdf")
plt.show()
 
 
# %%
# =============================================================================
# Plot 3: energy spectrum vs flux
# =============================================================================
fig, ax = plt.subplots(figsize=(8.0, 4.0))
 
for n in range(1, N_plot_levels):
    ax.plot(fx_scan, E_scan[:, n] / _2pi, lw=1.5)
 
ax.axvline(fx_opt, color="k",    ls="--", lw=1.2,
           label=rf"$f_{{x}}={fx_opt:.3f}$")

 
ax.set_xlabel(r"$\varphi_x$", fontsize=14)
ax.set_ylabel(r"$\omega_n / 2\pi$ (GHz)", fontsize=14)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
# fig.savefig("EnergySpectrum.pdf")
plt.show()
 

# %%

fx_scan = np.linspace(0.01, 0.49, fx_sweep)
Tdeph_l = np.zeros((fx_sweep, 4, 4))
for k in range(fx_sweep):
    for i in range(4):
        for j in range(4):
            _, _, Phi_t = Fluxonium_fft(EC_opt,EL_opt,EJ_opt,fx_scan[k])

            Tdeph_l[k,i,j] = Tdeph(i,j,Phi_t,EL_opt, fx_scan[k])

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.set_ylabel(r'$T_\varphi[\mu s]$')
ax.set_xlabel(r'$\varphi_e$')
ax.set_ylim(1e0,1e3)

ax.set_yscale('log')
for i in range(4):
    for j in range(i+1,4):
        ax.plot(fx_scan, Tdeph_l[:,i,j], label=f'{i,j}')

Tdeph_total=1/(1/Tdeph_l[:,0,1] + 1/Tdeph_l[:,0,2] + 1/Tdeph_l[:,1,3] + 1/Tdeph_l[:,2,3] + 1/Tdeph_l[:,1,2])

ax.plot(fx_scan, Tdeph_total, label=f'Total', color='r')
ax.axvline(fx_opt, color="k",    ls="--", lw=1.2,
           label=rf"$f_{{x}}={fx_opt:.3f}$")
ax.legend(loc = 'upper right')



