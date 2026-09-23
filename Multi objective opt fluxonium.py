# -*- coding: utf-8 -*-
"""
Fluxonium single-device two-flux-point optimization.
 
Find one set of circuit parameters (EC, EL, EJ) and two flux values
(fx1, fx2) such that:
 
    At fx1:  ω20 = ω31   AND   |n02| = |n13|
    At fx2:  ω10 = ω32   AND   |n10| = |n32|
 
where  ωij = Ei - Ej,  nij = |⟨i|Q̂|j⟩| (charge matrix elements),
and    phi_ij = |⟨i|φ̂|j⟩| (flux/phase matrix elements).
 
The joint cost is minimised over the 5-parameter vector (EC, EL, EJ, fx1, fx2).
After optimisation, a full flux sweep visualises both crossing conditions,
now including the φ̂ (phase) operator matrix elements alongside Q̂.
"""
 
# %%
# =============================================================================
# Imports and JAX configuration
# =============================================================================
import numpy as np
import scipy as sp
import jax as jx
import jax.numpy as jnp
from itertools import combinations

 
jx.config.update("jax_enable_x64", True)
jx.config.update("jax_platform_name", "cpu")
 
try:
    import optax
    _OPTAX_AVAILABLE = False
except ImportError:
    _OPTAX_AVAILABLE = True
    print("optax not found – skipping Adam warm-start.")
 
import matplotlib.pyplot as plt
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
 
# %%
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
 
 
# %%
# =============================================================================
# Parameter ranges
# =============================================================================
_2pi = 2.0 * jx.numpy.pi
 
## Bounds in GHz
EC_range = (0.2, 3.8)
EL_range = (0.5, 5)
EJ_range = (2.5, 20)
fx_range  = (0.4, 0.5)

## Bounds cost functions
min_anharm=0.25 # GHz
max_matrix_element=0.8
min_matrix_element=0.015
min_coherence_time=5 #\mu s

kill_transition_bool=False
kill_transitions_list=[(3,4)] # Can be several
max_element_to_kill=0.01 ## Max value of the matrix element transition that we want to kill


 
# Grid / truncation — 4 levels needed (indices 0–3)
N_grid        = 128
phi_max       = 12.0
N_trunc       = 6 # Number of energy levels that we consider in a truncated fluxonium (does not compute every level)
N_plot_levels = 10 # Number of energy levels that we consider in a full fluxonium diagonalization (computes every level but only shows N)
 
assert N_trunc >= 4, "N_trunc must be >= 4." # This is bc we want to do a 4 level syst.
 
# Cost weights. M: we will try to avoid this by using multiobj
_scale_w   = _2pi * 1.0   # normalise ω differences by 1 GHz
_eps       = 1e-12
 
w_crossing = 1.0    # (ωij - ωkl)² / scale_w²
w_matrix   = 1.0    # (nij - nkl)²
w_amp      = 0.01   # 1/(nij+nkl) regulariser against trivial zeros
w_flat     = 1.0    # (dω/dfx)² / scale_w² sweet-spot regulariser
 
 
# %%
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
 
 
# %%
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
 
 
# %%
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
 
 
# %%
# =============================================================================
# Scalar transition frequencies (for sweet-spot derivatives)
# =============================================================================
def _w20_of_fx(EC, EL, EJ, fx):
    E, _, _ = Fluxonium_fft(EC, EL, EJ, fx)
    return E[2] - E[0]
 
 
def _w10_of_fx(EC, EL, EJ, fx):
    E, _, _ = Fluxonium_fft(EC, EL, EJ, fx)
    return E[1] - E[0]
 
 
_dw20_dfx = jx.jit(jx.grad(_w20_of_fx, argnums=3))
_dw10_dfx = jx.jit(jx.grad(_w10_of_fx, argnums=3))
 
 
# %%
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

    omegas = (E[1:] - E[:-1])# GHz.

    anharms=omegas[1:] - omegas[:-1]


    ## Objective:

    cost_obj=(-1)*jnp.min(jnp.abs(anharms))

    # for anharm_i in anharms:
    #     # cost_obj += min_anharm ** 2 / (anharm_i ** 2 + min_anharm ** 2) #this goes to 0
    #     cost_obj+=(-1)*jx.numpy.abs(anharm_i) #this goes to -infty, te falta ponerlo adimesional

    ## Restriction:

    restriction1=min_anharm-jx.numpy.min(anharms) # My anharmocity has to be at least of 300 MHz
    
    # min_qubit_freq=2/_scale_w #GHz
    # restriction2=min_qubit_freq-jx.numpy.min(omegas[:4]) # only take into account the logical levels

    restrictions=[restriction1]

    return cost_obj,restrictions

@jx.jit
def _cost2_matrix_elements(Q):
    '''
    Matrix elements of n: In order to do the driving. Ideal: |<n>|~0.2. But look for 0.01<|<n>|<0.5.
    Maybe this can be passed as a restriction instead of an objective. Like bound it between the limits and that's it.
    We also put the restriction of killing the leakage channel.
    '''

    ## Objective:
   
    # cost_transition=0
    # for i, comb in enumerate(combinations(range(4),2)):## every combination of 2 elements within 4 levels
    #     cost_transition+=(jx.numpy.abs(Q[comb[0], comb[1]])-0.2)**2 ## this is a multiobj in disguise, reinforces the idea of going only for restrictions

    ## Restrictions
    retrictions=[]

    # kill the leakage
    if kill_transition_bool:
        kill_leakage_rest=[]
        for i in range(len(kill_transitions_list)):
            a,b=kill_transitions_list[i]
            kill_leakage_rest.append(jx.numpy.abs(Q[a, b])-max_element_to_kill)
        
        retrictions+=kill_leakage_rest
    
    
    # Max matrix elements
    n_max=[]


    for i, comb in enumerate(combinations(range(4),2)):## every combination of 2 elements within 4 levels
        n_max.append(jx.numpy.abs(Q[comb[0], comb[1]])-max_matrix_element) 

    retrictions+=n_max

    # Min matrix elements

    n_min=[]

    for i, comb in enumerate(combinations(range(4),2)):## every combination of 2 elements within 4 levels
        n_min.append(min_matrix_element-jx.numpy.abs(Q[comb[0], comb[1]])) 

    retrictions+=n_min


    return retrictions

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

    _dwij_dfx=jx.numpy.abs(dE_i-dE_j)*1e9


    A = 1e-6
    wlow = 1e-9*_2pi
    texp = 1e4

    gamma_deph = (jx.numpy.sqrt(2)*A*_dwij_dfx*jx.numpy.sqrt(jx.numpy.abs(jx.numpy.log(wlow*texp))))

    return 1e6/gamma_deph


@jx.jit
def _cost3_coherence_time(Phi, EL, fx):
    '''
    Phi: output of Fluxonium_fft, is a matrix
    EL, fx: scalars
    Write smth to maximize the inverse sum of the different coherence times (see formula). Also apply restricction of T>5 mu s
    '''

    Tphi_tot_inv=0
    for comb in combinations(range(4),2):
        Tphi=Tdeph(comb[0],comb[1],Phi,EL,fx)
        Tphi_tot_inv+=1/Tphi

    Tphi_tot=1/Tphi_tot_inv

    cost=(-1)*jx.numpy.abs(Tphi_tot)

    restriction1=min_coherence_time-Tphi_tot

    restrictions=[restriction1]


    return cost, restrictions


## Optimizator

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize


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
 
 
# --------------------------------------------------------------------------
# 2. Combine your three cost functions into a single JAX evaluation
# --------------------------------------------------------------------------
@jx.jit
def evaluate_single(params):
    EC, EL, EJ, fx = params
 
    E, Q, Phi = Fluxonium_fft(EC, EL, EJ, fx)
 
    cost1, r1 = _cost1_anharm(E)
    r2 = _cost2_matrix_elements(Q)
    cost3, r3 = _cost3_coherence_time(Phi, EL, fx)
 
    objectives = jx.numpy.stack([cost1, cost3])
    constraints = jx.numpy.concatenate([jx.numpy.stack(r1), jx.numpy.stack(r2), jx.numpy.stack(r3)])
 
    return objectives, constraints
 
 
# Vectorize over the whole population in one shot
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
    seed=1,
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








# %%

## Visualizing the results

F=res.F # Costs
X=res.X # Solutions (params)

# We flip the signs since we were maximizing the costs, now we want to see the magnitudes right
F_scale0=-1e3*F[:,0] ## In MHz
F_scale1=-F[:,1]


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



fig = plt.figure()

print('T_phi:', F_scale1[100])
print('Solution:', X[100,:])
plt.scatter(F_scale0,F_scale1,marker='.',s=0.7)
plt.xlabel(r'$\alpha_{min} \ [\textrm{MHz}]$')
plt.ylabel(r'$T_\varphi \ [\mu s]$')
plt.title(rf'Pareto for ${min_matrix_element}<|\langle \hat n\rangle|<{max_matrix_element}$')
plt.savefig(f'pareto for n_in_{min_matrix_element}_{max_matrix_element}.png')

plt.show()

EC,EL,EJ,fx=X[100,:]
EC,EL,EJ,fx=3.61326215 ,4.99994478, 5.33307494 ,0.48261268
 

E,_,_= Fluxonium_fft(EC, EL, EJ, fx)

print(E[4]-E[3])
print(E[3]-E[2])






# for i,name in enumerate(["EC","EL","EJ","fx"]):

#     plt.figure()

#     plt.scatter(X[:,i],F[:,0])

#     plt.xlabel(name)

#     plt.ylabel("cost1")








# %%

## =============================================================================
@jx.jit
def _cost_condition1(EC, EL, EJ, fx):
    """
    At flux fx:  ω20 = ω31   AND   |n02| = |n13|   AND   dω20/dfx = 0 (flat)
    """
    E, Q, Phi = Fluxonium_fft(EC, EL, EJ, fx) ## E are the energies, Q the matrix elements of n, Phi the matrix elements of phi
 
    w20 = E[2] - E[0]
    w31 = E[3] - E[1]
    n02 = jx.numpy.abs(Q[0, 2])
    n13 = jx.numpy.abs(Q[1, 3])
 
    dw   = (w20 - w31) / _scale_w # cost funcion for w20=w31, adimensional due to _scale_w
    dn   = jx.numpy.abs(n02 - n13)
    amp  = 1.0 / (n02 + n13 + _eps)
    flat = _dw20_dfx(EC, EL, EJ, fx) / _scale_w #cost function for dw/dfx, adimensional due to _scale_w
 
    return w_crossing * dw**2 + w_flat * flat**2 #+ w_matrix * dn**2 + w_amp * amp
 
 
@jx.jit
def _cost_condition2(EC, EL, EJ, fx):
    """
    At flux fx:  ω10 = ω32   AND   |n10| = |n32|   AND   dω10/dfx = 0 (flat)
    """
    E, Q, Phi = Fluxonium_fft(EC, EL, EJ, fx)
 
    w10 = E[1] - E[0]
    w32 = E[3] - E[2]
    n10 = jx.numpy.abs(Q[1, 0])
    n32 = jx.numpy.abs(Q[3, 2])
 
    dw   = (w10 - w32) / _scale_w
    dn   = jx.numpy.abs(n10 - n32)**2
    amp  = 1.0 / (n10 + n32 + _eps)
    flat = _dw10_dfx(EC, EL, EJ, fx) / _scale_w
 
    return w_crossing * dw**2 + w_flat * flat**2 #+ w_matrix * dn**2 + w_amp * amp
 
 
@jx.jit
def joint_cost_physical(EC, EL, EJ, fx1, fx2):
    """
    Joint cost over shared (EC, EL, EJ) and two independent flux points.
 
        condition 1 evaluated at fx1
        condition 2 evaluated at fx2
    """
    return _cost_condition1(EC, EL, EJ, fx1) + _cost_condition2(EC, EL, EJ, fx2)



 
 
# %%
# =============================================================================
# Parametrisation — 5-vector: (EC, EL, EJ, fx1, fx2)
# =============================================================================
'''
M: Here is the same as said in sigmoid, in this case we are dealing with parameters that can vary a lot, so doing the
optimization in log scale supresses that varyation so the optimizer doesn't get lost and then returns the values to the
initial scale.
'''
def map_log_bounded(y, lower, upper):
    s = sigmoid(y)
    return jx.numpy.exp(
        jx.numpy.log(lower) + s * (jx.numpy.log(upper) - jx.numpy.log(lower))
    )
 
 
def map_linear_bounded(y, lower, upper):
    return lower + sigmoid(y) * (upper - lower)
 
 
def inverse_map_log_bounded(x, lower, upper):
    z = (np.log(x) - np.log(lower)) / (np.log(upper) - np.log(lower))
    return inv_sigmoid(z)
 
 
def inverse_map_linear_bounded(x, lower, upper):
    z = (x - lower) / (upper - lower)
    return inv_sigmoid(z)
 
 
@jx.jit
def unpack_y5(y):
    """Unconstrained R⁵ → physical (EC, EL, EJ, fx1, fx2)."""
    EC  = map_log_bounded(y[0], EC_range[0], EC_range[1])
    EL  = map_log_bounded(y[1], EL_range[0], EL_range[1])
    EJ  = map_log_bounded(y[2], EJ_range[0], EJ_range[1])
    fx1 = map_linear_bounded(y[3], fx_range[0], fx_range[1])
    fx2 = map_linear_bounded(y[4], fx_range[0], fx_range[1])
    return EC, EL, EJ, fx1, fx2
 
 
def pack_y5(EC, EL, EJ, fx1, fx2):
    """Physical (EC, EL, EJ, fx1, fx2) → unconstrained R⁵."""
    return np.array([
        inverse_map_log_bounded(EC,  EC_range[0], EC_range[1]),
        inverse_map_log_bounded(EL,  EL_range[0], EL_range[1]),
        inverse_map_log_bounded(EJ,  EJ_range[0], EJ_range[1]),
        inverse_map_linear_bounded(fx1, fx_range[0], fx_range[1]),
        inverse_map_linear_bounded(fx2, fx_range[0], fx_range[1]),
    ], dtype=np.float64)
 
 
# %%
# =============================================================================
# Objective in unconstrained y-space
# =============================================================================
@jx.jit
def objective_y(y):
    return joint_cost_physical(*unpack_y5(y))
 
 
objective_and_grad = jx.jit(jx.value_and_grad(objective_y)) ## Esta funcion te devuelve el valor y el gradiente de ese valor usando autoatic differentiation
 
 
# %%
# =============================================================================
# SciPy wrapper
# =============================================================================
class ScipyObjectiveWrapper: ##Scipy quiere funciones de python normales, JAX devuelve objetos propios (DeviceArray, esto es el punte entre las dos)
    _CACHE_ATOL = 1e-15
 
    def __init__(self):
        self._last_x    = None
        self._last_val  = None
        self._last_grad = None
 
    def _evaluate(self, x):
        x = np.asarray(x, dtype=np.float64)
        if ( ## Esta condicion hace que solo calculemos el punto si es el primero o si NO está cerca del que ya tenemos
            self._last_x is None
            or not np.allclose(x, self._last_x, rtol=0.0, atol=self._CACHE_ATOL)
        ):
            val, grad       = objective_and_grad(jx.numpy.array(x))
            self._last_x    = x.copy()
            self._last_val  = float(np.asarray(val))
            self._last_grad = np.asarray(grad, dtype=np.float64)
        return self._last_val, self._last_grad
 
    def fun(self, x): return self._evaluate(x)[0]
    def jac(self, x): return self._evaluate(x)[1]
 
 
# %%
# =============================================================================
# Adam warm-start
# =============================================================================
def adam_warmup(y_init, n_steps=500, lr=0.05):
    if _OPTAX_AVAILABLE:
        return np.asarray(y_init, dtype=np.float64)
 
    optimizer = optax.adam(lr)
    y         = jx.numpy.array(y_init)
    opt_state = optimizer.init(y)
 
    @jx.jit
    def step(y, state):
        val, grads  = objective_and_grad(y)
        updates, ns = optimizer.update(grads, state)
        return optax.apply_updates(y, updates), ns, val
 
    for i in range(n_steps):
        y, opt_state, val = step(y, opt_state)
        if (i + 1) % 100 == 0:
            print(f"  Adam step {i+1:4d}  cost = {float(val):.6e}")
 
    return np.asarray(y, dtype=np.float64)
 
 
# %%
# =============================================================================
# Initial guess
# =============================================================================
EC_mid = np.sqrt(EC_range[0] * EC_range[1])
EL_mid = np.sqrt(EL_range[0] * EL_range[1])
EJ_mid = np.sqrt(EJ_range[0] * EJ_range[1])
 
# Start fx1 and fx2 at different points to break symmetry
y0 = pack_y5(EC_mid, EL_mid, EJ_mid, fx1=0.25, fx2=0.45)
 
 
# %%
# =============================================================================
# Adam warm-start then L-BFGS-B
# =============================================================================
if _OPTAX_AVAILABLE:
    print("Running Adam warm-start...")
    y0 = adam_warmup(y0, n_steps=500, lr=0.05)
    print("Adam warm-start complete.\n")
 
wrapper = ScipyObjectiveWrapper()
 
result = sp.optimize.minimize(
    fun=wrapper.fun,
    x0=y0,
    method="nelder-mead",
    options=dict(maxiter=50_000, ftol=1e-15, gtol=1e-15, maxls=50),
)
 
print("Optimisation finished.")
print("Success :", result.success)
print("Message :", result.message)
print("Final cost :", result.fun)
 
 
# %%
# =============================================================================
# Extract optimised parameters
# =============================================================================
# EC_opt, EL_opt, EJ_opt, fx1_opt, fx2_opt = [
#     float(v) for v in unpack_y5(jx.numpy.array(result.x))
# ]
EC_opt, EL_opt, EJ_opt, fx1_opt=[3.7999279 , 4.23905758 ,4.0765141 , 0.46757242]

print("\nOptimised parameters:")
print(f"  EC  = {EC_opt / _2pi:.6f} GHz")
print(f"  EL  = {EL_opt / _2pi:.6f} GHz")
print(f"  EJ  = {EJ_opt / _2pi:.6f} GHz")
print(f"  fx1 = {fx1_opt:.6f}   (condition 1: ω20=ω31, |n02|=|n13|)")
 
# %%
# =============================================================================
# Diagnostics at both flux points — now also reporting φ̂ matrix elements
# =============================================================================
@jx.jit
def diagnostic_data(EC, EL, EJ, fx):
    E, Q, Phi = Fluxonium_fft(EC, EL, EJ, fx)
 
    w10 = E[1] - E[0]
    w20 = E[2] - E[0]
    w31 = E[3] - E[1]
    w32 = E[3] - E[2]
 
    n02 = jx.numpy.abs(Q[0, 2])
    n13 = jx.numpy.abs(Q[1, 3])
    n10 = jx.numpy.abs(Q[1, 0])
    n32 = jx.numpy.abs(Q[3, 2])
 
    phi02 = jx.numpy.abs(Phi[0, 2])
    phi13 = jx.numpy.abs(Phi[1, 3])
    phi10 = jx.numpy.abs(Phi[1, 0])
    phi32 = jx.numpy.abs(Phi[3, 2])
 
    return (E, w10, w20, w31, w32,
            n02, n13, n10, n32,
            phi02, phi13, phi10, phi32)
 
 
(E_fx1, w10_1, w20_1, w31_1, w32_1,
n02_1, n13_1, n10_1, n32_1,
phi02_1, phi13_1, phi10_1, phi32_1) = diagnostic_data(
    EC_opt, EL_opt, EJ_opt, fx1_opt
)

 
print("\n" + "=" * 55)
print(f"At fx1 = {fx1_opt:.6f}  (target: ω20=ω31, |n02|=|n13|)")
print("=" * 55)
print(f"  Energies / 2π [GHz] : {np.asarray(E_fx1) / _2pi}")
print(f"  ω20 / 2π = {float(w20_1 / _2pi):.6f} GHz")
print(f"  ω31 / 2π = {float(w31_1 / _2pi):.6f} GHz")
print(f"  ω20 - ω31 [GHz]     = {float((w20_1 - w31_1) / _2pi):.3e}")
print(f"  |n02|  = {float(n02_1):.6f}")
print(f"  |n13|  = {float(n13_1):.6f}")
print(f"  |n02| - |n13|       = {float(n02_1 - n13_1):.3e}")
print(f"  |phi02| = {float(phi02_1):.6f}")
print(f"  |phi13| = {float(phi13_1):.6f}")
print(f"  |phi02| - |phi13|   = {float(phi02_1 - phi13_1):.3e}")
print(f"  dω20/dfx [GHz]      = {float(_dw20_dfx(EC_opt, EL_opt, EJ_opt, fx1_opt)) / _2pi:.3e}")
 

 
 
# %%
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
 
 
# %%
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
 
 
# %%
# =============================================================================
# Flux sweep
# =============================================================================
fx_scan = np.linspace(0.01, 0.49, 201)
 
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
 
ax.axvline(fx1_opt, color="k",    ls="--", lw=1.2,
           label=rf"$f_{{x1}}={fx1_opt:.3f}$ ($\omega_{{20}}=\omega_{{31}}$)")
ax.axvline(fx2_opt, color="gray", ls="--", lw=1.2,
           label=rf"$f_{{x2}}={fx2_opt:.3f}$ ($\omega_{{10}}=\omega_{{32}}$)")
 
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
 
ax.axvline(fx1_opt, color="k",    ls="--", lw=1.2,
           label=rf"$f_{{x1}}={fx1_opt:.3f}$")
ax.axvline(fx2_opt, color="gray", ls="--", lw=1.2,
           label=rf"$f_{{x2}}={fx2_opt:.3f}$")
 
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
 
ax.axvline(fx1_opt, color="k",    ls="--", lw=1.2,
           label=rf"$f_{{x1}}={fx1_opt:.3f}$")
ax.axvline(fx2_opt, color="gray", ls="--", lw=1.2,
           label=rf"$f_{{x2}}={fx2_opt:.3f}$")
 
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
 
ax.axvline(fx1_opt, color="k",    ls="--", lw=1.2,
           label=rf"$f_{{x1}}={fx1_opt:.3f}$")
ax.axvline(fx2_opt, color="gray", ls="--", lw=1.2,
           label=rf"$f_{{x2}}={fx2_opt:.3f}$")
 
ax.set_xlabel(r"$\varphi_x$", fontsize=14)
ax.set_ylabel(r"$\omega_n / 2\pi$ (GHz)", fontsize=14)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
# fig.savefig("EnergySpectrum.pdf")
plt.show()
 
# %%
 
''' 
Pablo's function to compute the coherence time.
'''

def _wij_of_fx(EC, EL, EJ, i,j,fx):
    E, _, _ = Fluxonium_fft(EC, EL, EJ, fx)
    return E[i] - E[j]
_dwij_dfx = jx.jit(jx.grad(_wij_of_fx, argnums=5))
 
def Tdeph(i,j,fx):

    A = 1e-6
    wlow = 1e-9*(2*np.pi)
    texp = 1e4

    gamma_deph = (np.sqrt(2)*A*(abs(_dwij_dfx(EC_opt, EL_opt, EJ_opt, i, j, fx))*1e9)*np.sqrt(np.abs(np.log(wlow*texp))))

    return 1e6/gamma_deph

fx_scan = np.linspace(0.01, 0.49, 201)
Tdeph_l = np.zeros((201, 4, 4))
for k in range(201):
    for i in range(4):
        for j in range(4):
            Tdeph_l[k,i,j] = Tdeph(i,j,fx_scan[k])

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.set_ylabel(r'$T_1[\mu s]$')
ax.set_xlabel(r'$\varphi_e$')

ax.set_yscale('log')
for i in range(4):
    for j in range(i+1,4):
        ax.plot(fx_scan, Tdeph_l[:,i,j], label=f'{i,j}')
ax.axvline(fx1_opt, color="k",    ls="--", lw=1.2,
           label=rf"$f_{{x1}}={fx1_opt:.3f}$")
ax.legend(loc = 'upper right')
# %%

# fx_scan = np.linspace(0.01, 0.49, 201)
Tdeph_l = np.zeros((201, 4, 4))
for k in range(201):
    for i in range(4):
        for j in range(4):
            Tdeph_l[k,i,j] = Tdeph(i,j,fx_scan[k])

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.set_ylabel(r'$T_1[\mu s]$')
ax.set_xlabel(r'$\varphi_e$')
ax.set_ylim(1e0,1e3)

ax.set_yscale('log')
for i in range(4):
    for j in range(i+1,4):
        ax.plot(fx_scan, Tdeph_l[:,i,j], label=f'{i,j}')

Tdeph_total=1/(1/Tdeph_l[:,0,1] + 1/Tdeph_l[:,0,2] + 1/Tdeph_l[:,1,3] + 1/Tdeph_l[:,2,3] + 1/Tdeph_l[:,1,2])

ax.plot(fx_scan, Tdeph_total, label=f'Total', color='r')
ax.axvline(fx1_opt, color="k",    ls="--", lw=1.2,
           label=rf"$f_{{x1}}={fx1_opt:.3f}$")
ax.legend(loc = 'upper right')
# %%
