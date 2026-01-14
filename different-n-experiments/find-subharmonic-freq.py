
import numpy as np
from qutip import basis
from scipy.optimize import minimize_scalar
from matplotlib import pyplot as plt
from qutip import *

def qubit_integrate_labframe(omega_0, omega_d, rabi, theta,psi0, solver, phi = 0, g1 = 0, g2 = 0, tlist=np.linspace(0,5000,10000)):

    H0 = (omega_0/2) * sigmaz()
    H1 = 2 * rabi * np.sin(theta) * sigmax()
    H2 = 2 * rabi * np.cos(theta) * sigmaz()
    
    def H1_coeff(t, args):
        return np.cos(omega_d*t+phi)
        
    def H2_coeff(t, args):
        return np.cos(omega_d*t+phi)
    
    # collapse operators
    c_ops = []

    if g1 > 0.0:
        c_ops.append(np.sqrt(g1) * sigmam())

    if g2 > 0.0:
        c_ops.append(np.sqrt(g2) * sigmaz())

    e_ops = [sigmax(), sigmay(), sigmaz()]
    
    H = [H0, [H1,H1_coeff],  [H2,H2_coeff]]
    
    if solver == "me":
        output = mesolve(H, psi0, tlist, c_ops, e_ops)  
    elif solver == "es":
        output = essolve(H, psi0, tlist, c_ops, e_ops)  
    elif solver == "mc":
        ntraj = 250
        output = mcsolve(H, psi0, tlist, ntraj, c_ops, e_ops)  
    else:
        raise ValueError("unknown solver")
        
    return output.expect[0], output.expect[1], output.expect[2]

if __name__ == "__main__":
    # Parameters
    N = 4 # Subharmonic number
    theta = 0.25 * np.pi # Driving angle, 
    rabi = 0.27 * np.pi / np.sin(theta) # Drive amplitude

    omega_0 = 1.0 * np.pi
    psi0 = basis(2,0)
    t_max = 300 
    samples = 10000
    tlist = np.linspace(0, t_max, samples)

    # Two steps to find the subharmonic frequency estimation: 1. "coarse" scanning, 2. finer scanning around the lower estimate.
    omega_est = 0.272 * np.pi  # Initial estimate for N
    coarse_range = np.linspace(omega_est - 0.002, omega_est + 0.002 * np.pi, 20)
    
    print(f"Coarse scanning for N={N}")
    depths = []
    for od in coarse_range:
        _, _, sz = qubit_integrate_labframe(omega_0, od, rabi, theta, psi0, "me", 0, 0, 0, tlist)
        depths.append(np.min(np.real(sz)))
    
    # Second step
    coarse_estimate = coarse_range[np.argmin(depths)]
    fine_range = np.linspace(coarse_estimate - 0.0001*np.pi, coarse_estimate + 0.0001*np.pi, 30)
    
    print(f"Fine scanning around {coarse_estimate/np.pi:.32f}π")
    fine_depths = []
    for od in fine_range:
        _, _, sz = qubit_integrate_labframe(omega_0, od, rabi, theta, psi0, "me", 0, 0, 0, tlist)
        fine_depths.append(np.min(np.real(sz)))

    min_omega_d = fine_range[np.argmin(fine_depths)]
    print(f"Optimal omega_d: {min_omega_d/np.pi:.64f}π")

    tlist_final = np.linspace(0, t_max, samples)
    _, _, sz_final = qubit_integrate_labframe(omega_0, min_omega_d, rabi, theta, psi0, "me", 0, 0, 0, tlist_final)

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(tlist_final, np.real(sz_final), 'g', label="sz")
    ax.plot((0, t_max), (-0.95, -0.95), 'r--', label="target")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f"N={N} Resonance at w_d = {min_omega_d/np.pi:.10f}π")
    plt.legend()
    plt.show()