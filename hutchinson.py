import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from IPython.display import clear_output

def initial_condition(t, N0):
    """Initial history function"""
    return N0 * np.cos(0.01 * t)

def get_delayed_value(N, i, n_delay, t, tau, N0):
    """Get the delayed value N(t-tau), using initial condition if necessary"""
    if i < n_delay:
        return initial_condition(t - tau, N0)
    return N[i - n_delay]

def dN_dt(N_t, N_tau, r, K):
    """The right-hand side of the DDE: dN/dt = r*N(t)*(1 - N(t-tau)/K)"""
    return r * N_t * (1 - N_tau/K)

def solve_delay_logistic_rk4(r, K, tau, N0, t_max=200, dt=0.1):
    """
    Solve the delayed logistic equation using RK4 method:
    dN/dt = r*N(t)*(1 - N(t-tau)/K)
    """
    # Create time array
    t = np.arange(0, t_max, dt)
    n_points = len(t)
    
    # Calculate number of delay steps
    n_delay = int(tau/dt)
    
    # Initialize solution array
    N = np.zeros(n_points)
    
    # Set initial history using the initial condition function
    N[:n_delay] = initial_condition(t[:n_delay], N0)
    
    # Solve using RK4 method
    for i in range(n_delay - 1, n_points - 1):
        # Current time and value
        t_i = t[i]
        N_i = N[i]
        
        # RK4 coefficients
        # k1 = f(t_i, y_i)
        N_tau = get_delayed_value(N, i, n_delay, t_i, tau, N0)
        k1 = dN_dt(N_i, N_tau, r, K)
        
        # k2 = f(t_i + dt/2, y_i + dt*k1/2)
        N_tau = get_delayed_value(N, i + 1//2, n_delay, t_i + dt/2, tau, N0)
        k2 = dN_dt(N_i + dt*k1/2, N_tau, r, K)
        
        # k3 = f(t_i + dt/2, y_i + dt*k2/2)
        N_tau = get_delayed_value(N, i + 1//2, n_delay, t_i + dt/2, tau, N0)
        k3 = dN_dt(N_i + dt*k2/2, N_tau, r, K)
        
        # k4 = f(t_i + dt, y_i + dt*k3)
        N_tau = get_delayed_value(N, i + 1, n_delay, t_i + dt, tau, N0)
        k4 = dN_dt(N_i + dt*k3, N_tau, r, K)
        
        # Update solution using RK4 formula
        N[i + 1] = N_i + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, N, n_delay

def plot_solution(r=0.16, K=1.1, tau=10, N0=1.0, max_time=200):
    """Create plots for the solution"""
    clear_output(wait=True)
    
    # Solve the equation with the specified max_time
    t, N, n_delay = solve_delay_logistic_rk4(r, K, tau, N0, t_max=max_time)
    
    # Create figure with a single subplot (only time series plot)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Time series plot (skip initial transients)
    skip_points = n_delay  # Skip first few time units
    ax1.plot(t[skip_points:], N[skip_points:], 'b-', lw=1.5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('N(t)')
    ax1.grid(True)
    ax1.set_title(f'Time Series (r={r:.3f}, K={K:.3f}, τ={tau:.3f}, N₀={N0:.3f}, T={max_time})')
    
    plt.tight_layout()
    plt.show()

# Interactive interface
interact(plot_solution,
        r=FloatSlider(min=0.05, max=0.5, step=0.001, value=0.16, description='r'),
        K=FloatSlider(min=0.5, max=100.0, step=0.001, value=1.1, description='K'),
        tau=FloatSlider(min=0.1, max=20.0, step=0.001, value=10, description='τ'),
        N0=FloatSlider(min=0.1, max=10.0, step=0.1, value=1.0, description='N₀'),
        max_time=FloatSlider(min=50, max=500, step=1, value=200, description='T')) 
