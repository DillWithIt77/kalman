import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import correlate
import tensorflow as tf
from tensorflow.keras import layers, models

# ------------------------
# MODEL PARAMETERS
# ------------------------
N = 40       # Number of Lorenz-96 variables
# F = 8.0       # Forcing

# ------------------------
# ENSEMBLE PARAMETERS
# ------------------------
Ne = 80
inflation = 1.05 #(book says to use r = 0.05)

# ------------------------
# OBSERVATION PARAMETERS
# ------------------------
M = 20
P = N // M  # spacing
obs_idx = np.arange(0, N, P)  # indices of observed variables
R_val = 3.0 # this is the r^o term in the book
obs_noise = np.sqrt(R_val)

# ------------------------
# SIMULATION PARAMETERS
# ------------------------
cycles = 5000
n_obs = 5
dt = 1/64
seed = 42 #set this so that we can keep runs the same

# ------------------------
# LORENZ-96 (regular and normalized) VECTORISED RHS
# ------------------------
def lorenz96_rhs(x, F):
    """
    Vectorized Lorenz-96 RHS for single state or ensemble.
    
    Args:
        x : ndarray of shape (N,) or (Ne, N)
        F : forcing parameter
        clip_value : optional, clip values to avoid overflow
    Returns:
        dx : ndarray of same shape as x
    """
    x = np.array(x, dtype=float)  # ensure float
    if x.ndim == 1:
        dx = (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    elif x.ndim == 2:
        dx = (np.roll(x, -1, axis=1) - np.roll(x, 2, axis=1)) * np.roll(x, 1, axis=1) - x + F
    else:
        raise ValueError("x must be 1D or 2D array")
    # Clip to avoid overflow
    # return np.clip(dx, -clip_value, clip_value)
    return dx

def lorenz96norm_rhs(x, F, Ep):
    ####need to run the regular L96 to get the E value used in the normalization
    """
    Normalized Lorenz-96 RHS
    Computes mean and energy from the current state.
    
    Args:
        x : ndarray, shape (N,) or (Ne, N)
            Normalized state variables (tilde{u}_j)
        F : float
            Forcing parameter
        clip_value : float
            Clip large values to avoid overflow
            
    Returns:
        dx : ndarray of same shape as x
            Time derivative d(tilde{u})/dt_tilde
    """
    x = np.array(x, dtype=float)

    if x.ndim == 1:
        u_bar = np.mean(x)
        dx = ((np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1)
              + Ep**(-0.5) * ((np.roll(x, -1) - np.roll(x, 2)) * u_bar - x)
              + Ep**(-1.0) * (F - u_bar))

    elif x.ndim == 2:
        # Ensemble of states
        u_bar = np.mean(x, axis=1, keepdims=True)
        dx = ((np.roll(x, -1, axis=1) - np.roll(x, 2, axis=1)) * np.roll(x, 1, axis=1)
              + Ep**(-0.5) * ((np.roll(x, -1, axis=1) - np.roll(x, 2, axis=1)) * u_bar - x)
              + Ep**(-1.0) * (F - u_bar))
    else:
        raise ValueError("x must be 1D or 2D array")

    ##clipping to stop run away values
    return np.clip(dx, -clip_value, clip_value)
    # return dx
#------------------------
# 1-D UNet Model
#------------------------
def build_unet_1d_cov(N, num_filters, kernel_size):
    """
    Build a 1D U-Net to predict an N x N covariance matrix from a 1D state vector.
    """
    # Input: shape (N, 1)
    inputs = layers.Input(shape=(N, 1))
    
    # Encoder
    c1 = layers.Conv1D(num_filters, kernel_size, padding='same', activation='relu')(inputs)
    c2 = layers.Conv1D(num_filters, kernel_size, padding='same', activation='relu')(c1)
    p1 = layers.MaxPooling1D(pool_size=2)(c2)
    
    c3 = layers.Conv1D(num_filters*2, kernel_size, padding='same', activation='relu')(p1)
    c4 = layers.Conv1D(num_filters*2, kernel_size, padding='same', activation='relu')(c3)
    p2 = layers.MaxPooling1D(pool_size=2)(c4)
    
    # Bottleneck
    b1 = layers.Conv1D(num_filters*4, kernel_size, padding='same', activation='relu')(p2)
    
    # Decoder
    u1 = layers.UpSampling1D(size=2)(b1)
    concat1 = layers.Concatenate()([u1, c4])
    c5 = layers.Conv1D(num_filters*2, kernel_size, padding='same', activation='relu')(concat1)
    
    u2 = layers.UpSampling1D(size=2)(c5)
    concat2 = layers.Concatenate()([u2, c2])
    c6 = layers.Conv1D(num_filters, kernel_size, padding='same', activation='relu')(concat2)
    
    # Output: flatten and dense to N x N
    flatten = layers.Flatten()(c6)
    dense_out = layers.Dense(N*N)(flatten)
    output = layers.Reshape((N, N))(dense_out)
    
    model = models.Model(inputs=inputs, outputs=output)
    return model

# ------------------------
# RK4 INTEGRATION
# ------------------------
def rk4_integrate(x0, dt, nsteps, rhs, F):
    """
    Integrate a single state or ensemble forward in time using RK4.

    Parameters
    ----------
    x0 : ndarray (N,) or (Ne, N)
        Initial state(s).
    dt : float
        Time step.
    nsteps : int
        Number of integration steps.
    rhs : callable
        Function that computes dx/dt given x and F.
    F : float or ndarray
        Forcing term.

    Returns
    -------
    x : ndarray
        Integrated state(s) after nsteps.
    """
    x = np.atleast_2d(x0).copy()
    single = (x0.ndim == 1)

    for _ in range(nsteps):
        k1 = rhs(x, F)
        k2 = rhs(x + 0.5 * dt * k1, F)
        k3 = rhs(x + 0.5 * dt * k2, F)
        k4 = rhs(x + dt * k3, F)
        x += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return x[0] if single else x
#-------------------------
# Functions for Table 11.1 Values page 194
#-------------------------
def lorenz96_jacobian(x):
    """Jacobian of Lorenz-96 system"""
    N = len(x)
    J = np.zeros((N, N))
    for i in range(N):
        J[i, i] = -1
        J[i, (i-2)%N] = -x[i-1]
        J[i, (i-1)%N] = x[(i+1)%N] - x[i-2]
        J[i, (i+1)%N] = x[i-1]
    return J

def lyapunov_spectrum(F, N, dt, cut, steps):
    """Compute full Lyapunov spectrum via QR method using rk4_integrate"""
    x = F * np.ones(N)
    x[0] += 0.01

    # --- Spinup phase ---
    n_spinup = int(cut / dt)
    x = rk4_integrate(x, dt, n_spinup, lorenz96_rhs, F)

    m = N
    Q = np.eye(N)
    lyap_sum = np.zeros(m)

    true_steps = int(steps / dt)
    eps = 1e-8  # small tolerance if needed

    for step in range(true_steps):
        # integrate base trajectory one step forward
        x = rk4_integrate(x, dt, 1, lorenz96_rhs, F)

        # tangent linear evolution using Jacobian
        J = lorenz96_jacobian(x)
        Y = Q + dt * (J @ Q)  # Euler step for tangent dynamics

        # re-orthonormalize
        Q, R = np.linalg.qr(Y)
        lyap_sum += np.log(np.abs(np.diag(R)))

    lyaps = lyap_sum / (true_steps * dt)
    return np.sort(lyaps)[::-1]  # descending order

def corr_time(series, dt):
    """Integral correlation time until 1/e decay."""
    series = series - np.mean(series)
    ac = correlate(series, series, mode='full')
    ac = ac[ac.size // 2:] / ac[ac.size // 2]
    cutoff = np.argmax(ac < 1/np.e)
    if cutoff == 0:
        cutoff = len(ac)
    return np.trapz(ac[:cutoff], dx=dt)

# ------------------------
# L96 Verification
#-------------------------
def L96_test(N, F, dt,steps, rhs):
    rng = np.random.default_rng(seed)
    perturb = 0.01
    x_true = F * np.ones(N) + perturb * rng.normal(size=N)
    x = x_true.copy()
    x_vals = []
    x_vals.append(x)
    for _ in range(steps):
        x = rk4_integrate(x[None, :], dt, 1, rhs, F)[0]
        x_vals.append(x)
    return np.array(x_vals)
#-------------------------
# Filter Functions
#-------------------------
# def H_operator(x, obs_idx):
#     if x.ndim == 1:
#         return x[obs_idx]
#     else:
#         return x[:, obs_idx]

def H_operator_fft(x, modes):
    x_fft = np.fft.fft(x, axis=-1)
    if x.ndim == 1:
        return (x_fft[modes]/np.sqrt(len(x))).real  # take real part for observations
    else:
        return (x_fft[:, modes]/np.sqrt(len(x))).real

def H_basis_vector_fft(N, mode_j):
    """Return H^T e_j for a Fourier-mode observation"""
    basis = np.zeros(N)
    basis[mode_j] = 1.0
    return np.real(np.fft.ifft(basis)) * np.sqrt(N)

def sqr_ensemble_update(ensemble_f, y_obs, R, obs_idx, inflation, method, **kwargs):
    """
    Deterministic square-root ensemble update (ETKF or EAKF), numerically stable.
    ensemble_f: Ne x N
    y_obs: ny
    Returns: ensemble_a (Ne x N)
    """
    num_filters = kwargs.get('num_filters', 0)
    kernel_size = kwargs.get('kernel_size', 0)

    Ne, N = ensemble_f.shape
    ny = len(obs_idx)

    # Forecast mean and anomalies
    xf_mean = ensemble_f.mean(axis=0)  # (N,)
    Xf = (ensemble_f - xf_mean).T * inflation  # (N, Ne)

    # Forecast in observation space
    Yf = ensemble_f[:, obs_idx]  # (Ne, ny)
    yf_mean = Yf.mean(axis=0)
    Yf_ano = (Yf - yf_mean).T  # (ny, Ne)

    Rmat = R*np.eye(ny) if np.isscalar(R) else R

    if method.upper() == 'ETKF':
        # 1. Compute ensemble-space covariance in obs space
        PfHT = Yf_ano.T / np.sqrt(Ne - 1)  # (Ne, ny)
        R_inv = np.linalg.inv(Rmat)
        # Matrix in ensemble space
        C = PfHT @ R_inv @ PfHT.T  # (Ne, Ne)
        
        # 2. Eigen-decomposition for stable square root
        eigvals, eigvecs = np.linalg.eigh(np.eye(Ne) + C)
        T = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T  # (Ne, Ne)

        # 3. Transform anomalies
        Xa = (Xf @ T).T  # (Ne, N)

        # 4. Update mean
        dx_mean = Xf @ (PfHT @ R_inv @ (y_obs - yf_mean) / np.sqrt(Ne - 1))
        xa_mean = xf_mean + dx_mean

        # 5. Add anomalies to mean
        ensemble_a = xa_mean + Xa

    elif method.upper() == 'EAKF':
        Ne, N = ensemble_f.shape
        ny = len(obs_idx)
        Rmat = R*np.eye(ny) if np.isscalar(R) else R

        # Forecast mean and anomalies
        xf_mean = ensemble_f.mean(axis=0)           # (N,)
        Xf = (ensemble_f - xf_mean) * inflation     # (Ne, N)

        # Forecast in observation space
        # Yf = ensemble_f[:, obs_idx]                 # (Ne, ny)
        
        Yf = H_operator_fft(ensemble_f, obs_idx)
        yf_mean = Yf.mean(axis=0)
        Yf_ano = Yf - yf_mean                       # (Ne, ny)

        # Ensemble-space Kalman gain for observations
        ensemble_a = np.zeros_like(ensemble_f)
        xf_mean_a = xf_mean.copy()
        Xf_a = Xf.copy()

        for j in range(ny):
            y_f = Yf[:, j]
            y_f_mean = yf_mean[j]
            y_f_ano = y_f - y_f_mean
            var_y_f = np.var(y_f_ano, ddof=1)
            Rj = Rmat[j, j]

            cov_xy = (Xf.T @ y_f_ano) / (Ne - 1)  # (N,)
            K_gain = cov_xy / (var_y_f + Rj)

            # Update mean
            xf_mean_a += K_gain * (y_obs[j] - y_f_mean)

            # Update anomalies deterministically
            alpha = np.sqrt(1 - var_y_f / (var_y_f + Rj))
            Xf_a = Xf_a - (1 - alpha) * np.outer(y_f_ano, K_gain) / var_y_f
            Xf_a -= Xf_a.mean(axis=0)  # recenter anomalies along ensemble axis

        # Posterior ensemble
        ensemble_a = xf_mean_a + Xf_a

    # elif method.upper() == "UNET":
    #     Ne, N = ensemble_f.shape
    #     ny = len(obs_idx)
    #     Rmat = R*np.eye(ny) if np.isscalar(R) else R
    #     unet_model = build_unet_1d_cov(N,num_filters = num_filters, kernel_size = kernel_size)
    #     unet_model.compile(optimizer='adam', loss='mse')

    #     # Forecast mean and anomalies
    #     xf_mean = ensemble_f.mean(axis=0)           # (N,)
    #     Xf = (ensemble_f - xf_mean) * inflation     # (Ne, N)
        
    #     Yf = H_operator_fft(ensemble_f, obs_idx)
    #     yf_mean = Yf.mean(axis=0)
    #     Yf_ano = Yf - yf_mean                       # (Ne, ny)

    #     # UNet predicted prior covariance (N x N)
    #     B_unet = unet_model.predict(xf_mean)
    #     B_unet = 0.5 * (B_unet + B_unet.T)  # ensure symmetry


    #     # Ensemble-space Kalman gain for observations
    #     ensemble_a = np.zeros_like(ensemble_f)
    #     xf_mean_a = xf_mean.copy()
    #     Xf_a = Xf.copy()

    #     for j in range(ny):
    #         h_j = H_basis_vector_fft(N, obs_idx[j])
    #         cov_xy = B_unet @ h_j              # N x 1
    #         var_y_f = h_j @ cov_xy             # scalar
    #         Rj = Rmat[j, j]

    #         cov_xy = (Xf.T @ y_f_ano) / (Ne - 1)  # (N,)
    #         K_gain = cov_xy / (var_y_f + Rj)

    #         # Update mean
    #         xf_mean_a += K_gain * (y_obs[j] - y_f_mean)

    #         # Update anomalies deterministically
    #         alpha = np.sqrt(1 - var_y_f / (var_y_f + Rj))
    #         Xf_a = Xf_a - (1 - alpha) * np.outer(y_f_ano, K_gain) / var_y_f
    #         Xf_a -= Xf_a.mean(axis=0)  # recenter anomalies along ensemble axis

    #     # Posterior ensemble
    #     ensemble_a = xf_mean_a + Xf_a

    else:
        raise ValueError("method must be 'ETKF' or 'EAKF'")

    return ensemble_a

# ------------------------
# RUN FILTER EXPERIMENT
# ------------------------
def run_sqrt_filter_experiment(
    N, M, Ne, cycles, n_obs, dt,
    R_val, inflation, method, seed,
    obs_idx, steps, F, **kwargs
):
    """
    Run ETKF/EAKF experiment with full storage of prior/posterior ensembles.
    
    Parameters
    ----------
    N : int
        Number of model variables
    M : int
        Number of observed variables
    Ne : int
        Ensemble size
    cycles : int
        Number of assimilation cycles
    n_obs : int
        Number of model steps between observations
    dt : float
        Model time step
    R_val : float
        Observation variance
    inflation : float
        Multiplicative inflation coefficient
    method : str
        'ETKF' or 'EAKF'
    seed : int
        Random seed
    obs_idx : array-like
        Indices of observed variables. If None, generated every N/M
    
    Returns
    -------
    results : dict
        Dictionary with keys:
        - times, rmse, spread, obs_idx, truth, prior_ensemble, posterior_ensemble
    """
    rng = np.random.default_rng(seed)

    num_filters = kwargs.get('num_filters',0)
    kernel_size = kwargs.get('kernel_size',0)

    # Set observation indices if not provided
    if obs_idx is None:
        P = N // M
        obs_idx = np.arange(0, N, P)

    # True initial state
    rng = np.random.default_rng(seed)
    perturb = 0.01
    x_true = F * np.ones(N) + perturb * rng.normal(size=N)
    x_true = rk4_integrate(x_true[None, :], dt, steps, lorenz96_rhs, F)[0]

    # # Long integration to estimate climatology
    # spinup_steps_long = 2000*64  # or at least a few thousand
    # x_clim = rk4_integrate(x_true[None, :], dt, steps_long, lorenz96_rhs, F)[0]
    # var_per_component = np.var(x_clim, axis=0)
    # E = np.mean(var_per_component)

    # preallocate array to store trajectory
    nsteps = 2000*64
    traj = np.zeros((nsteps+1, N))

    # initial condition
    x = x_true.copy()
    traj[0] = x

    # integrate step by step
    for i in range(1, nsteps+1):
        x = rk4_integrate(x, dt, 1, lorenz96_rhs, F)  # just 1 step at a time
        traj[i] = x

    # compute temporal mean per component
    u_bar = np.mean(traj, axis=0)

    # compute variance over time per component
    var_per_component = np.mean((traj - u_bar)**2, axis=0)

    # climatological energy
    Ep = 0.5 * np.mean(var_per_component)


    # Ensemble initialization
    ensemble = np.array([
        x_true + rng.normal(0, np.sqrt(Ep), N)
        for _ in range(Ne)
    ])  

    # Storage
    rmse_history = []
    spread_history = []
    time_history = []
    prior_ensemble_history = []
    posterior_ensemble_history = []
    truth_history = []

    time = 0.0

    for cycle in range(cycles):
        # Forecast step: integrate ensemble and truth
        ensemble = rk4_integrate(ensemble, dt, n_obs, lorenz96_rhs, F)
        x_true = rk4_integrate(x_true[None, :], dt, n_obs, lorenz96_rhs, F)[0]
        time += n_obs * dt

        # Generate observation
        # y_obs = H_operator(x_true, obs_idx) + rng.normal(0, np.sqrt(R_val), size=len(obs_idx))
        # positive Fourier modes to observe
        # modes = np.arange(1, M+1)  # first M positive Fourier modes
        modes = np.arange(1, min(M+1, N))

        # observation in Fourier space
        y_obs = H_operator_fft(x_true, modes) + rng.normal(0, np.sqrt(R_val), size=len(modes))
        obs_idx = modes  # update obs_idx to match Fourier-space indices

        # Save prior ensemble
        prior_ensemble = ensemble.copy()
        prior_ensemble_history.append(prior_ensemble)

        # Analysis update
        ensemble = sqr_ensemble_update(
            ensemble, y_obs, R_val, obs_idx,
            inflation=inflation, method=method, num_filters = num_filters, kernel_size = kernel_size
        )

        # Save posterior ensemble
        posterior_ensemble_history.append(ensemble.copy())

        # Diagnostics
        x_mean = ensemble.mean(axis=0)
        rmse_val = np.sqrt(np.mean((x_mean - x_true)**2))
        spread_val = np.sqrt(np.mean(np.var(ensemble, axis=0, ddof=1)))

        rmse_history.append(rmse_val)
        spread_history.append(spread_val)
        time_history.append(time)
        truth_history.append(x_true.copy())

    results = {
        "times": np.array(time_history),
        "rmse": np.array(rmse_history),
        "spread": np.array(spread_history),
        "obs_idx": np.array(obs_idx),
        "truth": np.array(truth_history),
        "prior_ensemble": np.array(prior_ensemble_history),
        "posterior_ensemble": np.array(posterior_ensemble_history),
    }

    return results

# ------------------------
# Plotting Functions
# ------------------------
###time vs space plots from book
def plot_L96(x_vals, F, cut):
    # Set up axes
    t_plot = np.arange(x_vals.shape[0]) * (1/64)
    x_plot = np.arange(x_vals.shape[1])

    x_vals_eq = x_vals[cut:, :]
    t_plot_eq = t_plot[cut:]

    plt.figure(figsize=(6,6))
    plt.contourf(x_plot, t_plot_eq, x_vals_eq, levels=20, cmap="RdBu_r")
    plt.xlabel("Space index n")
    plt.ylabel("Time")
    plt.title(f"Lorenz-96 space–time diagram (F={F}, post-spinup)")
    plt.colorbar(label="u")
    plt.show()

###distribution plots from book
def plot_L96_var(x_vals, F, var_idx, cut):
    # Drop first 1000 time units (i.e., 1000 * 64 steps)
    var = x_vals[cut:, var_idx]   # variable index 6 = u7

    # Compute sample mean & std
    mean_var = np.mean(var)
    std_var = np.std(var, ddof=1)

    # Set up histogram and Gaussian for comparison
    bins = 100
    plt.hist(var, bins=bins, density=True, histtype='step', linestyle='--', color='k', label=f'Empirical (x_{var_idx+1})')

    # Gaussian curve
    x_range = np.linspace(var.min(), var.max(), 400)
    plt.plot(x_range, norm.pdf(x_range, mean_var, std_var), 'r-', label='Gaussian (same μ, σ)')

    plt.xlabel(f"x_{var_idx+1} value")
    plt.ylabel("Probability density")
    plt.legend()
    plt.xlim(-20, 30)
    plt.ylim(0, 0.15)
    plt.title(f"Marginal probability distribution of x_{var_idx+1} (F={F})")
    plt.show()

def plot_sqrt_filter_results(results, snapshot_cycles=None):
    """
    Plot RMSE, spread, and value vs. state snapshots for ETKF/EAKF experiments.

    Parameters
    ----------
    results : dict
        Output of run_sqrt_filter_experiment, must include:
        'truth', 'prior_ensemble', 'posterior_ensemble', 'obs_idx', 'times', 'rmse', 'spread'
    snapshot_cycles : list of int, optional
        Cycles at which to plot ensemble snapshots. Default: first, middle, last.
    """
    # --- RMSE and Spread ---
    times = results['times']
    rmse = results['rmse']
    spread = results['spread']

    plt.figure(figsize=(10,5))
    plt.plot(times, rmse, label='RMSE')
    plt.plot(times, spread, label='Ensemble Spread')
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("RMSE and Ensemble Spread")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Ensemble snapshots ---
    if snapshot_cycles is None:
        snapshot_cycles = [0, len(times)//2, len(times)-1]

    truth = results['truth']
    obs_idx = results['obs_idx']

    N = truth.shape[1]
    j = np.arange(N)

    for cyc in snapshot_cycles:
        plt.figure(figsize=(12,6))

        # Prior mean
        prior_mean = results['prior_ensemble'][cyc].mean(axis=0)
        plt.plot(j, prior_mean, 'b-', linewidth=2, label='Prior Mean')

        # Posterior mean
        posterior_mean = results['posterior_ensemble'][cyc].mean(axis=0)
        plt.plot(j, posterior_mean, 'b-', linewidth=1, label='Posterior Mean')

        # Truth
        plt.plot(j, truth[cyc], 'k--', linewidth=2, label='Truth')

        # Observations
        plt.scatter(obs_idx, truth[cyc][obs_idx],
                    facecolors='none', edgecolors='r', s=80, label='Observations')

        plt.title(f"Ensemble Snapshot at Cycle {cyc+1}")
        plt.xlabel("State Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 40)
        plt.ylim(-20, 20)
        plt.show()
# ------------------------
# RUN EXPERIMENT
# ------------------------
##tests that the L96 model roughly matches the plots in 11.1 page 193
# for F in [6,8,16]:
# 	##for L96 model
#     x_vals = L96_test(N = 40, F = F, dt = 1/64, steps = 40*64, rhs = lorenz96_rhs)
#     plot_L96(x_vals, F, cut =20*64)
    ##for L96 normed model (the plots this produce don't seem to match the book, so it seems like they just use the regular L96 for their testing)
    # x_vals_norm = L96_test(N = 40, F = F, dt = 1/64, spinup_steps = 20*64, rhs = lorenz96norm_rhs)
    # plot_L96(x_vals_norm, F, spinup_cut = 5*64)


##tests that the L96 model roughly matches the plots in 11.1 page 195 (the peak is a bit too high and the spread it not wide enough to match that of the book)
#I did test this with both the L96 norm version and with the clipping removed and the peaks were even higher than the regular L96 model, so this confirms I should be using the regular L96 model for my tests.
# for F in [6,8,16]:
#     x_vals = L96_test(N = 40, F = F, dt = 1/64, steps = 11000*64, rhs = lorenz96_rhs)
#     plot_L96_var(x_vals = x_vals, F = F, var_idx = 6, cut = 1000*64)

###recreating the Table 11.1 on page 194 (Takes awhile to run because of the *64) (Seems like only the T_corr value might be a bit off, but overall values seem consistent with table)
# for F in [6,8,16]:
#     print(f"\n=== F = {F} ===")
    
#     # Lyapunov spectrum
#     lyaps = lyapunov_spectrum(F, N=40, dt=1/64,cut=10*64, steps=1000*64)
#     lambda1 = lyaps[0]
#     N_plus = np.sum(lyaps > 0)
#     KS = np.sum(lyaps[lyaps > 0])

#     # Correlation time
#     x_vals = L96_test(N = 40, F = F, dt = 1/64, steps = 1000*64, rhs = lorenz96_rhs)

#     # Spatial mean
#     u_mean = np.mean(x_vals, axis=1, keepdims=True)
#     anomalies = x_vals - u_mean

#     # Global perturbation energy
#     Ep = np.sum(anomalies**2) * dt / (2 * x_vals.shape[0])

#     # Rescale
#     x_tilde = anomalies / np.sqrt(Ep)

#     # Compute correlation function for each component
#     n, N = x_tilde.shape
#     Tcorr_components = []
#     for j in range(N):
#         series = x_tilde[:, j] - np.mean(x_tilde[:, j])  # zero-mean in time
#         ac = np.correlate(series, series, mode='full')
#         ac = ac[n-1:] / ac[n-1]
#         # integrate positive part
#         Tcorr_components.append(np.trapz(ac[ac>0], dx=dt))

#     # Average across all components
#     Tcorr = np.mean(Tcorr_components)

#     print(f"λ₁   = {lambda1:.2f}")
#     print(f"N⁺   = {N_plus}")
#     print(f"KS   = {KS:.2f}")
#     print(f"Tcorr = {Tcorr:.2f}")


####tests to replicate results of 11.2 page
# for F in [6,8,16]:
for F in [8]: 
	results = run_sqrt_filter_experiment(N=N, M=M, Ne=Ne, cycles=cycles, n_obs=n_obs, dt=dt,
                                     	R_val=R_val, inflation=inflation, method='EAKF', seed=seed, obs_idx=obs_idx, F = F, steps = 1000*64)
    # results = run_sqrt_filter_experiment(N=N, M=M, Ne=Ne, cycles=cycles, n_obs=n_obs, dt=dt,
    #                                     R_val=R_val, inflation=inflation, method='UNET', seed=seed, obs_idx=obs_idx, F = F, steps = 1000*64, num_filters = 32, kernel_size = 3)

    plot_sqrt_filter_results(results, snapshot_cycles=[2499, 4999])

def compute_rms_corr(results, spinup=1000):
    """Compute time-averaged RMS and correlation after spinup."""
    rmse = results["rmse"][spinup:]
    truth = results["truth"][spinup:]
    posterior_means = np.array([ens.mean(axis=0) for ens in results["posterior_ensemble"][spinup:]])
    corr = [np.corrcoef(truth[i], posterior_means[i])[0, 1] for i in range(len(truth))]
    return np.mean(rmse), np.mean(corr)


# def run_book_table_eakf():
#     """
#     Reproduce the EAKF columns of the book's Table 11.2 for r^o = 3.
#     """
#     N = 40
#     Ne = 80
#     R_val = 3.0
#     inflation = 1.05  # r = 0.05 (book)
#     F = 8.0
#     dt = 1/64
#     cycles = 5000
#     seed = 42

#     # Observation intervals (n_obs = 5 → T_obs=0.078, n_obs=15 → T_obs=0.234)
#     n_obs_values = [5, 15]
#     P_values = [1, 2, 3, 4, 5]

#     print(f"{'P':>2} | {'Tobs=0.078 (n=5)':^28} | {'Tobs=0.234 (n=15)':^28}")
#     print("   |    RMS     corr         |     RMS     corr        ")
#     print("-"*65)

#     for P in P_values:
#         M = N // P
#         obs_idx = np.arange(0, N, P)

#         row = f"{P:>2} | "
#         for n_obs in n_obs_values:
#             results = run_sqrt_filter_experiment(
#                 N=N, M=M, Ne=Ne, cycles=cycles, n_obs=n_obs, dt=dt,
#                 R_val=R_val, inflation=inflation, method='EAKF',
#                 seed=seed, obs_idx=obs_idx, F=F, steps=1000*64
#             )

#             rms, corr = compute_rms_corr(results, spinup=1000)
#             row += f"{rms:6.2f}    {corr:4.2f}       | "

#         print(row)

# run_book_table_eakf()