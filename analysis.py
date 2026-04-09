"""
DA Experiment Analysis Script
==============================
Produces diagnostic figures for any experiment group defined in experiments.py.
Currently supports the Lorenz 96 (L96) model, and is structured to support the
1-layer quasi-geostrophic (QG1) model once it is implemented — no changes to
this file will be needed for plots that are model-agnostic.  Model-specific
plots (snapshot, hovmoller, energy_spectrum) dispatch on model_id and have
clearly labelled stubs for QG1.

Usage
-----
    python analysis.py                         # run ACTIVE_GROUP, all plots
    python analysis.py --plots rmse spread     # specific plots only
    python analysis.py --group lr_sweep        # override active group
    python analysis.py --help                  # list all options
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import xarray as xr
from experiments import GROUPS
from DA_framework import L96_DA_Experiment, QG1_DA_Experiment

# ─────────────────────────────────────────────
# CONFIG  — only line you normally need to change
# ─────────────────────────────────────────────
ACTIVE_GROUP = 'lr_sweep'

# Plots that are meaningful for all models
_UNIVERSAL_PLOTS = [
    'losses',
    'rmse',
    'spread',
    'rmse_vs_spread',
    'covariance_anim',
    'distribution',
]

# Plots that have model-specific implementations
_MODEL_PLOTS = [
    'hovmoller',       # 1D space–time for L96 | 2D field sequence for QG1
    'energy_spectrum', # 1D wavenumber for L96 | 2D isotropic spectrum for QG1
    'snapshot',        # 1D spatial cross-section for L96 | 2D field for QG1
]

AVAILABLE_PLOTS = _UNIVERSAL_PLOTS + _MODEL_PLOTS


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_base_dir(exp):
    """Base data directory — mirrors run_experiments.get_base_dir()."""
    return f'./data/{exp["model_id"]}'


def get_exp_dir(exp):
    """Per-experiment output directory."""
    return f'{get_base_dir(exp)}/{exp["exp_id"]}'


def build_file_name(exp, method=None):
    """
    Reconstruct the file-name stem produced by DA_Experiment.file_name().
    Must stay in sync with DA_framework.py.
    """
    m = method or exp['DA_method']
    if m == 'NoDA':
        return f'Control_N{exp["N_DA"]}'
    return (f'{m}_N{exp["N_DA"]}_ens{exp["nens"]}_freq{exp["obs_freq"]}'
            f'_nobs{exp["nobs"]}_err{exp["obs_err"]:.1e}')


def get_da_experiment(exp):
    """
    Instantiate the correct DA_Experiment subclass for obs reading.
    Add new model_id cases here as new models are implemented.
    """
    model_id = exp.get('model_id')
    if model_id == 'L96':
        return L96_DA_Experiment(**exp)
    elif model_id == 'QG1':
        return QG1_DA_Experiment(**exp)
    else:
        raise ValueError(f"Unknown model_id '{model_id}' in get_da_experiment(). "
                         f"Register the corresponding DA_Experiment subclass.")


def make_exp_label(exp):
    """
    Short human-readable label for plot titles.
    Uses model-specific parameters where they exist.
    """
    parts = [exp['DA_method'], f'N={exp["N_DA"]}', f'nens={exp["nens"]}']
    if 'F' in exp:
        parts.append(f'F={exp["F"]}')
    if exp.get('model_id') == 'QG1':
        if 'nx' in exp:
            parts.append(f'{exp["nx"]}x{exp.get("ny", exp["nx"])}')
    return '  '.join(parts)


def load_experiment_data(exp):
    """
    Load all datasets needed for diagnostic plots and return them in a dict.

    The returned dict contains:
        exp_dir        — str: per-experiment save directory
        b_folder       — str: file-name stem matching DA_framework output
        analysis_mean  — xr.Dataset  with variable 'x'  (n_cycles, N)
        forecast_mean  — xr.Dataset  with variable 'x'  (n_cycles, N)
        analysis_std   — xr.Dataset  with variable 'x'  (n_cycles, N)
        forecast_std   — xr.Dataset  with variable 'x'  (n_cycles, N)
        truth_aligned  — np.ndarray  (n_cycles, N)  physical-space, aligned to DA cycles
        truth_raw      — np.ndarray  (n_time, N)    full truth run (for spectra etc.)
    """
    exp_dir  = get_exp_dir(exp)
    b_folder = build_file_name(exp)

    data = {
        'exp_dir':  exp_dir,
        'b_folder': b_folder,
    }

    # ── DA output files (same structure for all models) ─────────────────────
    for tag, fname in [
        ('analysis_mean', f'EnsMean_{b_folder}.nc'),
        ('forecast_mean', f'ForecastMean_{b_folder}.nc'),
        ('analysis_std',  f'EnsStd_{b_folder}.nc'),
        ('forecast_std',  f'ForecastStd_{b_folder}.nc'),
    ]:
        path = f'{exp_dir}/{fname}'
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected DA output file not found: {path}\n"
                f"Has Phase 1 been run for experiment '{exp['exp_id']}'?")
        data[tag] = xr.open_dataset(path)

    # ── Truth file — path depends on the model's generate_truth() naming ────
    data['truth_raw'], data['truth_aligned'] = _load_truth(exp, data)

    return data


def _load_truth(exp, data):
    """
    Load the truth trajectory and return (truth_raw, truth_aligned).
    truth_raw     : (n_time, N)  — full run, used for spectra
    truth_aligned : (n_cycles, N) — sub-sampled to match DA cycle times

    Add a new elif branch here when implementing QG1 if its truth file uses
    a different naming convention than L96.
    """
    base_dir  = get_base_dir(exp)
    model_id  = exp['model_id']
    N         = exp['N_truth']
    dt        = exp['dt']
    steps     = exp['steps_truth']

    if model_id == 'L96':
        F          = exp['F']
        truth_file = f'{base_dir}/Truth_N{N}_F{F}_dt{dt}_{steps}steps.nc'
    elif model_id == 'QG1':
        # ── QG1: update this path once generate_truth() is implemented ──────
        # Expected convention (mirror L96 pattern):
        #   Truth_N{N}_dt{dt}_{steps}steps.nc
        truth_file = f'{base_dir}/Truth_N{N}_dt{dt}_{steps}steps.nc'
    else:
        raise ValueError(f"Unknown model_id '{model_id}' in _load_truth(). "
                         f"Add a truth-file path for this model.")

    if not os.path.exists(truth_file):
        raise FileNotFoundError(
            f"Truth file not found: {truth_file}\n"
            f"Has generate_truth() been run for this experiment?")

    truth_ds = xr.open_dataset(truth_file)
    if 'x' not in truth_ds:
        raise KeyError(
            f"Truth file '{truth_file}' has no variable 'x'. "
            f"Variables present: {list(truth_ds.data_vars)}")

    truth_raw = truth_ds['x'].values                # (n_time, N)

    obs_freq  = exp['obs_freq']
    n_cycles  = len(data['analysis_mean'].time)
    idx       = np.arange(obs_freq - 1, obs_freq * n_cycles, obs_freq)
    idx       = idx[idx < truth_raw.shape[0]]       # clamp to available steps
    truth_aligned = truth_raw[idx]                  # (n_cycles, N)

    return truth_raw, truth_aligned


def _compute_rmse_spread(data):
    """Return (analysis_rmse, forecast_rmse, analysis_spread, forecast_spread)."""
    a_mean = data['analysis_mean'].x.values         # (n_cycles, N)
    f_mean = data['forecast_mean'].x.values
    truth  = data['truth_aligned']

    n = min(len(a_mean), len(truth))
    a_rmse   = np.sqrt(np.mean((a_mean[:n] - truth[:n]) ** 2, axis=1))
    f_rmse   = np.sqrt(np.mean((f_mean[:n] - truth[:n]) ** 2, axis=1))
    a_spread = data['analysis_std'].x.values[:n].mean(axis=1)
    f_spread = data['forecast_std'].x.values[:n].mean(axis=1)
    return a_rmse, f_rmse, a_spread, f_spread


def _spatial_corr(a_mean, truth):
    """Per-cycle spatial Pearson correlation between analysis mean and truth."""
    n = min(len(a_mean), len(truth))
    return np.array([np.corrcoef(a_mean[t].ravel(), truth[t].ravel())[0, 1]
                     for t in range(n)])


# ─────────────────────────────────────────────
# UNIVERSAL PLOT FUNCTIONS
# (work identically for all models)
# ─────────────────────────────────────────────

def plot_losses(exp, data, width, **_):
    """UNet training / validation loss curves."""
    exp_dir  = data['exp_dir']
    csv_path = f'{exp_dir}/losses.csv'
    if not os.path.exists(csv_path):
        print(f'  [losses] Skipping: losses.csv not found in {exp_dir}')
        return
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df.index, df['train_loss'], label='Training Loss')
    ax.plot(df.index, df['valid_loss'], label='Validation Loss')
    ax.set_xlabel('Epoch');  ax.set_ylabel('MSE Loss')
    ax.set_title(f'Training vs Validation Loss\n{make_exp_label(exp)}')
    ax.legend();  ax.grid(True)
    plt.tight_layout()
    out = f'{exp_dir}/losses_plt.png'
    plt.savefig(out, dpi=150);  plt.close()
    print(f'  [losses] Saved {os.path.basename(out)}')


def plot_rmse(exp, data, **_):
    """Per-cycle RMSE for analysis and forecast, plus summary statistics."""
    exp_dir = data['exp_dir']
    a_rmse, f_rmse, _, _ = _compute_rmse_spread(data)
    cycles = np.arange(len(a_rmse))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cycles, f_rmse, label='Forecast RMSE', color='orange', alpha=0.8)
    ax.plot(cycles, a_rmse, label='Analysis RMSE', color='blue',   alpha=0.8)
    ax.set_xlabel('DA Cycle');  ax.set_ylabel('RMSE')
    ax.set_title(f'Filter Performance\n{make_exp_label(exp)}')
    ax.legend();  ax.grid(True)
    plt.tight_layout()
    out = f'{exp_dir}/rmse_plt.png'
    plt.savefig(out, dpi=150);  plt.close()
    print(f'  [rmse] Saved {os.path.basename(out)}')

    corrs = _spatial_corr(data['analysis_mean'].x.values, data['truth_aligned'])
    print(f'    Forecast RMSE  (mean): {f_rmse.mean():.4f}')
    print(f'    Analysis RMSE  (mean): {a_rmse.mean():.4f}')
    print(f'    Analysis corr  (mean): {corrs.mean():.4f}')


def plot_spread(exp, data, **_):
    """Per-cycle ensemble spread for analysis and forecast."""
    exp_dir = data['exp_dir']
    _, _, a_spread, f_spread = _compute_rmse_spread(data)
    cycles = np.arange(len(a_spread))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cycles, f_spread, label='Forecast Spread', color='orange', alpha=0.8)
    ax.plot(cycles, a_spread, label='Analysis Spread', color='blue',   alpha=0.8)
    ax.set_xlabel('DA Cycle');  ax.set_ylabel('Ensemble Spread (std)')
    ax.set_title(f'Ensemble Spread\n{make_exp_label(exp)}')
    ax.legend();  ax.grid(True)
    plt.tight_layout()
    out = f'{exp_dir}/spread_plt.png'
    plt.savefig(out, dpi=150);  plt.close()
    print(f'  [spread] Saved {os.path.basename(out)}')
    print(f'    Forecast Spread (mean): {f_spread.mean():.4f}')
    print(f'    Analysis Spread (mean): {a_spread.mean():.4f}')


def plot_rmse_vs_spread(exp, data, **_):
    """Overlay RMSE and spread to assess filter calibration."""
    exp_dir = data['exp_dir']
    a_rmse, f_rmse, a_spread, f_spread = _compute_rmse_spread(data)
    cycles = np.arange(len(a_rmse))
    label  = make_exp_label(exp)

    for tag, rmse, spread, fname in [
        ('Forecast', f_rmse, f_spread, 'forecast_rmse_vs_spread_plt.png'),
        ('Analysis', a_rmse, a_spread, 'analysis_rmse_vs_spread_plt.png'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(cycles, rmse,   label=f'{tag} RMSE',   color='orange')
        ax.plot(cycles, spread, label=f'{tag} Spread', color='blue', linestyle='--')
        ax.set_title(f'{tag} RMSE vs Spread\n{label}')
        ax.set_xlabel('DA Cycle');  ax.set_ylabel('Value')
        ax.legend();  ax.grid(True)
        plt.tight_layout()
        out = f'{exp_dir}/{fname}'
        plt.savefig(out, dpi=150);  plt.close()
        print(f'  [rmse_vs_spread] Saved {fname}')


def plot_covariance_anim(exp, data, width, **_):
    """
    Animate the diagonal of the ensemble B matrix over DA cycles.
    Works for any model — the diagonal is always a 1D vector of length N
    regardless of the underlying state dimensionality.
    """
    exp_dir    = data['exp_dir']
    b_ens_file = f'{exp_dir}/B_ens_{data["b_folder"]}.nc'

    if not os.path.exists(b_ens_file):
        print(f'  [covariance_anim] Skipping: {os.path.basename(b_ens_file)} not found.')
        return

    B_ens_ds = xr.open_dataset(b_ens_file)
    if 'B_ens' not in B_ens_ds:
        print(f'  [covariance_anim] Skipping: unrecognised B file format. '
              f'Variables: {list(B_ens_ds.data_vars)}')
        return

    n_samples   = len(B_ens_ds.cycle)
    spin_up     = int(n_samples * 0.2)
    decorr_skip = max(1, int(1 / exp['dt']))        # ~1 model time unit between frames

    B_ens      = B_ens_ds['B_ens'].isel(cycle=slice(spin_up, None, decorr_skip)).values
    n_frames   = B_ens.shape[0]
    cycle_idx  = np.arange(spin_up, n_samples, decorr_skip)[:n_frames]
    diag       = np.diagonal(B_ens, axis1=1, axis2=2)   # (n_frames, N)
    diag_mean  = diag.mean(axis=0)
    y_max      = np.percentile(np.abs(diag), 99) * 1.1
    x_axis     = np.arange(diag.shape[1])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ln_var,  = axes[0].plot(x_axis, diag[0],      color='blue', alpha=0.8,  label='Variance')
    ln_vm,   = axes[0].plot(x_axis, diag_mean,    color='blue', alpha=0.3,
                            linestyle=':', label='Time-mean')
    axes[0].set_ylim(0, y_max)
    axes[0].set_xlabel('State index');  axes[0].set_ylabel('Variance')
    axes[0].set_title(f'Variance — Cycle {cycle_idx[0]}')
    axes[0].legend();  axes[0].grid(True)

    ln_std,  = axes[1].plot(x_axis, np.sqrt(diag[0]),     color='blue', alpha=0.8,  label='Spread')
    ln_sm,   = axes[1].plot(x_axis, np.sqrt(diag_mean),   color='blue', alpha=0.3,
                            linestyle=':', label='Time-mean')
    axes[1].set_ylim(0, np.sqrt(y_max))
    axes[1].set_xlabel('State index');  axes[1].set_ylabel('Std Dev')
    axes[1].set_title(f'Spread — Cycle {cycle_idx[0]}')
    axes[1].legend();  axes[1].grid(True)
    plt.tight_layout()

    def _update(frame):
        ln_var.set_ydata(diag[frame])
        ln_std.set_ydata(np.sqrt(diag[frame]))
        axes[0].set_title(f'Variance — Cycle {cycle_idx[frame]}')
        axes[1].set_title(f'Spread   — Cycle {cycle_idx[frame]}')
        return ln_var, ln_std

    frames_dir = f'{exp_dir}/B_diagonal_frames_w{width}'
    os.makedirs(frames_dir, exist_ok=True)
    for i in range(n_frames):
        _update(i);  plt.tight_layout()
        fig.savefig(f'{frames_dir}/frame_{i:04d}.png', dpi=100)
    print(f'  [covariance_anim] Saved {n_frames} frames → {os.path.basename(frames_dir)}/')

    ani = FuncAnimation(fig, _update, frames=n_frames, interval=150, blit=True)
    out = f'{exp_dir}/B_diagonal_over_time_w{width}.gif'
    ani.save(out, writer='pillow', fps=15);  plt.close()
    print(f'  [covariance_anim] Saved {os.path.basename(out)}')


def plot_distribution(exp, data, n_cycles=1000, var_idx=0, **_):
    """
    Distribution of one state variable over time vs a Gaussian.
    Works for all models — var_idx is a flat index into the N-dimensional state.
    For QG1 this indexes into the flattened (nx*ny) state vector.
    """
    exp_dir = data['exp_dir']
    truth   = data['truth_aligned']               # (n_cycles, N)
    n       = min(n_cycles, len(truth))
    vals    = truth[:n, var_idx]

    mu     = vals.mean()
    sigma  = vals.std()
    x_rng  = np.linspace(vals.min(), vals.max(), 200)
    gauss  = np.exp(-0.5 * ((x_rng - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(vals, bins=50, density=True, alpha=0.6, color='steelblue',
            label=f'x[{var_idx}]')
    ax.plot(x_rng, gauss, 'r-', lw=2, label=f'Gaussian (μ={mu:.2f}, σ={sigma:.2f})')
    ax.set_xlabel('State value');  ax.set_ylabel('Density')
    ax.set_title(f'State distribution vs Gaussian\n{make_exp_label(exp)}  '
                 f'— var {var_idx}, {n} cycles')
    ax.legend();  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f'{exp_dir}/state_dist_var{var_idx}_{n}cycles.png'
    plt.savefig(out, dpi=150);  plt.close()
    print(f'  [distribution] Saved {os.path.basename(out)}')


# ─────────────────────────────────────────────
# MODEL-SPECIFIC PLOT FUNCTIONS
# Each function dispatches on model_id and calls
# the appropriate private implementation below.
# ─────────────────────────────────────────────

def plot_hovmoller(exp, data, **_):
    model_id = exp.get('model_id')
    if model_id == 'L96':
        _hovmoller_l96(exp, data)
    elif model_id == 'QG1':
        _hovmoller_qg1(exp, data)
    else:
        print(f'  [hovmoller] No implementation for model_id={model_id!r}, skipping.')


def plot_energy_spectrum(exp, data, **_):
    model_id = exp.get('model_id')
    if model_id == 'L96':
        _energy_spectrum_l96(exp, data)
    elif model_id == 'QG1':
        _energy_spectrum_qg1(exp, data)
    else:
        print(f'  [energy_spectrum] No implementation for model_id={model_id!r}, skipping.')


def plot_snapshot(exp, data, **_):
    model_id = exp.get('model_id')
    if model_id == 'L96':
        _snapshot_l96(exp, data)
    elif model_id == 'QG1':
        _snapshot_qg1(exp, data)
    else:
        print(f'  [snapshot] No implementation for model_id={model_id!r}, skipping.')


# ─────────────────────────────────────────────
# L96  —  model-specific implementations
# ─────────────────────────────────────────────

def _hovmoller_l96(exp, data, t_max=20.0):
    """Space–time Hovmöller diagram of the L96 truth field."""
    exp_dir  = data['exp_dir']
    truth    = data['truth_aligned']              # (n_cycles, N)
    dt_cycle = exp['obs_freq'] * exp['dt']
    times    = np.arange(len(truth)) * dt_cycle

    mask  = times <= t_max
    truth = truth[mask];  times = times[mask]
    N     = truth.shape[1]

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(np.arange(N), times, truth, levels=25, cmap='RdBu_r')
    ax.contour(np.arange(N), times, truth, levels=25,
               colors='black', linewidths=0.2, alpha=0.3)
    plt.colorbar(cf, ax=ax, label='State value')
    ax.set_xlabel('space');  ax.set_ylabel('Model time')
    ax.set_xlim(0, N - 1);  ax.set_ylim(0, t_max)
    ax.set_title(f'Truth Hovmöller  F={exp["F"]}')
    plt.tight_layout()
    out = f'{exp_dir}/truth_hovmoller_t{t_max}.png'
    plt.savefig(out, dpi=150);  plt.close()
    print(f'  [hovmoller] Saved {os.path.basename(out)}')


def _energy_spectrum_l96(exp, data, n_cycles=10000):
    """
    1D variance spectrum E_k (textbook Figure 12.1).
    E_k = variance of the k-th Fourier coefficient over time, computed on the
    climatologically normalised truth  ũ = (u − ū) / σ.
    """
    exp_dir = data['exp_dir']
    truth   = data['truth_aligned']
    n       = min(n_cycles, len(truth))
    N       = truth.shape[1]

    clim_mean    = truth[:n].mean()
    clim_std     = truth[:n].std()
    truth_sc     = (truth[:n] - clim_mean) / clim_std       # (n, N)

    x_hat = np.fft.rfft(truth_sc, axis=1) / N               # (n, N//2+1)
    E_k   = x_hat.real.var(axis=0) + x_hat.imag.var(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(len(E_k)), E_k, 'b-', lw=1.5, label=f'F={exp["F"]}')
    ax.set_xlabel('Wavenumber k');  ax.set_ylabel('$E_k$')
    ax.set_title(f'Energy Spectrum  F={exp["F"]}')
    ax.set_xlim(0, N // 2);  ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3);  ax.legend()
    plt.tight_layout()
    out = f'{exp_dir}/energy_spectrum_{n}cycles.png'
    plt.savefig(out, dpi=150);  plt.close()
    print(f'  [energy_spectrum] Saved {os.path.basename(out)}')


def _snapshot_l96(exp, data, cycles_to_plot=(2499, 4999)):
    """
    Spatial snapshot at chosen DA cycles (textbook Figure 11.4).
    Shows truth, prior mean, analysis mean, and observations.
    """
    exp_dir       = data['exp_dir']
    truth         = data['truth_aligned']
    analysis_mean = data['analysis_mean'].x.values
    forecast_mean = data['forecast_mean'].x.values
    N             = truth.shape[1]
    x_ax          = np.arange(1, N + 1)

    DA_exp = L96_DA_Experiment(**exp)
    obs_ds = DA_exp.read_obs()

    fig, axes = plt.subplots(len(cycles_to_plot), 1,
                             figsize=(12, 4 * len(cycles_to_plot)))
    if len(cycles_to_plot) == 1:
        axes = [axes]

    for ax, cycle in zip(axes, cycles_to_plot):
        if cycle >= len(truth):
            print(f'  [snapshot] Cycle {cycle} out of range, skipping.')
            continue

        obs_vals   = obs_ds.x[cycle].values
        obs_idx    = obs_ds.idx[cycle].values
        prior_corr = np.corrcoef(forecast_mean[cycle], truth[cycle])[0, 1]
        post_corr  = np.corrcoef(analysis_mean[cycle], truth[cycle])[0, 1]

        ax.plot(x_ax, truth[cycle],         'k--', lw=1.5, label='Truth')
        ax.plot(x_ax, forecast_mean[cycle], 'k-',  lw=2.0, label='Prior mean')
        ax.plot(x_ax, analysis_mean[cycle], 'b-',  lw=1.0, label='Analysis mean')
        ax.plot(obs_idx + 1, obs_vals, 'ko', ms=5, fillstyle='none', label='Obs')
        ax.set_xlabel('space');  ax.set_ylabel('State')
        ax.set_xlim(1, N)
        ax.set_title(
            f'{exp["DA_method"]}  F={exp["F"]}  P={exp["N_DA"] // exp["nobs"]}'
            f'  $T_{{obs}}$={exp["obs_freq"] * exp["dt"]:.5f}'
            f'  $r^o$={exp["obs_err"] ** 2:.0f}  cycle={cycle + 1}\n'
            f'Prior corr={prior_corr:.4f}  Analysis corr={post_corr:.4f}'
        )
        ax.legend(loc='upper right', fontsize=8);  ax.grid(True, alpha=0.3)

    plt.tight_layout()
    cycles_str = '_'.join(str(c + 1) for c in cycles_to_plot)
    out = f'{exp_dir}/snapshot_cycles{cycles_str}.png'
    plt.savefig(out, dpi=150);  plt.close()
    print(f'  [snapshot] Saved {os.path.basename(out)}')


# ─────────────────────────────────────────────
# QG1  —  stubs for model-specific plots
#
# Fill these in once QG1Model and QG1_DA_Experiment
# are fully implemented.  The universal plots
# (rmse, spread, losses, covariance_anim, distribution)
# will work automatically — only these three need
# QG1-specific implementations.
# ─────────────────────────────────────────────

def _hovmoller_qg1(exp, data):
    """
    QG1 analogue of the Hovmöller diagram.

    Suggested implementation: plot a sequence of 2D PV field snapshots as a
    figure with one panel per time step, or animate the field over time.

    Skeleton:
        nx, ny        = exp['nx'], exp['ny']
        truth         = data['truth_aligned']     # (n_cycles, nx*ny)
        truth_2d      = truth.reshape(-1, nx, ny) # (n_cycles, nx, ny)
        dt_cycle      = exp['obs_freq'] * exp['dt']
        # … plot selected frames as imshow panels …
    """
    print('  [hovmoller] QG1 implementation not yet available — skipping.')


def _energy_spectrum_qg1(exp, data):
    """
    QG1 isotropic kinetic energy spectrum.

    Suggested implementation: compute the 2D FFT of the PV field at each
    cycle, bin into radial wavenumber shells, and plot E(k) vs k.

    Skeleton:
        nx, ny        = exp['nx'], exp['ny']
        truth         = data['truth_aligned']        # (n_cycles, nx*ny)
        truth_2d      = truth.reshape(-1, nx, ny)
        # 2D rfft, compute isotropic E(k), average over time
        # … plot E(k) vs k on log-log axes …
    """
    print('  [energy_spectrum] QG1 implementation not yet available — skipping.')


def _snapshot_qg1(exp, data):
    """
    QG1 field snapshot: truth, prior mean, analysis mean, obs locations.

    Suggested implementation: use imshow / pcolormesh to show each 2D field
    side-by-side at a chosen DA cycle.

    Skeleton:
        nx, ny        = exp['nx'], exp['ny']
        truth         = data['truth_aligned']        # (n_cycles, nx*ny)
        analysis_mean = data['analysis_mean'].x.values
        forecast_mean = data['forecast_mean'].x.values
        DA_exp        = QG1_DA_Experiment(**exp)
        obs_ds        = DA_exp.read_obs()
        # reshape to (nx, ny) and use imshow for chosen cycles …
    """
    print('  [snapshot] QG1 implementation not yet available — skipping.')


# ─────────────────────────────────────────────
# DISPATCH TABLE
# ─────────────────────────────────────────────

PLOT_FUNCS = {
    'losses':          plot_losses,
    'rmse':            plot_rmse,
    'spread':          plot_spread,
    'rmse_vs_spread':  plot_rmse_vs_spread,
    'covariance_anim': plot_covariance_anim,
    'distribution':    plot_distribution,
    'hovmoller':       plot_hovmoller,
    'energy_spectrum': plot_energy_spectrum,
    'snapshot':        plot_snapshot,
}


# ─────────────────────────────────────────────
# CLI + MAIN
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='DA Experiment Analysis — supports L96, QG1 (stub)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Available plots:\n  ' + '\n  '.join(AVAILABLE_PLOTS),
    )
    parser.add_argument(
        '--group', default=None,
        help='Experiment group name (overrides ACTIVE_GROUP in script).'
    )
    parser.add_argument(
        '--plots', nargs='+', default=AVAILABLE_PLOTS,
        choices=AVAILABLE_PLOTS, metavar='PLOT',
        help='Which plots to generate (default: all).',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args         = parse_args()
    group_name   = args.group or ACTIVE_GROUP
    group        = GROUPS[group_name]
    experiments  = group['experiments']
    widths       = group['widths']

    # Identify which plot functions need the full data dict loaded
    DATA_PLOTS = {'rmse', 'spread', 'rmse_vs_spread', 'covariance_anim',
                  'distribution', 'hovmoller', 'energy_spectrum', 'snapshot'}

    print(f'Group : {group_name}')
    print(f'Plots : {args.plots}\n')

    for exp, width in zip(experiments, widths):
        name = exp['exp_id']
        print(f'{"=" * 52}')
        print(f'Experiment : {name}  ({exp["model_id"]})')
        print(f'{"=" * 52}')

        needs_data = any(p in DATA_PLOTS for p in args.plots)
        try:
            data = load_experiment_data(exp) if needs_data else {'exp_dir': get_exp_dir(exp)}
        except FileNotFoundError as e:
            print(f'  ERROR loading data: {e}\n  Skipping experiment.\n')
            continue

        for plot_name in args.plots:
            try:
                PLOT_FUNCS[plot_name](exp=exp, data=data, width=width)
            except Exception as e:
                print(f'  [{ plot_name}] ERROR: {e}')

        print()