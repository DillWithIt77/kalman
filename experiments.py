"""
Experiment definitions for Lorenz 96 UNetKF Training.

To add a new experiment:
1. Define a new DA config dict below
2. Add it to one of the run groups (or create a new one)
3. In your main script, import and set `active_group = your_group_name`
"""

# ─────────────────────────────────────────────
# BASE CONFIG (shared defaults)
# ─────────────────────────────────────────────
L96_BASE = {
    'N_truth': 40,
    'N_DA': 40,
    'obs_freq': 5,
    'obs_err': 1.73,
    'nobs': 20,
    'save_B': True,
    'inflate': 1.05,
    'F': 8.0,
    'dt': 0.015625,
    'model_id': 'L96',
    'steps_train': 400000,
    'steps_truth': 1200000,
    'steps_spin_up': 10000,
}

# QG1_BASE = {
#     'N_truth': ???,
#     'N_DA': ???,
#     'obs_freq': ????,
#     'obs_err': ???,
#     'nobs': ???,
#     'save_B': True,
#     'inflate': ???,
#     'dt': ???,
#     'model_id': 'QG1',
#     'steps_train': ???,
#     'steps_truth': ???,
#     'steps_spin_up': ???,
# }

def l_exp(**overrides):
    return {**L96_BASE, **overrides}

# def q_exp(**overides)
    # return {**QG1_BASE, **overrides}

# ─────────────────────────────────────────────
# INDIVIDUAL EXPERIMENT CONFIGS
# ─────────────────────────────────────────────

# --- EnKF, ensemble 20 ---
DA_exp_L20             = l_exp(exp_id = 'L20', nens=20, DA_method='EnKF', obs_type='regular')
DA_exp_L20_ran_1       = l_exp(exp_id = 'L20_ran_1',nens=20, DA_method='EnKF', obs_type='random_once')
DA_exp_L20_ran_every   = l_exp(exp_id = 'L20_ran_every',nens=20, DA_method='EnKF', obs_type='random_every')

# --- EnKF, ensemble 20, M 10 ---
DA_exp_L20_M10             = l_exp(exp_id = 'L20_M10',nens=20, nobs = 10, DA_method='EnKF', obs_type='regular')
DA_exp_L20_ran_1_M10       = l_exp(exp_id = 'L20_M10_ran_1',nens=20, nobs = 10, DA_method='EnKF', obs_type='random_once')
DA_exp_L20_ran_every_M10   = l_exp(exp_id = 'L20_M10_ran_every',nens=20, nobs = 10, DA_method='EnKF', obs_type='random_every')

# --- EnKF, ensemble 20, M 5 ---
DA_exp_L20_M5             = l_exp(exp_id = 'L20_M5',nens=20, nobs = 5, DA_method='EnKF', obs_type='regular')
DA_exp_L20_ran_1_M5       = l_exp(exp_id = 'L20_M5_ran_1',nens=20, nobs = 5, DA_method='EnKF', obs_type='random_once')
DA_exp_L20_ran_every_M5   = l_exp(exp_id = 'L20_M5_ran_every',nens=20, nobs = 5, DA_method='EnKF', obs_type='random_every')

# --- EnKF, ensemble 80 ---
DA_exp_L80             = l_exp(exp_id = 'L80',nens=80, DA_method='EnKF', obs_type='regular')
DA_exp_L80_ran_1       = l_exp(exp_id = 'L80_ran_1',nens=80, DA_method='EnKF', obs_type='random_once')
DA_exp_L80_ran_every   = l_exp(exp_id = 'L80_ran_every',nens=80, DA_method='EnKF', obs_type='random_every')

# --- EnKF, ensemble 80, M 10 ---
DA_exp_L80_M10             = l_exp(exp_id = 'L80_M10',nens=80, nobs = 10, DA_method='EnKF', obs_type='regular')
DA_exp_L80_ran_1_M10       = l_exp(exp_id = 'L80_M10_ran_1',nens=80, nobs = 10, DA_method='EnKF', obs_type='random_once')
DA_exp_L80_ran_every_M10   = l_exp(exp_id = 'L80_M10_ran_every',nens=80, nobs = 10, DA_method='EnKF', obs_type='random_every')

# --- EnKF, ensemble 80, M 5 ---
DA_exp_L80_M5             = l_exp(exp_id = 'L80_M5',nens=80, nobs = 5, DA_method='EnKF', obs_type='regular')
DA_exp_L80_ran_1_M5       = l_exp(exp_id = 'L80_M5_ran_1',nens=80, nobs = 5, DA_method='EnKF', obs_type='random_once')
DA_exp_L80_ran_every_M5   = l_exp(exp_id = 'L80_M5_ran_every',nens=80, nobs = 5, DA_method='EnKF', obs_type='random_every')

# --- ETKF and EAKF, ensemble 80, M 20, all Fs ---
DA_exp_L80_ETKF_M20_F6      = l_exp(exp_id = 'L80_ETKF_F6',nens=80, nobs = 20, F = 6, DA_method='ETKF', obs_type='regular')
DA_exp_L80_EAKF_M20_F6      = l_exp(exp_id = 'L80_EAKF_F6',nens=80, nobs = 20, F = 6, DA_method='EAKF', obs_type='regular')
DA_exp_L80_ETKF_M20_F8      = l_exp(exp_id = 'L80_ETKF_F8',nens=80, nobs = 20, DA_method='ETKF', obs_type='regular')
DA_exp_L80_EAKF_M20_F8      = l_exp(exp_id = 'L80_EAKF_F8',nens=80, nobs = 20, DA_method='EAKF', obs_type='regular')
DA_exp_L80_ETKF_M20_F16     = l_exp(exp_id = 'L80_ETKF_F16',nens=80, nobs = 20, F = 16, DA_method='ETKF', obs_type='regular')
DA_exp_L80_EAKF_M20_F16     = l_exp(exp_id = 'L80_EAKF_F16',nens=80, nobs = 20, F = 16, DA_method='EAKF', obs_type='regular')

# --- EnKF ensemble 80, M 20, all Fs ---
DA_exp_L80_M20_F6       = l_exp(exp_id = 'L80_F6',nens=80, F = 6, DA_method='EnKF', obs_type='regular')
DA_exp_L80_M20_F8       = l_exp(exp_id = 'L80_F8',nens=80, DA_method='EnKF', obs_type='regular')
DA_exp_L80_M20_F16      = l_exp(exp_id = 'L80_F16',nens=80, F = 16, DA_method='EnKF', obs_type='regular')

# --- EnKF ensemble 80, M 20, all learning rates ---
DA_exp_L80_lr002 = l_exp(exp_id = 'L80_lr002',nens=80, DA_method='EnKF', obs_type='regular')
DA_exp_L80_lr001 = l_exp(exp_id = 'L80_lr001',nens=80, DA_method='EnKF', obs_type='regular')
DA_exp_L80_lr0003 = l_exp(exp_id = 'L80_lr0003',nens=80, DA_method='EnKF', obs_type='regular')

# --- EnKF ensemble 80, M 20, all dropout rates ---
DA_exp_L80_dr03 = l_exp(exp_id = 'L80_dr03',nens=80, DA_method='EnKF', obs_type='regular')
DA_exp_L80_dr02 = l_exp(exp_id = 'L80_dr02',nens=80, DA_method='EnKF', obs_type='regular')
DA_exp_L80_dr01 = l_exp(exp_id = 'L80_dr01',nens=80, DA_method='EnKF', obs_type='regular')

# ─────────────────────────────────────────────
# RUN GROUPS
# Each group is a dict with parallel lists that
# the main script zips over.
# ─────────────────────────────────────────────

GROUPS = {

    'all_exps_strd': dict(
        experiments   = [DA_exp_L20,DA_exp_L20_M10,DA_exp_L20_M5,
        DA_exp_L20_ran_1,DA_exp_L20_ran_1_M10,DA_exp_L20_ran_1_M5,
        DA_exp_L20_ran_every,DA_exp_L20_ran_every_M10,DA_exp_L20_ran_every_M5,
        DA_exp_L80,DA_exp_L80_M10,DA_exp_L80_M5,
        DA_exp_L80_ran_1,DA_exp_L80_ran_1_M10,DA_exp_L80_ran_1_M5,
        DA_exp_L80_ran_every,DA_exp_L80_ran_every_M10,DA_exp_L80_ran_every_M5],
        widths        = [4] * 18,
        learning_rate = [0.002] * 18,
        dropout       = [0.1] * 18,
    ),

    # Learning-rate sweep (EnKF, ens 80)
    'lr_sweep': dict(
        experiments   = [DA_exp_L80_lr002, DA_exp_L80_lr001, DA_exp_L80_lr0003],
        widths        = [4] * 3,
        learning_rate = [0.002, 0.001, 0.0003],
        dropout       = [0] * 3,
    ),

    # Dropout sweep (EnKF, ens 80)
    'dropout_sweep': dict(
        experiments   = [DA_exp_L80_dr03, DA_exp_L80_dr02, DA_exp_L80_dr01],
        widths        = [4] * 3,
        learning_rate = [0.002] * 3,
        dropout       = [0.3, 0.2, 0.1],
    ),

    'ETKF_EAKF_reg': dict(
        experiments   = [DA_exp_L80_ETKF_M20_F6,DA_exp_L80_EAKF_M20_F6, 
        DA_exp_L80_ETKF_M20_F8,DA_exp_L80_EAKF_M20_F8,
        DA_exp_L80_ETKF_M20_F16,DA_exp_L80_EAKF_M20_F16],
        widths        = [4] * 6,
        learning_rate = [0.002] * 6,
        dropout       = [0] * 6,
    ),
    'all_forces_EnKF': dict(
        experiments   = [DA_exp_L80_M20_F6,DA_exp_L80_M20_F8,DA_exp_L80_M20_F16],
        exp_id      = ['L80_F6', 'L80_F8', 'L80_F16'],
        widths        = [4] * 3,
        learning_rate = [0.002] * 3,
        dropout       = [0] * 3,
    ),
    
}