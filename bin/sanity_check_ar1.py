"""
Bayesian Hierarchical AR(1) Hurdle Model (Sanity Check)

A two-step model to handle zero-inflated migration matrices:
Step 1 (Hurdle) : P(flow > 0 | d, t) = logistic(α_d + β_lag * is_mig_lag)
Step 2 (Volume) : log(flow) | flow > 0 ~ Normal(μ_d + φ_d*(log_flow_lag - μ_d), σ_d)

This architecture resolves the topological issues (Dirac mass at 0) caused by 
forcing a continuous Normal distribution onto sparse data arrays.
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from cmdstanpy import CmdStanModel

warnings.filterwarnings('ignore')


# 1. Data Loading & Subsetting

DATA_PATH = "/Users/romain/Desktop/Projets DS/ProjetStat/data/data_final/DF_GRAVITY_sans_NaN.csv"
df_main = pd.read_csv(DATA_PATH)

PAYS_STABLES    = ['FRA', 'USA', 'ESP', 'CAN', 'MEX']
PAYS_CHAOTIQUES = ['DZA', 'MMR', 'RWA', 'HTI', 'ZAF', 'NER']
PAYS_TEST       = PAYS_STABLES + PAYS_CHAOTIQUES

df = df_main[
    df_main['orig'].isin(PAYS_TEST) &
    df_main['dest'].isin(PAYS_TEST) &
    (df_main['orig'] != df_main['dest'])
].copy()

df = df.sort_values(['orig', 'dest', 'year']).reset_index(drop=True)


# 2. Feature Engineering & Temporal Lags

df['dyad'] = df['orig'] + "_" + df['dest']
dyades_uniques = sorted(df['dyad'].unique())
dyad_to_id = {d: i + 1 for i, d in enumerate(dyades_uniques)}
df['dyad_id'] = df['dyad'].map(dyad_to_id)
D = len(dyades_uniques)

if 'is_migration' not in df.columns:
    df['is_migration'] = (df['flow'] > 0).astype(int)

df['is_mig_lag'] = df.groupby('dyad')['is_migration'].shift(1)
df['log_flow'] = np.where(df['flow'] > 0, np.log(df['flow']), np.nan)
df['log_flow_lag'] = df.groupby('dyad')['log_flow'].shift(1)

df_clean = df.dropna(subset=['is_mig_lag']).copy().reset_index(drop=True)


# 3. Two-Step Problem Separation (Hurdle vs Volume)

# Step 1: All observations (for binary classification)
df_hurdle = df_clean.copy()
N_total = len(df_hurdle)

# Step 2: Only strictly positive flows with available lags
df_volume = df_clean[
    (df_clean['flow'] > 0) &
    df_clean['log_flow_lag'].notna()
].copy().reset_index(drop=True)

N_vol = len(df_volume)


# 4. Stan Dictionary Formulation

dyades_vol = sorted(df_volume['dyad'].unique())
dyad_to_id_vol = {d: i + 1 for i, d in enumerate(dyades_vol)}
df_volume['dyad_id_vol'] = df_volume['dyad'].map(dyad_to_id_vol)
D_vol = len(dyades_vol)

stan_data = {
    'N_h': N_total,
    'D_h': D,
    'dyad_id_h': df_hurdle['dyad_id'].astype(int).tolist(),
    'is_mig': df_hurdle['is_migration'].astype(int).tolist(),
    'is_mig_lag': df_hurdle['is_mig_lag'].astype(float).tolist(),

    'N_v': N_vol,
    'D_v': D_vol,
    'dyad_id_v': df_volume['dyad_id_vol'].astype(int).tolist(),
    'log_flow': df_volume['log_flow'].tolist(),
    'log_flow_lag': df_volume['log_flow_lag'].tolist(),
}


# 5. Stan Sampling

STAN_FILE = "/Users/romain/Desktop/Projets DS/ProjetStat/STAN/MCMC_AR1.stan"
model = CmdStanModel(stan_file=STAN_FILE)

fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=2000,
    seed=42,
    adapt_delta=0.95,
    max_treedepth=12,
    show_progress=True,
)


# 6. Diagnostics

resume = fit.summary()
params_globaux = [
    'mu_global', 'tau_mu', 'phi_global', 'tau_phi', 'sigma_global',
    'alpha_global', 'tau_alpha', 'beta_lag_global'
]

print(resume.loc[[p for p in params_globaux if p in resume.index], ['Mean', 'StdDev', 'R_hat', 'ESS_bulk']])


# 7. Visualizations & Posterior Predictive Checks

idata = az.from_cmdstanpy(
    posterior=fit,
    log_likelihood={'hurdle': 'log_lik_h', 'volume': 'log_lik_v'},
    posterior_predictive={'is_mig_hat': 'is_mig_hat', 'log_flow_hat': 'log_flow_hat'}
)

# Global Hyperparameters Traceplots
fig, axes = plt.subplots(len(params_globaux), 2, figsize=(14, 3 * len(params_globaux)))
colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

for i, param in enumerate(params_globaux):
    if param not in idata.posterior:
        axes[i, 0].set_visible(False)
        axes[i, 1].set_visible(False)
        continue
    chains_data = idata.posterior[param].values
    ax_t, ax_h = axes[i, 0], axes[i, 1]
    for c in range(chains_data.shape[0]):
        ax_t.plot(chains_data[c], alpha=0.7, lw=0.5, color=colors[c])
    ax_t.set_title(f'Trace : {param}', fontsize=9)
    all_d = chains_data.flatten()
    ax_h.hist(all_d, bins=60, color='#1565C0', alpha=0.7, density=True)
    ax_h.axvline(np.mean(all_d), color='red', lw=1.5, linestyle='--', label=f'μ={np.mean(all_d):.3f}')
    ax_h.set_title(f'Posterior : {param}', fontsize=9)
    ax_h.legend(fontsize=8)

plt.tight_layout()
plt.show()

# PPC Step 1: Hurdle (Bernoulli)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
is_mig_obs = np.array(stan_data['is_mig'])
is_mig_hat = idata.posterior_predictive['is_mig_hat'].values.reshape(-1, N_total)

obs_rate = is_mig_obs.mean()
pred_rates = is_mig_hat.mean(axis=1)

axes[0].hist(pred_rates, bins=40, color='#2196F3', alpha=0.7, density=True, label='Replications')
axes[0].axvline(obs_rate, color='red', lw=2, label=f'Observed = {obs_rate:.3f}')
axes[0].set_xlabel("Proportion of positive flows")
axes[0].set_title("Predicted vs Observed Active Corridors")
axes[0].legend()

dyade_obs_rate = df_hurdle.groupby('dyad')['is_migration'].mean().values
dyade_pred_rate = np.array([
    is_mig_hat[:, df_hurdle['dyad'] == d].mean()
    for d in dyades_uniques if d in df_hurdle['dyad'].values
])
axes[1].scatter(dyade_obs_rate, dyade_pred_rate, alpha=0.6, s=40, color='#1565C0')
axes[1].plot([0, 1], [0, 1], 'r--', lw=1.5)
axes[1].set_xlabel("Observed Proportion (by dyad)")
axes[1].set_ylabel("Predicted Proportion (median)")
axes[1].set_title("Dyad-level Calibration")

plt.tight_layout()
plt.show()

# PPC Step 2: Volume (Log-Normal)
log_flow_hat = idata.posterior_predictive['log_flow_hat'].values.reshape(-1, N_vol)
log_flow_obs = np.array(stan_data['log_flow'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(log_flow_obs, bins=40, color='black', alpha=0.6, density=True, label='Observed', zorder=3)
for i in range(min(150, log_flow_hat.shape[0])):
    axes[0].hist(log_flow_hat[i], bins=40, alpha=0.02, density=True, color='#2196F3')
axes[0].set_xlabel("log(flow) | flow > 0")
axes[0].set_title("Predicted vs Observed Volume Distribution")
axes[0].legend()

y_pred_med = np.median(log_flow_hat, axis=0)
axes[1].scatter(log_flow_obs, y_pred_med, alpha=0.4, s=15, color='#1565C0')
lim2 = [log_flow_obs.min(), log_flow_obs.max()]
axes[1].plot(lim2, lim2, 'r--', lw=1.5, label='Perfect Prediction')

resid = log_flow_obs - y_pred_med
mae_v = np.mean(np.abs(resid))
r2_v = 1 - np.sum(resid**2) / np.sum((log_flow_obs - log_flow_obs.mean())**2)

axes[1].text(0.05, 0.95, f"MAE(log) = {mae_v:.3f}\nR²       = {r2_v:.3f}",
             transform=axes[1].transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
axes[1].set_xlabel("Observed log(flow)")
axes[1].set_ylabel("Predicted log(flow) (median)")
axes[1].set_title("Observed vs Predicted (Volume Only)")
axes[1].legend()

plt.tight_layout()
plt.show()
