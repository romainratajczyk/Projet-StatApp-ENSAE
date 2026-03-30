// Point1: but de ce code: dans HMC_ARX_v4 la variance est calculée comme sigma_d[d] = fmax(sigma_cluster[...] * exp(...), 1e-4). Ne dépend pas de la magnitude du flux.
// on pense que c'est pour cela que la distribution des erreurs est positivement corrélée à la magnitude du flux pour les flux > 15 000. 

// Point 2: Aussi, le but premier est de remplacer la loi log-normale par une loi Negative Binomiale, qui gère nativement la nature discrète des micro-flux et converge asymptotiqueemnt vers une allure gaussienne qui ressemble à la log-normale précédente pour les macro-flux. 
// l'hétéroscédasticité est aussi native dans la loi NegBin! le point 1 est donc redondant. 

// Point3: Enfin, traiter de manière hiérarchique (comme les autres paramètres) le coefficient beta_lag du Hurdle. 


/*
Bayesian Hierarchical ARX Hurdle Model for Gravity Migration

Component A: Hurdle (Hierarchical Bernoulli)
Component B: Volume (Hierarchical ARX Zero-Truncated Negative Binomial)
Component C: Geographic Heteroscedasticity (Dispersion clustering)
*/

data {
  // 1. Hurdle
  int<lower=1> N_h;
  int<lower=1> D_h;
  int<lower=1> K_h;
  array[N_h] int<lower=1, upper=D_h> dyad_id_h;
  array[N_h] int<lower=0, upper=1>   is_mig;
  vector[N_h]                        is_mig_lag;
  matrix[N_h, K_h]                   X_h;

  // 2. Volume (ZTNB exige des données de comptage brutes)
  int<lower=1> N_v;
  int<lower=1> D_v;
  int<lower=1> K_v;
  array[N_v] int<lower=1, upper=D_v> dyad_id_v;
  array[N_v] int<lower=1>            flow;              // Entiers stricts > 0
  vector[N_v]                        log_flow_lag;      // Maintenu continu pour le processus AR(1)
  matrix[N_v, K_v]                   X_v;

  // 3. Clusters géographiques (M49)
  int<lower=1>                       K_clusters;
  array[D_h] int<lower=1, upper=K_clusters> cluster_h;
  array[D_v] int<lower=1, upper=K_clusters> cluster_v;

  // 4. Test OOS
  int<lower=0>                       N_test;
  array[N_test] int<lower=1, upper=D_h>     dyad_id_test_h;
  array[N_test] int<lower=0, upper=D_v>     dyad_id_test_v;
  matrix[N_test, K_h]                       X_h_test;
  vector[N_test]                            is_mig_lag_test;
  matrix[N_test, K_v]                       X_v_test;
  vector[N_test]                            log_flow_lag_test;
  array[N_test] int<lower=1, upper=K_clusters> cluster_test_h;

  // 5. Flags
  int<lower=0, upper=1> do_ppc;
}

parameters {
  // A. Hurdle
  real alpha_global;
  real<lower=0> tau_alpha;
  vector[K_h] beta_h;
  real mu_beta_lag;                     // Hyperparamètre espérance beta_lag
  real<lower=0> sigma_beta_lag;         // Hyperparamètre variance beta_lag
  vector[K_clusters] beta_lag_m49;      // Effet inertiel hiérarchisé par M49
  vector[D_h] alpha_raw;

  // B. Volume ARX (Rho remplace Phi pour l'inertie)
  real mu_intercept;
  real<lower=0> tau_mu;
  vector[K_v] beta_grav;
  real rho_global_raw;
  real<lower=0> tau_rho;
  vector[D_v] rho_raw;
  vector[D_v] mu_raw;

  // C. Dispersion (Phi remplace Sigma)
  real<lower=0> phi_disp_global;
  vector<lower=0>[K_clusters] phi_disp_cluster;
  vector[D_v] phi_disp_raw;
  real<lower=0> tau_phi_disp;           // Prior Half-Cauchy appliqué ici
}

transformed parameters {
  // A. Prédicteurs Hurdle
  vector[D_h] alpha_d;
  for (d in 1:D_h)
    alpha_d[d] = alpha_global + tau_alpha * alpha_raw[d];

  // B. Prédicteurs Volume ARX
  real rho_global = tanh(rho_global_raw);
  vector[D_v] alpha_V;
  vector[D_v] rho_d;
  for (d in 1:D_v) {
    alpha_V[d] = mu_intercept + tau_mu * mu_raw[d];
    rho_d[d]   = tanh(rho_global_raw + tau_rho * rho_raw[d]);
  }

  // C. Dispersion Hiérarchique (Hétéroscédasticité native)
  // phi_disp_d : paramètre de dispersion inverse de la NegBin
  vector<lower=0>[D_v] phi_disp_d;
  for (d in 1:D_v) {
    phi_disp_d[d] = fmax(phi_disp_cluster[cluster_v[d]] * exp(tau_phi_disp * phi_disp_raw[d]), 1e-4);
  }
  // Note : L'inflation de la variance AR(1) est mathématiquement supprimée. La NegBin gère la variance via E[Y] + E[Y]²/phi

  // Construction vectorisée des prédicteurs
  vector[N_h] lag_effect;
  for (n in 1:N_h) {
    lag_effect[n] = beta_lag_m49[cluster_h[dyad_id_h[n]]] * is_mig_lag[n];
  }
  vector[N_h] logit_p = alpha_d[dyad_id_h] + X_h * beta_h + lag_effect;

  vector[N_v] mu_dt = alpha_V[dyad_id_v] + X_v * beta_grav;

  vector[N_v] ar_pred; // Espérance log-linéaire (eta) de la NegBin
  for (n in 1:N_v) {
    int d = dyad_id_v[n];
    ar_pred[n] = mu_dt[n] + rho_d[d] * (log_flow_lag[n] - mu_dt[n]);
  }
}

model {
  // A. Priors Hurdle
  alpha_global   ~ normal(0.5, 2);
  tau_alpha      ~ exponential(1);
  beta_h[1]      ~ normal(-1, 1);
  beta_h[2]      ~ normal(2, 1);
  beta_h[3]      ~ normal(0.5, 1);
  beta_h[4]      ~ normal(1, 1);
  beta_h[5]      ~ normal(1, 1);
  beta_h[6]      ~ normal(0.5, 1);
  beta_h[7]      ~ normal(0.5, 1);
  beta_h[8]      ~ normal(0.5, 1);
  
  // Hiérarchisation stricte de beta_lag_m49
  mu_beta_lag    ~ normal(3.0, 2.0);
  sigma_beta_lag ~ exponential(1);
  beta_lag_m49   ~ normal(mu_beta_lag, sigma_beta_lag);
  alpha_raw      ~ std_normal();

  // B. Priors Volume
  mu_intercept   ~ normal(0, 2);
  tau_mu         ~ exponential(1);
  beta_grav      ~ normal(0, 1);
  rho_global_raw ~ normal(0.5, 0.5);
  tau_rho        ~ exponential(2);
  rho_raw        ~ std_normal();
  mu_raw         ~ std_normal();

  // C. Priors Dispersion
  phi_disp_global ~ exponential(1);
  for (k in 1:K_clusters)
    phi_disp_cluster[k] ~ normal(phi_disp_global, 0.5);
    
  tau_phi_disp ~ cauchy(0, 0.5); // Prior Half-Cauchy tronqué (lower=0 dans la déclaration)
  phi_disp_raw ~ std_normal();

  // Vraisemblance Hurdle
  is_mig ~ bernoulli_logit(logit_p);

  // Vraisemblance Volume (Zero-Truncated Negative Binomial)
  for (n in 1:N_v) {
    int d = dyad_id_v[n];
    // 1. Log-vraisemblance standard
    target += neg_binomial_2_log_lpmf(flow[n] | ar_pred[n], phi_disp_d[d]);
    // 2. Troncature stricte à zéro (renormalisation)
    target += -log1m_exp(neg_binomial_2_log_lpmf(0 | ar_pred[n], phi_disp_d[d]));
  }
}

generated quantities {
  vector[N_h] log_lik_h;
  vector[N_v] log_lik_v;

  for (n in 1:N_h)
    log_lik_h[n] = bernoulli_logit_lpmf(is_mig[n] | logit_p[n]);

  for (n in 1:N_v) {
    int d = dyad_id_v[n];
    log_lik_v[n] = neg_binomial_2_log_lpmf(flow[n] | ar_pred[n], phi_disp_d[d]) 
                   - log1m_exp(neg_binomial_2_log_lpmf(0 | ar_pred[n], phi_disp_d[d]));
  }

  array[do_ppc ? N_h : 0] int is_mig_hat;
  if (do_ppc) {
    for (n in 1:N_h)
      is_mig_hat[n] = bernoulli_logit_rng(logit_p[n]);
  }

  // Précalcul OOS
  vector[K_clusters] mu_cluster_k;
  {
    vector[K_clusters] n_cluster_k = rep_vector(0.0, K_clusters);
    mu_cluster_k = rep_vector(0.0, K_clusters);
    for (d in 1:D_v) {
      mu_cluster_k[cluster_v[d]] += alpha_V[d];
      n_cluster_k[cluster_v[d]]  += 1.0;
    }
    for (k in 1:K_clusters) {
      if (n_cluster_k[k] > 0)
        mu_cluster_k[k] /= n_cluster_k[k];
      else
        mu_cluster_k[k] = mu_intercept;
    }
  }

  vector[N_test] prob_mig_test;
  vector[N_test] mu_dt_test;     // η : log-espérance de la NegBin
  vector[N_test] phi_test;       // Dispersion extraite pour reconstruction Python

  for (n in 1:N_test) {
    int d_h = dyad_id_test_h[n];
    int d_v = dyad_id_test_v[n];
    int k = cluster_test_h[n];

    real logit_p_test = alpha_d[d_h]
                        + dot_product(X_h_test[n], beta_h)
                        + beta_lag_m49[k] * is_mig_lag_test[n];
    prob_mig_test[n] = inv_logit(logit_p_test);

    real mu_base = mu_intercept + dot_product(X_v_test[n], beta_grav);

    if (d_v > 0) {
      real mu_full = alpha_V[d_v] + dot_product(X_v_test[n], beta_grav);
      if (is_mig_lag_test[n] > 0) {
        mu_dt_test[n] = mu_full + rho_d[d_v] * (log_flow_lag_test[n] - mu_full);
      } else { 
        mu_dt_test[n] = mu_full;
      }
      phi_test[n] = phi_disp_d[d_v];
    } else {
      mu_dt_test[n] = mu_cluster_k[k] + dot_product(X_v_test[n], beta_grav);
      phi_test[n] = phi_disp_cluster[k];
    }
  }

  real rho_global_monitor = rho_global;
}