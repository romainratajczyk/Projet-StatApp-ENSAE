data {
  int<lower=1> N;           
  int<lower=1> D;           
  array[N] int dyad_id;     
  vector[N] log_y;          
  vector[N] log_y_lag;      
}

parameters {
  // Hyperparamètres
  real mu_global;
  real<lower=1e-6> sigma_mu; // FIX : On interdit le 0 strict
  
  // Paramètres par dyade
  vector[D] mu_dyad_raw;               // FIX : La variable "brute" (Non-Centered Parameterization)
  vector<lower=-1, upper=1>[D] phi;    
  vector<lower=1e-6>[D] sigma_dyad;    // FIX : On interdit le 0 strict pour les couloirs vides
}

transformed parameters {
  vector[D] mu_dyad;
  // FIX : "The Matt Trick" - Transformation déterministe pour détruire l'entonnoir géométrique
  mu_dyad = mu_global + mu_dyad_raw * sigma_mu;
}

model {
  // 1. Priors
  mu_global ~ normal(0, 5);
  sigma_mu ~ exponential(1);
  
  // Le Prior de la variable brute est une Loi Normale Standard (0, 1)
  mu_dyad_raw ~ std_normal();
  
  phi ~ uniform(-1, 1);
  sigma_dyad ~ exponential(1);
  
  // 2. Vraisemblance (Likelihood)
  for (n in 1:N) {
    log_y[n] ~ normal(mu_dyad[dyad_id[n]] + phi[dyad_id[n]] * log_y_lag[n], sigma_dyad[dyad_id[n]]);
  }
}



// exemples jags / exemples dans gelman