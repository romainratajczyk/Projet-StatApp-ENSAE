// 1) DATA
data {
  int<lower=0> N;                // Nombre d'observations (lignes)
  int<lower=0> K;                // Nombre de variables explicatives (colonnes)
  matrix[N, K] X;                // La matrice contenant toutes tes variables (Pop, Dist, PIB...)
  vector[N] y;                   // La cible (log_migrantCount)
}

// 2) PARAMETERS
parameters {
  real alpha;                    // L'intercept
  vector[K] beta;                // Un vecteur contenant les 20 coeffs (un par variable)
  real<lower=0> sigma;           // L'écart-type de l'erreur
}

// 3) MODEL
model {
  // A. Priors (Croyances initiales)
  // On suppose que les effets sont raisonnables (entre -10 et +10 environ)
  beta ~ normal(0, 5);           
  alpha ~ normal(0, 10);
  sigma ~ cauchy(0, 2.5);        // Prior standard pour l'erreur

  // B. Vraisemblance (Likelihood)
  // Produit matriciel : X * beta fait tout le calcul d'un coup
  y ~ normal(alpha + X * beta, sigma);
}