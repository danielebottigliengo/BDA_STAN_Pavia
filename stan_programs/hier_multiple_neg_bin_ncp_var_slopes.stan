// Hierarchical NB model with varying intercepts and slopes

functions {
  /*
  * Alternative to neg_binomial_2_log_rng() that
  * avoids potential numerical problems during warmup
  */
  int neg_binomial_2_log_safe_rng(real eta, real phi) {
    real gamma_rate = gamma_rng(phi, phi / exp(eta));
    if (gamma_rate >= exp(20.79))
      return -9;

    return poisson_rng(gamma_rate);
  }
}

data {

  int<lower=1> N;                     // number of observations
  int<lower=0> complaints[N];         // number of complaints
  vector<lower=0>[N] traps;           // number of traps
  vector[N] log_sq_foot;              // vector of exposure (offset)

  // building-level data
  int<lower=1> K;                       // number of building-level covs
  int<lower=1> J;                       // number of building
  int<lower=1, upper=J> building_idx[N];// id of the building
  matrix[J,K] building_data;            // building-level matrix

}

parameters {

  real<lower=0> inv_phi;     // inverse of the parameter phi

  // Non centered parameters for varying intercepts
  vector[J] mu_raw;        // auxiliary parameter
  real<lower=0> sigma_mu;  // sd of buildings-specific intercepts
  real alpha;              // 'global' intercept for buildings
  vector[K] zeta;          // coefficients on building-level predictors

  // Non centered parameters for varying slopes
  vector[J] kappa_raw;       // auxiliary parameter
  real<lower=0> sigma_kappa; // sd of buildings-specific slopes
  real beta;                 // 'global' slope on traps variable
  vector[K] gamma;           // coefficients on building-level predictors

}

transformed parameters {

  real phi = inv(inv_phi);  // original parameter phi

  // Original parameters mu and kappa
  vector[J] mu = alpha +
                 building_data * zeta +
                 sigma_mu * mu_raw;

  vector[J] kappa = beta +
                    building_data * gamma +
                    sigma_kappa * kappa_raw;

}

model {

  // Declare linear predictor Linear predictor
  vector[N] eta = mu[building_idx] +
                  kappa[building_idx] .* traps +
                  log_sq_foot;

  // Prior on inv_phi
  target += normal_lpdf(inv_phi | 0, 1) +

  // Prior on varying slopes parameters
            normal_lpdf(kappa_raw | 0, 1) +
            normal_lpdf(sigma_kappa | 0, 1) +
            normal_lpdf(beta | -0.25, 1) +
            normal_lpdf(gamma | 0, 1) +

  // Prior on varying intercepts parameters
            normal_lpdf(mu_raw | 0, 1) +
            normal_lpdf(sigma_mu | 0, 1) +
            normal_lpdf(alpha | log(4), 1) +
            normal_lpdf(zeta | 0, 1);

  // Likelihood
  target += neg_binomial_2_log_lpmf(complaints | eta, phi);

  // The symbol ".*" is element-wise multiplication to multiply
  // the slope of each building for the number of traps of that building

}

generated quantities {

  // Declare replicated data
  int y_rep[N];

  for (n in 1:N) {
    real eta_n = mu[building_idx[n]] +
                 kappa[building_idx[n]] * traps[n] +
                 log_sq_foot[n];

    y_rep[n] = neg_binomial_2_log_safe_rng(eta_n, phi);
  }

}

