// Hierarchical multiple Negative Binomial model

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
  int<lower = 1> N;
  int<lower = 0> complaints[N];
  vector<lower = 0>[N] traps;
  vector[N] log_sq_foot;

  // Declare the hierarchical part
  int<lower = 1> J;             // number of building
  int<lower = 1> K;             // number of building variables
  matrix[J, K] building_data;   // matrix of building data
  int<lower = 1, upper = J> building_idx[N];  // id of the building
}

parameters {

  real<lower = 0> inv_phi;  // inverse overdispersion's parameter
  real beta;                // coefficient on traps
  real alpha;               // intercept on the building
  vector[J] mu_star;        // parameter of the varying intercept
  real<lower = 0> sigma_mu; // sd of the varying intercept
  vector[K] zeta;           // vector of coefficient of building vars
}

transformed parameters {

  // get the original phi
  real phi = inv(inv_phi);
  // get the original parameter of the varying intercept
  vector[J] mu = alpha + building_data * zeta + sigma_mu * mu_star;
}

model {

  /*If you define the linear predictor outside of the likelihood,
  you must specify it as the first object of the model block.*/

  // Likelihood
  vector[N] eta = mu[building_idx] +   // Loop over buildings
                  beta * traps +
                  log_sq_foot;

  // complaints ~ neg_binomial_2_log(eta, phi);
  target += neg_binomial_2_log_lpmf(complaints | eta, phi);

  // Priors on alpha, beta and inv_phi
  // alpha ~ normal(log(4), 1);
  // beta ~ normal(-0.25, 1);
  // inv_phi ~ normal(0, 1);

  target += normal_lpdf(alpha| log(4), 1) +
            normal_lpdf(beta | -0.25, 1) +
            normal_lpdf(inv_phi | 0, 1);

  // Priors on the coefficients of buildings
  // zeta ~ normal(0, 1);
  // sigma_mu ~ normal(0, 1);

  target += normal_lpdf(zeta | 0, 1) +
            normal_lpdf(mu_star| 0, 1) +
            normal_lpdf(sigma_mu | 0, 1);
}

generated quantities {

  int y_rep[N];

  for (n in 1:N) {

    // Define linear predictor into the loop as a temporary real number
    real eta_rep = mu[building_idx[n]] +
                   beta * traps[n] +
                   log_sq_foot[n];

     y_rep[n] = neg_binomial_2_log_safe_rng(eta_rep, phi);

  }

}
