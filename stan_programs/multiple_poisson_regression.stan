/* Multi Poisson model*/

/*
* Alternative to poisson_log_rng() that
* avoids potential numerical problems during warmup
*/
functions {
  int poisson_log_safe_rng(real eta) {
    real pois_rate = exp(eta);
    if (pois_rate >= exp(20.79))
      return -9;
    return poisson_rng(pois_rate);
  }
}

// Declare the structure of the data
data {
  int<lower = 1> N;                       // number of observation
  vector<lower = 0>[N] n_traps;           // number of traps
  int<lower = 0> complaints[N];           // number of complaints
  vector[N] log_sq_foot;                  // log of square foot building
  vector<lower = 0, upper = 1>[N] live_in_super; // level of the apartment
}

// Declare the parameters of the model (the goal of our inferemce)
parameters {
  // Intercept and the slope of the linear predictor eta
  real alpha;
  real beta;
  real beta_super;
}

// Define the structure of the Poisson model: priors and likelihood
model {
  // Let's create the linear predictor: eta
  vector[N] eta = alpha +
                  beta * n_traps +
                  beta_super * live_in_super +
                  log_sq_foot;

  /*Let's declare our outcome variable and its pdf. Poisson_log
  function directly exponentiated the linear prediction*/
  target += poisson_log_lpmf(complaints | eta);
  /*or equivalently

  complaints ~ poisson_log(eta);

  */

  /*Let's declare our priors distributions. Let's put some reasonable
  priors. In particular, we expect that for higher number of traps
  there will be less complaints from people living in the building.*/
  target += normal_lpdf(alpha | log(4), 1) +
            normal_lpdf(beta | -0.25, 1) +
            normal_lpdf(beta_super | -0.5, 1);

  /*or equivalently

  alpha ~ normal(log(4), 1);
  beta ~ normal(-0.25, 1);

  */
}

// Simulate replicated data from the model
generated quantities {

  int y_rep[N];

  for (n in 1:N) {

    real eta_rep = alpha +
                   beta * n_traps[n] +
                   beta_super * live_in_super[n] +
                   log_sq_foot[n];

    y_rep[n] = poisson_log_safe_rng(eta_rep);

  }

}

