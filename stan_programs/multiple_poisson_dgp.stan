/* Generating data from a multiple Poisson regression model*/

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

data {
  // Number of observations
  int<lower=1> N;
}

model{

}

// Predictive posterior distributions in generated quantities
generated quantities {

  // Declare simulated data
  vector[N] log_sq_foot;
  int live_in_super[N];
  int n_traps[N];
  int complaints[N];

  // Generate parameters values from the prior predictive distribution
  real alpha = normal_rng(log(4), 1);
  real beta = normal_rng(-0.25, 1);
  real beta_super = normal_rng(-0.5, 1);

  // Generate simulated values of the outcome (number of complaints)
  for(n in 1:N) {

    // Generate fake data
    log_sq_foot[n] = normal_rng(1.5, 1);
    live_in_super[n] = bernoulli_rng(0.5);
    n_traps[n] = poisson_rng(8);

    // Generate simulated number of complaints
    complaints[n] = poisson_log_safe_rng(alpha +
                    log_sq_foot[n] + beta * n_traps[n] +
                    beta_super * live_in_super[n]);
  }
}


