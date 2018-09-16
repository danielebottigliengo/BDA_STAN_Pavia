// Multi NB data generating process

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
  int<lower = 1> N;         // Number of observations
}

parameters {
}

model {
}

generated quantities {

  // Declare simulated variables
  vector[N] log_sq_foot;
  int live_in_super[N];
  int traps[N];
  int complaints[N];

  // Generate parameter values from the prior predictive distribution
  real alpha = normal_rng(log(4), 0.1);
  real beta = normal_rng(-0.25, 0.1);
  real beta_super = normal_rng(-0.5, 0.1);
  real inv_phi = fabs(normal_rng(0, 1));

  // Generate fake data
  for(n in 1:N) {

    // Generate covariates
    log_sq_foot[n] = normal_rng(1.5, 0.1);
    live_in_super[n] = bernoulli_rng(0.5);
    traps[n] = poisson_rng(8);

    // Generate outcome
    complaints[n] = neg_binomial_2_log_safe_rng(
      alpha +
      beta * traps[n] +
      beta_super * live_in_super[n] +
      log_sq_foot[n],
      inv(inv_phi)
    );
  }
}

