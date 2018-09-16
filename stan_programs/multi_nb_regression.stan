// Multiple NB regression

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

  int<lower = 1> N;             // number of observations
  vector[N] log_sq_foot;        // log square foot of the building
  vector<lower = 0, upper = 1>[N] live_in_super; // is the bulding in super?
  vector<lower = 0>[N] traps;     // number of traps in the building
  int<lower = 0> complaints[N];   // number of complaints in the building

}

parameters {

  real alpha;                 // intercept
  real beta;                  // coefficients on the traps
  real beta_super;            // coefficients on live in super
  real<lower = 0> inv_phi;    // inverse of phi coefficients

}

transformed parameters {

  real phi = inv(inv_phi);    // phi coefficient

}

model {

  // Linear predictor
  vector[N] eta = alpha +
                  beta * traps +
                  beta_super * live_in_super +
                  log_sq_foot;

  // Priors
  target += normal_lpdf(alpha | log(4), 1) +
            normal_lpdf(beta | -0.25, 1) +
            normal_lpdf(beta_super | -0.5, 1) +
            normal_lpdf(inv_phi | 0, 1);

  // Likelihood
  target += neg_binomial_2_log_lpmf(complaints| eta, phi);

}

generated quantities {

  // Declare simulated data from the model
  int y_rep[N];

  for(n in 1:N) {

    real eta_rep = alpha +
                   beta * traps[n] +
                   beta_super * live_in_super[n] +
                   log_sq_foot[n];

    y_rep[n] = neg_binomial_2_log_safe_rng(eta_rep, phi);

  }
}


