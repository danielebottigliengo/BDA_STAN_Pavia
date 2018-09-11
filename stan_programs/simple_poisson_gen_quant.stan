/* Simple Poisson model*/

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
  int<lower = 1> N;                     // number of observation
  vector<lower = 0>[N] n_traps;         // number of traps
  int<lower = 0> complaints[N];         // number of complaints
}

// Declare the parameters of the model (the goal of our inferemce)
parameters {
  // Intercept and the slope of the linear predictor eta
  real alpha;
  real beta;
}

// Define the structure of the Poisson model: priors and likelihood
model {
  // Let's create the linear predictor: eta
  vector[N] eta = alpha + beta * n_traps;

  // likelihood
  target += poisson_log_lpmf(complaints | eta);

  // Priors on the parameters
  target += normal_lpdf(alpha | log(4), 1) +
            normal_lpdf(beta | -0.25, 1);
}

// Predictive posterior distributions in generated quantities
generated quantities {

  int y_rep[N];       // replicated y as array with vector of integers

  // Simulate data from the fitted modedl
  for (n in 1:N) {

    real eta_n = alpha + beta * n_traps[n];   // linear predictor
    y_rep[n] = poisson_log_safe_rng(eta_n);   // replicated complaints

  }
}

