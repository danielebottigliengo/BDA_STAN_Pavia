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
  vector<lower = 0>[N] n_traps;
  vector<lower = 0, upper = 1>[N] live_in_super;
  vector[N] log_sq_foot;
  int<lower=0> complaints[N];
}

parameters {
  real alpha;
  real beta;
  real beta_super;
  real<lower = 0> inv_phi;
}

transformed parameters {
  real phi = inv(inv_phi);
}

model {

  vector[N] eta = alpha + beta * n_traps + beta_super * live_in_super +
                  log_sq_foot;

  target += neg_binomial_2_log_lpmf(complaints | eta, phi);

  target += normal_lpdf(alpha | log(4), 1) +
            normal_lpdf(beta | -0.25, 1) +
            normal_lpdf(beta_super | -0.5, 1) +
            normal_lpdf(inv_phi | 0, 1);
}

generated quantities {
  int y_rep[N];
  for (n in 1:N)
    y_rep[n] = neg_binomial_2_log_safe_rng(
      alpha +
      beta * n_traps[n] +
      beta_super * live_in_super[n] +
      log_sq_foot[n],
      phi
    );

}
