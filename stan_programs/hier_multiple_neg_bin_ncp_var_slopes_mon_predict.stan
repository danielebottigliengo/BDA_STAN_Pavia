// Predictions using hierarchical NB model with varying intercepts
// and slopes and month effect

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

  // 'exposure'
  vector[N] log_sq_foot;

  // building-level data
  int<lower = 1> K;     // number of building-level predictors
  int<lower = 1> J;     // number of buildings
  int<lower = 1, upper = J> building_idx[N]; // building id
  matrix[J, K] building_data; // building-level matrix

  // month info
  int<lower = 1> M;
  int<lower = 1, upper = M> mo_idx[N];

  // To use in the generated quantities block
  int<lower = 1> M_forward;
  vector[J] log_sq_foot_pred;

  // Number of traps used to predict number of complaints
  int N_hypo_traps;
  int hypo_traps[N_hypo_traps];

  // Lost revenue for one complaint
  real lost_rev;

}

parameters {

  real<lower = 0> inv_phi;   // inverse of phi

  // Varying intercept for the buildings
  vector[J] mu_raw;        // auxiliary parameter
  real<lower = 0> sigma_mu;// sd of buildings-specific intercepts
  real alpha;              // 'global' intercept
  vector[K] zeta;          // coefficients on building-level predictors

  // Varying slopes for the buildings
  vector[J] kappa_raw;       // auxiliary parameter
  real<lower = 0> sigma_kappa; // sd of buildings-specific slopes
  real beta;                 // 'global' slope on traps variable
  vector[K] gamma;           // coefficients on building-level predictors

  // month-specific parameters
  real<lower = 0,upper = 1> rho_raw;  // used to construct rho
  vector[M] mo_raw;
  real<lower = 0> sigma_mo;

}

transformed parameters {

  real phi = inv(inv_phi);    // original phi

  // non-centered parameterization of building-specific
  vector[J] mu = alpha +
                 building_data * zeta +
                 sigma_mu * mu_raw;

  vector[J] kappa = beta +
                    building_data * gamma +
                    sigma_kappa * kappa_raw;

  // AR(1) process priors
  real rho = 2.0 * rho_raw - 1.0;
  vector[M] mo = sigma_mo * mo_raw;
  mo[1] /= sqrt(1 - rho^2);   //   mo[1] = mo[1]/sqrt(1-rho^2)

  // add in the dependence on previous month

  for(m in 2:M) {

    mo[m] += rho * mo[m - 1];

  }
}

model {

  // Likelihood
  vector[N] eta = mu[building_idx] +
                  kappa[building_idx] .* traps +
                  mo[mo_idx] +
                  log_sq_foot;

  target += neg_binomial_2_log_lpmf(complaints | eta, phi);

  // Priors
  target += normal_lpdf(inv_phi | 0, 1) +
            // Priors on non-centered slopes
            normal_lpdf(kappa_raw | 0, 1) +
            normal_lpdf(sigma_kappa | 0, 1) +
            normal_lpdf(beta | -0.25, 1) +
            normal_lpdf(gamma | 0, 1) +
            // Priors on non-centered intercepts
            normal_lpdf(mu_raw | 0, 1) +
            normal_lpdf(sigma_mu | 0, 1) +
            normal_lpdf(alpha | log(4), 1) +
            normal_lpdf(zeta | 0, 1) +
            // Priors on non-centered months
            beta_lpdf(rho_raw | 10, 5) +
            normal_lpdf(mo_raw | 0, 1) +
            normal_lpdf(sigma_mo | 0, 1);

}

generated quantities {

  /*  we'll predict number of complaints and revenue lost for each
  building at each hypothetical number of traps for M_forward months in
  the future*/

  int y_pred[J, N_hypo_traps];
  matrix[J, N_hypo_traps] rev_pred;

  for (j in 1:J) {  // loop over buildings

    for (i in 1:N_hypo_traps) {  // loop over hypothetical traps

      int y_pred_by_month[M_forward]; // monthly predictions
      vector[M_forward] mo_forward;   // number of month forward

      // first future month depends on last observed month
      mo_forward[1] = normal_rng(rho * mo[M], sigma_mo);

      for (m in 2:M_forward) {

        mo_forward[m] = normal_rng(rho * mo_forward[m-1], sigma_mo);

      }

      for (m in 1:M_forward) {
        real eta = mu[j] +
                   kappa[j] * hypo_traps[i] +
                   mo_forward[m] +
                   log_sq_foot_pred[j];

        y_pred_by_month[m] = neg_binomial_2_log_safe_rng(eta, phi);

      }

      // Sum the number of complaints by month for each number of
      // traps in each building
      y_pred[j, i] = sum(y_pred_by_month);

      /* We  were were told every 10 complaints has additional
      exterminator cost of $100, so $10 lose per complaint.*/
      rev_pred[j,i] = y_pred[j,i] * (-lost_rev);
    }
  }
}



