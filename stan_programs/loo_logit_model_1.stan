// Logistic regression full model

data{

  int<lower = 1> N;                    // number of observations
  int<lower =1> K;                     // number of coefficients
  vector[K] beta;                      // vector of coefficients
  vector[N] x1;                        // continuous predictor
  vector[N] x2;                        // continuous predictor
  int<lower = 0, upper = 1> y[N];      // outcome

}

parameters {

  real alpha;
  real beta_x1;
  real beta_x2;

}

model {

  // Linear predictor
  vector[N] eta;

  // Prios
  target += student_t_lpdf(alpha | 7, beta[1], 1) +
            student_t_lpdf(beta_x1 | 7, beta[2], 1) +
            student_t_lpdf(beta_x2 | 7, beta[3], 1);

  // Linear predictor
  eta = alpha +
        beta_x1 * x1 +
        beta_x2 * x2;

  // Likelihood
  target += bernoulli_logit_lpmf(y | eta);

}

generated quantities{

  vector[N] y_rep;       // replicated data
  vector[N] log_lik;     // pointwise log likelihood for LOO-CV

  for(n in 1:N) {

    real eta_rep = alpha +
                   beta_x1 * x1[n] +
                   beta_x2 * x2[n];

    y_rep[n] = bernoulli_logit_rng(eta_rep);

    log_lik[n] = bernoulli_logit_lpmf(y[n] | eta_rep);

  }


}
