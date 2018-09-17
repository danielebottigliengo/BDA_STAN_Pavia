// Logistic regression Data Generating Process

data{

  int<lower = 1> N;                     // number of observations
  int<lower = 1> K;                     // number of coefficients
  vector[K] beta; // coefficients

}

parameters{

}

model {

}

generated quantities {

  // Declare simulated variables
  vector[N] x1;
  vector[N] x2;
  int<lower = 0, upper = 1> x3[N];
  int<lower = 0, upper = 1> y[N];

  // Generate parameters value from the prior predictive distribution
  // (try a t_student)
  real alpha = student_t_rng(7, beta[1], 0.1);
  real beta_x1 = student_t_rng(7, beta[2], 0.1);
  real beta_x2 = student_t_rng(7, beta[3], 0.1);
  real beta_x3 = student_t_rng(7, beta[4], 0.1);
  real beta_x1_x3 = student_t_rng(7, beta[5], 0.1);

  // Declare linear predictor
  real eta;

  // Generate simulated values of the outcome
  for (n in 1:N) {

    // Generate covariates
    x1[n] = normal_rng(0, 1);
    x2[n] = normal_rng(0, 1);
    x3[n] = bernoulli_rng(0.4);

    // Linear predictor
    eta = alpha +
          beta_x1 * x1[n] +
          beta_x2 * x2[n] +
          beta_x3 * x3[n] +
          beta_x1_x3 * x1[n] * x3[n];

    y[n] = bernoulli_logit_rng(eta);

  }

}
