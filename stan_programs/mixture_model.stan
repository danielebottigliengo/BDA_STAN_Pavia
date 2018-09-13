// Mixture model

data{

  int<lower = 1> K;
  int<lower = 1> N;
  real y[N];

}

parameters {

  simplex[K] theta;
  real mu[K];
  real<lower = 0> sigma[K];

}

model {

  real ps[K];



}
