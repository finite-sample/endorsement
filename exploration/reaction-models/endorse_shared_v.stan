// Same model as endorse_shared.stan, with the likelihood vectorized by
// (item, endorser, income) cell: rows must be sorted so each cell is a
// contiguous block [start[g], end[g]] sharing one scale and one cutpoint vector.
// Endorsed latent index: beta_j * (x_i + lambda[k, c] + shared * eta_i + u_ik), u ~ N(0, omega_u[k]^2),
// with u integrated out, which rescales the probit for that cell.
data {
  int<lower=1> N; int<lower=1> P; int<lower=1> J; int<lower=1> K; int<lower=1> C; int<lower=1> G;
  array[N] int<lower=1, upper=P> person;
  array[N] int<lower=1, upper=5> y;
  array[G] int<lower=1, upper=N> start;
  array[G] int<lower=1, upper=N> end;
  array[G] int<lower=1, upper=J> cell_item;
  array[G] int<lower=0, upper=K> cell_group;
  array[G] int<lower=1, upper=C> cell_inc;
  array[P] int<lower=1, upper=C> inc;
  int<lower=1, upper=C> ref;
  int<lower=0, upper=1> shared;
  int<lower=0, upper=1> by_inc;
  int<lower=0, upper=1> free_scale;  // treated-item latent SD is a free s_k instead of sqrt(1 + beta^2 omega_u^2)
  int<lower=0, upper=1> additive;    // reaction enters the response unscaled: rho * beta_j * x + lambda + eta, not beta_j * (x + lambda + eta)
  int<lower=0, upper=1> free_loading;  // reaction loads on item j with free gamma_j (eta then has SD 1); nests BIS and additive
}
parameters {
  vector<lower=0>[J] beta;
  array[J] ordered[4] tau;
  vector[C - 1] mu_free;
  vector<lower=0>[C - 1] sd_free;
  matrix[K, by_inc ? C : 1] lambda;
  vector<lower=0>[(shared && !free_loading) ? (by_inc ? C : 1) : 0] omega_eta;
  vector<lower=0>[free_loading ? J : 0] gamma;
  vector<lower=0>[free_scale ? 0 : K] omega_u;
  vector<lower=0>[free_scale ? K : 0] s_k;
  vector<lower=0>[additive] rho;  // treated-arm loading on the ideal point relative to control
  vector[P] z_x;
  vector[shared ? P : 0] z_eta;
}
transformed parameters {
  vector[C] mu_x; vector[C] sd_x;
  {
    int f = 1;
    for (c in 1:C) {
      if (c == ref) { mu_x[c] = 0; sd_x[c] = 1; }
      else { mu_x[c] = by_inc ? mu_free[f] : 0; sd_x[c] = by_inc ? sd_free[f] : 1; f += 1; }
    }
  }
}
model {
  vector[P] x = mu_x[inc] + sd_x[inc] .* z_x;
  vector[P] eta;
  if (free_loading) eta = z_eta;
  else if (shared) {
    if (by_inc) eta = omega_eta[inc] .* z_eta; else eta = omega_eta[1] * z_eta;
  } else eta = rep_vector(0, P);
  beta ~ normal(0, 2);
  for (j in 1:J) tau[j] ~ normal(0, 5);
  mu_free ~ normal(0, 1);
  sd_free ~ lognormal(0, 0.5);
  to_vector(lambda) ~ normal(0, 1);
  omega_eta ~ normal(0, 1);
  omega_u ~ normal(0, 1);
  s_k ~ lognormal(0, 0.5);
  rho ~ lognormal(0, 0.5);
  gamma ~ lognormal(0, 1);
  z_x ~ std_normal();
  z_eta ~ std_normal();
  for (g in 1:G) {
    int j = cell_item[g]; int k = cell_group[g];
    if (k == 0) {
      y[start[g]:end[g]] ~ ordered_probit(beta[j] * x[person[start[g]:end[g]]], tau[j]);
    } else {
      int c = by_inc ? cell_inc[g] : 1;
      real s = free_scale ? s_k[k] : sqrt(1 + square(beta[j] * omega_u[k]));
      vector[end[g] - start[g] + 1] xp = x[person[start[g]:end[g]]];
      vector[end[g] - start[g] + 1] ep = eta[person[start[g]:end[g]]];
      if (additive)
        y[start[g]:end[g]] ~ ordered_probit((rho[1] * beta[j] * xp + (free_loading ? gamma[j] : 1) * (lambda[k, c] + ep)) / s, tau[j] / s);
      else
        y[start[g]:end[g]] ~ ordered_probit(beta[j] * (xp + lambda[k, c] + ep) / s, tau[j] / s);
    }
  }
}
