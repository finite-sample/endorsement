// Latent-class version of the shared-reaction model: every treated respondent is an
// opponent (shifts all four answers down by delta_neg), indifferent (no shift) or a
// supporter (up by delta_pos); class shares pi are estimated, not assumed.
// Control rows:  y* = beta_j x_i + e,                      e ~ N(0, 1)
// Treated rows:  y* = rho beta_j x_i + shift_i + e,        e ~ N(0, s_T^2)
// The shift is additive on the response scale (same for every item), which is what the
// treated-arm covariance structure supports. Control persons' rows come first, then treated
// persons' rows grouped by person; mixed-arm respondents are dropped upstream.
data {
  int<lower=1> N; int<lower=1> P; int<lower=1> J;
  int<lower=0, upper=N> N_c;            // control rows come first (rows 1..N_c), then treated persons' rows
  int<lower=0, upper=P> P_t;            // number of treated persons
  array[N] int<lower=1, upper=P> person;
  array[N] int<lower=1, upper=J> item;
  array[N] int<lower=1, upper=5> y;
  array[P_t] int<lower=1, upper=N> t_start;  // row range of each treated person
  array[P_t] int<lower=1, upper=N> t_end;
  int<lower=0, upper=1> symmetric;  // delta_pos = delta_neg: the paper's symmetric-effects case with d estimated
  int<lower=0, upper=1> flex_x;     // ideal points ~ (1-w) N(0,1) + w N(m_bump, s_bump), shared by both arms
  int<lower=0, upper=1> top_class;  // a share w_top answers the top category on every item in either arm
  int<lower=0, upper=P> P_c;
  array[P_c] int<lower=1, upper=N> c_start;
  array[P_c] int<lower=1, upper=N> c_end;
  array[P] int<lower=0, upper=1> all_top;  // indexed by person, controls first then treated
}
parameters {
  vector<lower=0>[J] beta;
  array[J] ordered[4] tau;
  vector[P] z_x;
  vector<lower=0, upper=1>[flex_x] w_bump;
  vector<lower=0, upper=1>[top_class] w_top;
  vector[flex_x] m_bump;
  vector<lower=0>[flex_x] s_bump;
  simplex[3] pi;              // opponent, indifferent, supporter
  // A shift above 2 noise-SDs sends a respondent past a full category on every item; the only
  // posterior mode beyond that is the degenerate one that relabels the ceiling cluster as supporters.
  real<lower=0, upper=2> delta_neg;
  vector<lower=0, upper=2>[symmetric ? 0 : 1] delta_pos_free;
  real<lower=0> rho;
  real<lower=0> s_T;
}
transformed parameters {
  real delta_pos = symmetric ? delta_neg : delta_pos_free[1];
}
model {
  vector[N] lo; vector[N] hi; vector[N] bx;
  vector[3] shift = [-delta_neg, 0, delta_pos]';
  beta ~ normal(0, 2);
  for (j in 1:J) tau[j] ~ normal(0, 5);
  if (flex_x) {
    // the bump lets the control arm explain its own ceiling cluster, so a treatment class cannot
    w_bump ~ beta(2, 8);
    m_bump ~ normal(2, 1);
    s_bump ~ lognormal(-1, 0.5);
    for (p in 1:P) target += log_mix(w_bump[1], normal_lpdf(z_x[p] | m_bump[1], s_bump[1]), normal_lpdf(z_x[p] | 0, 1));
  } else z_x ~ std_normal();
  pi ~ dirichlet(rep_vector(2, 3));
  delta_neg ~ normal(0, 1);
  delta_pos_free ~ normal(0, 1);
  rho ~ lognormal(0, 0.5);
  s_T ~ lognormal(0, 0.5);
  w_top ~ beta(2, 8);
  for (n in 1:N) {
    // +-50 stand in for the open-ended outer categories; Phi(50) = 1 and Phi(-50) = 0 exactly
    lo[n] = y[n] == 1 ? -50 : tau[item[n]][y[n] - 1];
    hi[n] = y[n] == 5 ? 50 : tau[item[n]][y[n]];
    bx[n] = beta[item[n]] * z_x[person[n]];
  }
  {
    // control rows: P(y) = Phi(hi - mu) - Phi(lo - mu). Computing the difference both ways and
    // taking the larger keeps it accurate when both arguments sit far in one tail.
    vector[N_c] a = hi[1:N_c] - bx[1:N_c];
    vector[N_c] b = lo[1:N_c] - bx[1:N_c];
    vector[N_c] lc = log(fmax(Phi(a) - Phi(b), Phi(-b) - Phi(-a)) + 1e-300);
    if (top_class) {
      for (p in 1:P_c) {
        real reg = sum(lc[c_start[p]:c_end[p]]);
        target += all_top[p] ? log_mix(w_top[1], 0, reg) : log1m(w_top[1]) + reg;
      }
    } else target += sum(lc);
  }
  if (P_t > 0) {
    int M = N - N_c;
    matrix[M, 3] lp;
    for (c in 1:3) {
      vector[M] mu = (rho * bx[(N_c + 1):N] + shift[c]) / s_T;
      vector[M] a = hi[(N_c + 1):N] / s_T - mu;
      vector[M] b = lo[(N_c + 1):N] / s_T - mu;
      lp[, c] = log(fmax(Phi(a) - Phi(b), Phi(-b) - Phi(-a)) + 1e-300);
    }
    for (p in 1:P_t) {
      vector[3] acc = log(pi);
      for (c in 1:3) acc[c] += sum(lp[(t_start[p] - N_c):(t_end[p] - N_c), c]);
      if (top_class) {
        real reg = log_sum_exp(acc);
        target += all_top[P_c + p] ? log_mix(w_top[1], 0, reg) : log1m(w_top[1]) + reg;
      } else target += log_sum_exp(acc);
    }
  }
}
generated quantities {
  // shares among responsive respondents; whole-sample bounds let the always-top group be none or all favorable
  real top_share = top_class ? w_top[1] : 0;
  real favorable_lower = (1 - top_share) * pi[3];
  real favorable_upper = (1 - top_share) * pi[3] + top_share;
  real share_favorable = pi[3];
  real share_opposed = pi[1];
  real share_indifferent = pi[2];
  real mean_shift = pi[3] * delta_pos - pi[1] * delta_neg;
}
