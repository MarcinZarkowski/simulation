#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

using namespace std;

float round_to_grid(float S) {
  if (S < 25.0f)
    return round(S * 2.0f) / 2.0f;
  else
    return round(S);
}

float strike_step(float S) {
  if (S < 25.0f)
    return 0.5f;
  else
    return 1.0f;
}

vector<float> generate_strikes(float S0, int num_each_side = 40) {
  float S0_rounded = round_to_grid(S0);
  float step = strike_step(S0_rounded);
  vector<float> strikes;
  for (int i = -num_each_side; i < num_each_side; i++) {
    float strike_price = S0_rounded + i * step;
    if (strike_price >= 0 &&
        (strikes.empty() || strikes.back() < strike_price)) {
      strikes.push_back(strike_price);
    }
  }
  return strikes;
}

tuple<float, float> get_tree_params(float implied_vol, int T, int N) {
  float dt = ((float)T / 252.0f) / (float)N;
  float u = expf(implied_vol * sqrtf(dt));
  float d = 1.0f / u;
  return {u, d};
}

/**
 * Price an American option using a binomial tree.
 * Returns {price, delta, gamma, theta}
 */
vector<float> price_binomial_tree(float S0, float K, float r, int T, int N,
                                  float u, float d, string option_type) {
  float dt = (float)T / 252.0f / (float)N;
  float q = (expf(r * dt) - d) / (u - d);
  float disc = expf(-r * dt);

  // Initial stock prices at time N
  vector<float> S(N + 1);
  for (int j = 0; j <= N; j++) {
    S[j] = S0 * powf(u, j) * powf(d, N - j);
  }

  // Terminal payoffs at time N
  vector<float> C(N + 1);
  for (int j = 0; j <= N; j++) {
    if (option_type == "PUT") {
      C[j] = max(0.0f, K - S[j]);
    } else {
      C[j] = max(0.0f, S[j] - K);
    }
  }

  vector<float> C1, S1, C2, S2;

  // Backward induction
  for (int i = N - 1; i >= 0; i--) {
    vector<float> next_S(i + 1);
    for (int j = 0; j <= i; j++) {
      next_S[j] = S0 * powf(u, j) * powf(d, i - j);
      C[j] = disc * (q * C[j + 1] + (1.0f - q) * C[j]);

      // Early exercise check (American option)
      if (option_type == "PUT") {
        C[j] = max(C[j], K - next_S[j]);
      } else {
        C[j] = max(C[j], next_S[j] - K);
      }
    }
    C.resize(i + 1);

    // Store intermediate values for Greek calculations
    if (i == 2) {
      C2 = C;
      S2 = next_S;
    } else if (i == 1) {
      C1 = C;
      S1 = next_S;
    }
  }

  float price = C[0];
  float delta = (C1[1] - C1[0]) / (S1[1] - S1[0]);

  // Gamma = (Delta_up - Delta_down) / dS
  float delta_up = (C2[2] - C2[1]) / (S2[2] - S2[1]);
  float delta_down = (C2[1] - C2[0]) / (S2[1] - S2[0]);
  float gamma = (delta_up - delta_down) / (0.5f * (S2[2] - S2[0]));

  // Theta = (C2_middle - Price) / (2 * dt)
  float theta = (C2[1] - price) / (2.0f * dt);

  return {price, delta, gamma, theta};
}
