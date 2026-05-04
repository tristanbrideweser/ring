"""
naive_baseline.py
-----------------

Naive distributed IRL baseline that IGNORES the coupled Riccati structure.

For each agent i, this baseline solves an independent single-agent inverse LQR
problem: assume agent i acts in isolation under dynamics (A, B, Q_i, R_i) with
no coupling D_ij and no other agents' policies, and recover (Q_i, R_i) such
that the resulting LQR gain matches the locally-observed gain restricted to
agent i's own state block.

This is the natural baseline a practitioner would implement if they treated
the multi-agent IRL problem as n independent single-agent problems. As shown
in Sec. V-B of the RING paper, it fails by roughly two orders of magnitude
in MSE because each agent's true gain depends on every other agent's policy
through the coupled CAREs -- single-agent inverse LQR cannot represent that.

Usage:
    from naive_baseline import run_naive_baseline
    result = run_naive_baseline(params, K_hat)
    print(result['mse_K'])
"""

import numpy as np
from scipy.linalg import solve_discrete_are

from forward_sim import GameParams
from centralized_irl import (compute_K_and_Acl, normalize_params,
                              project_psd, project_pd)


# ---------------------------------------------------------------------------
# Single-agent inverse LQR for one agent (Adam over single-agent DARE)
# ---------------------------------------------------------------------------

def _single_agent_dare_gain(A, B, Q, R):
    """Solve single-agent DARE and return (P, K_local) where K_local has
    shape (nu, nx) -- only acts on the agent's OWN state."""
    P = solve_discrete_are(A, B, Q, R)
    S = R + B.T @ P @ B
    K_local = np.linalg.solve(S, B.T @ P @ A)
    return P, K_local


def _own_block_of_gain(K_full, agent_idx, nx):
    """Extract the nx-block of a stacked gain K_full corresponding to the
    agent's own state. K_full has shape (nu, n*nx)."""
    return K_full[:, agent_idx * nx:(agent_idx + 1) * nx]


def _single_agent_loss(Q, R, A, B, K_target_local, mu):
    """Forward-only loss for the single-agent inverse LQR problem."""
    try:
        _, K_loc = _single_agent_dare_gain(A, B, Q, R)
    except Exception:
        return float('inf'), None
    res = K_loc - K_target_local
    loss = (np.linalg.norm(res, 'fro') ** 2
            + mu * (np.linalg.norm(Q, 'fro') ** 2
                    + np.linalg.norm(R, 'fro') ** 2))
    return loss, K_loc


def _fd_grad(f, X, eps=1e-6):
    """Symmetric finite-difference gradient of f: matrix -> scalar.
    f(X) returns a scalar. Returns dX with same shape as X."""
    g = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] += eps
            fp = f(X)
            X[i, j] -= 2 * eps
            fm = f(X)
            X[i, j] += eps
            g[i, j] = (fp - fm) / (2 * eps)
    return g


def _single_agent_irl(A, B, K_target_local, nu, nx,
                       mu=1e-3, lr=5e-3, max_iter=500, tol=1e-8):
    """
    Solve single-agent inverse LQR: find (Q, R) such that the single-agent
    DARE gain matches K_target_local.

    Uses Adam with PSD/PD projection and trace normalization. Gradients are
    computed by finite differences -- nx and nu are tiny here (typically 2
    and 1), so this is fast and avoids any single-vs-multi-agent algebra
    pitfalls. The point of this baseline is correctness, not speed.

    Returns (Q_est, R_est, K_recovered_local).
    """
    rng = np.random.default_rng(99)
    Q = np.eye(nx) + 0.2 * rng.standard_normal((nx, nx))
    Q = project_psd(Q + Q.T)
    R = np.eye(nu) + 0.05 * rng.standard_normal((nu, nu))
    R = project_pd(R + R.T)
    Q, R = normalize_params(Q, R)

    # Adam state
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    mQ = np.zeros_like(Q); vQ = np.zeros_like(Q)
    mR = np.zeros_like(R); vR = np.zeros_like(R)

    prev_loss = np.inf

    for t in range(max_iter):
        loss, K_loc = _single_agent_loss(Q, R, A, B, K_target_local, mu)
        if not np.isfinite(loss):
            break
        if abs(prev_loss - loss) < tol and t > 20:
            break
        prev_loss = loss

        # Finite-difference gradients (small dims -> fine)
        grad_Q = _fd_grad(lambda X: _single_agent_loss(X, R, A, B,
                                                        K_target_local, mu)[0],
                           Q.copy())
        grad_R = _fd_grad(lambda X: _single_agent_loss(Q, X, A, B,
                                                        K_target_local, mu)[0],
                           R.copy())
        grad_Q = 0.5 * (grad_Q + grad_Q.T)
        grad_R = 0.5 * (grad_R + grad_R.T)

        # Adam update
        step = t + 1
        mQ = beta1 * mQ + (1 - beta1) * grad_Q
        vQ = beta2 * vQ + (1 - beta2) * grad_Q ** 2
        Q = Q - lr * (mQ / (1 - beta1 ** step)) / \
            (np.sqrt(vQ / (1 - beta2 ** step)) + eps_adam)

        mR = beta1 * mR + (1 - beta1) * grad_R
        vR = beta2 * vR + (1 - beta2) * grad_R ** 2
        R = R - lr * (mR / (1 - beta1 ** step)) / \
            (np.sqrt(vR / (1 - beta2 ** step)) + eps_adam)

        Q = project_psd(Q)
        R = project_pd(R)
        Q, R = normalize_params(Q, R)

    try:
        _, K_loc = _single_agent_dare_gain(A, B, Q, R)
    except Exception:
        K_loc = None
    return Q, R, K_loc


# ---------------------------------------------------------------------------
# Naive baseline: per-agent independent inverse LQR
# ---------------------------------------------------------------------------

def run_naive_baseline(params: GameParams,
                        K_hat: list,
                        mu: float = 1e-3,
                        lr: float = 5e-3,
                        max_iter: int = 500,
                        tol: float = 1e-8,
                        verbose: bool = False) -> dict:
    """
    Naive baseline: for each agent i, run an independent single-agent
    inverse LQR using only the own-state block of the observed gain.

    No coupling D_ij is modeled; no consensus is performed; no other agents'
    policies are considered. This is the "obvious wrong thing" that motivates
    why distributing inverse game theory is non-trivial.

    Parameters
    ----------
    params   : GameParams
    K_hat    : list of n stacked gains (nu, n*nx) from least-squares
    mu, lr, max_iter, tol : passed to single-agent inverse LQR
    verbose  : print per-agent progress

    Returns
    -------
    dict with keys:
        Q_est, R_est : recovered cost matrices (independent per agent)
        K_recovered  : list of n stacked gains recovered by plugging
                       (Q_est, R_est) back into the FULL coupled Nash solver
                       -- this is the fair comparison: how well does the
                       naive cost recovery explain the actual coupled gains?
        mse_K        : mean squared error vs K_hat (full stacked)
        mse_K_own_block : MSE on just the own-block portion (the part the
                          naive baseline actually fits)
    """
    n, nx, nu = params.n, params.nx, params.nu

    Q_est = []
    R_est = []
    for i in range(n):
        K_target = _own_block_of_gain(K_hat[i], i, nx)
        if verbose:
            print(f"  Agent {i}: fitting single-agent inverse LQR...")
        Q_i, R_i, _ = _single_agent_irl(
            params.A, params.B, K_target, nu, nx,
            mu=mu, lr=lr, max_iter=max_iter, tol=tol)
        Q_est.append(Q_i)
        R_est.append(R_i)

    # --- Fair evaluation: plug naive (Q,R) back into the coupled Nash solver ---
    # This is the key step: the naive baseline RECOVERS Q,R as if agents were
    # isolated, but the truth is that agents play a coupled game. To measure
    # how much error this produces, we compute what gains the coupled CAREs
    # would produce given these naive cost estimates.
    try:
        K_recovered, _, _ = compute_K_and_Acl(params, Q_est, R_est)
        mse_K = np.mean([np.linalg.norm(K_recovered[i] - K_hat[i], 'fro') ** 2
                         for i in range(n)])
    except Exception as e:
        if verbose:
            print(f"  Coupled Nash solve failed on naive (Q,R): {e}")
        K_recovered = None
        mse_K = float('nan')

    # Also report the metric the naive baseline thinks it's optimizing:
    # own-block-only MSE
    mse_K_own_block = 0.0
    if K_recovered is not None:
        for i in range(n):
            own_rec = _own_block_of_gain(K_recovered[i], i, nx)
            own_hat = _own_block_of_gain(K_hat[i], i, nx)
            mse_K_own_block += np.linalg.norm(own_rec - own_hat, 'fro') ** 2
        mse_K_own_block /= n

    return {
        'Q_est': Q_est,
        'R_est': R_est,
        'K_recovered': K_recovered,
        'mse_K': mse_K,
        'mse_K_own_block': mse_K_own_block,
    }


# ---------------------------------------------------------------------------
# Sweep across the same 20 random games used for price-of-decentralization
# ---------------------------------------------------------------------------

def run_naive_sweep(n_games: int = 20,
                     n_agents: int = 4,
                     T: int = 200,
                     M: int = 50,
                     sigma_w: float = 0.01,
                     verbose: bool = True) -> dict:
    """
    Run the naive baseline across n_games random LQ Nash games to produce
    the row added to Table I.

    Returns dict with mean/std MSE(K) and per-game results.
    """
    from forward_sim import (build_game, solve_nash,
                              generate_trajectories, estimate_gains)

    mse_list = []
    own_block_mse_list = []

    for g in range(n_games):
        params = build_game(n=n_agents, nx=2, nu=1, seed=g)
        nash = solve_nash(params, verbose=False)
        traj = generate_trajectories(params, nash, M=M, T=T,
                                      sigma_w=sigma_w, seed=42 + g)
        K_hat = estimate_gains(traj, params)
        result = run_naive_baseline(params, K_hat, verbose=False)
        mse_list.append(result['mse_K'])
        own_block_mse_list.append(result['mse_K_own_block'])
        if verbose:
            print(f"  Game {g:2d}: naive MSE(K) = {result['mse_K']:.4e}  "
                  f"(own-block: {result['mse_K_own_block']:.4e})")

    mse_arr = np.array(mse_list)
    own_arr = np.array(own_block_mse_list)
    finite = np.isfinite(mse_arr)

    summary = {
        'mean_mse_K':  float(np.mean(mse_arr[finite])) if finite.any() else float('nan'),
        'std_mse_K':   float(np.std(mse_arr[finite]))  if finite.any() else float('nan'),
        'mean_mse_K_own': float(np.mean(own_arr[finite])) if finite.any() else float('nan'),
        'std_mse_K_own':  float(np.std(own_arr[finite]))  if finite.any() else float('nan'),
        'per_game_mse_K': mse_arr,
        'per_game_mse_K_own': own_arr,
        'n_failed': int((~finite).sum()),
    }

    if verbose:
        print(f"\n  Naive baseline over {n_games} games:")
        print(f"    Mean MSE(K)        = {summary['mean_mse_K']:.4e}")
        print(f"    Std  MSE(K)        = {summary['std_mse_K']:.4e}")
        print(f"    Mean MSE(K) own    = {summary['mean_mse_K_own']:.4e}")
        print(f"    Failures (NaN)     = {summary['n_failed']}/{n_games}")

    return summary


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Naive Independent-IRL Baseline -- Validation")
    print("=" * 60)

    from forward_sim import (build_game, solve_nash,
                              generate_trajectories, estimate_gains)

    # Quick sanity check on one game
    print("\n[1] Single game sanity check (n=4, seed=0)...")
    params = build_game(n=4, nx=2, nu=1, seed=0)
    nash = solve_nash(params, verbose=False)
    traj = generate_trajectories(params, nash, M=50, T=200,
                                  sigma_w=0.01, seed=42)
    K_hat = estimate_gains(traj, params)

    result = run_naive_baseline(params, K_hat, verbose=True)
    print(f"\n  Naive MSE(K)            = {result['mse_K']:.4e}")
    print(f"  Naive MSE(K) own-block  = {result['mse_K_own_block']:.4e}")
    print("  (Compare to RING ~6e-2 and centralized ~9e-3 from Table I)")

    # Full sweep
    print("\n[2] Full sweep over 20 games...")
    summary = run_naive_sweep(n_games=20, verbose=True)

    print("\nNaive baseline OK.")