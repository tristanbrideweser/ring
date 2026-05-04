"""
forward_sim.py
--------------

Solves the coupled Algebraic Riccati Equations (CAREs) for an n-agent
linear-quadratic Nash game via best-response iteration, then generates
noisy trajectories from the resulting Nash equilibrium policies.

"""

import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class GameParams(NamedTuple):
    """All parameters that define an LQ Nash game."""
    A:    np.ndarray          # (nx, nx)  shared dynamics
    B:    np.ndarray          # (nx, nu)  shared input matrix
    D:    np.ndarray          # (n, n, nx, nx) coupling matrices D[i,j] = D_ij
    Q:    list                # list of n (nx, nx) state cost matrices
    R:    list                # list of n (nu, nu) input cost matrices
    n:    int                 # number of agents
    nx:   int                 # state dimension per agent
    nu:   int                 # input dimension per agent


class NashSolution(NamedTuple):
    """Output of the Nash solver."""
    K:    list                # list of n gain matrices (nu, n*nx)
    P:    list                # list of n Riccati solutions (nx, nx)
    A_cl: np.ndarray          # (n*nx, n*nx) closed-loop stacked dynamics
    converged: bool
    iterations: int


# ---------------------------------------------------------------------------
# Game construction helpers
# ---------------------------------------------------------------------------

def build_game(n=2, nx=2, nu=1, seed=0) -> GameParams:
    """
    Build a random stable LQ Nash game for testing.

    Parameters
    ----------
    n   : number of agents
    nx  : state dimension per agent
    nu  : input dimension per agent
    seed: random seed for reproducibility

    Returns
    -------
    GameParams with stable A, controllable (A,B), random Q/R, small coupling D
    """
    rng = np.random.default_rng(seed)

    # Stable shared dynamics: random A scaled to have spectral radius < 1
    A_raw = rng.standard_normal((nx, nx))
    eigvals = np.linalg.eigvals(A_raw)
    sr = np.max(np.abs(eigvals))
    A = A_raw / (sr + 0.5)   # spectral radius ~ 0.67, safely stable

    # Input matrix
    B = rng.standard_normal((nx, nu))

    # Coupling matrices: small to keep game well-conditioned
    D = np.zeros((n, n, nx, nx))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = 0.05 * rng.standard_normal((nx, nx))

    # Cost matrices: Q PSD, R PD
    Q = []
    R = []
    for i in range(n):
        q_half = rng.standard_normal((nx, nx))
        Q.append(q_half @ q_half.T + 0.1 * np.eye(nx))   # PSD + small ridge
        r_half = rng.standard_normal((nu, nu))
        R.append(r_half @ r_half.T + 0.5 * np.eye(nu))   # PD

    return GameParams(A=A, B=B, D=D, Q=Q, R=R, n=n, nx=nx, nu=nu)


# ---------------------------------------------------------------------------
# Nash solver: best-response iteration over coupled CAREs
# ---------------------------------------------------------------------------

def _stacked_closed_loop(params: GameParams, K: list) -> np.ndarray:
    """
    Build the stacked closed-loop dynamics matrix A_bar given gains K.

    The stacked state is x = [x1; x2; ...; xn], shape (n*nx,).
    Each agent i's dynamics:
        x_i(k+1) = A x_i + B u_i + sum_{j!=i} D_ij x_j
                 = (A - B K_i^{(i)}) x_i
                   - B K_i^{(j)} x_j  (cross terms from gain)
                   + D_ij x_j          (coupling)

    where K_i^{(j)} is the block of K_i corresponding to agent j's state.
    """
    n, nx, nu = params.n, params.nx, params.nu
    A_cl = np.zeros((n * nx, n * nx))

    for i in range(n):
        for j in range(n):
            row = slice(i * nx, (i + 1) * nx)
            col = slice(j * nx, (j + 1) * nx)
            # Gain block: K_i maps full stacked state -> input
            K_ij = K[i][:, j * nx:(j + 1) * nx]   # (nu, nx)
            if i == j:
                A_cl[row, col] = params.A - params.B @ K_ij
            else:
                A_cl[row, col] = params.D[i, j] - params.B @ K_ij

    return A_cl


def _solve_single_care(params: GameParams, i: int, K: list):
    """
    Solve agent i's DARE given other agents' gains K_{-i} fixed.

    We extract an effective (A_eff, B) for agent i by absorbing the
    other agents' contributions into A_eff, then call scipy's DARE solver.

    Returns (P_i, K_i)
    """
    n, nx, nu = params.n, params.nx, params.nu

    # Build A_eff: agent i's perspective of the dynamics
    # x_i(k+1) = A x_i + B u_i + sum_{j!=i}(D_ij - B K_j^{(i)}) x_i
    #           + sum_{j!=i}(D_ij - B K_j^{(j')}) x_{j'} ...
    # We need the full stacked A_cl and then extract the relevant blocks.
    A_cl = _stacked_closed_loop(params, K)

    # Agent i's effective dynamics when it deviates:
    # Fix all j != i at their current K_j, only agent i chooses u_i freely.
    # The relevant DARE for agent i uses:
    #   A_bar_{-i}: the closed-loop matrix with agent i's gain removed
    #
    # Specifically, agent i faces:
    #   x(k+1) = A_bar_{-i} x(k) + B_ext u_i(k)
    # where B_ext = [0;...;B;...;0] (B in agent i's block)
    # and the full state cost for agent i is x^T blkdiag(Q_i, 0,...) x

    # A_bar_{-i}: replace agent i's contribution with open-loop
    K_no_i = [K[j] if j != i else np.zeros_like(K[i]) for j in range(n)]
    A_bar_minus_i = _stacked_closed_loop(params, K_no_i)

    # Extended B: agent i's input only affects its own block
    B_ext = np.zeros((n * nx, nu))
    B_ext[i * nx:(i + 1) * nx, :] = params.B

    # Extended Q: agent i only pays for its own state
    Q_ext = np.zeros((n * nx, n * nx))
    Q_ext[i * nx:(i + 1) * nx, i * nx:(i + 1) * nx] = params.Q[i]

    # Solve DARE: P = Q_ext + A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A
    try:
        P_ext = solve_discrete_are(A_bar_minus_i, B_ext, Q_ext, params.R[i])
    except Exception as e:
        raise RuntimeError(f"DARE failed for agent {i}: {e}")

    # Compute gain
    S = params.R[i] + B_ext.T @ P_ext @ B_ext          # (nu, nu)
    K_i_full = np.linalg.solve(S, B_ext.T @ P_ext @ A_bar_minus_i)  # (nu, n*nx)

    # Extract agent i's own Riccati matrix (its block)
    P_i = P_ext[i * nx:(i + 1) * nx, i * nx:(i + 1) * nx]

    return P_i, K_i_full


def solve_nash(params: GameParams,
               max_iter: int = 500,
               tol: float = 1e-8,
               verbose: bool = True) -> NashSolution:
    """
    Solve the LQ Nash game via best-response iteration.

    Algorithm:
        1. Initialize each K_i from independent LQR (ignoring coupling).
        2. Cycle through agents, updating each K_i with other K_{-i} fixed.
        3. Repeat until max gain change < tol.

    Parameters
    ----------
    params   : GameParams
    max_iter : maximum best-response iterations
    tol      : convergence threshold on max||K_i^{t+1} - K_i^t||_F
    verbose  : print convergence info

    Returns
    -------
    NashSolution
    """
    n, nx, nu = params.n, params.nx, params.nu

    # --- Initialize with independent LQR (zero coupling) ---
    K = []
    P = []
    for i in range(n):
        try:
            p0 = solve_discrete_are(params.A, params.B, params.Q[i], params.R[i])
            S0 = params.R[i] + params.B.T @ p0 @ params.B
            k0_local = np.linalg.solve(S0, params.B.T @ p0 @ params.A)  # (nu, nx)
            # Embed into full stacked gain (nu, n*nx), only agent i's block nonzero
            k0_full = np.zeros((nu, n * nx))
            k0_full[:, i * nx:(i + 1) * nx] = k0_local
            K.append(k0_full)
            P.append(p0)
        except Exception:
            K.append(np.zeros((nu, n * nx)))
            P.append(np.eye(nx))

    # --- Best-response iteration ---
    converged = False
    for t in range(max_iter):
        K_old = [k.copy() for k in K]

        for i in range(n):
            P[i], K[i] = _solve_single_care(params, i, K)

        # Convergence check
        delta = max(np.linalg.norm(K[i] - K_old[i], 'fro') for i in range(n))
        if verbose and (t % 50 == 0 or delta < tol):
            print(f"  Iter {t:4d} | max ||ΔK||_F = {delta:.2e}")

        if delta < tol:
            converged = True
            if verbose:
                print(f"  Converged in {t+1} iterations.")
            break

    if not converged and verbose:
        print(f"  Warning: did not converge in {max_iter} iterations. "
              f"Final delta = {delta:.2e}")

    A_cl = _stacked_closed_loop(params, K)

    # Verify stability
    eigs = np.linalg.eigvals(A_cl)
    sr = np.max(np.abs(eigs))
    if verbose:
        print(f"  Closed-loop spectral radius: {sr:.4f} "
              f"({'stable' if sr < 1 else 'UNSTABLE'})")

    return NashSolution(K=K, P=P, A_cl=A_cl,
                        converged=converged, iterations=t + 1)


# ---------------------------------------------------------------------------
# Nash verification
# ---------------------------------------------------------------------------

def verify_nash(params: GameParams, nash: NashSolution,
                tol: float = 1e-5) -> bool:
    """
    Verify Nash equilibrium: for each agent i, check that K_i is the
    best response to K_{-i} by comparing against a fresh DARE solve.

    Returns True if all agents satisfy Nash conditions within tol.
    """
    n = params.n
    all_ok = True
    for i in range(n):
        _, K_br = _solve_single_care(params, i, nash.K)
        err = np.linalg.norm(K_br - nash.K[i], 'fro')
        ok = err < tol
        print(f"  Agent {i}: best-response deviation = {err:.2e}  "
              f"{'✓' if ok else '✗ FAILED'}")
        if not ok:
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def generate_trajectories(params: GameParams,
                           nash: NashSolution,
                           M: int = 50,
                           T: int = 100,
                           sigma_w: float = 0.01,
                           sigma_0: float = 1.0,
                           seed: int = 42) -> dict:
    """
    Generate M noisy trajectories of length T from the Nash equilibrium.

    Parameters
    ----------
    params  : GameParams
    nash    : NashSolution (must be stable)
    M       : number of trajectories
    T       : trajectory length (time steps)
    sigma_w : process noise standard deviation
    sigma_0 : initial state standard deviation
    seed    : random seed

    Returns
    -------
    dict with keys:
        'X' : (M, T, n*nx)  stacked state trajectories
        'U' : (M, T, n, nu) control inputs per agent
        'K' : list of n gain matrices (ground truth)
    """
    rng = np.random.default_rng(seed)
    n, nx, nu = params.n, params.nx, params.nu
    A_cl = nash.A_cl
    full_nx = n * nx

    X = np.zeros((M, T, full_nx))
    U = np.zeros((M, T, n, nu))

    for m in range(M):
        x = rng.standard_normal(full_nx) * sigma_0
        for t in range(T):
            X[m, t] = x
            # Compute each agent's control
            for i in range(n):
                u_i = -nash.K[i] @ x         # (nu,)
                U[m, t, i] = u_i
            # Advance state with process noise
            noise = rng.standard_normal(full_nx) * sigma_w
            x = A_cl @ x + noise

    return {'X': X, 'U': U, 'K': nash.K}


# ---------------------------------------------------------------------------
# Gain estimation from trajectories (least squares)
# ---------------------------------------------------------------------------

def estimate_gains(traj: dict, params: GameParams) -> list:
    """
    Estimate Nash gains K_hat_i from trajectory data via least squares.

    For each agent i:
        u_i(t) = -K_i x(t)  =>  solve min ||U_i + X K_i^T||_F^2

    Parameters
    ----------
    traj   : output of generate_trajectories
    params : GameParams

    Returns
    -------
    K_hat : list of n estimated gain matrices (nu, n*nx)
    """
    M, T, full_nx = traj['X'].shape
    n, nu = params.n, params.nu

    # Reshape: (M*T, full_nx) and (M*T, nu) per agent
    X_flat = traj['X'].reshape(-1, full_nx)        # (M*T, n*nx)
    U_flat = traj['U'].reshape(-1, n, nu)          # (M*T, n, nu)

    K_hat = []
    for i in range(n):
        u_i = U_flat[:, i, :]                      # (M*T, nu)
        # u_i = -K_i x  =>  K_i = -(X^T X)^{-1} X^T U
        # i.e. least squares: min ||u_i + X K_i^T||
        K_i_T, _, _, _ = np.linalg.lstsq(X_flat, -u_i, rcond=None)
        K_hat.append(K_i_T.T)                      # (nu, n*nx)

    return K_hat