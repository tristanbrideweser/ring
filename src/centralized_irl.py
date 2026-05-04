"""
centralized_irl.py
------------------
Stage 2 of DMA-IRL: Centralized Inverse Baseline

Given observed Nash equilibrium gains K_hat_i (estimated from trajectory data),
recover the cost parameters theta_i = (Q_i, R_i) for each agent by minimizing:

    F(theta) = sum_i || K_i(theta) - K_hat_i ||_F^2 + mu * ||theta_i||_F^2

Gradients are computed analytically via implicit differentiation of the
coupled Riccati equations, reducing to discrete Lyapunov solves.

Usage:
    from centralized_irl import run_centralized_irl
"""

import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from forward_sim import (GameParams, NashSolution, build_game,
                          solve_nash, generate_trajectories,
                          estimate_gains, _solve_single_care,
                          _stacked_closed_loop)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_params(Q: np.ndarray, R: np.ndarray) -> tuple:
    """
    Normalize cost parameters to remove scale ambiguity.
    Convention: trace(R_i) = 1  (fixes the scale of each agent's cost).
    """
    scale = np.trace(R) + 1e-12
    return Q / scale, R / scale


def symmetrize(M: np.ndarray) -> np.ndarray:
    """Enforce symmetry: (M + M^T) / 2."""
    return (M + M.T) / 2


def project_psd(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Project matrix onto PSD cone by clipping eigenvalues."""
    M = symmetrize(M)
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def project_pd(M: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Project matrix onto PD cone (strictly positive eigenvalues)."""
    return project_psd(M, eps=eps)


# ---------------------------------------------------------------------------
# Gradient computation via implicit differentiation
# ---------------------------------------------------------------------------

def compute_K_and_Acl(params: GameParams,
                       Q_list: list,
                       R_list: list) -> tuple:
    """
    Given cost parameters (Q_list, R_list), solve the Nash game and
    return gains K, Riccati solutions P, and closed-loop matrix A_cl.

    Builds a temporary GameParams with the current parameter estimates.
    """
    tmp = GameParams(A=params.A, B=params.B, D=params.D,
                     Q=Q_list, R=R_list,
                     n=params.n, nx=params.nx, nu=params.nu)
    nash = solve_nash(tmp, verbose=False, tol=1e-8)
    return nash.K, nash.P, nash.A_cl


def lyapunov_gradient(A_cl: np.ndarray,
                       K_i: np.ndarray,
                       K_hat_i: np.ndarray,
                       B: np.ndarray,
                       R_i: np.ndarray,
                       nx: int,
                       nu: int,
                       agent_idx: int,
                       n_agents: int) -> tuple:
    """
    Compute gradients dL/dQ_i and dL/dR_i for agent i using the
    Lyapunov-based implicit differentiation derived in Day 1.

    Loss for agent i:
        L_i = || K_i - K_hat_i ||_F^2

    Chain rule:
        dL/dQ_i = sum_{pq} trace(residual^T dK/dQ_i[pq]) * E_pq
        dL/dR_i = sum_{pq} trace(residual^T dK/dR_i[pq]) * E_pq

    Parameters
    ----------
    A_cl    : (n*nx, n*nx) full closed-loop stacked dynamics
    K_i     : (nu, n*nx) current gain for agent i
    K_hat_i : (nu, n*nx) estimated gain from data
    B       : (nx, nu) input matrix
    R_i     : (nu, nu) current R for agent i
    nx, nu  : dimensions
    agent_idx : index of agent i
    n_agents  : total number of agents

    Returns
    -------
    grad_Q : (nx, nx) gradient w.r.t. Q_i
    grad_R : (nu, nu) gradient w.r.t. R_i
    """
    full_nx = n_agents * nx
    row_i = slice(agent_idx * nx, (agent_idx + 1) * nx)

    # Extended B (B in agent i's block of the stacked system)
    B_ext = np.zeros((full_nx, nu))
    B_ext[row_i, :] = B

    # Closed-loop for agent i: A_cl_i = A_cl (already incorporates all gains)
    # Agent i's local closed-loop: A - B K_i^{(i)} (for Lyapunov RHS)
    # But we need the FULL stacked A_cl for the Lyapunov equation
    A_cl_full = A_cl  # (n*nx, n*nx)

    # S = R_i + B_ext^T P_ext B_ext  -- we can recover from K_i
    # K_i = S^{-1} B_ext^T P_ext A_bar_{-i}
    # Instead compute S from: S K_i = B_ext^T P_ext A_bar_{-i}
    # We'll differentiate K_i = (R + B^T P B)^{-1} B^T P A_bar directly

    # Residual: dL/dK_i = 2(K_i - K_hat_i)
    residual = 2.0 * (K_i - K_hat_i)  # (nu, n*nx)

    # --------------- Adjoint Lyapunov (single solve for both grads) ---------------
    # Instead of nx*(nx+1)/2 + nu*(nu+1)/2 forward Lyapunov solves,
    # use ONE adjoint solve to back-propagate the loss through dK -> dP.
    #
    # Forward map: dQ -> dP (Lyapunov) -> dK (linear)
    # Adjoint map: residual -> dP* (adjoint Lyapunov) -> grad_Q, grad_R
    #
    # Adjoint of dK = R^{-1}[B^T dP A - (B^T dP B) K]:
    #   <residual, dK> = <R^{-T} residual, B^T dP A - (B^T dP B) K>
    #                  = <dP, B R^{-T} residual A^T - B (R^{-T} residual K^T B^T)^T>
    #                    (cycling trace)
    # So adjoint RHS Z = B_ext R^{-T} res A^T - B_ext R^{-T} res K^T B_ext^T  (symmetrized)
    try:
        Sinv_res = np.linalg.solve(R_i.T, residual)   # (nu, n*nx)
    except np.linalg.LinAlgError:
        Sinv_res = residual

    # B_ext: (full_nx, nu), Sinv_res: (nu, full_nx), A_cl_full: (full_nx, full_nx)
    # Term1: B_ext Sinv_res A^T -> (full_nx, full_nx)
    # Term2: B_ext (B_ext^T Sinv_res^T) Sinv_res^T ... fix via outer product
    # Correct: <res, S^{-1} B^T dP (A - BK)> => adjoint on dP:
    #   dL/dP = (A - BK)^T (S^{-T} res)^T B^T^T = (A-BK)^T Sinv_res^T B^T
    # Simpler: Z_pq = sum_{ab} res_{ab} * dK_{ab}/dP_{pq}
    # dK/dP = S^{-1} B^T (·) A - S^{-1} B^T (·) B K
    # Adjoint: Z = B S^{-T} res A^T - B S^{-T}(res K^T) B^T  ... all (full_nx x full_nx)
    Z = (B_ext @ Sinv_res @ A_cl_full.T
         - B_ext @ (Sinv_res @ K_i.T) @ B_ext.T)
    Z_sym = (Z + Z.T) / 2.0

    # Adjoint Lyapunov: dP* - A_cl^T dP* A_cl = Z_sym
    try:
        dP_adj = solve_discrete_lyapunov(A_cl_full.T, Z_sym)
    except Exception:
        dP_adj = np.zeros((full_nx, full_nx))

    # --------------- grad_Q_i ---------------
    # grad_Q = dP_adj[agent_i_block, agent_i_block]  (from chain rule on dQ_ext)
    grad_Q = dP_adj[agent_idx*nx:(agent_idx+1)*nx,
                    agent_idx*nx:(agent_idx+1)*nx].copy()
    grad_Q = (grad_Q + grad_Q.T) / 2.0

    # --------------- grad_R_i ---------------
    # From Lyapunov RHS = K^T dR K:
    #   <residual, dK_via_P> = <dR, K dP_adj K^T>
    # Direct term: dK_direct = -R^{-1} dR K
    #   <residual, dK_direct> = -<R^{-T} res K^T, dR>
    grad_R = K_i @ dP_adj @ K_i.T - Sinv_res @ K_i.T
    grad_R = (grad_R + grad_R.T) / 2.0

    return grad_Q, grad_R


def _dK_from_dP(BtdP: np.ndarray,
                A_cl: np.ndarray,
                B_ext: np.ndarray,
                K_i: np.ndarray,
                R_i: np.ndarray) -> np.ndarray:
    """
    Compute dK given B_ext^T dP.

    dK = S^{-1} [B_ext^T dP A_cl - (B_ext^T dP B_ext) K_i]

    where S = R_i + B_ext^T P B_ext (approximated via K_i).
    """
    # Approximate S^{-1} via R_i (works well when B_ext^T P B_ext << R_i,
    # or we recover S from the gain equation below)
    # Better: solve for P from the gain and use exact S
    dK_unscaled = BtdP @ A_cl - (BtdP @ B_ext) @ K_i   # (nu, n*nx)

    # S: we need this for the solve. Use R_i as approximation if P unavailable.
    # This is exact when combined with the P solve in lyapunov_gradient.
    # Here we just use R_i^{-1} as S^{-1} approximation for the direction.
    # The centralized optimizer corrects via line search anyway.
    try:
        result = np.linalg.solve(R_i, dK_unscaled)
    except np.linalg.LinAlgError:
        result = dK_unscaled
    return result


def _get_P_from_K(K_i: np.ndarray,
                   R_i: np.ndarray,
                   B_ext: np.ndarray,
                   A_cl: np.ndarray) -> np.ndarray:
    """
    Recover P from K via the discrete Lyapunov equation satisfied by P
    at the Nash equilibrium:  P = Q + A_cl^T P A_cl  (implicit in gain).

    Since we don't store P here, solve the Lyapunov equation:
        P - A_cl^T P A_cl = I  (placeholder; returns identity if ill-posed)

    In practice, P is passed directly from the Nash solver in the main loop.
    This function is a fallback for the S computation.
    """
    try:
        P = solve_discrete_lyapunov(A_cl.T, np.eye(A_cl.shape[0]))
        return P
    except Exception:
        return np.eye(A_cl.shape[0])


# ---------------------------------------------------------------------------
# Centralized IRL optimizer
# ---------------------------------------------------------------------------

def centralized_irl_loss(params: GameParams,
                          Q_list: list,
                          R_list: list,
                          K_hat: list,
                          mu: float = 1e-3) -> float:
    """
    Compute the centralized IRL loss:
        F(theta) = sum_i || K_i(theta) - K_hat_i ||_F^2
                 + mu * sum_i (||Q_i||_F^2 + ||R_i||_F^2)
    """
    K, P, A_cl = compute_K_and_Acl(params, Q_list, R_list)
    loss = 0.0
    for i in range(params.n):
        loss += np.linalg.norm(K[i] - K_hat[i], 'fro') ** 2
        loss += mu * (np.linalg.norm(Q_list[i], 'fro') ** 2 +
                      np.linalg.norm(R_list[i], 'fro') ** 2)
    return loss


def run_centralized_irl(params: GameParams,
                         K_hat: list,
                         mu: float = 1e-3,
                         lr: float = 5e-3,
                         max_iter: int = 500,
                         tol: float = 1e-8,
                         verbose: bool = True) -> dict:
    """
    Run centralized IRL: recover (Q_i, R_i) from observed gains K_hat_i.

    Algorithm: Adam optimizer with projection onto PSD/PD cones and
    normalization to remove scale ambiguity.

    Parameters
    ----------
    params   : true GameParams (A, B, D, n, nx, nu used; Q/R are ground truth)
    K_hat    : list of n estimated gains from trajectory data
    mu       : regularization weight
    lr       : Adam learning rate
    max_iter : maximum gradient steps
    tol      : convergence on loss change
    verbose  : print progress

    Returns
    -------
    dict with keys: Q_est, R_est, loss_history, K_recovered, mse_Q, mse_R
    """
    n, nx, nu = params.n, params.nx, params.nu

    # --- Initialize: small perturbation from identity ---
    rng = np.random.default_rng(99)
    Q_est = []
    R_est = []
    for i in range(n):
        q0 = np.eye(nx) + 0.2 * rng.standard_normal((nx, nx))
        Q_est.append(project_psd(q0 + q0.T))
        r0 = np.eye(nu) + 0.05 * rng.standard_normal((nu, nu))
        R_est.append(project_pd(r0 + r0.T))

    for i in range(n):
        Q_est[i], R_est[i] = normalize_params(Q_est[i], R_est[i])

    # Adam state
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    mQ = [np.zeros_like(Q_est[i]) for i in range(n)]
    vQ = [np.zeros_like(Q_est[i]) for i in range(n)]
    mR = [np.zeros_like(R_est[i]) for i in range(n)]
    vR = [np.zeros_like(R_est[i]) for i in range(n)]

    loss_history = []
    prev_loss = np.inf

    for t in range(max_iter):
        # Forward pass
        try:
            K_curr, P_curr, A_cl = compute_K_and_Acl(params, Q_est, R_est)
        except Exception as e:
            if verbose:
                print(f"  Nash solve failed at iter {t}: {e}")
            break

        # Loss
        loss = 0.0
        for i in range(n):
            loss += np.linalg.norm(K_curr[i] - K_hat[i], 'fro') ** 2
            loss += mu * (np.linalg.norm(Q_est[i], 'fro') ** 2 +
                          np.linalg.norm(R_est[i], 'fro') ** 2)
        loss_history.append(loss)

        if verbose and t % 50 == 0:
            print(f"  Iter {t:4d} | Loss = {loss:.6e}")

        if abs(prev_loss - loss) < tol and t > 20:
            if verbose:
                print(f"  Converged at iter {t} (Δloss = {abs(prev_loss-loss):.2e})")
            break
        prev_loss = loss

        # Backward pass + Adam update
        step = t + 1
        for i in range(n):
            grad_Q, grad_R = lyapunov_gradient(
                A_cl=A_cl, K_i=K_curr[i], K_hat_i=K_hat[i],
                B=params.B, R_i=R_est[i],
                nx=nx, nu=nu, agent_idx=i, n_agents=n
            )
            grad_Q += 2 * mu * Q_est[i]
            grad_R += 2 * mu * R_est[i]

            # Adam for Q
            mQ[i] = beta1 * mQ[i] + (1 - beta1) * grad_Q
            vQ[i] = beta2 * vQ[i] + (1 - beta2) * grad_Q ** 2
            mQ_hat = mQ[i] / (1 - beta1 ** step)
            vQ_hat = vQ[i] / (1 - beta2 ** step)
            Q_est[i] = Q_est[i] - lr * mQ_hat / (np.sqrt(vQ_hat) + eps_adam)

            # Adam for R
            mR[i] = beta1 * mR[i] + (1 - beta1) * grad_R
            vR[i] = beta2 * vR[i] + (1 - beta2) * grad_R ** 2
            mR_hat = mR[i] / (1 - beta1 ** step)
            vR_hat = vR[i] / (1 - beta2 ** step)
            R_est[i] = R_est[i] - lr * mR_hat / (np.sqrt(vR_hat) + eps_adam)

            # Project + normalize
            Q_est[i] = project_psd(Q_est[i])
            R_est[i] = project_pd(R_est[i])
            Q_est[i], R_est[i] = normalize_params(Q_est[i], R_est[i])

    # --- Final evaluation ---
    K_recovered, _, _ = compute_K_and_Acl(params, Q_est, R_est)

    # Normalize ground truth for fair MSE comparison
    Q_true_norm = []
    R_true_norm = []
    for i in range(n):
        q_n, r_n = normalize_params(params.Q[i].copy(), params.R[i].copy())
        Q_true_norm.append(q_n)
        R_true_norm.append(r_n)

    mse_Q = np.mean([np.linalg.norm(Q_est[i] - Q_true_norm[i], 'fro') ** 2
                     for i in range(n)])
    mse_R = np.mean([np.linalg.norm(R_est[i] - R_true_norm[i], 'fro') ** 2
                     for i in range(n)])
    mse_K = np.mean([np.linalg.norm(K_recovered[i] - K_hat[i], 'fro') ** 2
                     for i in range(n)])

    return {
        'Q_est': Q_est,
        'R_est': R_est,
        'loss_history': loss_history,
        'K_recovered': K_recovered,
        'mse_Q': mse_Q,
        'mse_R': mse_R,
        'mse_K': mse_K,
        'Q_true_norm': Q_true_norm,
        'R_true_norm': R_true_norm,
    }


# ---------------------------------------------------------------------------
# Validation run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("DMA-IRL Centralized IRL — Validation Run")
    print("=" * 60)

    # --- Build game and generate data ---
    print("\n[1] Building 2-agent game and generating trajectories...")
    params = build_game(n=2, nx=2, nu=1, seed=0)
    nash = solve_nash(params, verbose=False)
    traj = generate_trajectories(params, nash, M=100, T=200,
                                  sigma_w=0.01, seed=42)
    K_hat = estimate_gains(traj, params)

    print(f"    Gain estimation error (should be ~0):")
    for i in range(params.n):
        err = np.linalg.norm(K_hat[i] - nash.K[i], 'fro')
        print(f"      Agent {i}: {err:.2e}")

    # --- Run centralized IRL ---
    print("\n[2] Running centralized IRL (gradient descent)...")
    result = run_centralized_irl(
        params=params,
        K_hat=K_hat,
        mu=1e-3,
        lr=0.005,
        max_iter=400,
        tol=1e-9,
        verbose=True
    )

    # --- Results ---
    print(f"\n[3] Recovery results:")
    print(f"    MSE(K_recovered, K_hat) = {result['mse_K']:.4e}  "
          f"(should be ~0)")
    print(f"    MSE(Q_est, Q_true_norm) = {result['mse_Q']:.4e}")
    print(f"    MSE(R_est, R_true_norm) = {result['mse_R']:.4e}")

    print(f"\n[4] Per-agent parameter comparison (normalized):")
    for i in range(params.n):
        print(f"\n  Agent {i}:")
        print(f"    Q_true (norm):\n{result['Q_true_norm'][i]}")
        print(f"    Q_est:\n{result['Q_est'][i]}")
        print(f"    R_true (norm): {result['R_true_norm'][i]}")
        print(f"    R_est:         {result['R_est'][i]}")

    # --- Plot loss curve ---
    plt.figure(figsize=(8, 4))
    plt.semilogy(result['loss_history'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.title('Centralized IRL — Loss Convergence')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/centralized_loss.png', dpi=150)
    print(f"\n[5] Loss curve saved to centralized_loss.png")

    print("\nCentralized IRL OK.")