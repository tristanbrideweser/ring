"""
distributed_irl.py
------------------
Stage 3 of DMA-IRL: Distributed Inverse Problem

Each observer node j holds local trajectory data from agents in its
neighborhood N_j. Nodes collaboratively recover all agents' cost parameters
using the gradient tracking algorithm (equations 12-13 from the paper):

    theta_j^{t+1} = sum_{l in N_j} w_jl * theta_l^t - alpha * s_j^t
    s_j^{t+1}     = sum_{l in N_j} w_jl * s_l^t
                    + grad f_j(theta_j^{t+1}) - grad f_j(theta_j^t)

where w_jl are Metropolis-Hastings weights from the communication graph G.

Usage:
    from distributed_irl import run_distributed_irl, build_graph, run_topology_sweep
"""

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

from forward_sim import (GameParams, build_game, solve_nash,
                          generate_trajectories, estimate_gains,
                          _stacked_closed_loop)
from centralized_irl import (run_centralized_irl, compute_K_and_Acl,
                               lyapunov_gradient, normalize_params,
                               project_psd, project_pd)


# ---------------------------------------------------------------------------
# Graph construction & weight matrix
# ---------------------------------------------------------------------------

def build_graph(n: int, topology: str, seed: int = 0) -> np.ndarray:
    """
    Build adjacency matrix for n nodes with given topology.

    Topologies: 'complete', 'ring', 'star', 'random_geometric', 'erdos_renyi'

    Returns
    -------
    adj : (n, n) binary adjacency matrix (symmetric, no self-loops)
    """
    rng = np.random.default_rng(seed)

    if topology == 'complete':
        G = nx.complete_graph(n)

    elif topology == 'ring':
        G = nx.cycle_graph(n)

    elif topology == 'star':
        G = nx.star_graph(n - 1)   # n-1 leaves + 1 center = n nodes

    elif topology == 'random_geometric':
        # Place nodes randomly in unit square, connect if within radius r
        r = np.sqrt(2.5 * np.log(n) / n) + 0.1   # ensures connectivity whp
        while True:
            pos = {i: rng.uniform(0, 1, 2) for i in range(n)}
            G = nx.random_geometric_graph(n, r, pos=pos, seed=int(rng.integers(1e6)))
            if nx.is_connected(G):
                break

    elif topology == 'erdos_renyi':
        p = np.log(n) / n + 0.2    # above connectivity threshold
        while True:
            G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(1e6)))
            if nx.is_connected(G):
                break

    else:
        raise ValueError(f"Unknown topology: {topology}")

    return nx.to_numpy_array(G, dtype=float)


def metropolis_hastings_weights(adj: np.ndarray) -> np.ndarray:
    """
    Compute Metropolis-Hastings doubly stochastic weight matrix from adjacency.

    w_jl = 1 / (1 + max(d_j, d_l))  for (j,l) in E
    w_jj = 1 - sum_{l in N_j} w_jl

    Returns W: (n, n) doubly stochastic matrix
    """
    n = adj.shape[0]
    degrees = adj.sum(axis=1)
    W = np.zeros((n, n))

    for j in range(n):
        for l in range(n):
            if j != l and adj[j, l] > 0:
                W[j, l] = 1.0 / (1.0 + max(degrees[j], degrees[l]))
        W[j, j] = 1.0 - W[j, :].sum()

    return W


def algebraic_connectivity(adj: np.ndarray) -> float:
    """
    Compute algebraic connectivity lambda_2(G) — the Fiedler value.
    Second-smallest eigenvalue of the graph Laplacian L = D - A.
    """
    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj
    eigvals = np.sort(np.linalg.eigvalsh(L))
    return float(eigvals[1])   # lambda_2


# ---------------------------------------------------------------------------
# Local objective and gradient for one observer node
# ---------------------------------------------------------------------------

def local_gradient(params: GameParams,
                    node_j: int,
                    neighborhood: list,
                    theta_j: dict,
                    K_hat_local: dict,
                    mu: float = 1e-3) -> tuple:
    """
    Compute local loss f_j(theta) and its gradient at node j.

    f_j(theta) = sum_{i in N_j} || K_i(theta) - K_hat_i^{(j)} ||_F^2
               + mu * sum_i (||Q_i||_F^2 + ||R_i||_F^2)

    Node j only has K_hat for agents in its neighborhood, but theta
    contains estimates for ALL agents (shared via consensus).

    Parameters
    ----------
    params       : GameParams
    node_j       : index of this observer node
    neighborhood : list of agent indices observed by node j
    theta_j      : dict with 'Q': list of n Q matrices, 'R': list of n R matrices
    K_hat_local  : dict {agent_idx: K_hat_i} for agents in neighborhood
    mu           : regularization weight

    Returns
    -------
    loss    : scalar local loss
    grad_Q  : list of n gradient matrices for Q
    grad_R  : list of n gradient matrices for R
    """
    n, nx, nu = params.n, params.nx, params.nu
    Q_list = theta_j['Q']
    R_list = theta_j['R']

    # Forward pass with current theta
    try:
        K_curr, _, A_cl = compute_K_and_Acl(params, Q_list, R_list)
    except Exception:
        # Return zero gradients if Nash solve fails
        return 0.0, [np.zeros((nx, nx)) for _ in range(n)], \
                    [np.zeros((nu, nu)) for _ in range(n)]

    # Local loss: only over observed agents
    loss = 0.0
    for i in neighborhood:
        loss += np.linalg.norm(K_curr[i] - K_hat_local[i], 'fro') ** 2
    for i in range(n):
        loss += mu * (np.linalg.norm(Q_list[i], 'fro') ** 2 +
                      np.linalg.norm(R_list[i], 'fro') ** 2)

    # Gradients: nonzero only for agents in neighborhood (others get reg only)
    grad_Q = [2 * mu * Q_list[i] for i in range(n)]
    grad_R = [2 * mu * R_list[i] for i in range(n)]

    for i in neighborhood:
        gQ, gR = lyapunov_gradient(
            A_cl=A_cl, K_i=K_curr[i], K_hat_i=K_hat_local[i],
            B=params.B, R_i=R_list[i],
            nx=nx, nu=nu, agent_idx=i, n_agents=n
        )
        grad_Q[i] += gQ
        grad_R[i] += gR

    return loss, grad_Q, grad_R


# ---------------------------------------------------------------------------
# Parameter vector packing / unpacking
# ---------------------------------------------------------------------------

def pack_theta(Q_list: list, R_list: list) -> np.ndarray:
    """Flatten all Q_i and R_i into a single vector."""
    parts = []
    for Q in Q_list:
        parts.append(Q.flatten())
    for R in R_list:
        parts.append(R.flatten())
    return np.concatenate(parts)


def unpack_theta(vec: np.ndarray, n: int, nx: int, nu: int) -> tuple:
    """Unpack vector back into Q_list, R_list."""
    Q_list, R_list = [], []
    idx = 0
    for _ in range(n):
        Q_list.append(vec[idx:idx + nx * nx].reshape(nx, nx))
        idx += nx * nx
    for _ in range(n):
        R_list.append(vec[idx:idx + nu * nu].reshape(nu, nu))
        idx += nu * nu
    return Q_list, R_list


def project_theta(Q_list: list, R_list: list, n: int) -> tuple:
    """Project and normalize all Q_i, R_i."""
    Q_out, R_out = [], []
    for i in range(n):
        q = project_psd(Q_list[i])
        r = project_pd(R_list[i])
        q, r = normalize_params(q, r)
        Q_out.append(q)
        R_out.append(r)
    return Q_out, R_out


# ---------------------------------------------------------------------------
# Main distributed IRL algorithm
# ---------------------------------------------------------------------------

def run_distributed_irl(params: GameParams,
                         adj: np.ndarray,
                         K_hat_global: list,
                         mu: float = 1e-3,
                         alpha: float = 5e-3,
                         max_iter: int = 500,
                         tol: float = 1e-8,
                         verbose: bool = True) -> dict:
    """
    Run distributed IRL via gradient tracking over communication graph G.

    Each node j observes agents in its closed neighborhood N_j and
    collaboratively recovers all agents' cost parameters.

    Parameters
    ----------
    params        : GameParams
    adj           : (n, n) adjacency matrix of communication graph
    K_hat_global  : list of n ground-truth estimated gains (for evaluation)
    mu            : regularization weight
    alpha         : step size
    max_iter      : maximum iterations
    tol           : convergence on consensus disagreement
    verbose       : print progress

    Returns
    -------
    dict with keys: theta_avg, loss_history, disagreement_history,
                    mse_K_history, K_recovered
    """
    n, nx, nu = params.n, params.nx, params.nu
    W = metropolis_hastings_weights(adj)
    lam2 = algebraic_connectivity(adj)

    if verbose:
        print(f"  Graph: n={n}, lambda_2={lam2:.4f}, "
              f"spectral gap={1 - np.sort(np.abs(np.linalg.eigvals(W)))[n-2]:.4f}")

    # Each node j observes agents in its closed neighborhood
    neighborhoods = []
    for j in range(n):
        nbrs = [j] + [i for i in range(n) if adj[j, i] > 0]
        neighborhoods.append(nbrs)

    # Local K_hat: node j only sees K_hat for its neighbors
    K_hat_local = []
    for j in range(n):
        local = {i: K_hat_global[i] for i in neighborhoods[j]}
        K_hat_local.append(local)

    # --- Initialize: all nodes start from same perturbed identity ---
    rng = np.random.default_rng(7)
    theta = []   # theta[j] = {'Q': [...], 'R': [...]}
    for j in range(n):
        Q0 = []
        R0 = []
        for i in range(n):
            q0 = np.eye(nx) + 0.2 * rng.standard_normal((nx, nx))
            Q0.append(project_psd(q0 + q0.T))
            r0 = np.eye(nu) + 0.05 * rng.standard_normal((nu, nu))
            R0.append(project_pd(r0 + r0.T))
        Q0, R0 = project_theta(Q0, R0, n)
        theta.append({'Q': Q0, 'R': R0})

    # Initialize gradient trackers s_j = grad f_j(theta_j^0)
    s = []
    for j in range(n):
        _, gQ, gR = local_gradient(params, j, neighborhoods[j],
                                    theta[j], K_hat_local[j], mu)
        s.append({'Q': gQ, 'R': gR})

    # Pack everything into vectors for consensus step
    theta_vec = np.array([pack_theta(theta[j]['Q'], theta[j]['R'])
                           for j in range(n)])   # (n, d)
    s_vec = np.array([pack_theta(s[j]['Q'], s[j]['R'])
                       for j in range(n)])        # (n, d)

    loss_history = []
    disagreement_history = []
    mse_K_history = []

    # Cache grad at theta^0 (already computed for s init — reuse it)
    grad_old_vec_cache = s_vec.copy()   # s^0 = grad f(theta^0)

    for t in range(max_iter):
        # ---- Consensus step on theta ----
        theta_vec_new = W @ theta_vec - alpha * s_vec   # (n, d)

        # Project each node's estimate back onto feasible set
        for j in range(n):
            Q_j, R_j = unpack_theta(theta_vec_new[j], n, nx, nu)
            Q_j, R_j = project_theta(Q_j, R_j, n)
            theta_vec_new[j] = pack_theta(Q_j, R_j)

        # ---- Gradient tracking update ----
        # Compute new gradients at theta^{t+1}
        grad_new_vec = np.zeros_like(s_vec)
        local_losses = []
        for j in range(n):
            Q_j, R_j = unpack_theta(theta_vec_new[j], n, nx, nu)
            theta_j_new = {'Q': Q_j, 'R': R_j}
            loss_j, gQ_new, gR_new = local_gradient(
                params, j, neighborhoods[j], theta_j_new,
                K_hat_local[j], mu)
            grad_new_vec[j] = pack_theta(gQ_new, gR_new)
            local_losses.append(loss_j)

        # s^{t+1} = W s^t + grad(theta^{t+1}) - grad(theta^t)
        # Use cached grad(theta^t) — avoids recomputing Nash solve twice per iter
        s_vec_new = W @ s_vec + grad_new_vec - grad_old_vec_cache
        grad_old_vec_cache = grad_new_vec.copy()

        # ---- Metrics ----
        # Consensus disagreement: max ||theta_j - theta_mean||
        theta_mean = theta_vec_new.mean(axis=0)
        disagreement = np.max(np.linalg.norm(
            theta_vec_new - theta_mean[None, :], axis=1))
        disagreement_history.append(disagreement)

        # Global loss (using mean parameter estimate)
        total_loss = np.mean(local_losses)
        loss_history.append(total_loss)

        # MSE on K recovery using mean estimate
        Q_mean, R_mean = unpack_theta(theta_mean, n, nx, nu)
        Q_mean, R_mean = project_theta(Q_mean, R_mean, n)
        try:
            K_rec, _, _ = compute_K_and_Acl(params, Q_mean, R_mean)
            mse_K = np.mean([np.linalg.norm(K_rec[i] - K_hat_global[i], 'fro') ** 2
                             for i in range(n)])
        except Exception:
            mse_K = float('nan')
        mse_K_history.append(mse_K)

        if verbose and t % 50 == 0:
            print(f"  Iter {t:4d} | Loss = {total_loss:.4e} | "
                  f"Disagreement = {disagreement:.4e} | MSE(K) = {mse_K:.4e}")

        # Convergence: consensus reached and loss stable
        if disagreement < tol and t > 20:
            if verbose:
                print(f"  Converged at iter {t} "
                      f"(disagreement = {disagreement:.2e})")
            break

        # Advance
        theta_vec = theta_vec_new
        s_vec = s_vec_new

    # Final result: average over all nodes
    theta_final_mean = theta_vec.mean(axis=0)
    Q_final, R_final = unpack_theta(theta_final_mean, n, nx, nu)
    Q_final, R_final = project_theta(Q_final, R_final, n)

    try:
        K_recovered, _, _ = compute_K_and_Acl(params, Q_final, R_final)
    except Exception:
        K_recovered = None

    return {
        'Q_est': Q_final,
        'R_est': R_final,
        'K_recovered': K_recovered,
        'loss_history': loss_history,
        'disagreement_history': disagreement_history,
        'mse_K_history': mse_K_history,
        'lambda2': lam2,
        'W': W,
    }


# ---------------------------------------------------------------------------
# Topology sweep
# ---------------------------------------------------------------------------

def run_topology_sweep(params: GameParams,
                        K_hat: list,
                        topologies: list = None,
                        mu: float = 1e-3,
                        alpha: float = 5e-3,
                        max_iter: int = 300,
                        verbose: bool = False) -> dict:
    """
    Run distributed IRL across multiple graph topologies and record
    convergence rate vs algebraic connectivity lambda_2(G).

    Returns
    -------
    dict mapping topology name -> result dict (with lambda2, mse_K_history, etc.)
    """
    if topologies is None:
        topologies = ['complete', 'ring', 'star',
                      'random_geometric', 'erdos_renyi']

    results = {}
    for topo in topologies:
        print(f"\n  [{topo}]")
        adj = build_graph(params.n, topo, seed=42)
        res = run_distributed_irl(
            params=params,
            adj=adj,
            K_hat_global=K_hat,
            mu=mu,
            alpha=alpha,
            max_iter=max_iter,
            verbose=verbose
        )
        final_mse_K = res['mse_K_history'][-1] if res['mse_K_history'] else float('nan')
        print(f"    lambda_2 = {res['lambda2']:.4f} | "
              f"final MSE(K) = {final_mse_K:.4e} | "
              f"iters = {len(res['loss_history'])}")
        results[topo] = res

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_convergence_vs_topology(sweep_results: dict,
                                   save_path: str = 'topology_sweep.png'):
    """Plot MSE(K) convergence curves for each topology."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(sweep_results)))

    # Left: convergence curves
    ax = axes[0]
    for (topo, res), c in zip(sweep_results.items(), colors):
        mse = res['mse_K_history']
        ax.semilogy(mse, label=f"{topo} (λ₂={res['lambda2']:.3f})", color=c)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE(K) — log scale')
    ax.set_title('Distributed IRL Convergence by Topology')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: final MSE(K) vs lambda_2
    ax2 = axes[1]
    lam2s = [res['lambda2'] for res in sweep_results.values()]
    final_mses = [res['mse_K_history'][-1] if res['mse_K_history']
                  else float('nan') for res in sweep_results.values()]
    topos = list(sweep_results.keys())

    for i, (topo, lam2, mse) in enumerate(zip(topos, lam2s, final_mses)):
        ax2.scatter(lam2, mse, color=colors[i], s=100, zorder=5)
        ax2.annotate(topo, (lam2, mse),
                     textcoords='offset points', xytext=(5, 5), fontsize=8)

    ax2.set_xlabel('Algebraic Connectivity λ₂(G)')
    ax2.set_ylabel('Final MSE(K)')
    ax2.set_title('Convergence Quality vs Graph Connectivity')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")


def plot_disagreement(result: dict, save_path: str = 'disagreement.png'):
    """Plot consensus disagreement over iterations."""
    plt.figure(figsize=(7, 4))
    plt.semilogy(result['disagreement_history'])
    plt.xlabel('Iteration')
    plt.ylabel('Max ||θ_j - θ_mean|| (log scale)')
    plt.title('Consensus Disagreement — Distributed IRL')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")


def plot_distributed_vs_centralized(dist_result: dict,
                                     cent_result: dict,
                                     save_path: str = 'dist_vs_cent.png'):
    """Compare distributed and centralized MSE(K) convergence."""
    plt.figure(figsize=(7, 4))
    plt.semilogy(dist_result['mse_K_history'], label='Distributed (complete graph)')
    cent_mse = [np.mean([np.linalg.norm(
        cent_result['K_recovered'][i] - cent_result['K_recovered'][i], 'fro') ** 2
        for i in range(len(cent_result['K_recovered']))])] * len(dist_result['mse_K_history'])
    final_cent = cent_result['mse_K']
    plt.axhline(final_cent, color='red', linestyle='--',
                label=f'Centralized baseline (MSE={final_cent:.2e})')
    plt.xlabel('Iteration')
    plt.ylabel('MSE(K) — log scale')
    plt.title('Distributed vs Centralized IRL')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Validation run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("DMA-IRL Distributed IRL — Validation Run")
    print("=" * 60)

    # --- Setup ---
    print("\n[1] Building game and generating data...")
    params = build_game(n=4, nx=2, nu=1, seed=0)
    nash = solve_nash(params, verbose=False)
    traj = generate_trajectories(params, nash, M=100, T=200,
                                  sigma_w=0.01, seed=42)
    K_hat = estimate_gains(traj, params)

    # --- Test on complete graph first (easiest, should match centralized) ---
    print("\n[2] Distributed IRL on complete graph (n=4)...")
    adj_complete = build_graph(params.n, 'complete')
    result_complete = run_distributed_irl(
        params=params,
        adj=adj_complete,
        K_hat_global=K_hat,
        mu=1e-3,
        alpha=5e-3,
        max_iter=400,
        verbose=True
    )
    print(f"\n  Final MSE(K) = {result_complete['mse_K_history'][-1]:.4e}")

    # --- Centralized baseline for comparison ---
    print("\n[3] Running centralized baseline...")
    cent_result = run_centralized_irl(
        params=params, K_hat=K_hat,
        mu=1e-3, lr=5e-3, max_iter=400,
        tol=1e-9, verbose=False
    )
    print(f"  Centralized MSE(K) = {cent_result['mse_K']:.4e}")

    # --- Topology sweep ---
    print("\n[4] Topology sweep...")
    sweep = run_topology_sweep(
        params=params, K_hat=K_hat,
        mu=1e-3, alpha=5e-3, max_iter=300
    )

    # --- Plots ---
    print("\n[5] Generating plots...")
    plot_convergence_vs_topology(sweep, save_path='topology_sweep.png')
    plot_disagreement(result_complete, save_path='disagreement.png')
    plot_distributed_vs_centralized(result_complete, cent_result,
                                     save_path='dist_vs_cent.png')

    # --- Summary table ---
    print("\n[6] Summary:")
    print(f"  {'Topology':<20} {'lambda_2':>10} {'Final MSE(K)':>14} {'Iters':>8}")
    print("  " + "-" * 56)
    for topo, res in sweep.items():
        final = res['mse_K_history'][-1] if res['mse_K_history'] else float('nan')
        print(f"  {topo:<20} {res['lambda2']:>10.4f} {final:>14.4e} "
              f"{len(res['loss_history']):>8}")

    print("\nDistributed IRL OK.")