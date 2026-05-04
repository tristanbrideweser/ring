"""
ring_experiments.py
--------------------
RING: Reward Inference on Networked Games
Supplementary experiments for the paper.

Run from your dma-irl/src/ directory:
    python ring_experiments.py

Outputs (saved to ./figures/):
    1. scalability_sweep.png          -- MSE(K) and iters vs. n_agents
    2. step_size_sensitivity.png      -- convergence heatmap: alpha x topology
    3. noise_robustness.png           -- MSE(K) vs. observation noise sigma
    4. trajectory_length.png          -- MSE(K) vs. trajectory length T
    5. price_of_decentralization.png  -- distributed vs. centralized over 20 random games
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)

from forward_sim import (
    build_game, solve_nash, generate_trajectories, estimate_gains
)
from centralized_irl import run_centralized_irl
from distributed_irl import (
    run_distributed_irl, build_graph, algebraic_connectivity
)

# ── style ─────────────────────────────────────────────────────────────────────
PALETTE = ["#2d6a9f", "#e05c2e", "#2ca02c", "#9467bd",
           "#8c564b", "#17becf", "#d4a017", "#d62728"]
plt.rcParams.update({
    "font.family":       "serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

# ── shared helpers ─────────────────────────────────────────────────────────────

def _get_K_hat(params, sigma_w=0.01, T=200, M=50, seed=42):
    """Solve Nash, generate trajectories, return estimated gains."""
    nash = solve_nash(params, verbose=False)
    traj = generate_trajectories(params, nash, M=M, T=T,
                                 sigma_w=sigma_w, seed=seed)
    return estimate_gains(traj, params), nash


def _run_dist(params, adj, K_hat, alpha=5e-3, max_iter=300, tol=1e-4):
    """Single distributed RING run. Returns (final_mse, iters, result_dict)."""
    res = run_distributed_irl(
        params=params, adj=adj, K_hat_global=K_hat,
        mu=1e-3, alpha=alpha, max_iter=max_iter, tol=tol, verbose=False
    )
    final_mse = res["mse_K_history"][-1] if res["mse_K_history"] else np.nan
    iters     = len(res["loss_history"])
    return final_mse, iters, res


def _run_cent(params, K_hat, max_iter=300):
    """Single centralized run. Returns final MSE(K)."""
    res = run_centralized_irl(
        params=params, K_hat=K_hat,
        mu=1e-3, lr=5e-3, max_iter=max_iter, tol=1e-8, verbose=False
    )
    mse = np.mean([
        np.linalg.norm(res["K_recovered"][i] - K_hat[i], "fro") ** 2
        for i in range(params.n)
    ])
    return mse


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 1 — Scalability sweep  (n = 4, 6, 8, 10)
# ═══════════════════════════════════════════════════════════════════════════════

def exp_scalability():
    print("\n" + "="*60)
    print("EXP 1: Scalability sweep")
    print("="*60)

    agent_counts = [4, 6, 8, 10]
    final_mse    = []
    iter_counts  = []

    for n in agent_counts:
        print(f"  n={n} ...", end=" ", flush=True)
        params      = build_game(n=n, nx=2, nu=1, seed=0)
        K_hat, _    = _get_K_hat(params, T=200, M=50)
        adj         = build_graph(n, "complete", seed=42)
        mse, iters, _ = _run_dist(params, adj, K_hat,
                                   alpha=5e-3, max_iter=400, tol=1e-4)
        final_mse.append(mse)
        iter_counts.append(iters)
        print(f"MSE={mse:.3e}  iters={iters}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(agent_counts, final_mse, "o-", color=PALETTE[0], lw=2, ms=7)
    ax1.set_xlabel("Number of agents $n$")
    ax1.set_ylabel("Final MSE$(K)$")
    ax1.set_title("Recovery accuracy vs. scale")
    ax1.set_yscale("log")
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax2.plot(agent_counts, iter_counts, "s-", color=PALETTE[1], lw=2, ms=7)
    ax2.set_xlabel("Number of agents $n$")
    ax2.set_ylabel("Iterations to convergence")
    ax2.set_title("Convergence speed vs. scale")
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))

    fig.suptitle("RING — Scalability (complete graph)", fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/scalability_sweep.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> figures/scalability_sweep.png")


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 2 — Step-size sensitivity  (alpha x topology heatmap)
# ═══════════════════════════════════════════════════════════════════════════════

def exp_step_size_sensitivity():
    print("\n" + "="*60)
    print("EXP 2: Step-size sensitivity")
    print("="*60)

    params   = build_game(n=4, nx=2, nu=1, seed=0)
    K_hat, _ = _get_K_hat(params, T=200, M=50)

    topologies = ["complete", "random_geometric", "ring", "star", "erdos_renyi"]
    alphas     = [0.001, 0.005, 0.01, 0.02, 0.05]

    converged_grid = np.zeros((len(topologies), len(alphas)), dtype=bool)
    mse_grid       = np.full((len(topologies), len(alphas)), np.nan)

    for ti, topo in enumerate(topologies):
        adj  = build_graph(params.n, topo, seed=42)
        lam2 = algebraic_connectivity(adj)
        print(f"  {topo:20s} (lambda_2={lam2:.2f})")
        for ai, alpha in enumerate(alphas):
            mse, iters, res = _run_dist(params, adj, K_hat,
                                        alpha=alpha, max_iter=300, tol=1e-4)
            disagree_final = (res["disagreement_history"][-1]
                              if res["disagreement_history"] else np.inf)
            conv = disagree_final < 1e-4
            converged_grid[ti, ai] = conv
            mse_grid[ti, ai]       = mse
            print(f"    alpha={alpha:.3f}  MSE={mse:.3e}  "
                  f"conv={'YES' if conv else 'NO '}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Convergence binary map
    cmap_conv = LinearSegmentedColormap.from_list("conv",
                                                   ["#d62728", "#2ca02c"])
    ax1.imshow(converged_grid.astype(float), cmap=cmap_conv,
               vmin=0, vmax=1, aspect="auto")
    ax1.set_xticks(range(len(alphas)))
    ax1.set_xticklabels([str(a) for a in alphas])
    ax1.set_yticks(range(len(topologies)))
    ax1.set_yticklabels(topologies)
    ax1.set_xlabel("Step size $\\alpha$")
    ax1.set_title("Converged? (green = yes)")
    for ti in range(len(topologies)):
        for ai in range(len(alphas)):
            ax1.text(ai, ti,
                     "Y" if converged_grid[ti, ai] else "N",
                     ha="center", va="center",
                     fontsize=11, fontweight="bold", color="white")

    # MSE heatmap
    log_mse = np.log10(np.clip(mse_grid, 1e-10, None))
    im2 = ax2.imshow(log_mse, cmap="viridis_r", aspect="auto")
    ax2.set_xticks(range(len(alphas)))
    ax2.set_xticklabels([str(a) for a in alphas])
    ax2.set_yticks(range(len(topologies)))
    ax2.set_yticklabels(topologies)
    ax2.set_xlabel("Step size $\\alpha$")
    ax2.set_title("Final $\\log_{10}$ MSE$(K)$")
    cb = fig.colorbar(im2, ax=ax2, shrink=0.85)
    cb.set_label("$\\log_{10}$ MSE")
    for ti in range(len(topologies)):
        for ai in range(len(alphas)):
            ax2.text(ai, ti, f"{mse_grid[ti, ai]:.1e}",
                     ha="center", va="center", fontsize=6.5, color="white")

    fig.suptitle("RING — Step-size sensitivity: $\\alpha$ $\\times$ topology",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/step_size_sensitivity.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> figures/step_size_sensitivity.png")


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 3 — Noise robustness  (sigma_w sweep)
# ═══════════════════════════════════════════════════════════════════════════════

def exp_noise_robustness():
    print("\n" + "="*60)
    print("EXP 3: Noise robustness")
    print("="*60)

    params   = build_game(n=4, nx=2, nu=1, seed=0)
    nash     = solve_nash(params, verbose=False)
    adj_comp = build_graph(params.n, "complete", seed=42)
    adj_ring = build_graph(params.n, "ring",     seed=42)

    noise_levels = [0.0, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5]
    res_comp, res_ring, res_cent = [], [], []

    for sigma in noise_levels:
        print(f"  sigma={sigma:.3f} ...", end=" ", flush=True)
        traj  = generate_trajectories(params, nash, M=50, T=200,
                                      sigma_w=sigma, seed=42)
        K_hat = estimate_gains(traj, params)

        mse_c, _, _ = _run_dist(params, adj_comp, K_hat,
                                 alpha=5e-3, max_iter=300, tol=1e-4)
        mse_r, _, _ = _run_dist(params, adj_ring, K_hat,
                                 alpha=1e-3, max_iter=500, tol=1e-4)
        mse_cent    = _run_cent(params, K_hat, max_iter=300)

        res_comp.append(mse_c)
        res_ring.append(mse_r)
        res_cent.append(mse_cent)
        print(f"complete={mse_c:.3e}  ring={mse_r:.3e}  cent={mse_cent:.3e}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(noise_levels, res_comp, "o-",  color=PALETTE[0], lw=2, ms=6,
            label="RING (complete, $\\lambda_2$=4)")
    ax.plot(noise_levels, res_ring, "s--", color=PALETTE[1], lw=2, ms=6,
            label="RING (ring, $\\lambda_2$=2)")
    ax.plot(noise_levels, res_cent, "^:",  color=PALETTE[2], lw=2, ms=6,
            label="Centralized baseline")
    ax.set_xlabel("Observation noise $\\sigma_w$")
    ax.set_ylabel("Final MSE$(K)$")
    ax.set_title("RING — Robustness to observation noise")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/noise_robustness.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> figures/noise_robustness.png")


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 4 — Trajectory length  (T sweep)
# ═══════════════════════════════════════════════════════════════════════════════

def exp_trajectory_length():
    print("\n" + "="*60)
    print("EXP 4: Trajectory length")
    print("="*60)

    params   = build_game(n=4, nx=2, nu=1, seed=0)
    nash     = solve_nash(params, verbose=False)
    adj_comp = build_graph(params.n, "complete", seed=42)
    adj_ring = build_graph(params.n, "ring",     seed=42)

    T_values = [25, 50, 100, 200, 500, 1000]
    res_comp, res_ring, res_cent = [], [], []

    for T in T_values:
        print(f"  T={T} ...", end=" ", flush=True)
        traj  = generate_trajectories(params, nash, M=50, T=T,
                                      sigma_w=0.01, seed=42)
        K_hat = estimate_gains(traj, params)

        mse_c, _, _ = _run_dist(params, adj_comp, K_hat,
                                 alpha=5e-3, max_iter=300, tol=1e-4)
        mse_r, _, _ = _run_dist(params, adj_ring, K_hat,
                                 alpha=1e-3, max_iter=500, tol=1e-4)
        mse_cent    = _run_cent(params, K_hat, max_iter=300)

        res_comp.append(mse_c)
        res_ring.append(mse_r)
        res_cent.append(mse_cent)
        print(f"complete={mse_c:.3e}  ring={mse_r:.3e}  cent={mse_cent:.3e}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(T_values, res_comp, "o-",  color=PALETTE[0], lw=2, ms=6,
            label="RING (complete, $\\lambda_2$=4)")
    ax.plot(T_values, res_ring, "s--", color=PALETTE[1], lw=2, ms=6,
            label="RING (ring, $\\lambda_2$=2)")
    ax.plot(T_values, res_cent, "^:",  color=PALETTE[2], lw=2, ms=6,
            label="Centralized baseline")
    ax.set_xlabel("Trajectory length $T$")
    ax.set_ylabel("Final MSE$(K)$")
    ax.set_title("RING — Data requirements: MSE vs. trajectory length")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/trajectory_length.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> figures/trajectory_length.png")


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 5 — Price of decentralization  (20 random games, mean +/- std)
# ═══════════════════════════════════════════════════════════════════════════════

def exp_price_of_decentralization():
    print("\n" + "="*60)
    print("EXP 5: Price of decentralization (20 random games)")
    print("="*60)

    N_GAMES    = 20
    topologies = ["complete", "ring", "star"]
    alphas     = {"complete": 5e-3, "ring": 1e-3, "star": 1e-3}

    dist_mses = {t: [] for t in topologies}
    cent_mses = []

    for seed in range(N_GAMES):
        print(f"  Game {seed+1:2d}/{N_GAMES} ...", end=" ", flush=True)
        try:
            params = build_game(n=4, nx=2, nu=1, seed=seed)
            K_hat, _ = _get_K_hat(params, T=200, M=50, seed=seed)
        except Exception as e:
            print(f"SKIP ({e})")
            for t in topologies:
                dist_mses[t].append(np.nan)
            cent_mses.append(np.nan)
            continue

        try:
            mse_c = _run_cent(params, K_hat, max_iter=300)
        except Exception:
            mse_c = np.nan
        cent_mses.append(mse_c)

        for topo in topologies:
            try:
                adj       = build_graph(params.n, topo, seed=42)
                mse, _, _ = _run_dist(params, adj, K_hat,
                                      alpha=alphas[topo],
                                      max_iter=400, tol=1e-4)
            except Exception:
                mse = np.nan
            dist_mses[topo].append(mse)

        print("  ".join(
            [f"cent={mse_c:.2e}"] +
            [f"{t}={dist_mses[t][-1]:.2e}" for t in topologies]
        ))

    # ── summary stats ─────────────────────────────────────────────────────────
    def _stats(arr):
        a = np.array(arr, dtype=float)
        return np.nanmean(a), np.nanstd(a)

    cent_mu, cent_std = _stats(cent_mses)
    print(f"\n  Summary (mean +/- std, {N_GAMES} games):")
    print(f"    Centralized : {cent_mu:.3e} +/- {cent_std:.3e}")
    for topo in topologies:
        mu, std = _stats(dist_mses[topo])
        print(f"    RING ({topo:8s}): {mu:.3e} +/- {std:.3e}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart: mean +/- std
    labels = ["Centralized"] + [f"RING\n({t})" for t in topologies]
    means  = [cent_mu] + [np.nanmean(dist_mses[t]) for t in topologies]
    stds   = [cent_std] + [np.nanstd(dist_mses[t])  for t in topologies]
    colors = [PALETTE[2]] + PALETTE[:len(topologies)]
    x      = np.arange(len(labels))

    bars = ax1.bar(x, means, yerr=stds, color=colors, alpha=0.82,
                   capsize=5, error_kw={"lw": 1.5})
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Mean MSE$(K)$ $\\pm$ std")
    ax1.set_title("Price of decentralization")
    ax1.set_yscale("log")
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 mean * 1.8, f"{mean:.1e}",
                 ha="center", va="bottom", fontsize=7.5)

    # Scatter: per-game centralized vs. best distributed
    best_dist = [
        min((dist_mses[t][i] for t in topologies), default=np.nan)
        for i in range(N_GAMES)
    ]
    valid = [(c, d) for c, d in zip(cent_mses, best_dist)
             if not (np.isnan(c) or np.isnan(d))]
    if valid:
        cx, dy = zip(*valid)
        ax2.scatter(cx, dy, color=PALETTE[0], alpha=0.7, s=55, zorder=3)
        lo = min(min(cx), min(dy)) * 0.5
        hi = max(max(cx), max(dy)) * 2.0
        ax2.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.5,
                 label="$y=x$ (equal performance)")
        ax2.set_xlabel("Centralized MSE$(K)$")
        ax2.set_ylabel("Best distributed MSE$(K)$")
        ax2.set_title("Per-game: centralized vs. distributed")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.legend(fontsize=8)

    fig.suptitle(
        "RING — Price of decentralization (20 random LQ Nash games)",
        fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig("figures/price_of_decentralization.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> figures/price_of_decentralization.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("RING — All supplementary experiments")
    print("="*60)

    exp_scalability()
    exp_step_size_sensitivity()
    exp_noise_robustness()
    exp_trajectory_length()
    exp_price_of_decentralization()

    print("\n" + "="*60)
    print("Done. All figures in ./figures/")
    print("="*60)