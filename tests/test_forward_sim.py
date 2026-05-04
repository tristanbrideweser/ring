import numpy as np
import pytest
from src.forward_sim import build_game, solve_nash, generate_trajectories, estimate_gains, verify_nash

def test_nash_convergence():
    params = build_game(n=2, nx=2, nu=1, seed=0)
    nash = solve_nash(params, verbose=False)
    assert nash.converged

def test_closed_loop_stable():
    params = build_game(n=2, nx=2, nu=1, seed=0)
    nash = solve_nash(params, verbose=False)
    sr = np.max(np.abs(np.linalg.eigvals(nash.A_cl)))
    assert sr < 1.0, f"Spectral radius {sr:.3f} >= 1"

def test_nash_verification():
    params = build_game(n=2, nx=2, nu=1, seed=0)
    nash = solve_nash(params, verbose=False)
    assert verify_nash(params, nash)

def test_gain_estimation_accuracy():
    params = build_game(n=2, nx=2, nu=1, seed=0)
    nash = solve_nash(params, verbose=False)
    traj = generate_trajectories(params, nash, M=50, T=100, seed=42)
    K_hat = estimate_gains(traj, params)
    for i in range(params.n):
        err = np.linalg.norm(K_hat[i] - nash.K[i], 'fro')
        assert err < 1e-10, f"Agent {i} gain error {err:.2e} too large"

@pytest.mark.parametrize("n", [2, 4])
def test_scalability(n):
    params = build_game(n=n, nx=2, nu=1, seed=1)
    nash = solve_nash(params, verbose=False)
    assert nash.converged