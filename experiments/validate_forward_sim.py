# experiments/validate_forward_sim.py
from src.forward_sim import build_game, solve_nash, generate_trajectories, estimate_gains, verify_nash

if __name__ == "__main__":
    print("=== Forward Simulator Validation ===")
    params = build_game(n=2, nx=2, nu=1, seed=0)
    nash = solve_nash(params, verbose=True)
    assert verify_nash(params, nash)
    traj = generate_trajectories(params, nash, M=50, T=100)
    K_hat = estimate_gains(traj, params)
    for i in range(params.n):
        err = __import__('numpy').linalg.norm(K_hat[i] - nash.K[i], 'fro')
        print(f"Agent {i} gain error: {err:.2e}")
    print("OK")