import numpy as np
from mosek.fusion import Model, Domain, Expr, ObjectiveSense
from time import time
np.random.seed(1)

# Parameters
k = 50
n = k * 100
T = 10
gamma = 1.0
d = 1.0
x0 = np.zeros(n)

# Generate data
D_diag = np.random.rand(n) * np.sqrt(k)
D_sqrt = np.sqrt(D_diag)
F = np.random.randn(n, k) * (np.random.rand(n, k) < 0.5)

# Generate factor covariance matrix Ω (symmetric positive definite matrix) - consistent with Julia code
Omega_temp = np.random.randn(k, k)
Omega = (Omega_temp @ Omega_temp.T) / k  # Ensure eigenvalues are of moderate size

# Cholesky decomposition of factor covariance matrix for optimized quadratic form calculations
Omega_chol = np.linalg.cholesky(Omega)

mu_matrix = (3.0 + 9.0 * np.random.rand(n, T)) / 100.0

with Model("multi_periods") as M:
    M.setSolverParam("numThreads", 1)
    M.setSolverParam("presolveUse", "off")
    # Define variables as matrices
    x = M.variable("x", [T, n], Domain.inRange(0.0, 0.1))
    y = M.variable("y", [T, k], Domain.inRange(0.0, 0.1))
    
    # Add transaction volume variables - consistent with Julia code
    z = M.variable("z", [T, n], Domain.greaterThan(0.0))

    # Scalar variables alpha[t], beta[t]
    # alpha[t] represents asset-specific risk x[t]'*D*x[t]
    # beta[t] represents factor risk y[t]'*Omega*y[t]
    alpha = [M.variable() for _ in range(T)]
    beta = [M.variable() for _ in range(T)]

    # Add transaction volume constraints for the first period - consistent with Julia code
    for i in range(n):
        # z[0,i] >= x[0,i] - x0[i]
        M.constraint(Expr.sub(z.index(0, i), Expr.sub(x.index(0, i), x0[i])), Domain.greaterThan(0.0))
        # z[0,i] >= x0[i] - x[0,i]
        M.constraint(Expr.sub(z.index(0, i), Expr.sub(x0[i], x.index(0, i))), Domain.greaterThan(0.0))
    
    # Add transaction volume constraints for subsequent periods - consistent with Julia code
    for t in range(1, T):
        for i in range(n):
            # z[t,i] >= x[t,i] - x[t-1,i]
            M.constraint(Expr.sub(z.index(t, i), Expr.sub(x.index(t, i), x.index(t-1, i))), Domain.greaterThan(0.0))
            # z[t,i] >= x[t-1,i] - x[t,i]
            M.constraint(Expr.sub(z.index(t, i), Expr.sub(x.index(t-1, i), x.index(t, i))), Domain.greaterThan(0.0))

    # Define transaction cost rate - consistent with Julia code
    transaction_cost_rate = 0.002

    # Build constraints and rotated cones
    objective_terms = []
    for t in range(T):
        # Constraint y[t] = F' * x[t] - consistent with Julia code
        x_t = x.slice([t, 0], [t+1, n])
        for j in range(k):
            M.constraint(
                Expr.sub(
                    y.index(t, j),
                    Expr.dot(F[:, j], x_t)
                ),
                Domain.equalsTo(0.0)
            )
        
        # Budget constraints to establish time sequence relationships
        if t == 0:
            # First period: use the given initial weights x0
            M.constraint(Expr.sum(x.slice([0, 0], [1, n])), 
                         Domain.equalsTo(d + np.sum(x0)))
        else:
            # Subsequent periods: maintain the same total investment as the previous period
            M.constraint(
                Expr.sub(
                    Expr.sum(x.slice([t, 0], [t+1, n])),
                    Expr.sum(x.slice([t-1, 0], [t, n]))
                ),
                Domain.equalsTo(0.0)
            )
        
        # Handle asset-specific risk x[t]'*D*x[t] using rotated second-order cone
        s_t_expr = []
        for i in range(n):
            s_t_expr.append(Expr.mul(D_sqrt[i], x.index(t, i)))
        
        M.constraint(Expr.vstack(alpha[t], 1.0, Expr.vstack(s_t_expr)), 
                    Domain.inRotatedQCone())
        
        # Handle factor risk y[t]'*Omega*y[t] using rotated second-order cone
        p_t_expr = []
        
        for j in range(k):
            # Optimization: Cholesky matrix is lower triangular, only consider non-zero elements
            row_expr = Expr.constTerm(0.0)
            for j2 in range(j+1):  # Only consider lower triangular part
                if Omega_chol[j, j2] != 0.0:  # Skip zero elements
                    row_expr = Expr.add(row_expr, 
                                      Expr.mul(Omega_chol[j, j2], 
                                              y.index(t, j2)))
            p_t_expr.append(row_expr)
        
        M.constraint(Expr.vstack(beta[t], 1.0, Expr.vstack(p_t_expr)), 
                    Domain.inRotatedQCone())

        # Objective function terms: add transaction costs - consistent with Julia code
        # alpha[t] + beta[t] - (1/gamma) * mu[t]'*x[t] + transaction_cost_rate * sum(z[t,:])
        sum_mu_x = Expr.dot(mu_matrix[:, t], x_t)
        sum_z_t = Expr.sum(z.slice([t, 0], [t+1, n]))
        objective_terms.append(Expr.add(
            Expr.sub(
                Expr.add(alpha[t], beta[t]),
                Expr.mul(1.0/gamma, sum_mu_x)
            ),
            Expr.mul(transaction_cost_rate, sum_z_t)
        ))

    # Overall objective function
    obj_expr = objective_terms[0]
    for t in range(1, T):
        obj_expr = Expr.add(obj_expr, objective_terms[t])
    
    M.objective(ObjectiveSense.Minimize, obj_expr)
    start_time = time()
    M.solve()
    print("Time elapsed =", time() - start_time)
    
    # Extract results
    x_sol = np.zeros((T, n))
    y_sol = np.zeros((T, k))
    z_sol = np.zeros((T, n))
    
    for t in range(T):
        for i in range(n):
            x_sol[t, i] = x.index(t, i).level()[0]
            z_sol[t, i] = z.index(t, i).level()[0]
        for j in range(k):
            y_sol[t, j] = y.index(t, j).level()[0]

    # Print results
    obj_val = M.primalObjValue()
    print("Optimal Objective Value =", obj_val)
    print("\nFactor Covariance Matrix Ω (partial display):")
    print(Omega[:5, :5])  # Only display a subset
    
    print("\nAsset Allocation Results by Period:")
    for t in range(T):
        expected_return = np.dot(mu_matrix[:, t], x_sol[t, :])
        print(f"\nPeriod {t+1}:")
        print(f"Expected Return = {expected_return:.6f}")
        
        # Add transaction volume information
        total_transaction = np.sum(z_sol[t, :])
        print(f"Total Transaction Volume = {total_transaction:.6f}")
        print(f"Transaction Cost = {transaction_cost_rate * total_transaction:.6f}")
        
        print("Top 10 Largest Weights and Their Indices:")
        top10_idx = np.argsort(-x_sol[t, :])[:10]
        for rank, idx in enumerate(top10_idx, start=1):
            print(f"Rank {rank:2d}: Asset {idx:4d}, Weight = {x_sol[t,idx]:.6f}")
