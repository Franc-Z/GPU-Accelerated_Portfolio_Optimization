# Import necessary libraries for numerical computing, sparse matrices, optimization, and GPU computing
using LinearAlgebra, SparseArrays, Random, Clarabel, JuMP, CUDA, Printf

# Set random seed for reproducibility
rng = Random.MersenneTwister(1)

# Model dimensions
k = 50 # Number of risk factors
n = k * 100 # Number of assets (same as the original single-period model)
T = 2 # Number of time periods in the multi-period optimization

# Generate asset-specific risk vector (idiosyncratic risks)
# Each element represents the specific risk for each asset, scaled by sqrt(k)
D = rand(rng, n) .* sqrt(k)

# Generate factor loading matrix (sparse) - maps risk factors to assets
# sprandn creates a sparse matrix with approximately 50% non-zero elements
F_T = sprandn(rng, k, n, 0.5)

# Risk aversion parameter (trade-off between risk and return)
γ = 1.0

# Generate factor covariance matrix Ω (symmetric positive definite)
# First create a random matrix, then convert it to a symmetric positive definite matrix
Ω_temp = randn(rng, k, k)
Ω = Ω_temp * Ω_temp' / k  # Ensure eigenvalues are of moderate size

# Alternative approach to generate a simpler diagonal factor covariance matrix:
# Ω = Diagonal(rand(rng, k) .* 0.5 .+ 0.5) # Diagonal elements in range [0.5, 1.0]

# Cholesky decomposition of the factor covariance matrix for optimized quadratic form calculations
Ω_chol = cholesky(Ω).L

# Generate expected returns for each asset across all time periods
mu_matrix = Matrix{Float64}(undef, n, T)
for t in 1:T
    # Random returns for each period, approximately in range [-3%, 12%]
    mu_matrix[:, t] = (-3 .+ 15 .* rand(rng, n)) ./ 100
end

# Target investment amount (same as in single-period model)
d = 1.0 

# Initial portfolio holdings (assumed to be zero)
x0 = zeros(n)

# Initialize optimization model using Clarabel solver
CUDA.@time begin
    
    model = JuMP.Model(Clarabel.Optimizer)

    # Configure solver parameters for GPU acceleration
    set_optimizer_attribute(model, "direct_solve_method", :cudss)
    set_optimizer_attribute(model, "iterative_refinement_enable", false)
    set_optimizer_attribute(model, "iterative_refinement_max_iter", 1)
    set_optimizer_attribute(model, "iterative_refinement_reltol", 1e-8)
    set_optimizer_attribute(model, "iterative_refinement_abstol", 1e-8)
    set_optimizer_attribute(model, "presolve_enable", true)
    set_optimizer_attribute(model, "static_regularization_enable", true)
    set_optimizer_attribute(model, "dynamic_regularization_enable", true)
    set_optimizer_attribute(model, "chordal_decomposition_enable", true)
    set_optimizer_attribute(model, "equilibrate_max_iter", 1)
    # Define decision variables
    @variables(model, begin
        0.1 >= x[1:T, 1:n] >= 0.0  # Portfolio weights for each asset at each time period, bounded by 0 and 0.1
        0.1 >= y[1:T, 1:k] >= 0.0  # Factor exposures at each time period
    end)
    #=
    # Set initial values for x to be proportional to positive values in mu_matrix and sum to 1.0
    for t in 1:T
        # Get positive returns for current period
        positive_returns = max.(mu_matrix[:, t], 0.0)
        
        # If there are positive returns, normalize them to sum to 1.0
        if sum(positive_returns) > 0
            initial_weights = positive_returns ./ sum(positive_returns)
        else
            # If no positive returns, use uniform weights
            initial_weights = fill(1.0/n, n)
        end
        
        # Set initial values for x variables
        for i in 1:n
            set_start_value(x[t, i], min(initial_weights[i], 0.1))
        end
    end
    =#
    # Define transaction volume variables for implementing trading costs
    @variable(model, z[1:T, 1:n] >= 0)  # Transaction volume variables (absolute changes in weights)

    # Transaction volume constraints for the first period (comparing with initial holdings)
    @constraint(model, z[1, :] .>= x[1, :] .- x0)     # Buying constraint
    @constraint(model, z[1, :] .>= x0 .- x[1, :])     # Selling constraint

    # Transaction volume constraints for subsequent periods
    for t in 2:T
        @constraint(model, z[t, :] .>= x[t, :] .- x[t-1, :])     # Buying constraint
        @constraint(model, z[t, :] .>= x[t-1, :] .- x[t, :])     # Selling constraint
    end

    # Transaction cost rate - applied linearly to the transaction volume
    transaction_cost_rate = 0.002  # Can be adjusted based on market conditions

    # Budget constraint for the first period
    # Sum of weights equals target investment amount plus initial holdings
    @constraint(model, sum(x[1, :]) == d + sum(x0))

    # Budget constraints for subsequent periods
    # Maintain constant investment amount across periods
    for t in 2:T
        @constraint(model, sum(x[t, :]) == sum(x[t-1, :]))
    end

    # Factor exposure constraints - link portfolio weights to factor exposures
    for t in 1:T
        @constraint(model, y[t, :] .== F_T * x[t, :])
    end

    # Objective function: minimize the weighted sum of:
    # 1. Asset-specific risk (x[t]' * D * x[t])
    # 2. Factor risk (y[t]' * Ω * y[t])
    # 3. Negative expected return scaled by risk aversion (-1/γ * mu_matrix[:, t]' * x[t, :])
    # 4. Transaction costs (transaction_cost_rate * sum(z[t, :]))
    # Summed across all time periods
    @objective(model, Min, sum(dot(x[t,:], D .* x[t, :]) + 
                            dot(y[t,:], Ω * y[t, :]) -
                            (1 / γ) * (dot(mu_matrix[:, t], x[t, :]) +
                            transaction_cost_rate * sum(z[t, :]))
                            for t in 1:T)
    )
    optimize!(model)
end
# Allow scalar operations in CUDA environment and solve the model
CUDA.@allowscalar begin    
    # Solve the optimization problem
        
    # Extract optimal solutions
    x_opt = value.(x)    # Optimal portfolio weights
    y_opt = value.(y)    # Optimal factor exposures
    
    # Calculate expected returns for each period based on optimal weights
    expected_returns = [dot(mu_matrix[:, t], x_opt[t, :]) for t in 1:T]

    # Print results
    println("Optimal Objective Value = ", objective_value(model))
    println("\nFactor Covariance Matrix Ω:")
    #display(Ω)
    println("\nAsset Allocation Results by Period:")
    
    # Display allocation results for each period
    for t in 1:T
        println("\nPeriod $t:")
        println("Expected Return = ", expected_returns[t])
        println("Top 10 Largest Weights and Their Indices:")
        # Find indices of the 10 largest weights
        top10_idx = sortperm(x_opt[t, :], rev=true)[1:10]
        for (i, idx) in enumerate(top10_idx)
            @printf("Rank %2d: Asset %4d, Weight = %.6f\n", i, idx, x_opt[t, idx])
        end
    end
end
