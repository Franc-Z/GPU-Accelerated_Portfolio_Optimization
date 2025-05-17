using LinearAlgebra, JuMP, Clarabel, PythonCall
import CUDA
import PythonCall: pyconvert
CUDA.allowscalar(true)

MyFloat = Float64

mutable struct PortfolioModel
    n_assets::Int
    n_style::Int
    λ_risk::MyFloat
    x0::Vector{MyFloat}
    y0::Vector{MyFloat}
    cov::Matrix{MyFloat}
    expo_T::Matrix{MyFloat}
    bias::Vector{MyFloat}
    cost::Vector{MyFloat}
    return_ratio::Vector{MyFloat}
    model::Model
    new_q::Union{CUDA.CuArray{MyFloat}, Nothing} # New Q matrix for the solver
    new_b::Union{CUDA.CuArray{MyFloat}, Nothing} # New b vector for the solver
end

# 构造函数
function CreatePortfolioModel(n_assets::Int, n_style::Int, λ_risk::MyFloat, 
                      py_x0::PyVector{MyFloat}, py_cov::PyMatrix{MyFloat}, 
                      py_expo::PyMatrix{MyFloat}, py_bias::PyVector{MyFloat}, 
                      py_cost::PyVector{MyFloat}, py_u_cpu::PyVector{MyFloat})
    x0 = pyconvert(Vector{MyFloat}, py_x0)
    cov = pyconvert(Matrix{MyFloat}, py_cov)
       
    expo = pyconvert(Matrix{MyFloat}, py_expo)
    bias = pyconvert(Vector{MyFloat}, py_bias)
    
    cost = pyconvert(Vector{MyFloat}, py_cost)
    return_ratio = pyconvert(Vector{MyFloat}, py_u_cpu)
    
    model = Model(Clarabel.Optimizer)
    set_optimizer_attribute(model, "direct_solve_method", :cudss)       # :cudssmixed or :cudss
    set_optimizer_attribute(model, "verbose", true)
    set_optimizer_attribute(model, "iterative_refinement_enable", false)
    set_optimizer_attribute(model, "presolve_enable", false)
    set_optimizer_attribute(model, "static_regularization_enable", true)
    set_optimizer_attribute(model, "dynamic_regularization_enable", true)
    set_optimizer_attribute(model, "chordal_decomposition_enable", true)
    set_optimizer_attribute(model, "equilibrate_enable", true)
    return PortfolioModel(n_assets, n_style, λ_risk, x0, zeros(MyFloat, n_style), cov, expo', bias, cost, return_ratio, model, nothing, nothing)
end

function setup_model!(pm::PortfolioModel)
    @variable(pm.model, x[1:pm.n_assets])
    set_start_value.(x, pm.x0)
    @variable(pm.model, y[1:pm.n_style])
    @variable(pm.model, t >= 0.0)
    @variable(pm.model, x_buy_sell[1:pm.n_assets])

    # 模型约束
    @constraint(pm.model, y .== pm.expo_T * x)
    @constraint(pm.model, 0.1 .>= x .>= 0.0)
    @constraint(pm.model, sum(x) == 1.0)
    
    # 根据实际n_style设置约束
    
    @constraint(pm.model, 3.0 .>= y[1:min(10,pm.n_style)] .>= -3.0)
    if pm.n_style > 10
        @constraint(pm.model, 0.1 .>= y[11:min(40,pm.n_style)] .>= 0.0)
    end
    if pm.n_style >= 41
        @constraint(pm.model, y[41] == 1.0)
    end
    
    @constraint(pm.model, x_buy_sell .>= 0.0)
    @constraint(pm.model, x_buy_sell .>= x - pm.x0)
    @constraint(pm.model, x_buy_sell .>= pm.x0 - x)       
    @objective(pm.model, Min, dot(y, pm.cov*y) + dot(x, pm.bias.*x) + (1 / pm.λ_risk) * (dot(pm.cost, x_buy_sell) - dot(pm.return_ratio, x)))
end

function initial_solve!(pm::PortfolioModel)
    CUDA.@time optimize!(pm.model)  
    status = termination_status(pm.model)
    if status != MOI.OPTIMAL && status != MOI.ALMOST_OPTIMAL
        println("Initial solve failed")
    end  
    pm.new_q = CUDA.similar(pm.model.moi_backend.optimizer.model.optimizer.solver.data.q, MyFloat)
    pm.new_b = CUDA.similar(pm.model.moi_backend.optimizer.model.optimizer.solver.data.b, MyFloat)
end

# --- Update Return Data for Subsequent Solves ---
function update_return_ratio!(pm::PortfolioModel, py_u_cpu::PyVector{MyFloat})
    # Convert new return data and update the model struct
    pm.return_ratio = pyconvert(Vector{MyFloat}, py_u_cpu)
    println("Expected return ratios updated.")
end

# --- Find the index range of the x variable ---
function find_indice_in_q(pm::PortfolioModel, target_value::MyFloat)
    # 将 CUDA 稀疏矩阵转换到 CPU 后再进行查找
    CUDA.copyto!(pm.new_q, pm.model.moi_backend.optimizer.model.optimizer.solver.data.q)
    if pm.my_solver.settings.equilibrate_enable
        @. pm.new_q *= pm.model.moi_backend.optimizer.model.optimizer.solver.data.equilibration.dinv / pm.model.moi_backend.optimizer.model.optimizer.solver.data.equilibration.c
        #@. pm.new_b *= pm.model.moi_backend.optimizer.model.optimizer.solver.data.equilibration.einv
    end
    cpu_q = Array(pm.new_q)
    indices = findall(x -> isapprox(x, target_value), cpu_q)
    println("Indices of elements approximately equal to $target_value: ", indices)
end

# --- Re-solve the Optimization Problem After Updates ---
function resolve!(pm::PortfolioModel)
    println("Starting re-solve...")
    # --- Update Warm-start Values and Constraints ---
    my_solver = pm.model.moi_backend.optimizer.model.optimizer.solver
    
    CUDA.copyto!(pm.new_q, my_solver.data.q)
    CUDA.copyto!(pm.new_b, my_solver.data.b)

    if my_solver.settings.equilibrate_enable
        CUDA.@. pm.new_q *= my_solver.data.equilibration.dinv / my_solver.data.equilibration.c
        CUDA.@. pm.new_b *= my_solver.data.equilibration.einv
        CUDA.synchronize()
    end
    # update the return ratio in the new_q
    
    CUDA.@allowscalar begin
        # update the initial weights in the new_b
        #=
        idx = 3*(pm.n_assets) + 2
        CUDA.copyto!(CUDA.view(pm.new_b, idx:idx+(n-1)), my_solver.solution.x[1:n])
        CUDA.copyto!(CUDA.view(pm.new_b, idx+n:idx+(2*n-1)), -my_solver.solution.x[1:n])
        =#
        CUDA.copyto!(CUDA.view(pm.new_q, 1:pm.n_assets), -pm.return_ratio[:])

        Clarabel.update_q!(my_solver, pm.new_q)
        Clarabel.update_b!(my_solver, pm.new_b)
    end
    
    # --- Re-optimize the Model ---
    solve_time = CUDA.@elapsed Clarabel.solve!(my_solver, true)
    println("Re-solve completed in $solve_time seconds.")

    status = my_solver.info.status
    if status != Clarabel.SOLVED && status != Clarabel.ALMOST_SOLVED
        println("Warning: Re-solve did not reach optimality. Status: ", status)
    else
        println("Re-solve successful. Status: ", status)
    end
end
    
# --- Get the Calculated Risk Value (Standard Deviation) ---
function get_risk(pm::PortfolioModel)
    return pm.model.moi_backend.optimizer.model.optimizer.solver.solution.x[pm.n_assets+2]  # Return the first element of the solution vector
end
