using LinearAlgebra, JuMP, Clarabel, PythonCall
import CUDA
import PythonCall: pyconvert

CUDA.allowscalar(true)

mutable struct PortfolioModel
    n_assets::Int
    n_style::Int
    λ_risk::Float64
    x0::Vector{Float64}
    U_cov::Matrix{Float64}
    expo_T::Matrix{Float64}
    bias_sqrt::Vector{Float64}
    cost::Vector{Float64}
    ret_ratio::Vector{Float64}
    model::JuMP.Model
    x::Vector{JuMP.VariableRef}
    t::Union{JuMP.VariableRef, Nothing}
    y::Vector{JuMP.VariableRef}
    con_1::Vector{JuMP.ConstraintRef}
    con_2::Vector{JuMP.ConstraintRef}
end


# 构造函数
function CreatePortfolioModel(n_assets::Int, n_style::Int, λ_risk::Float64, 
                      py_x0::PyVector{Float64}, py_cov::PyMatrix{Float64}, 
                      py_expo::PyMatrix{Float64}, py_bias::PyVector{Float64}, 
                      py_cost::PyVector{Float64}, py_u_cpu::PyVector{Float64})
    x0 = pyconvert(Vector{Float64}, py_x0)
    cov = pyconvert(Matrix{Float64}, py_cov)
    U_cov = cholesky(Hermitian(cov)).U
   
    expo = pyconvert(Matrix{Float64}, py_expo)
    bias = pyconvert(Vector{Float64}, py_bias)
    bias_sqrt = sqrt.(bias)
    cost = pyconvert(Vector{Float64}, py_cost)
    ret_ratio = pyconvert(Vector{Float64}, py_u_cpu)
    
    model = Model(Clarabel.Optimizer)
    set_optimizer_attribute(model, "direct_solve_method", :cudss)       # cudssmixed or cudss
    set_optimizer_attribute(model, "verbose", true)
    set_optimizer_attribute(model, "presolve_enable", true)
    set_optimizer_attribute(model, "static_regularization_enable", true)
    set_optimizer_attribute(model, "dynamic_regularization_enable", true)
    set_optimizer_attribute(model, "equilibrate_enable", true)
    set_optimizer_attribute(model, "iterative_refinement_enable", false)
    set_optimizer_attribute(model, "chordal_decomposition_enable", true)
    
    # 空初始化
    var_vector_empty = Vector{JuMP.VariableRef}()
    con_empty = Vector{JuMP.ConstraintRef}()
    
    return PortfolioModel(n_assets, n_style, λ_risk, x0, U_cov, expo', bias_sqrt, cost, ret_ratio, model, var_vector_empty, nothing, var_vector_empty, con_empty, con_empty)
end

function setup_model!(pm::PortfolioModel)
    @variable(pm.model, x[1:pm.n_assets])
    set_start_value.(x, pm.x0)
    @variable(pm.model, y[1:pm.n_style])
    @variable(pm.model, t)
    @variable(pm.model, x_buy_sell[1:pm.n_assets])
    # 更新引用
    pm.x = x
    pm.y = y
    pm.t = t
    
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
    
    @expression(pm.model, risk1, pm.U_cov * y)
    @expression(pm.model, risk2, pm.bias_sqrt .* x)
    @constraint(pm.model, [t; risk1; risk2] in SecondOrderCone())
    @constraint(pm.model, x_buy_sell .>= 0.0)
    pm.con_1 = @constraint(pm.model, x_buy_sell .>= x - pm.x0)
    pm.con_2 = @constraint(pm.model, x_buy_sell .>= pm.x0 - x)
    @objective(pm.model, Min, t + dot(pm.cost, x_buy_sell) - (1 / pm.λ_risk) * dot(pm.ret_ratio, x))
end

function initial_solve!(pm::PortfolioModel)
    optimize!(pm.model)  
    if termination_status(pm.model) != MOI.OPTIMAL
        println("Initial solve failed")
    end  
end

function update_return_ratio!(pm::PortfolioModel, py_u_cpu::PyVector{Float64})
    pm.ret_ratio = pyconvert(Vector{Float64}, py_u_cpu)    
end

function resolve!(pm::PortfolioModel)
    # 更新目标函数
    begin
        copyto!(pm.x0, value.(pm.x)) 
        #set_start_value.(pm.x, pm.x0)
        set_normalized_rhs.(pm.con_1, -pm.x0)
        set_normalized_rhs.(pm.con_2, pm.x0)
        coef = - (1 / pm.λ_risk) .* pm.ret_ratio
        set_objective_coefficient.(pm.model, pm.x, coef)
    end
    
    CUDA.@time optimize!(pm.model)    
    if termination_status(pm.model) != MOI.OPTIMAL
        println("Initial solve failed")
    end  
end

function get_risk(pm::PortfolioModel)
    CUDA.@allowscalar return value(pm.t)
end

