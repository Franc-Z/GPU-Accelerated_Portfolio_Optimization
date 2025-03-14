using LinearAlgebra, JuMP, Clarabel, PythonCall, NVTX
import CUDA
import PythonCall: pyconvert

Clarabel.CUDA.allowscalar(true)

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
    #v_buffer::Vector{MyFloat}
    model::JuMP.Model
    x::Vector{JuMP.VariableRef}
    t::Union{JuMP.VariableRef, Nothing}
    y::Vector{JuMP.VariableRef}
    con_1::Vector{JuMP.ConstraintRef}
    con_2::Vector{JuMP.ConstraintRef}
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
    set_optimizer_attribute(model, "presolve_enable", true)
    #set_optimizer_attribute(model, "device", :gpu)
    set_optimizer_attribute(model, "static_regularization_enable", true)
    set_optimizer_attribute(model, "dynamic_regularization_enable", true)
    set_optimizer_attribute(model, "equilibrate_enable", true)
    set_optimizer_attribute(model, "iterative_refinement_enable", true)
    set_optimizer_attribute(model, "chordal_decomposition_enable", true)

    # 空初始化
    var_vector_empty = Vector{JuMP.VariableRef}()
    con_empty = Vector{JuMP.ConstraintRef}()
    
    return PortfolioModel(n_assets, n_style, λ_risk, x0, zeros(MyFloat, n_style), cov, expo', bias, cost, return_ratio, model, var_vector_empty, nothing, var_vector_empty, con_empty, con_empty)
end

function setup_model!(pm::PortfolioModel)
    @variable(pm.model, x[1:pm.n_assets])
    set_start_value.(x, pm.x0)
    @variable(pm.model, y[1:pm.n_style])
    @variable(pm.model, t >= 0.0)
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
    
    @constraint(pm.model, x_buy_sell .>= 0.0)
    pm.con_1 = @constraint(pm.model, x_buy_sell .>= x - pm.x0)
    pm.con_2 = @constraint(pm.model, x_buy_sell .>= pm.x0 - x)
    @constraint(pm.model, dot(y, pm.cov*y) + dot(x, pm.bias.*x) <= t)
    @objective(pm.model, Min, t + dot(pm.cost, x_buy_sell) - (1 / pm.λ_risk) * dot(pm.return_ratio, x))
end

function initial_solve!(pm::PortfolioModel)
    optimize!(pm.model)  
end

function update_return_ratio!(pm::PortfolioModel, py_u_cpu::PyVector{MyFloat})
    pm.return_ratio = pyconvert(Vector{MyFloat}, py_u_cpu)    
end

function resolve!(pm::PortfolioModel)
    # 更新目标函数
    begin
        copyto!(pm.x0, value.(pm.x)) 
        copyto!(pm.y0, value.(pm.y))
        t_value::MyFloat = value(pm.t)
        set_normalized_rhs.(pm.con_1, -pm.x0)
        set_normalized_rhs.(pm.con_2, pm.x0)
        pm.return_ratio .*= - (1 / pm.λ_risk) 
        set_objective_coefficient.(pm.model, pm.x, pm.return_ratio)

        set_start_value.(pm.x, pm.x0)
        set_start_value.(pm.y, pm.y0)
        set_start_value(pm.t, t_value)
    end
    
    NVTX.@range "my_kernel" begin
        optimize!(pm.model)
    end
          
    if is_solved_and_feasible(pm.model)
        println("已正确求解！")
    else
        println("求解失败！")
    end  
end

function get_risk(pm::PortfolioModel)
    CUDA.@allowscalar return value(pm.t)
end

