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
    #v_buffer::Vector{MyFloat}
    model::JuMP.Model
    x::Vector{JuMP.VariableRef}
    #t::Union{JuMP.VariableRef, Nothing}
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
    set_optimizer_attribute(model, "iterative_refinement_enable", false)
    set_optimizer_attribute(model, "iterative_refinement_max_iter", 1)
    set_optimizer_attribute(model, "presolve_enable", true)
    set_optimizer_attribute(model, "static_regularization_enable", true)
    set_optimizer_attribute(model, "dynamic_regularization_enable", true)
    set_optimizer_attribute(model, "chordal_decomposition_enable", true)
    set_optimizer_attribute(model, "equilibrate_max_iter", 1)
    # 空初始化
    var_vector_empty = Vector{JuMP.VariableRef}()
    con_empty = Vector{JuMP.ConstraintRef}()
    
    return PortfolioModel(n_assets, n_style, λ_risk, x0, zeros(MyFloat, n_style), cov, expo', bias, cost, return_ratio, model, var_vector_empty, var_vector_empty, con_empty, con_empty)
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
    #pm.t = t
    
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
    @objective(pm.model, Min, dot(y, pm.cov*y) + dot(x, pm.bias.*x) + (1 / pm.λ_risk) * (dot(pm.cost, x_buy_sell) - dot(pm.return_ratio, x)))
end

function initial_solve!(pm::PortfolioModel)
    CUDA.@time optimize!(pm.model)  
    status = termination_status(pm.model)
    if status != MOI.OPTIMAL && status != MOI.ALMOST_OPTIMAL
        println("Initial solve failed")
    end  
end

function update_return_ratio!(pm::PortfolioModel, py_u_cpu::PyVector{MyFloat})
    pm.return_ratio = pyconvert(Vector{MyFloat}, py_u_cpu)    
end

function resolve!(pm::PortfolioModel)
    # 更新目标函数
    begin
        copyto!(pm.x0, value.(pm.x)) 
        copyto!(pm.y0, value.(pm.y))
        #t_value::MyFloat = value(pm.t)
        set_normalized_rhs.(pm.con_1, -pm.x0)
        set_normalized_rhs.(pm.con_2, pm.x0)
        
        # 修改这一行，不再使用累积乘法，而是直接设置目标系数
        # pm.return_ratio .*= - (1 / pm.λ_risk)  # 移除这行，防止累积修改
        set_objective_coefficient.(pm.model, pm.x, pm.return_ratio .* (- (1 / pm.λ_risk)))

        set_start_value.(pm.x, pm.x0)
        set_start_value.(pm.y, pm.y0)
        #set_start_value(pm.t, t_value)
    end
    
    optimize!(pm.model)
end

function get_risk(pm::PortfolioModel)
    return value(dot(pm.y, pm.cov*pm.y) + dot(pm.x, pm.bias.*pm.x))
end
