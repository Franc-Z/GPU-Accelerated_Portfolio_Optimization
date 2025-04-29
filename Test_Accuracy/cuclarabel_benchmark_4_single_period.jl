using NPZ, LinearAlgebra, SparseArrays, Random, Clarabel, JuMP, CUDA, Printf

# Load data
D_diag = npzread("/nvtest/Test_Accuracy/D_diag.npy")
F = npzread("/nvtest/Test_Accuracy/F.npy")
Ω = npzread("/nvtest/Test_Accuracy/Omega.npy")
mu_matrix = npzread("/nvtest/Test_Accuracy/mu_matrix.npy")

println("\nFactor Covariance Matrix Ω (partial display):")
for i in 1:5
    @printf("[%2d] %s\n", i, join(Ω[i, 1:5], ", "))
end

D_sqrt = sqrt.(D_diag)

n, k = size(F)
T = size(mu_matrix, 2)
x0 = ones(n)./ n
γ = 1.0
d = 1.0
transaction_cost_rate = 0.002

# 初始化优化模型，使用Clarabel求解器
begin
    model = JuMP.Model(Clarabel.Optimizer)

    # 调整求解器参数以平衡求解精度和速度
    set_optimizer_attribute(model, "direct_solve_method", :cudss)
    set_optimizer_attribute(model, "iterative_refinement_enable", false)
    set_optimizer_attribute(model, "presolve_enable", true)
    set_optimizer_attribute(model, "static_regularization_enable", true)
    set_optimizer_attribute(model, "dynamic_regularization_enable", true)
    set_optimizer_attribute(model, "chordal_decomposition_enable", true)
    set_optimizer_attribute(model, "equilibrate_enable", false)  # 增加平衡迭代次数

    # 使用单一@variables块定义所有变量以减少JuMP内部开销
    @variables(model, begin
        0.1 >= x[1:T, 1:n] >= 0.0     # 添加上限约束提高求解效率
        0.1 >= y[1:T, 1:k] >= 0.0     # 因子暴露
        z[1:T, 1:n] >= 0.0            # 交易量变量      
    end)

    # 批量添加交易量约束(第一个时间段)
    con_1 = @constraint(model, z[1,:] .>= x[1,:] - x0[:])
    con_2 = @constraint(model, z[1,:] .>= x0[:] - x[1,:])

    # 批量添加后续时间段的交易量约束
    for t in 2:T
        @constraints(model, begin
            z[t,:] .>= x[t,:] - x[t-1,:]
            z[t,:] .>= x[t-1,:] - x[t,:]
        end)
    end

    # 批量添加预算约束
    @constraint(model, sum(x[1,:]) == d + sum(x0))
    @constraint(model, [t=2:T], sum(x[t,:]) == sum(x[t-1,:]))

    # 使用矩阵向量乘法形式添加因子暴露约束(避免双循环)
    for t in 1:T
        # 使用列向量表达式一次性添加所有约束
        F_t = F'  # 预计算转置矩阵
        @constraint(model, y[t,:] .== F_t * x[t,:])
    end

    # 目标函数: 预先计算常量项以减少求解器的计算量
    @expression(model, expected_returns[t=1:T], dot(mu_matrix[:,t], x[t,:]))
    @expression(model, transaction_costs[t=1:T], transaction_cost_rate * sum(z[t,:]))

    @objective(model, Min, 
        sum(-expected_returns[t] + transaction_costs[t] + γ*(dot(y[t,:], Ω*y[t,:])+dot(x[t,:], D_sqrt.*x[t,:]))  for t in 1:T)
    )

    # 求解模型
    optimize!(model)    
end

# 下面为调用CuClarabel底层的求解器，进行多次重复求解，从而进行准确计时。由于未直接调用JuMP.optimize!()，所以省去了问题设置的CPU耗时和H2D的耗时。
my_solver = model.moi_backend.optimizer.model.optimizer.solver
new_q = CUDA.similar(my_solver.data.q, Float64)
CUDA.copyto!(new_q, my_solver.data.q)

new_b = CUDA.similar(my_solver.data.b, Float64)
CUDA.copyto!(new_b, my_solver.data.b)

CUDA.copyto!(new_q[1:n], mu_matrix[:,1])
begin
    #=
    target_value = -1.0/n  # 替换为您想比较的值
    # 将 CUDA 稀疏矩阵转换到 CPU 后再进行查找
    cpu_b = Array(my_solver.data.b)
    indices = findall(x -> isapprox(x, target_value), cpu_b)
    println("Indices of elements approximately equal to $target_value: ", indices)
    =#
    new_b[15152:20151] .= my_solver.solution.x[1:n]
    new_b[20152:25151] .= -my_solver.solution.x[1:n]
end
CUDA.@time for i in 1:1        
    Clarabel.update_q!(my_solver, new_q)
    Clarabel.update_b!(my_solver, new_b)
    Clarabel.solve!(my_solver)
end
#println("Objective value: ", my_solver.solution.objective_value)    
CUDA.@allowscalar begin
    local x_opt = vec(my_solver.solution.x[1:n])
    local top10_idx = partialsortperm(x_opt, 1:10, rev=true)
    #println("Optimal solution x: ", top10_idx)
    for (i, idx) in enumerate(top10_idx)
        @printf("排名 %2d: 资产 %4d, 权重 = %.6f\n", i, idx, x_opt[idx])
    end
end
