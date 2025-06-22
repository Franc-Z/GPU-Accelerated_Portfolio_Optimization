using NPZ, LinearAlgebra, SparseArrays, Random, Clarabel, JuMP, CUDA, Printf
#import MathOptInterface: SecondOrderCone
# 新的优化实现已经整合到主函数中，会自动检测并使用最优的实现
# Clarabel.update_scaling_soc! = Clarabel.update_scaling_soc_enhanced!
# Clarabel.mul_Hs_soc! = Clarabel.mul_Hs_soc_enhanced!
# Load data
D_diag = npzread("/nvtest/Test_Accuracy/D_diag.npy")
F = npzread("/nvtest/Test_Accuracy/F.npy")
Ω = npzread("/nvtest/Test_Accuracy/Omega.npy")
mu_matrix = npzread("/nvtest/Test_Accuracy/mu_matrix.npy")

F_t = F'
D_sqrt = sqrt.(D_diag)

n, k = size(F)
T = size(mu_matrix, 2)
#println("T = ", T)
x0 = zeros(n)
γ = 1.0
d = 1.0 - sum(x0)
transaction_cost_rate = 0.002

# 初始化优化模型，使用Clarabel求解器
begin
    model = JuMP.Model(Clarabel.Optimizer)
    # 调整求解器参数 - 使用优化后的GPU配置
    set_optimizer_attribute(model, "direct_solve_method", :cudss)
    set_optimizer_attribute(model, "iterative_refinement_enable", true)
    set_optimizer_attribute(model, "presolve_enable", false)
    set_optimizer_attribute(model, "static_regularization_enable", false)
    set_optimizer_attribute(model, "dynamic_regularization_enable", false)  # 关闭以提高稳定性
    set_optimizer_attribute(model, "chordal_decomposition_enable", false)    # GPU不支持
    set_optimizer_attribute(model, "equilibrate_enable", false)              # 关闭以避免缩放问题
    set_optimizer_attribute(model, "verbose", true)
    set_optimizer_attribute(model, "tol_feas", 1e-6)
    set_optimizer_attribute(model, "tol_gap_abs", 1e-6)
    set_optimizer_attribute(model, "tol_gap_rel", 1e-6)
    #set_optimizer_attribute(model, "static_regularization_constant", 1e-6)   # 增加正则化

    # 使用单一@variables块定义所有变量以减少JuMP内部开销
    @variables(model, begin
        0.1 >= x[1:n, 1:T] >= 0.0       # 添加上限约束提高求解效率
        0.1 >= y[1:k, 1:T] >= 0.0       # 因子暴露
        z[1:n, 1:T]                     # 交易量变量      
        risk[1:T] >= 0.0                # 风险变量
    end)

    # 批量添加交易量约束(第一个时间段)
    @constraint(model, z[:,1] .>= x[:,1] - x0[:])
    @constraint(model, z[:,1] .>= x0[:] - x[:,1])

    # 批量添加后续时间段的交易量约束
    for t in 2:T
        @constraints(model, begin
            z[:,t] .>= x[:,t] - x[:,t-1]
            z[:,t] .>= x[:,t-1] - x[:,t]
        end)
    end

    # 批量添加预算约束
    @constraint(model, sum(x[:,1]) == d + sum(x0))
    if T > 1    
        @constraint(model, [t=2:T], sum(x[:,t]) == sum(x[:,t-1]))
    end

    # 使用列向量表达式一次性添加所有约束
    @constraint(model, [t=1:T], y[:,t] .== F_t * x[:,t])

    cov_U = cholesky(Ω).U
    soc_dim = 1 + size(cov_U, 1) + length(D_sqrt)
    @constraint(model, [t=1:T], [risk[t]; cov_U * y[:,t]; D_sqrt .* x[:,t]] in MOI.SecondOrderCone(soc_dim))

    # 目标函数: 预先计算常量项以减少求解器的计算量
    @expression(model, expected_returns[t=1:T], dot(mu_matrix[:,t], x[:,t]))
    @expression(model, transaction_costs[t=1:T], transaction_cost_rate * sum(z[:,t]))

    @objective(model, Min, 
        sum(-expected_returns[t] + transaction_costs[t] + γ*risk[t]  for t in 1:T)
    )

    println("开始求解模型...")
    optimize!(model)

    status = termination_status(model)
    if status != MOI.OPTIMAL && status != MOI.ALMOST_OPTIMAL
        println("警告: 模型未能达到最优解, 状态: $status")
    end
end
# 下面为调用CuClarabel底层的求解器，进行多次重复求解，从而进行准确计时。由于未直接调用JuMP.optimize!()，所以省去了问题设置的CPU耗时和H2D的耗时。
my_solver = model.moi_backend.optimizer.model.optimizer.solver
my_solution = zeros(Float64, n)  # 修正：只需要n个元素，不是n*T

# 修正：对于T=1的情况，直接提取前n个元素，并使用数组操作避免元素级索引
copyto!(my_solution, view(my_solver.solution.x, 1:n))

# 打印排名前10的资产
let
    x_opt = vec(my_solution)
    top10_idx = partialsortperm(x_opt, 1:10, rev=true)
    for (i, idx) in enumerate(top10_idx)
        @printf("排名 %2d: 资产 %4d, 权重 = %.6f\n", i, idx, x_opt[idx])
    end
    println("")
end

new_q = similar(my_solver.data.q, Float64)
new_b = similar(my_solver.data.b, Float64)

for i in 1:1    
    #=
    target_value = 1.222222222  # 替换为您想比较的值
    # 将 CUDA 稀疏矩阵转换到 CPU 后再进行查找
    cpu_b = Array(my_solver.data.b)
    indices = findall(x -> isapprox(x, target_value), cpu_b)[1]
    println("Indices of elements approximately equal to $target_value: ", indices)
    =#
    copyto!(new_b, my_solver.data.b)
    copyto!(new_q, my_solver.data.q)
    # 在此部分我希望给mu_matrix的原有值加上其(-3% ~ 3%)之内的随机扰动
    #random_noise = 0.05 .* randn(size(mu_matrix)) .- 0.025
    #mu_matrix .*= (1.0 .+ random_noise)
    #println("随机扰动后的mu_matrix: ", mu_matrix)

    if my_solver.settings.equilibrate_enable
        @. new_q *= my_solver.data.equilibration.dinv / my_solver.data.equilibration.c
        @. new_b *= my_solver.data.equilibration.einv
        #CUDA.synchronize()
    end
    
    idx = 2*T*n + 3*T*k + 2*T + 1
    new_b[idx:idx+(n-1)] .= my_solver.solution.x[1+(T-1)*n:T*n]
    new_b[idx+n:idx+(2*n-1)] .= -my_solver.solution.x[1+(T-1)*n:T*n]
    
    # 修正：使用数组操作以避免元素级索引
    copyto!(view(new_q, 1:(n*T)), vec(-mu_matrix))
      
    Clarabel.update_q!(my_solver, new_q)
    Clarabel.update_b!(my_solver, new_b)
    
    # 尝试冷启动
    CUDA.@time Clarabel.solve!(my_solver, true)
    
    # 检查求解状态
    
    println("Solver Status: ", my_solver.solution.status)
    println("Iterations: ", my_solver.solution.iterations)
    println("Primal residual: ", my_solver.solution.r_prim)
    println("Dual residual: ", my_solver.solution.r_dual)
    

    # 修正：安全地提取解，避免元素级索引
    copyto!(my_solution, view(my_solver.solution.x, 1:n))
    
    let 
        x_opt = vec(my_solution)
        top10_idx = partialsortperm(x_opt, 1:10, rev=true)
        for (i, idx) in enumerate(top10_idx)
            @printf("排名 %2d: 资产 %4d, 权重 = %.6f\n", i, idx, x_opt[idx])
        end
        println("求解结束，耗时: ", my_solver.solution.solve_time)
    end
        
end
#println("Objective value: ", my_solver.solution.objective_value)
