# Import necessary libraries for numerical computing, sparse matrices, optimization, and GPU computing
using LinearAlgebra, SparseArrays, Random, Clarabel, JuMP, CUDA, Printf

# Set random seed for reproducibility
rng = Random.MersenneTwister(1)

# 参数设置 - 匹配Python代码
k = 50                    # 风险因子数量
n = k * 100               # 资产数量
T = 2                     # 时间周期数
γ = 1.0                   # 风险厌恶系数
d = 1.0                   # 初始投资金额
x0 = zeros(n)             # 初始持仓权重

# 生成资产特定风险向量(使用稀疏对角矩阵提高效率)
D_diag = rand(rng, n) .* sqrt(k)
D_sqrt = sqrt.(D_diag)

# 生成稀疏因子载荷矩阵(使用稀疏表示提高计算和内存效率)
F = randn(rng, n, k) .* (rand(rng, n, k) .< 0.5)

# 优化的因子协方差矩阵生成
Ω_temp = randn(k, k)
Ω = (Ω_temp * Ω_temp') / k
#Ω_chol = cholesky(Symmetric(Ω)).L  # 使用Symmetric确保数值稳定性

# 预先计算并存储预期收益矩阵
mu_matrix = (3.0 .+ 9.0 .* rand(rng, n, T)) ./ 100.0

# 交易成本率参数
transaction_cost_rate = 0.002

# 初始化优化模型，使用Clarabel求解器
CUDA.@time begin
    model = JuMP.Model(Clarabel.Optimizer)
    
    # 调整求解器参数以平衡求解精度和速度
    set_optimizer_attribute(model, "direct_solve_method", :cudss)
    set_optimizer_attribute(model, "iterative_refinement_enable", false)
    set_optimizer_attribute(model, "presolve_enable", true)
    set_optimizer_attribute(model, "static_regularization_enable", true)
    set_optimizer_attribute(model, "dynamic_regularization_enable", true)
    set_optimizer_attribute(model, "chordal_decomposition_enable", true)
    set_optimizer_attribute(model, "equilibrate_max_iter", 1)  # 增加平衡迭代次数
    
    # 使用单一@variables块定义所有变量以减少JuMP内部开销
    @variables(model, begin
        0.1 >= x[1:T, 1:n] >= 0.0     # 添加上限约束提高求解效率
        0.1 >= y[1:T, 1:k] >= 0.0                  # 因子暴露
        z[1:T, 1:n] >= 0.0            # 交易量变量      
    end)
    
    # 批量添加交易量约束(第一个时间段)
    @constraints(model, begin
        [i=1:n], z[1,i] >= x[1,i] - x0[i]
        [i=1:n], z[1,i] >= x0[i] - x[1,i]
    end)
    
    # 批量添加后续时间段的交易量约束
    for t in 2:T
        @constraints(model, begin
            [i=1:n], z[t,i] >= x[t,i] - x[t-1,i]
            [i=1:n], z[t,i] >= x[t-1,i] - x[t,i]
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

# 允许标量操作并高效提取结果
CUDA.@allowscalar begin    
    # 快速检查求解状态
    status = termination_status(model)
    if status != MOI.OPTIMAL && status != MOI.ALMOST_OPTIMAL
        println("警告: 模型未能达到最优解, 状态: $status")
    end
    
    # 高效提取解向量
    x_opt = value.(x)
    y_opt = value.(y)
    z_opt = value.(z)
    
    # 使用矩阵运算计算预期收益(避免循环)
    expected_returns = [dot(mu_matrix[:,t], x_opt[t,:]) for t in 1:T]
    
    # 打印结果
    println("最优目标函数值 = ", objective_value(model))
    
    # 按时间段显示资产配置结果
    println("\n按时间段的资产配置结果:")
    for t in 1:T
        # 使用高效的向量求和
        total_transaction = sum(z_opt[t,:])
        
        println("\n时间段 $t:")
        println("预期收益 = ", expected_returns[t])
        println("总交易量 = ", total_transaction)
        println("交易成本 = ", transaction_cost_rate * total_transaction)
        
        println("前10个最大权重及其指数:")
        # 使用高效部分排序而非完全排序
        top10_idx = partialsortperm(vec(x_opt[t,:]), 1:10, rev=true)
        for (i, idx) in enumerate(top10_idx)
            @printf("排名 %2d: 资产 %4d, 权重 = %.6f\n", i, idx, x_opt[t, idx])
        end
    end
    
    # 打印模型规模信息
    println("\n模型规模统计:")
    println("变量数量: ", num_variables(model))
    println("约束数量: ", num_constraints(model, count_variable_in_set_constraints=true))
end
