using NPZ, LinearAlgebra, SparseArrays, Random, JuMP, MosekTools, Printf, MathOptInterface

# Load data
D_diag = npzread("/nvtest/PO/GPU-Accelerated_Portfolio_Optimization/Test_Accuracy/D_diag.npy")
F = npzread("/nvtest/PO/GPU-Accelerated_Portfolio_Optimization/Test_Accuracy/F.npy")
Ω = npzread("/nvtest/PO/GPU-Accelerated_Portfolio_Optimization/Test_Accuracy/Omega.npy")
mu_matrix = npzread("/nvtest/PO/GPU-Accelerated_Portfolio_Optimization/Test_Accuracy/mu_matrix.npy")

println("\nFactor Covariance Matrix Ω (partial display):")
for i in 1:5
    @printf("[%2d] %s\n", i, join(Ω[i, 1:5], ", "))
end

D_sqrt = sqrt.(D_diag)

n, k = size(F)
T = size(mu_matrix, 2)
x0 = zeros(n)
γ = 1.0
d = 1.0
transaction_cost_rate = 0.002

println("\n问题规模: n=$n, k=$k, T=$T")



# 初始化优化模型，使用Mosek求解器
model = JuMP.Model(Mosek.Optimizer)

# 调整求解器参数
set_optimizer_attribute(model, "INTPNT_CO_TOL_PFEAS", 1e-8)
set_optimizer_attribute(model, "INTPNT_CO_TOL_DFEAS", 1e-8)
set_optimizer_attribute(model, "INTPNT_CO_TOL_REL_GAP", 1e-8)
set_optimizer_attribute(model, "INTPNT_CO_TOL_INFEAS", 1e-10)
set_optimizer_attribute(model, "INTPNT_CO_TOL_MU_RED", 1e-8)
set_optimizer_attribute(model, "OPTIMIZER_MAX_TIME", 600.0)
set_attribute(model, "MSK_IPAR_NUM_THREADS", 12)
set_optimizer_attribute(model, "LOG", 1)

# 使用单一@variables块定义所有变量以减少JuMP内部开销
@variables(model, begin
    0.1 >= x[1:T, 1:n] >= 0.0     # 添加上限约束提高求解效率
    0.1 >= y[1:T, 1:k] >= 0.0     # 因子暴露
    z[1:T, 1:n] >= 0.0            # 交易量变量      
end)

# 批量添加交易量约束(第一个时间段)
@constraints(model, begin
    z[1,:] .>= x[1,:] - x0[:]
    z[1,:] .>= x0[:] - x[1,:]
end)

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

# 先调用一次optimize!确保模型已经构建完成
println("开始求解模型...")
optimize!(model)

# 获取底层的Mosek优化器和任务
my_optimizer = backend(model)

# 直接使用MOI接口进行多次求解
@time for i in 1:20        
    MathOptInterface.optimize!(my_optimizer)
end

# 快速检查求解状态
status = termination_status(model)
if status != MOI.OPTIMAL && status != MOI.ALMOST_OPTIMAL
    println("警告: 模型未能达到最优解, 状态: $status")
end

# 高效提取解向量
x_opt = value.(x)
y_opt = value.(y)
z_opt = value.(z)
expected_returns_val = value.(expected_returns)
transaction_costs_val = value.(transaction_costs)

# 打印结果
println("最优目标函数值 = ", objective_value(model))

# 按时间段显示资产配置结果
println("\n按时间段的资产配置结果:")
for t in 1:T
    println("\n时间段 $t:")
    println("预期收益 = ", expected_returns_val[t])
    println("交易成本 = ", transaction_costs_val[t])

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
