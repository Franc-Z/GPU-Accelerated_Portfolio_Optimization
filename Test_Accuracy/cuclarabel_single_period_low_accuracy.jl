using NPZ, LinearAlgebra, SparseArrays, Random, Clarabel, JuMP, CUDA, Printf

# 加载数据
D_diag = npzread("/nvtest/Test_Accuracy/D_diag.npy")
F = npzread("/nvtest/Test_Accuracy/F.npy")
Ω = npzread("/nvtest/Test_Accuracy/Omega.npy")
mu_matrix = npzread("/nvtest/Test_Accuracy/mu_matrix.npy")
F_t = F'  # 预计算转置矩阵
D_sqrt = sqrt.(D_diag)

n, k = size(F)

x0 = zeros(n)
γ = 1.0
d = 1.0 - sum(x0)

# 初始化优化模型，使用Clarabel求解器
model = JuMP.Model(Clarabel.Optimizer)

# 调整求解器参数以平衡求解精度和速度
set_optimizer_attribute(model, "direct_solve_method", :cudss)
set_optimizer_attribute(model, "iterative_refinement_enable", false)
set_optimizer_attribute(model, "presolve_enable", false)
set_optimizer_attribute(model, "static_regularization_enable", false)
set_optimizer_attribute(model, "dynamic_regularization_enable", false)
set_optimizer_attribute(model, "chordal_decomposition_enable", false)
set_optimizer_attribute(model, "equilibrate_enable", false)
set_optimizer_attribute(model, "verbose", true)
set_optimizer_attribute(model, "tol_gap_abs", 1e-8)
set_optimizer_attribute(model, "tol_gap_rel", 1e-8)
set_optimizer_attribute(model, "tol_feas", 1e-8)

# 定义变量
@variables(model, begin
    0.1 >= x[1:n] >= 0.0  # 添加上限约束提高求解效率
    0.1 >= y[1:k] >= 0.0  # 因子暴露
    z[1:n] >= 0.0         # 交易量变量
end)

# 添加约束
@constraint(model, sum(x) == d + sum(x0))

@constraint(model, y .== F_t * x)

@constraint(model, z .>= x - x0)
@constraint(model, z .>= x0 - x)

# 定义目标函数
@expression(model, expected_return, dot(mu_matrix, x))
@expression(model, transaction_fee, 0.002*sum(z))
@objective(model, Min, 
    -expected_return + γ * (dot(y, Ω * y) + dot(x, D_sqrt .* x)) + transaction_fee
)

# 求解模型
optimize!(model)

# 获取底层求解器
my_solver = model.moi_backend.optimizer.model.optimizer.solver
# 输出结果
CUDA.@allowscalar begin
    local x_opt = vec(my_solver.solution.x[1:n])
    local top10_idx = partialsortperm(x_opt, 1:10, rev = true)
    for (i, idx) in enumerate(top10_idx)
        @printf("排名 %2d: 资产 %4d, 权重 = %.6f\n", i, idx, x_opt[idx])
    end
    println("")
end


# 优化数据拷贝操作
new_q = CUDA.similar(my_solver.data.q, Float64)
new_b = CUDA.similar(my_solver.data.b, Float64)
CUDA.copyto!(new_q, my_solver.data.q)
CUDA.copyto!(new_b, my_solver.data.b)

# 以下使用cuclarabel的底层API直接进行参数更新及求解
CUDA.@time begin
    #=
    target_value = -mu_matrix[n]  # 替换为您想比较的值
    # 将 CUDA 稀疏矩阵转换到 CPU 后再进行查找
    cpu_q = Array(my_solver.data.q)
    indices = findall(x -> isapprox(x, target_value), cpu_q)
    println("Indices of elements approximately equal to $target_value: ", indices)
    =#
    # 在此部分我希望给mu_matrix的原有值加上其(-3% ~ 3%)之内的随机扰动
    println(mu_matrix[1:5])
    random_noise = 0.1 .* randn(size(mu_matrix))
    mu_matrix .*= (1.0 .+ random_noise)
    println(mu_matrix[1:5])
    # 将新的目标函数系数和约束右侧值传递给求解器
    CUDA.copyto!(CUDA.view(new_q, 1:n), -mu_matrix[:])
    println(new_q[1:5])
    Clarabel.update_q!(my_solver, new_q)
    Clarabel.update_b!(my_solver, new_b)
    solution = Clarabel.solve!(my_solver, true)
end

# 输出结果
CUDA.@allowscalar begin
    local x_opt = vec(solution.x[1:n])
    local top10_idx = partialsortperm(x_opt, 1:10, rev = true)
    for (i, idx) in enumerate(top10_idx)
        @printf("排名 %2d: 资产 %4d, 权重 = %.6f\n", i, idx, x_opt[idx])
    end
    println("")
end

# 以下使用JuMP提供的API来执行参数更新和优化操作
CUDA.@allowscalar for i in 1:1
    #copyto!(x0, value.(x[:]))
    #set_normalized_rhs.(con_1, -x0)
    #set_normalized_rhs.(con_2, x0)
    
    set_objective_coefficient.(model, x[:], -mu_matrix[:])
    
    println("设置目标函数系数和初始值完成")
    @time optimize!(model)
end

# 输出结果
CUDA.@allowscalar begin
    println("预期收益 = ", value(expected_return))
    println("交易成本 = ", value(transaction_fee))

    println("前10个最大权重及其指数:")
    # 使用高效部分排序而非完全排序
    local x_opt = vec(value.(x))
    local top10_idx = partialsortperm(x_opt, 1:10, rev=true)
    for (i, idx) in enumerate(top10_idx)
        @printf("排名 %2d: 资产 %4d, 权重 = %.6f\n", i, idx, x_opt[idx])
    end
end
