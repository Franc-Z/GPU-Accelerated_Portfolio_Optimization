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
set_optimizer_attribute(model, "tol_gap_abs", 1e-4)
set_optimizer_attribute(model, "tol_gap_rel", 1e-4)
set_optimizer_attribute(model, "tol_feas", 1e-4)

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
@expression(model, expected_returns, dot(mu_matrix, x))
@expression(model, transaction_fee, 0.002*sum(z))
@objective(model, Min, 
    -expected_returns + γ * (dot(y, Ω * y) + dot(x, D_sqrt .* x)) + transaction_fee
)

# 求解模型
optimize!(model)

# 获取底层求解器
my_solver = model.moi_backend.optimizer.model.optimizer.solver

# 优化数据拷贝操作
new_q = CUDA.similar(my_solver.data.q, Float64)
new_b = CUDA.similar(my_solver.data.b, Float64)
CUDA.copyto!(new_q, my_solver.data.q)
CUDA.copyto!(new_b, my_solver.data.b)


# 重复求解并计时
CUDA.@time begin
    CUDA.copyto!(new_q[1:n], -mu_matrix)
    Clarabel.update_q!(my_solver, new_q)
    Clarabel.update_b!(my_solver, new_b)
    Clarabel.solve!(my_solver)
end

# 输出结果
CUDA.@allowscalar begin
    local x_opt = vec(my_solver.solution.x[1:n])
    local top10_idx = partialsortperm(x_opt, 1:10, rev = true)
    for (i, idx) in enumerate(top10_idx)
        @printf("排名 %2d: 资产 %4d, 权重 = %.6f\n", i, idx, x_opt[idx])
    end
    println("")
end
