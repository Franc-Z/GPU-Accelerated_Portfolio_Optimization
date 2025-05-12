using NPZ, LinearAlgebra, SparseArrays, Random, Clarabel, JuMP, CUDA, Printf

# Load data
D_diag = npzread("/nvtest/Test_Accuracy/D_diag.npy")
F = npzread("/nvtest/Test_Accuracy/F.npy")
Ω = npzread("/nvtest/Test_Accuracy/Omega.npy")
mu_matrix = npzread("/nvtest/Test_Accuracy/mu_matrix.npy")
F_t = F'
D_sqrt = sqrt.(D_diag)

n, k = size(F)
T = size(mu_matrix, 2)
x0 = zeros(n)
γ = 1.0
d = 1.0 - sum(x0)
transaction_cost_rate = 0.002

# 初始化优化模型，使用Clarabel求解器
begin
    model = JuMP.Model(Clarabel.Optimizer)

    # 调整求解器参数以平衡求解精度和速度，如需要直接通过矩阵或向量更新数据，下面（1）和（2）尽量设为false
    set_optimizer_attribute(model, "direct_solve_method", :cudss)
    set_optimizer_attribute(model, "iterative_refinement_enable", false)
    set_optimizer_attribute(model, "presolve_enable", false)                 # （1）
    set_optimizer_attribute(model, "static_regularization_enable", false)
    set_optimizer_attribute(model, "dynamic_regularization_enable", false)
    set_optimizer_attribute(model, "chordal_decomposition_enable", false)    # （2）
    set_optimizer_attribute(model, "equilibrate_enable", false)  # 增加平衡迭代次数
    set_optimizer_attribute(model, "verbose", true)

    # 使用单一@variables块定义所有变量以减少JuMP内部开销
    @variables(model, begin
        0.1 >= x[1:n, 1:T] >= 0.0     # 添加上限约束提高求解效率
        0.1 >= y[1:k, 1:T] >= 0.0     # 因子暴露
        z[1:n, 1:T] >= 0.0            # 交易量变量      
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
    @constraint(model, [t=2:T], sum(x[:,t]) == sum(x[:,t-1]))

    # 使用列向量表达式一次性添加所有约束
    @constraint(model, [t=1:T], y[:,t] .== F_t * x[:,t])
    

    # 目标函数: 预先计算常量项以减少求解器的计算量
    @expression(model, expected_returns[t=1:T], dot(mu_matrix[:,t], x[:,t]))
    @expression(model, transaction_costs[t=1:T], transaction_cost_rate * sum(z[:,t]))

    @objective(model, Min, 
        sum(-expected_returns[t] + transaction_costs[t] + γ*(dot(y[:,t], Ω*y[:,t])+dot(x[:,t], D_sqrt.*x[:,t]))  for t in 1:T)
    )

    # 求解模型
    optimize!(model)    
end

# 下面为调用CuClarabel底层的求解器，进行多次重复求解，从而进行准确计时。由于未直接调用JuMP.optimize!()，所以省去了问题设置的CPU耗时和H2D的耗时。
my_solver = model.moi_backend.optimizer.model.optimizer.solver
my_solution = zeros(Float64, n*T)
CUDA.copyto!(my_solution, my_solver.solution.x[1+(T-1)*n:T*n])
CUDA.@allowscalar begin
    local x_opt, top10_idx
    x_opt = vec(my_solution)
    top10_idx = partialsortperm(x_opt, 1:10, rev=true)
    for (i, idx) in enumerate(top10_idx)
        @printf("排名 %2d: 资产 %4d, 权重 = %.6f\n", i, idx, x_opt[idx])
    end
    println("")
end
new_q = CUDA.similar(my_solver.data.q, Float64)
new_b = CUDA.similar(my_solver.data.b, Float64)

for i in 1:1
    #=
    target_value = -1/n  # 替换为您想比较的值
    # 将 CUDA 稀疏矩阵转换到 CPU 后再进行查找
    cpu_b = Array(my_solver.data.b)
    indices = findall(x -> isapprox(x, target_value), cpu_b)
    println("Indices of elements approximately equal to $target_value: ", indices)
    =#
    CUDA.copyto!(new_b, my_solver.data.b)
    CUDA.copyto!(new_q, my_solver.data.q)
    # 在此部分我希望给mu_matrix的原有值加上其(-3% ~ 3%)之内的随机扰动
    random_noise = 0.1 .* randn(size(mu_matrix))
    mu_matrix .*= (1.0 .+ random_noise)
    #println("随机扰动后的mu_matrix: ", mu_matrix)

    if my_solver.settings.equilibrate_enable
        @. new_q *= my_solver.data.equilibration.dinv / my_solver.data.equilibration.c
        @. new_b *= my_solver.data.equilibration.einv
        #CUDA.synchronize()
    end

    idx = 3*T*(n + k) + T + 1
    
    #new_b[idx:idx+(n-1)] .= my_solver.solution.x[1+(T-1)*n:T*n]
    CUDA.copyto!(CUDA.view(new_b, idx:idx+(n-1)), my_solver.solution.x[1+(T-1)*n:T*n])
    #new_b[idx+n:idx+(2*n-1)] .= -my_solver.solution.x[1+(T-1)*n:T*n]
    CUDA.copyto!(CUDA.view(new_b, idx+n:idx+(2*n-1)), -my_solver.solution.x[1+(T-1)*n:T*n])
    for t in 1:T
        CUDA.copyto!(CUDA.view(new_q, 1+(t-1)*n:t*n), -mu_matrix[:,t])
    end
       
    Clarabel.update_q!(my_solver, new_q)
    Clarabel.update_b!(my_solver, new_b)
    CUDA.@time Clarabel.solve!(my_solver, true)
    
    CUDA.@allowscalar for t in 1:T
        local x_opt, top10_idx
        x_opt = vec(my_solver.solution.x[1+(t-1)*n:t*n])    
        top10_idx = partialsortperm(x_opt, 1:10, rev=true)
        println("时间段 $t:")
        for (i, idx) in enumerate(top10_idx)
            @printf("排名 %2d: 资产 %4d, 权重 = %.6f\n", i, idx, x_opt[idx])
        end
        println("")
    end
end
