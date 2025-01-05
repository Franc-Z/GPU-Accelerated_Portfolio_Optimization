# 本算法实现的优化形式请参考https://docs.mosek.com/portfolio-cookbook/appendix.html#quadratic-cones-and-riskless-solution

# 导入必要的模块，使用 import
import NLPModels: AbstractNLPModel, NLPModelMeta, Counters
import NLPModels
import MadNLP
import MadNLPGPU
import CUDA
import LinearAlgebra: isposdef, cholcopy, issymmetric, norm_sqr
import PythonCall

# 如果需要在全局进行CUDA标量操作（即给CUDA数组中的元素分别赋值【会影响性能】），下面一行代码就不要注释掉
CUDA.allowscalar(true)

T = Float64             # 目前可以设置为使用Float32或Float64两种精度，推荐使用Float64，因为Float32精度较低，可能会导致小幅计算误差
VT = CUDA.CuVector{T}   # 使用T类型的向量类型CuVector
MT = CUDA.CuMatrix{T}   # 使用T类型的矩阵类型CuMatrix

# 设置快速计算模式，并利用TensorCore加速计算, 但是这个设置有可能影响计算精度
CUDA.math_mode!(CUDA.FAST_MATH)

# ==========================
# 定义自定义的 NLPModel
# ==========================

# 自定义的 NLPModel，用于在 GPU 上执行计算
mutable struct PortfolioNLPModelCUDA{
    T,
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T},
    } <: AbstractNLPModel{T,VT}  # 继承自 AbstractNLPModel
    meta::NLPModelMeta{T, VT}       # x0每次求解需要更新
    counters::Counters
    µ::VT                           # 平均log回报率u，每次求解需要更新
    V_buffer::VT                    # 存放向量的缓存
    Σ::MT                           # 协方差矩阵Σ
    #L::MT                          # L矩阵每次求解需要更新
    U::MT                           # 对协方差矩阵进行LU分解的U矩阵，U矩阵每次求解需要更新
    r_f::T                          # 对数(log)无风险利率r_f
    λ_risk::T                       # 风险厌恶系数delta
end

# 构造函数
function PortfolioNLPModelCUDA_Construct(
    u::VT, 
    r_f::T, 
    delta::T,  # delta is λ_risk
    Q::MT,
    n_con::Int,
    x0::VT, 
    #y0::AbstractVector{T},
    lvar::VT, 
    uvar::VT, 
    lcon::VT, 
    ucon::VT,
    ) where {T, VT, MT}
    nvar = length(x0)
    n_assets = nvar - 1
    
    x0_ = x0
    u_ = u .- r_f
    Σ_ = Q
    cholesky_result = CUDA.CUSOLVER.cholesky(CUDA.CUBLAS.Hermitian(Σ_)) # 在 GPU 上计算，确保矩阵是 Hermitian
    U_ = cholcopy(cholesky_result.U)  #U就是L'
    #L_ = cholcopy(cholesky_result.L)
    lvar_ = lvar
    uvar_ = uvar
    lcon_ = lcon
    ucon_ = ucon
    v_buffer_ = CUDA.similar(x0_, n_assets)
    meta = NLPModelMeta(nvar,
                        ncon = n_con,   # 二阶锥约束和资金总和约束
                        x0 = x0_,
                        #y0 = y0,
                        lvar = lvar_,
                        uvar = uvar_,
                        lcon = lcon_,
                        ucon = ucon_,
                        minimize = true # 最小化目标函数
                        )    
    return PortfolioNLPModelCUDA(meta, Counters(), u_, v_buffer_, Σ_, U_, r_f, delta)
end

# ==========================
# 实现 NLPModels 所需的函数
# ==========================

# 目标函数
function NLPModels.obj(nlp::PortfolioNLPModelCUDA, x::AbstractVector{T}) where T
    nlp.counters.neval_obj += 1
    n = nlp.meta.nvar - 1   # 资产数量
    x_var = x[1:n]          # 前n个值为资产权重
    t = x[end]              # 最后一个值为t，用于二阶锥约束，t >= sqrt(w'Σw)，
    # 计算目标函数值（取负号，因为默认是最小化）
    return (-((CUDA.dot(nlp.µ, x_var)+ nlp.r_f) - nlp.λ_risk * t))
end

# 目标函数梯度（g为梯度向量）
function NLPModels.grad!(nlp::PortfolioNLPModelCUDA, x::AbstractVector{T}, g::AbstractVector{T}) where T
    nlp.counters.neval_grad += 1
    n = nlp.meta.nvar - 1
    # 对 x_i 的偏导数
    g[1:n] .= - nlp.µ
    # 对 t 的偏导数
    g[end] = nlp.λ_risk
    return g
end

# 约束函数，目前约束有两项：二阶锥约束和资金总和约束（c为约束方程组的左边部分，c的元素个数就是约束的总数）
function NLPModels.cons!(nlp::PortfolioNLPModelCUDA, x::AbstractVector{T}, c::AbstractVector{T}) where T
    nlp.counters.neval_cons += 1
    n = nlp.meta.nvar - 1
    x_var = x[1:n]
    t = x[end]
    # 在 GPU 上计算
    CUDA.CUBLAS.mul!(nlp.V_buffer, nlp.U, x_var)
    norm_Ux_sqr = norm_sqr(nlp.V_buffer)
    c[1] = norm_Ux_sqr - t*t      # 二阶锥约束
    c[2] = CUDA.sum(x_var)  # 资金总和约束
    return c
end

# 约束方程的Jacobian 矩阵（即约束方程的一阶导数矩阵，密集矩阵形式）
function MadNLP.jac_dense!(nlp::PortfolioNLPModelCUDA, x::AbstractVector{T}, J::AbstractMatrix{T}) where T
    nlp.counters.neval_jac += 1
    nvar = nlp.meta.nvar
    n = nvar - 1
    x_var = x[1:n]
    t = x[end]

    # 第一行对应二阶锥约束，对 x[i] 的偏导数为2*Σ*x[i]，对 t 的偏导数为-2*t
    CUDA.CUBLAS.mul!(nlp.V_buffer, nlp.Σ, x_var)
    J[1, 1:n] = T(2).*nlp.V_buffer'    
    J[1, end] = T(-2)*t

    # 第二行对应资金总和约束，对 x[i] 的偏导数为1
    J[2, 1:n] .= one(T)
    J[2, end] = zero(T)
    return J
end

# 拉格朗日函数的Hessian矩阵（即目标函数的二阶导数矩阵，密集形式）（y向量为约束方程的系数，即拉格朗日系数，维度即为约束的个数）
function MadNLP.hess_dense!(nlp::PortfolioNLPModelCUDA, x::AbstractVector{T}, y::AbstractVector{T}, H::AbstractMatrix{T}; obj_weight=1.0) where T
    nlp.counters.neval_hess += 1
    n = nlp.meta.nvar - 1
    #println("拉格朗日H的维度：", size(H))
    # 初始化 Hessian 矩阵为零
    CUDA.fill!(H, zero(T))
    # 只需要考虑锥约束的Hessian矩阵。因为目标函数是线性的，其二阶导数为全零。资金总额约束也是线性的，其二阶导数也是全零。
    H[1:n, 1:n] .= T(2*y[1]) .* nlp.Σ 
    H[end,end] = -2*y[1]
    return H
end
####################################################################################################################
# 以下是为了在 Python 中调用 Julia中的CUDA函数而编写的代码
function cupy_to_cumatrix(ptr, rows, cols)
    cu_ptr = CUDA.CuPtr{T}(pyconvert(UInt, ptr))  # 将指针转换为CuPtr
    return CUDA.unsafe_wrap(CUDA.CuMatrix{T}, cu_ptr, (rows, cols))  # 转换为CuMatrix
end

function cupy_to_cuvector(ptr, rows)
    cu_ptr = CUDA.CuPtr{T}(pyconvert(UInt, ptr))  # 将指针转换为CuPtr
    return CUDA.unsafe_wrap(CUDA.CuVector{T}, cu_ptr, rows)  # 转换为CuVector
end

function update_parameters(nlp::NLPModels.AbstractNLPModel, x0::CUDA.CuVector{T}, u::CUDA.CuVector{T}, Q::CUDA.CuMatrix{T}, y0::CUDA.CuVector{T}) where T
    CUDA.copyto!(NLPModels.get_x0(nlp), x0)
    CUDA.copyto!(NLPModels.get_y0(nlp), y0)
    nlp.µ = u
    nlp.Σ = Q
    cholesky_result = CUDA.CUSOLVER.cholesky(CUDA.CUBLAS.Hermitian(nlp.Σ)) # 在 GPU 上计算，确保矩阵是 Hermitian（正定的）
    nlp.U = cholcopy(cholesky_result.U)     #U就是L'
    #nlp.L = cholcopy(cholesky_result.L)
end
