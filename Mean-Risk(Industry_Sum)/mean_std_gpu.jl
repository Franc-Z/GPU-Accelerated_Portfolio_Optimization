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
MyInt = Int64           # 整数类型，用于表示股票的行业分类，一定要是Int64类型
VI = CUDA.CuVector{MyInt}  # 使用MyInt类型的向量类型CuVector

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
    VI <: AbstractVector{MyInt},
    } <: AbstractNLPModel{T, VT}  # 继承自 AbstractNLPModel
    meta::NLPModelMeta{T, VT}       # x0每次求解需要更新
    counters::Counters
    µ::VT                           # 平均log回报率u，每次求解需要更新
    Cost::VT                        # 交易成本向量，维数与µ相同
    Class::VI                       # 每个资产所属的类别编号（必须连续，且从1开始），维数与µ相同
    w0::VT                          # 优化前的权重向量，维数与µ相同，用于计算交易成本
    V_Buffer::VT                    # 存放向量的缓存
    Σ::MT                           # 协方差矩阵Σ
    #L::MT                          # L矩阵每次求解需要更新
    U::MT                           # 对协方差矩阵进行LU分解的U矩阵，U矩阵每次求解需要更新
    r_f::T                          # 对数(log)无风险利率r_f
    λ_risk::T                       # 风险厌恶系数delta
    Class_Count::MyInt              # 行业的数量(标量)
end

# 构造函数
function PortfolioNLPModelCUDA_Construct(
    u::VT, 
    r_f::T, 
    cost::VT,        # 交易成本向量
    cls::VI,         # 每个资产所属的类别向量
    w0::VT,          # 优化前的权重向量
    delta::T,  # delta is λ_risk
    Q::MT,
    n_con::Int,
    x0::VT, 
    #y0::AbstractVector{T},
    lvar::VT, 
    uvar::VT, 
    lcon::VT, 
    ucon::VT,
    cls_count::Int   # 行业类别数量
    ) where {T, VT, MT}
    nvar = length(x0)
    n_assets = nvar - 1
    
    x0_ = x0
    u_ = u .- r_f
    cost_ = cost
    cls_ = cls
    w0_ = w0
    cls_count_ = cls_count
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
    return PortfolioNLPModelCUDA(meta, Counters(), u_, cost_, cls_, w0_, v_buffer_, Σ_, U_, r_f, delta, cls_count_)
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
    return (-((CUDA.dot(nlp.µ, x_var)+ nlp.r_f) - CUDA.dot(CUDA.FastMath.abs_fast.(x_var - nlp.w0), nlp.Cost) - nlp.λ_risk * t))
end

# 目标函数梯度（g为梯度向量）
function NLPModels.grad!(nlp::PortfolioNLPModelCUDA, x::AbstractVector{T}, g::AbstractVector{T}) where T
    nlp.counters.neval_grad += 1
    n = nlp.meta.nvar - 1
    x_var = x[1:n]
    # 对 x_i 的偏导数
    g[1:n] .=  CUDA.FastMath.sign_fast.(x_var - nlp.w0) .* nlp.Cost - nlp.µ
    # 对 t 的偏导数
    g[end] = nlp.λ_risk
    return g
end

# CUDA核函数，用于求各行业的权重和。x为所以股票的权重数组，cls为维度与x相同的整数编号数组（用于表示每只股票的行业分类），cls_sum为每类行业的权重和，n为股票的总支数；
function sum_by_class_kernel!(x, cls, cls_sum, n, cls_count)
    idx = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if idx <= n
        cls_idx = cls[idx]
        if cls_idx <= cls_count && cls_idx >= 1
            #CUDA.@atomic cls_sum[cls_idx] += x[idx]
            CUDA.atomic_add!(CUDA.pointer(cls_sum, cls_idx), x[idx])
        end
    end   
    return nothing
end

# CUDA核函数，用于设定条件约束的Jacobian矩阵。
# 其中cls为维度与x相同的整数编号数组（用于表示每只股票的行业分类），J为条件约束的雅可比矩阵，n_con为条件约束的个数，n为股票的总支数，start_idx为条件约束的起始序号（一般为2，第一个是资金总和约束）
function set_Jacobian_by_class_kernel!(cls, J, n_con, n, start_idx)
    idx = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if idx <= n
        cls_idx = cls[idx] + start_idx - 1
        if cls_idx <= n_con && cls_idx >= start_idx
            J[cls_idx, idx] = 1.0
        end
    end   
    return nothing
end

# 约束函数，目前约束有两项：二阶锥约束和资金总和约束（c为约束方程组的左边部分，c的元素个数就是约束的总数）
function NLPModels.cons!(nlp::PortfolioNLPModelCUDA, x::AbstractVector{T}, c::AbstractVector{T}) where T
    nlp.counters.neval_cons += 1
    n = nlp.meta.nvar - 1
    x_var = x[1:n]
    t = x[end]

    #CUDA.fill!(c, zero(T))
    
    # 在 GPU 上计算
    CUDA.CUBLAS.mul!(nlp.V_Buffer, nlp.U, x_var)
    norm_Ux_sqr = norm_sqr(nlp.V_Buffer)
    c[1] = norm_Ux_sqr - t*t      # 二阶锥约束
    c[2] = CUDA.sum(x_var)  # 资金总和约束

    # 调用 CUDA 核函数
    CUDA.fill!(nlp.V_Buffer, zero(T))           # 后面用nlp.V_Buffer来存储每类行业的权重和，所以必须先将nlp.V_Buffer清零
    threads = 64  # 每个线程块的线程数
    blocks = ceil(Int, n / threads)  # 计算需要的线程块数
    CUDA.@cuda threads=threads blocks=blocks sum_by_class_kernel!(x_var, nlp.Class, nlp.V_Buffer, n, nlp.Class_Count)
    c[3:2+nlp.Class_Count] .= nlp.V_Buffer[1:nlp.Class_Count]
    return c
end

# 约束方程的Jacobian 矩阵（即约束方程的一阶导数矩阵，密集矩阵形式）
function MadNLP.jac_dense!(nlp::PortfolioNLPModelCUDA, x::AbstractVector{T}, J::AbstractMatrix{T}) where T
    nlp.counters.neval_jac += 1
    nvar = nlp.meta.nvar
    n = nvar - 1
    x_var = x[1:n]
    t = x[end]
    CUDA.fill!(J, zero(T))

    # 第一行对应二阶锥约束，对 x[i] 的偏导数为2*Σ*x[i]，对 t 的偏导数为-2*t
    CUDA.CUBLAS.mul!(nlp.V_Buffer, nlp.Σ, x_var)
    J[1, 1:n] = T(2).*nlp.V_Buffer'    
    J[1, end] = T(-2)*t

    # 第二行对应资金总和约束，对 x[i] 的偏导数为1
    J[2, 1:n] .= one(T)
    #J[2, end] = zero(T)

    threads = 64           # 每个线程块的线程数
    blocks = ceil(Int, n / threads)  # 计算需要的线程块数
    CUDA.@cuda threads=threads blocks=blocks set_Jacobian_by_class_kernel!(nlp.Class, J, nlp.meta.ncon, n, 3)
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

function cupy_to_cuvector_precision(ptr, rows, my_precision)
    cu_ptr = CUDA.CuPtr{my_precision}(pyconvert(UInt, ptr))  # 将指针转换为CuPtr
    return CUDA.unsafe_wrap(CUDA.CuVector{my_precision}, cu_ptr, rows)  # 转换为CuVector
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
