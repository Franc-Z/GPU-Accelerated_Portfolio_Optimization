# 导入必要的模块，使用 import
import NLPModels: AbstractNLPModel, NLPModelMeta, Counters 
import NLPModels
import MadNLP
import MadNLPGPU
import CUDA
import LinearAlgebra: isposdef, cholcopy, norm_sqr, BLAS, norm
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
# 定义自定义的 NLPModel,以此定义优化模型的数据结构
# ==========================

# 自定义的 NLPModel，用于在 GPU 上执行计算
mutable struct PortfolioNLPModelCUDA{
    T,
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T},
    VI <: AbstractVector{MyInt},
    } <: AbstractNLPModel{T, VT}
    meta::NLPModelMeta{T, VT}       # NLPModel元数据(由 NLPModels 定义的结构体), 包括变量和约束的数量等信息, 以及初始点和上下界, 是否最小化等信息。具体参考https://jso.dev/NLPModels.jl/v0.18/models/#NLPModels.NLPModelMeta
    Counters::Counters              # 用于记录计算次数的计数器
    µ::VT                           # 平均log回报率向量µ
    Cost::VT                        # 交易成本向量，维数与µ相同
    Class::VI                       # 每个资产所属的类别编号（必须连续，且从1开始），维数与µ相同
    w0::VT                          # 优化前的权重向量，维数与µ相同，用于计算交易成本
    V_Buffer::VT                    # 临时变量，用于存储中间计算结果
    #Σ::MT                          # Σ矩阵，即考虑暴露因子后的协方差矩阵（一般为50x50）
    Q::MT                           # 实际计算Hessian时用到的协方差矩阵, 维度为(资产数量×资产数量), 为expo * Σ * expo'
    U::MT                           # 对Σ矩阵进行Cholesky分解后的LU矩阵中的U矩阵，即Σ = LU (L = U', U = L')
    λ_risk::T                       # 风险厌恶系数（标量）
    Class_Count::MyInt              # 行业的数量(标量)
end

# 构造函数
function PortfolioNLPModelCUDA_Construct(
    u::VT,           # 平均log回报率向量
    cost::VT,        # 交易成本向量
    cls::VI,         # 每个资产所属的类别向量
    w0::VT,          # 优化前的权重向量
    λ_risk::T,       # λ_risk是风险厌恶系数
    Q::MT,           # 协方差矩阵
    n_con::Int,      # 约束数量,包括等式约束和不等式约束
    x0::VT,          # 自变量向量的初始赋值
    y0::VT,          # 拉格朗日方程中，约束项的系数的初始赋值，其维数与约束数量相同
    lvar::VT,        # 自变量向量的下界，即 x[i] >= lvar[i]
    uvar::VT,        # 自变量向量的上界，即 x[i] <= uvar[i]
    lcon::VT,        # 约束向量的下界，即 c[i] >= lcon[i]
    ucon::VT,        # 约束向量的上界，即 c[i] <= ucon[i]
    cls_count::Int   # 行业类别数量
    ) where {T}
    nvar = length(x0)               # 自变量的维数
    n_assets = nvar                 # 在Mean-Variance问题中，资产数量即为自变量的维数
    #n_expos = size(Σ,1)            # 暴露度矩阵的行数即为暴露度的维数
    
    x0_ = x0
    y0_ = y0
    u_ = u
    cost_ = cost
    cls_ = cls
    w0_ = w0
    #Σ_ = CUDA.copyto!(CUDA.similar(x0_, n_expos, n_expos), Σ)              # 输入的协方差矩阵的维度一般为(因子数量×因子数量)
    #Expo_ = CUDA.copyto!(CUDA.similar(x0_, n_assets, n_expos), Expo)       # 输入的暴露度矩阵的维度一般为(资产数量×因子数量)
    Q_ = Q
    cholesky_result = CUDA.CUSOLVER.cholesky(CUDA.CUBLAS.Hermitian(Q_))     # 在 GPU 上计算，确保矩阵是 Hermitian
    U_ = cholcopy(cholesky_result.U)                                        # U就是L'
    lvar_ = lvar
    uvar_ = uvar
    lcon_ = lcon
    ucon_ = ucon
    v_buffer_ = CUDA.similar(x0_, n_assets)
    
    meta = NLPModelMeta(nvar,
                        ncon = n_con,   
                        x0 = x0_,
                        y0 = y0_,
                        lvar = lvar_,
                        uvar = uvar_,
                        lcon = lcon_,
                        ucon = ucon_,
                        minimize = true
                        )    
    return PortfolioNLPModelCUDA(   meta, 
                                    Counters(), 
                                    u_, 
                                    cost_, 
                                    cls_,
                                    w0_, 
                                    v_buffer_, 
                                    Q_, 
                                    #Expo_,
                                    U_, 
                                    #U_, 
                                    #J, 
                                    λ_risk, 
                                    cls_count,
                                )
end

# ==========================
# 实现 NLPModels 所需的函数
# ==========================
# 目标函数
function NLPModels.obj(nlp::PortfolioNLPModelCUDA{T,VT,MT}, x::AbstractVector{T}) where {T, VT, MT}
    nlp.Counters.neval_obj += 1
    h_var = norm_sqr(nlp.U * x)
    # 计算目标函数值（取负号，因为默认是最小化）
    return -(CUDA.dot(nlp.µ, x) - CUDA.dot(CUDA.FastMath.abs_fast.(x - nlp.w0), nlp.Cost) - nlp.λ_risk * h_var)
end

# 目标函数梯度
function NLPModels.grad!(nlp::PortfolioNLPModelCUDA{T,VT,MT}, x::AbstractVector{T}, g::AbstractVector{T}) where {T, VT, MT}
    nlp.Counters.neval_grad += 1
    # 对 x_i 的偏导数
    CUDA.CUBLAS.mul!(g, nlp.Q, x)
    g .*= T(2.0) * nlp.λ_risk
    g .-= nlp.µ - CUDA.FastMath.sign_fast.(x - nlp.w0) .* nlp.Cost
    return g
end

# CUDA核函数，用于求各行业的权重和。x为所以股票的权重数组，cls为维度与x相同的整数编号数组（用于表示每只股票的行业分类），cls_sum为每类行业的权重和，n为股票的总支数；
function sum_by_class_kernel!(x, cls, cls_sum, n, cls_count)
    idx = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if idx <= n
        cls_idx = cls[idx]
        val = x[idx]
        if cls_idx <= cls_count && cls_idx >= 1
            CUDA.@atomic cls_sum[cls_idx] += val
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

# 约束函数
function NLPModels.cons!(nlp::PortfolioNLPModelCUDA{T,VT,MT}, x::AbstractVector{T}, c::AbstractVector{T}) where {T, VT, MT}
    nlp.Counters.neval_cons += 1
    n = nlp.meta.nvar
    CUDA.fill!(c, zero(T))
    CUDA.fill!(nlp.V_Buffer, zero(T))
    c[1] = CUDA.sum(x)  # 资金总和约束
    
    # 调用 CUDA 核函数
    threads = 256  # 每个线程块的线程数
    blocks = ceil(Int, n / threads)  # 计算需要的线程块数
    CUDA.@cuda threads=threads blocks=blocks sum_by_class_kernel!(x, nlp.Class, nlp.V_Buffer, n, nlp.Class_Count)
    c[2:1+nlp.Class_Count] .= nlp.V_Buffer[1:nlp.Class_Count]
    #println(c)
    return c
end

# 约束的Jacobian 矩阵（密集形式）
function MadNLP.jac_dense!(nlp::PortfolioNLPModelCUDA{T,VT,MT}, x::AbstractVector{T}, J::AbstractMatrix{T}) where {T, VT, MT}
    nlp.Counters.neval_jac += 1
    n = nlp.meta.nvar
    n_con = nlp.meta.ncon
    CUDA.fill!(J, zero(T))
    J[1, 1:end] .= one(T)   # 约束条件只有一个，即权重求和项，因此对x[i]的导数均为1，因此此处填充J为1
    threads = 256           # 每个线程块的线程数
    blocks = ceil(Int, n / threads)  # 计算需要的线程块数
    CUDA.@cuda threads=threads blocks=blocks set_Jacobian_by_class_kernel!(nlp.Class, J, n_con, n, 2)
    return J                #对于线性约束，J不随x变化
end

# 拉格朗日函数的Hessian 矩阵（密集形式）
function MadNLP.hess_dense!(nlp::PortfolioNLPModelCUDA{T,VT,MT}, x::AbstractVector{T}, y::AbstractVector{T}, H::AbstractMatrix{T}; obj_weight=1.0) where {T, VT, MT}
    nlp.Counters.neval_hess += 1
    CUDA.copyto!(H, T(2*obj_weight*nlp.λ_risk).* nlp.Q)
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

function update_parameters(nlp::NLPModels.AbstractNLPModel, x0::CUDA.CuVector{T}, u::CUDA.CuVector{T}, Q::CUDA.CuMatrix{T}, y0::CUDA.CuVector{T}) where {T}
    CUDA.copyto!(NLPModels.get_x0(nlp), x0)
    CUDA.copyto!(NLPModels.get_y0(nlp), y0)
    nlp.µ = u
    nlp.Q = Q
    cholesky_result = CUDA.CUSOLVER.cholesky(CUDA.CUBLAS.Hermitian(Q)) # 在 GPU 上计算，确保矩阵是 Hermitian（正定的）
    nlp.U = cholcopy(cholesky_result.U)     #U就是L'
    return nothing
end
