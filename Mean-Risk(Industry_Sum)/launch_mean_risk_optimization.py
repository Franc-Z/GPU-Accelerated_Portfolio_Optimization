import cupy as cp
from cupy.cuda.memory import MemoryPointer, UnownedMemory
from numpy import float32, float64
from juliacall import Main as jl
from juliacall import VectorValue as jl_VectorValue     # Julia的广义向量类型，既可以是CuVector也可以是Vector
from os.path import exists
from time import time

# 如果初始数据是DataFrame，可以导入cuDF (使用方法类似pandas)
#import cudf as pd

# Julia代码路径，最好能给容器里的绝对路径
JULIA_CODE_PATH = "/nvtest/GPU-Accelerated_Portfolio_Optimization/Mean-Risk(Industry_sum)/mean_std_gpu.jl"
# 检查Julia代码路径是否存在
if not exists(JULIA_CODE_PATH):
    raise FileNotFoundError(f"Julia代码路径 {JULIA_CODE_PATH} 不存在")

# 在python中加载Julia代码
jl.include(JULIA_CODE_PATH)

# 数据类型，这里我们使用float64，也可以使用float32，但必须与Julia中的数据类型一致
if jl.T == jl.Float32:
    T = cp.float32      # GPU数组的数据类型
    TOL = 5e-6          # 求解精度容忍度(本身为python的float精度), 对于float32，可以适当放宽到5e-6
    MyFloat = float32   # Python中浮点数的数据类型
else:
    T = cp.float64      # GPU数组的数据类型
    TOL = 1.2e-8        # 求解精度容忍度(本身为python的float精度), 对于float64，1.2e-8是一个合理的值
    MyFloat = float64   # Python中浮点数的数据类型
  
# 定义一个函数，将Julia的CuVector转换为CuPy的数组
def JuliaCuVector2CuPyArray(jl_arr:jl_VectorValue):
    # 假设我们从Juliacall获得了一个CUDA指针pDevice
    pDevice = (jl.Int(jl.pointer(jl_arr)))                # 从Juliacall获取的CUDA指针
    #print(type(pDevice))
    span = jl.size(jl_arr)                       # 数组长度
    dtype = jl.eltype(jl_arr)                    # 数据类型
    # 判断Julia的CuVector的数据类型，如果是Float32，则转换为CuPy的float32类型，如果是Float64，则转换为CuPy的float64类型
    if dtype==jl.Float64:
        dtype = cp.float64
    else:
        dtype = cp.float32
    # 计算数组的字节大小
    sizeByte = int(span[0] * cp.dtype(dtype).itemsize)      #这里我们只考虑了一维向量的情况
    #print(type(sizeByte))

    # 创建CuPy的UnownedMemory对象
    mem = UnownedMemory(pDevice, sizeByte, owner=None)
    
    # 创建MemoryPointer对象
    memptr = MemoryPointer(mem, 0)
    
    # 创建CuPy数组
    arr = cp.ndarray(shape=span, dtype=dtype, memptr=memptr)
    
    return arr


def CuPyArray2JuliaCuVector(arr:cp.ndarray):
    # 获取CuPy数组的指针和形状
    ptr = arr.data.ptr  # 获取指针
    rows = arr.shape    # 获取形状
    return jl.cupy_to_cuvector(ptr, rows)

def CuPyArray2JuliaCuIntVector(arr:cp.ndarray):
    # 获取CuPy数组的指针和形状
    ptr = arr.data.ptr  # 获取指针
    rows = arr.shape    # 获取形状
    if arr.dtype == cp.int64:
        MyType = jl.Int64
    else:
        MyType = jl.Int32
    return jl.cupy_to_cuvector_precision(ptr, rows, MyType)

def CuPyArray2JuliaCuMatrix(arr:cp.ndarray):
    # 获取CuPy数组的指针和形状
    ptr = arr.data.ptr  # 获取指针
    rows, cols = arr.shape    # 获取形状
    return jl.cupy_to_cumatrix(ptr, rows, cols)

def PortfolioNLPModelCUDA_Construct(CovMat,             # 协方差矩阵, CuPy数组, shape=(N_Assets, N_Assets), 如果有暴露因子的情况，需要预先将暴露因子合并到协方差矩阵中
                                    Mean_Returns,       # 各资产的平均收益率, CuPy数组, shape=(N_Assets,)
                                    Cost,               # 各资产的交易成本, CuPy数组, shape=(N_Assets,)
                                    Cls,                # 各资产的分类向量, CuPy数组, shape=(N_Assets,)
                                    W0,                 # 各资产上一时间点的权重,用于计算买卖的比重, CuPy数组, shape=(N_Assets,)
                                    Lambda_risk,        # 风险厌恶系数, MyFloat类型
                                    W_lb,               # 各资产的权重下限, CuPy数组, shape=(N_Assets,)
                                    W_ub,               # 各资产的权重上限, CuPy数组, shape=(N_Assets,)
                                    X0,                 # 各资产的权重初始值, CuPy数组, shape=(N_Assets,)
                                    n_con,              # 约束条件（包含等式约束和不等式约束）的个数, int类型    
                                    #Y0,                 # 拉格朗日乘子向量，即拉格朗日方程中约束部分的初始系数（里面的元素默认为1）, CuPy数组, shape=(n_con,)
                                    Lcon,               # 约束条件的下限, CuPy数组, shape=(n_con,)
                                    Ucon,               # 约束条件的上限, CuPy数组, shape=(n_con,)
                                    r_f,                # 无风险利率, MyFloat类型
                                    ):
    # 将CuPy数组转换为Julia的CuMatrix或CuVector
    cov_mat = CuPyArray2JuliaCuMatrix(CovMat)
    mean = CuPyArray2JuliaCuVector(Mean_Returns)
    cost = CuPyArray2JuliaCuVector(Cost)
    w0 = CuPyArray2JuliaCuVector(W0)
    w_lb = CuPyArray2JuliaCuVector(W_lb)
    w_ub = CuPyArray2JuliaCuVector(W_ub)
    x0 = CuPyArray2JuliaCuVector(X0)
    #y0 = CuPyArray2JuliaCuVector(Y0)
    lcon = CuPyArray2JuliaCuVector(Lcon)
    ucon = CuPyArray2JuliaCuVector(Ucon)
    cls = CuPyArray2JuliaCuIntVector(Cls)
    cls_stat = cp.unique(Cls)
    cls_count = len(cls_stat)                # 行业类别的数量
    print(f"行业类别个数={cls_count}")
    return jl.PortfolioNLPModelCUDA_Construct(mean, r_f, cost, cls, w0, Lambda_risk, cov_mat, n_con, x0, w_lb, w_ub, lcon, ucon, cls_count)

if __name__ == "__main__":
    Stocks_Mean_LR = cp.loadtxt('/nvtest/stocks_lr_mean.csv', dtype=T)  # 读取股票的平均收益率
    Cov_Mat = cp.loadtxt('/nvtest/cov_mat.csv', dtype=T)                # 读取股票的协方差矩阵
    N_Assets = Stocks_Mean_LR.shape[0]              # 股票数量  
    jl.println("Julia后端调用正常！")               # 调用Julia的println函数，验证Julia环境是否正常

    Cost = cp.full(N_Assets, 0.002, dtype=T)        # 交易成本，这里我们假设所有股票的交易成本都是0.2%
    W0 = cp.full(N_Assets, 1.0/N_Assets, dtype=T)                # 上一时间点的权重向量
    # 为了验证结果，我们将部分股票的权重设置为非0值
    '''
    W0[1301] = T(0.39949818)                
    W0[1415] = T(0.32822413)
    W0[256] = T(0.14476788)
    W0[797] = T(0.07121188)
    W0[1299] = T(0.05629793)
    '''
    Industry_Count = 40                            # 行业类别数量
    N_Var = N_Assets + 1        # 变量个数，最后一个为辅助变量t，代表锥约束
    Lambda_risk = MyFloat(1.0)  # 风险厌恶系数，由于此参数需要导入到Julia中，所以需要与Julia中的数据类型一致，这里我们使用MyFloat。注意，这里不能使用T类型，因为T类型是GPU上的CuPy数组的数据类型
    W_min = 0.0                 # 权重下限
    W_max = 0.1                 # 权重上限
    Risk_min = 0.0              # 风险下限
    Risk_max = 1.0              # 风险上限
    Risk_Free_Rate = MyFloat(0.04)   # 无风险利率
    Industry_Weight_Max = 0.1    # 行业权重上限
    W_lb = cp.full(N_Var, W_min, dtype=T)        # 权重下限向量（添加对自变量的约束，可以改这里的数组元素）
    W_ub = cp.full(N_Var, W_max, dtype=T)        # 权重上限向量（添加对自变量的约束，可以改这里的数组元素）
    W_lb[-1] = Risk_min
    W_ub[-1] = Risk_max
    X0 = cp.full(N_Var, 1.0/N_Assets, dtype=T)   # 权重计算的初始值向量，这里我们假设所有股票的权重都是相等的，即1/N_Assets。注意这里的X0不参与交易成本的计算，只是用于优化的初始值
    X0[-1] = Risk_min+0.1
    N_con:int = 1 + 1 + Industry_Count              # 约束条件的个数（必须是int64类型），这里我们考虑锥约束、权重之和为1的约束和行业权重约束
    Y0 = cp.full(N_con, 1.0, dtype=T)               # 拉格朗日乘子向量，即拉格朗日方程中约束部分的初始系数（里面的元素默认为1）
    Lcon = cp.full(N_con, 0.0, dtype=T)             # 约束条件的下限，这里我们只考虑权重之和最小为0的约束
    Lcon[0] = cp.NINF                               # 权重总和约束下限
    Ucon = cp.full(N_con, Industry_Weight_Max, dtype=T)             # 约束条件的上限，这里我们只考虑权重之和最大为1的约束 
    Ucon[0] = 0.0                                   # 锥约束上限
    Ucon[1] = 1.0                                   # 权重总和约束上限
    #Cls = cp.random.randint(1, Industry_Count+1, size=N_Assets, dtype=cp.int64)  # 分类向量（从1开始，与Julia代码中MyInt的格式相同），用于分类权重和约束，这里我们随机生成一个分类向量，每个元素的值为1,2,3,4,...,N_con中的一个
    Cls = cp.loadtxt('/nvtest/industry_labels.csv', dtype=cp.int64)
    print("开始构建非线性规划的NLPModel模型")
    # 构建NLPModel模型，这里我们直接使用CuPy数组为参数
    julia_model = PortfolioNLPModelCUDA_Construct(Cov_Mat, Stocks_Mean_LR, Cost, Cls, W0, Lambda_risk, W_lb, W_ub, X0, N_con, Lcon, Ucon, Risk_Free_Rate)
    print("NLPModel模型构建完成，开始调用MadNLP构建求解环境")
    # 创建MadNLP求解器
    julia_solver = jl.MadNLPGPU.MadNLPSolver(   julia_model,        # NLPModels的模型  
                                                tol = TOL,          # 精度容忍度，当两次迭代的目标函数值差小于该值时，停止迭代。对于float64，1.2e-8是一个合理的值, 对于float32，可以适当放宽到1.2e-6
                                                callback = jl.MadNLP.DenseCallback, # 回调函数，用于在每次迭代后记录信息或进行其他操作。MadNLP.DenseCallback是一个简单的回调函数，用于记录每次迭代的信息。
                                                kkt_system = jl.MadNLP.DenseCondensedKKTSystem, # KKT系统求解器，MadNLP.DenseCondensedKKTSystem是一个适用于稠密问题的求解器。
                                                max_iter = 200,    # 最大迭代次数，当迭代次数达到该值时，停止迭代。
                                                jacobian_constant = False,   # 是否为常数雅可比矩阵, 对于Mean-Var模型，雅可比矩阵是常数
                                                hessian_constant = False,    # 是否为常数黑塞矩阵, 对于Mean-Var模型，黑塞矩阵是常数
                                                linear_solver = jl.MadNLPGPU.LapackGPUSolver,   # 线性求解器，MadNLPGPU.LapackGPUSolver是一个基于cuSolver的求解器。
                                                lapack_algorithm = jl.MadNLP.CHOLESKY,          # 求解器的算法，MadNLP.CHOLESKY是一个求解器的算法。适用于KKT矩阵为对称正定矩阵的情况。
                                                equality_treatment = jl.MadNLP.RelaxEquality, # 等式约束的处理方式，MadNLP.EnforceEquality是一个处理等式约束的方法。
                                                print_level = jl.MadNLP.INFO,                   # 输出信息的级别，具体见https://madnlp.github.io/MadNLP.jl/stable/options/#Output-options
                                            )
    # 首次调用MadNLP求解，注意Julia首次执行会包括编译时间，所以首次求解会相对慢一些。后续求解会快很多
    print("开始第一次调用MadNLP求解")
    julia_results = jl.MadNLPGPU.solve_b(julia_solver)          # 求解最佳权重向量,实际上就是Julia中的MadNLPGPU.solve!()函数，带叹号的函数在python中用_b表示
    print("不计Warm-up求解时间，从第二次开始正式计时")
    start_time = time()
    julia_results = jl.MadNLPGPU.solve_b(julia_solver)
    print(f"求解用时 = {time()-start_time}")
    
    # 打印结果
    print("求解状态: ", julia_results.status)                   # 求解状态
    print("目标函数值: ", -(julia_results.objective))           # 目标函数最优化取值
    
    TopK = 15
    print(f"TOP-{TopK}权重向量: ")
    x_result = JuliaCuVector2CuPyArray(julia_results.solution[:-1])
    print(cp.flip(cp.partition(x_result, -TopK)[-TopK:]))
    print(f'平均回报率 = {x_result @ Stocks_Mean_LR}')
    print(f"Risk = {julia_results.solution[-1]}")

    # 更新参数，重新求解，这里我们只更新了均值和协方差矩阵
    jl.update_parameters(julia_model,                               # NLPModels的模型 
                         julia_results.solution,                    # 上一次求解的权重向量结果，作为下一次求解的初始值
                         CuPyArray2JuliaCuVector(Stocks_Mean_LR),   # 更新后的均值回报率向量
                         CuPyArray2JuliaCuMatrix(Cov_Mat),          # 更新后的协方差矩阵
                         CuPyArray2JuliaCuVector(Y0)                # 更新后的拉格朗日乘子向量
                         )
    
    print("更新参数后，开始第二次调用MadNLP求解")
    julia_results = jl.MadNLPGPU.solve_b(julia_solver, mu_init = 1e-7)
    # 打印结果
    print("求解状态: ", julia_results.status)                   # 求解状态
    print("目标函数值: ", -(julia_results.objective))           # 目标函数最优化取值
    print(f"TOP-{TopK}权重向量: ")
    x_result = JuliaCuVector2CuPyArray(julia_results.solution[:-1])
    print(cp.flip(cp.partition(x_result, -TopK)[-TopK:]))
    print(f'平均回报率 = {x_result @ Stocks_Mean_LR}')
    print(f"Risk = {julia_results.solution[-1]}")
