import cupy as cp
from cupy.cuda.memory import MemoryPointer, UnownedMemory
from numpy import float32, float64, inf
from juliacall import Main as jl
from juliacall import VectorValue as jl_VectorValue     # Julia的广义向量类型，既可以是CuVector也可以是Vector
from os.path import exists
import time

# 如果初始数据是DataFrame，可以导入cuDF (使用方法类似pandas)进行GPU上的数据处理。
import cudf as pd

# Julia代码路径，最好能给容器里的绝对路径
JULIA_CODE_PATH = "/nvtest/mean-risk/mean_std_gpu.jl"
# 检查Julia代码路径是否存在
if not exists(JULIA_CODE_PATH):
    raise FileNotFoundError(f"Julia代码路径 {JULIA_CODE_PATH} 不存在")

# 在python中加载Julia代码
jl.include(JULIA_CODE_PATH)

# 数据类型，这里我们使用float64，也可以使用float32，但必须与Julia中的数据类型一致
if jl.T == jl.Float32:
    T = cp.float32      # GPU数组的数据类型
    TOL = 1.2e-6        # 求解精度容忍度(本身为python的float精度), 对于float32，可以适当放宽到5e-6
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

def CuPyArray2JuliaCuMatrix(arr:cp.ndarray):
    # 获取CuPy数组的指针和形状
    ptr = arr.data.ptr      # 获取指针
    rows, cols = arr.shape  # 获取形状
    return jl.cupy_to_cumatrix(ptr, rows, cols)

def PortfolioNLPModelCUDA_Construct(CovMat,             # 协方差矩阵, CuPy数组, shape=(N_Assets, N_Assets), 如果有暴露因子的情况，需要预先将暴露因子合并到协方差矩阵中
                                    Mean_Returns,       # 各资产的平均收益率, CuPy数组, shape=(N_Assets,)
                                    #Cost,              # 各资产的交易成本, CuPy数组, shape=(N_Assets,)
                                    #W0,                # 各资产上一时间点的权重,用于计算买卖的比重, CuPy数组, shape=(N_Assets,)
                                    Lambda_risk,        # 风险厌恶系数, MyFloat类型
                                    W_lb,               # (各资产+Risk)的权重下限, CuPy数组, shape=(N_Assets+1,)
                                    W_ub,               # (各资产+Risk)的权重上限, CuPy数组, shape=(N_Assets+1,)
                                    X0,                 # 各资产的权重初始值, CuPy数组, shape=(N_Assets+1,)
                                    n_con,              # 约束条件（包含等式约束和不等式约束）的个数, int类型    
                                    #Y0,                # 拉格朗日乘子向量，即拉格朗日方程中约束部分的初始系数（里面的元素默认为1）, CuPy数组, shape=(n_con,)
                                    Lcon,               # 约束条件的下限, CuPy数组, shape=(n_con,)
                                    Ucon,               # 约束条件的上限, CuPy数组, shape=(n_con,)
                                    r_f = MyFloat(0.04),          # 无风险利率, 默认为0.04, MyFloat类型
                                    ):
    # 将CuPy数组转换为Julia的CuMatrix或CuVector
    cov_mat = CuPyArray2JuliaCuMatrix(CovMat)
    mean = CuPyArray2JuliaCuVector(Mean_Returns)
    #cost = CuPyArray2JuliaCuVector(Cost)
    #w0 = CuPyArray2JuliaCuVector(W0)
    w_lb = CuPyArray2JuliaCuVector(W_lb)
    w_ub = CuPyArray2JuliaCuVector(W_ub)
    x0 = CuPyArray2JuliaCuVector(X0)
    #y0 = CuPyArray2JuliaCuVector(Y0)
    lcon = CuPyArray2JuliaCuVector(Lcon)
    ucon = CuPyArray2JuliaCuVector(Ucon)
    
    # 调用Julia的函数构建NLPModel模型, 返回一个NLPModel对象. 注意，这里的参数必顫与Julia中的数据类型一致
    return jl.PortfolioNLPModelCUDA_Construct( mean,
                                               r_f,
                                               Lambda_risk,
                                               cov_mat,
                                               n_con,
                                               x0,
                                               w_lb,
                                               w_ub,
                                               lcon,
                                               ucon)

if __name__ == "__main__":
    # 读取量价数据，数据的格式为parquet，包含日期[DATE_TIME,为索引列]和股票代码[共4558列]，数据为股票的量价数据
    df = pd.read_parquet('/nvtest/2022_minute_level/merged.parquet')
    df = df.sort_index()
    df.index = pd.to_datetime(df.index) # 将索引列转换为时间戳，而非字符串

    # 选择在特定日期范围内的行
    #start_date = pd.to_datetime('2022-01-06')
    #end_date = pd.to_datetime('2022-02-06')
    #df = df[start_date:end_date]
    df = df.dropna(axis=1)              # 删除包含缺失值的行
    # 如果需要只选择前2000列，可以使用iloc方法
    #df = df.iloc[:, :2000]

    #print(df.shape)
    N_Count, N_Assets = df.shape        # 数据的行数和列数（即股票的总数）
    #print(df.head())

    pct = df.pct_change().dropna(axis=0)# 计算收益率，并删除包含缺失值的行
    stocks_lr = cp.log(1.0+pct.values)  # 计算对数收益率，并转换为CuPy数组
    #print(stocks_lr[1:5,:])
    N_Data_Count = stocks_lr.shape[0]
    print(f"N_data_count = {N_Data_Count}")

    Stocks_Mean_LR = cp.sum(stocks_lr, axis=0).T
    Cov_Mat = cp.cov(stocks_lr, rowvar=False) * (N_Count)
    #Stocks_Mean_LR = cp.loadtxt('/nvtest/stocks_lr_mean.csv', dtype=T)  # 读取股票的平均收益率
    #Cov_Mat = cp.loadtxt('/nvtest/cov_mat.csv', dtype=T)                # 读取股票的协方差矩阵
    #N_Assets = Stocks_Mean_LR.shape[0]              # 股票数量  
    print("股票数量: ", N_Assets)
    print("Cov_Mat Shape: ", Cov_Mat.shape)
    jl.println("Julia后端调用正常！")               # 调用Julia的println函数，验证Julia环境是否正常

    # 定义模型参数
    N_Var = N_Assets + 1        # 变量的个数，即股票数量+1，多出来的一个变量是辅助变量sqrt(w'Σw)
    Lambda_risk = MyFloat(1.0)  # 风险厌恶系数，由于此参数需要导入到Julia中，所以需要与Julia中的数据类型一致，这里我们使用MyFloat。注意，这里不能使用T类型，因为T类型是GPU上的CuPy数组的数据类型
    Risk_Free_Rate = MyFloat(0.04)  # 无风险利率
    W_min = 0.0                 # 权重下限
    W_max = 1.0                 # 权重上限
    Risk_min = 0.0              # 风险下限，如果需要固定风险收益分析，可以设置风险下限和风险上限都为同一常量
    Risk_max = 1.0              # 风险上限
    W_lb = cp.full(N_Var, W_min, dtype=T)       # 权重下限向量，如果需要对每支股票的权重上下限进行微调，可以在这里进行微调
    W_lb[-1] = T(Risk_min)                      # 最后一个变量是辅助变量sqrt(w'Σw)，所以它的权重下限为0
    W_ub = cp.full(N_Var, W_max, dtype=T)       # 权重上限向量，如果需要对每支股票的权重上下限进行微调，可以在这里进行微调
    W_ub[-1] = T(Risk_max)                      # 最后一个变量是辅助变量sqrt(w'Σw)，所以它的权重上限为1
    X0 = cp.full(N_Var, 1.0/N_Assets, dtype=T)  # 权重计算的初始值向量，这里我们假设所有股票的权重都是相等的，即1/N_Assets。注意这里的X0不参与交易成本的计算，只是用于优化的初始值
    X0[-1] = T(0.2)                             # 最后一个变量是辅助变量sqrt(w'Σw)，所以它的初始值为0.2
    N_con:int = 2                                   # 约束条件的个数（必须是int64类型），这里我们只考虑两项约束，第一为锥约束（即对权重最后一项的辅助量的约束），第二为权重之和为1的约束
    Y0 = cp.full(N_con, 1.0, dtype=T)               # 拉格朗日乘子向量，即拉格朗日方程中约束部分的初始系数（里面的元素默认为1）
    Lcon = cp.array([-10.0, 0.0], dtype=T)      # 约束条件的下限，这里我们只考虑两项，第一为锥约束，第二为权重之和最小为0的约束
    Ucon = cp.array([0.0, 1.0], dtype=T)            # 约束条件的上限，这里我们只考虑两项，第一为锥约束，第二为权重之和最大为1的约束 
    print("开始构建非线性规划的NLPModel模型")
    # 构建NLPModel模型，这里我们直接使用CuPy数组为参数
    julia_model = PortfolioNLPModelCUDA_Construct(Cov_Mat, Stocks_Mean_LR, 
                                                  Lambda_risk, W_lb, W_ub, X0, N_con, 
                                                  Lcon, Ucon, Risk_Free_Rate)
    print("NLPModel模型构建完成，开始调用MadNLP构建求解环境")
    # 创建MadNLP求解器
    julia_solver = jl.MadNLPGPU.MadNLPSolver(   julia_model,        # NLPModels的模型  
                                                tol = TOL,          # 精度容忍度，当两次迭代的目标函数值差小于该值时，停止迭代。对于float64，1.2e-8是一个合理的值, 对于float32，可以适当放宽到1.2e-6
                                                callback = jl.MadNLP.DenseCallback, # 回调函数，用于在每次迭代后记录信息或进行其他操作。MadNLP.DenseCallback是一个简单的回调函数，用于记录每次迭代的信息。
                                                kkt_system = jl.MadNLP.DenseCondensedKKTSystem, # KKT系统求解器，MadNLP.DenseCondensedKKTSystem是一个适用于稠密问题的求解器。
                                                max_iter = 500,    # 最大迭代次数，当迭代次数达到该值时，停止迭代。
                                                jacobian_constant = False,   # 是否为常数雅可比矩阵, 对于Mean-Risk模型，雅可比矩阵不是常数
                                                hessian_constant = False,    # 是否为常数黑塞矩阵, 对于Mean-Risk模型，黑塞矩阵不是常数
                                                linear_solver = jl.MadNLPGPU.LapackGPUSolver,   # 线性求解器，MadNLPGPU.LapackGPUSolver是一个基于cuSolver的求解器。
                                                lapack_algorithm = jl.MadNLP.CHOLESKY,          # 求解器的算法，MadNLP.CHOLESKY是一个求解器的算法。适用于KKT矩阵为对称正定矩阵的情况。
                                                equality_treatment = jl.MadNLP.EnforceEquality, # 等式约束的处理方式，MadNLP.EnforceEquality是一个处理等式约束的方法。
                                                print_level = jl.MadNLP.INFO,                   # 输出信息的级别，具体见https://madnlp.github.io/MadNLP.jl/stable/options/#Output-options
                                            )
    # 首次调用MadNLP求解，注意Julia首次执行会包括编译时间，所以首次求解会相对慢一些。后续求解会快很多
    print("开始第一次调用MadNLP求解")
    julia_results = jl.MadNLPGPU.solve_b(julia_solver)          # 求解最佳权重向量,实际上就是Julia中的MadNLPGPU.solve!()函数，带叹号的函数在python中用_b表示
    # 打印结果
    print("求解状态: ", julia_results.status)                     # 求解状态
    print("目标函数值: ", -(julia_results.objective))           # 目标函数最优化取值
    print("TOP-5权重向量: ")
    print(cp.flip(cp.partition(JuliaCuVector2CuPyArray(julia_results.solution)[:-1], -5)[-5:]))
    print("TOP-5权重的对应股票序号(从0开始)：")
    print(cp.flip(cp.argpartition(JuliaCuVector2CuPyArray(julia_results.solution)[:-1], -5)[-5:]))
    print("Standard Deviation(Risk): ", julia_results.solution[-1])   # 标准差

    # 更新参数，重新求解，这里我们只更新了均值和协方差矩阵
    start_time = time.time()    
    jl.update_parameters(julia_model,                               # NLPModels的模型 
                         julia_results.solution,                    # 上一次求解的权重向量结果，作为下一次求解的初始值
                         CuPyArray2JuliaCuVector(Stocks_Mean_LR),   # 更新后的均值回报率向量
                         CuPyArray2JuliaCuMatrix(Cov_Mat),          # 更新后的协方差矩阵
                         CuPyArray2JuliaCuVector(Y0)                # 更新后的拉格朗日乘子向量
                        )    
    print("更新参数耗时: ", time.time() - start_time)
    # 再次调用MadNLP求解
    print("更新参数后，开始第二次调用MadNLP求解")
    julia_results = jl.MadNLPGPU.solve_b(julia_solver, mu_init = 1e-7)        # mu_init参数表示初始的障碍参数（barrier parameter），如果是以上次求解的结果为输入的情况，可以把mu_init设置为1e-7，如果不设置，默认为1e-1
    # 打印结果, 注意，这里的结果是第二次求解的结果
    print("求解状态", julia_results.status)                     # 求解状态
    print("目标函数值: ", -(julia_results.objective))           # 目标函数最优化取值
    print("TOP-5权重向量: ")
    print(cp.flip(cp.partition(JuliaCuVector2CuPyArray(julia_results.solution)[:-1], -5)[-5:]))
    print("TOP-5权重的对应股票序号(从0开始):")
    print(cp.flip(cp.argpartition(JuliaCuVector2CuPyArray(julia_results.solution)[:-1], -5)[-5:]))
    print("Standard Deviation(Risk): ", julia_results.solution[-1])   # 标准差

    # 使用cuDF和CuPy来从量价数据中计算对数收益率和协方差矩阵
