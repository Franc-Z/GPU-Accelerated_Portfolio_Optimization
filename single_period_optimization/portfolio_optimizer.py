import numpy as np
from time import time
from os import path, environ
from juliacall import Main as jl
# 设置Julia环境变量
environ['PYTHON_JULIACALL_THREADS'] = '1'
environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = 'yes'
environ['PYTHON_JULIACALL_OPTIMIZE'] = '3'
environ['PYTHON_JULIACALL_COMPILE'] = 'yes'  # 最小编译 (可选)
# 其他可选的Julia性能设置
environ['JULIA_CPU_TARGET'] = 'native'  # 为本地CPU架构优化
#environ['JULIA_MAX_NUM_PRECOMPILE_FILES'] = '100'  # 增加预编译文件上限
environ['JULIA_PKG_PRECOMPILE_AUTO'] = '1'  # 自动预编译

# 加载Julia脚本并获取模块
julia_script_path = '/nvtest/GPU-Accelerated_Portfolio_Optimization/single_period_optimization/Mean-Variance.jl'  
#julia_script_path = '/nvtest/GPU-Accelerated_Portfolio_Optimization/single_period_optimization/Mean-Risk.jl'
assert path.exists(julia_script_path), f"Julia script not found: {julia_script_path}"

jl.include(julia_script_path)

if jl.MyFloat == jl.Float64:
    MyFloat = np.float64
else:
    MyFloat = np.float32

class PortfolioOptimizer:
    def __init__(self, n_assets, n_style, lbd_risk):
        self.n_assets = n_assets
        self.n_style = n_style
        self.lbd_risk = lbd_risk
        self.pm = None

    def setup_model(self, x0, cov, expo, bias, cost, u_cpu=None):
        if u_cpu is None:
            u_cpu = np.zeros(self.n_assets, dtype=MyFloat)
        # 使用模块中的函数
        self.pm = jl.CreatePortfolioModel(
            self.n_assets, self.n_style, self.lbd_risk,
            x0, cov, expo, bias, cost, u_cpu
        )
        jl.setup_model_b(self.pm)

    def initial_solve(self):
        # 获取并返回优化结果
        jl.initial_solve_b(self.pm)
        
    def resolve(self, new_u_cpu):
        # 更新回报率并重新优化
        jl.update_return_ratio_b(self.pm, new_u_cpu)
        jl.resolve_b(self.pm)
        

def load_data():
    cov = np.loadtxt('/nvtest/GPU-Accelerated_Portfolio_Optimization/single_period_optimization/cov_41_41.csv', delimiter=',', dtype=MyFloat)
    cov *= 12.0
    expo = np.loadtxt('/nvtest/GPU-Accelerated_Portfolio_Optimization/single_period_optimization/expo_4558_41.csv', delimiter=',', dtype=MyFloat)
    expo = np.nan_to_num(expo)
    bias = np.loadtxt('/nvtest/GPU-Accelerated_Portfolio_Optimization/single_period_optimization/bias_4558.csv', dtype=MyFloat)
    bias = np.nan_to_num(bias)*12.0
    return_ratio = np.loadtxt('/nvtest/GPU-Accelerated_Portfolio_Optimization/single_period_optimization/stock_return_2023.csv', dtype=MyFloat)
    cost = np.full_like(return_ratio, 0.002, dtype=MyFloat)
    return cov, expo, bias, return_ratio, cost

def generate_data(N = 50000, N_style = 41):
    #cov = np.loadtxt('/nvtest/cov_41_41.csv', delimiter=',', dtype=MyFloat)
    #cov *= 12.0
    cov = np.random.randn(N_style, N_style)
    cov = np.dot(cov, cov.T) * 0.5 + np.eye(N_style) * 1e-8
    expo = np.random.randn(N, N_style)
    bias = np.random.randn(N)
    bias *= bias
    return_ratio = (np.random.randn(N)*9.0 + 3.0) / 100.0
    cost = np.full_like(return_ratio, 0.002, dtype=MyFloat)
    return cov, expo, bias, return_ratio, cost
    
def test_4558():
    cov, expo, bias, return_ratio, cost = load_data()
    return_ratio_new = np.loadtxt("/nvtest/GPU-Accelerated_Portfolio_Optimization/single_period_optimization/stock_return_2023.csv", dtype=MyFloat)
    n_assets = len(return_ratio)
    n_style = expo.shape[1]
    x0 = np.full(n_assets, 1.0/n_assets, dtype=MyFloat)
    optimizer = PortfolioOptimizer(n_assets, n_style, MyFloat(1.0))
    optimizer.setup_model(x0, cov, expo, bias, cost, u_cpu=return_ratio)
    optimizer.initial_solve()
    print(f"risk = {jl.get_risk(optimizer.pm)}")
    print(f'objective = {jl.JuMP.objective_value(optimizer.pm.model)}')
    start = time()
    #with nvtx.annotate(message="my_loop", color="green"):
    for i in range(1,2):        
        optimizer.resolve(return_ratio_new)
    print(f'each optimizing time: {(time()-start)/i}')
    print(f"risk = {jl.get_risk(optimizer.pm)}")
    print(f'objective = {jl.JuMP.objective_value(optimizer.pm.model)}')
    
if __name__ == "__main__":
    test_4558()
