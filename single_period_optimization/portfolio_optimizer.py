from os import path, environ
# 设置Julia环境变量
environ['PYTHON_JULIACALL_THREADS'] = '1'
environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = 'yes'
environ['PYTHON_JULIACALL_OPTIMIZE'] = '3'
environ['PYTHON_JULIACALL_COMPILE'] = 'yes'  # 最小编译 (可选)
# 其他可选的Julia性能设置
environ['JULIA_CPU_TARGET'] = 'native'  # 为本地CPU架构优化
#environ['JULIA_MAX_NUM_PRECOMPILE_FILES'] = '100'  # 增加预编译文件上限
environ['JULIA_PKG_PRECOMPILE_AUTO'] = '1'  # 自动预编译

import numpy as np
from time import time
from juliacall import Main as jl

# 加载Julia脚本并获取模块
#julia_script_path = '/nvtest/single_period_optimization/Mean-Variance.jl'  
julia_script_path = '/nvtest/single_period_optimization/Mean-Risk.jl'
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
        """
        Performs the initial solve (warm-up) of the Julia model.
        This includes JIT compilation time.
        """
        if self.pm is None:
            raise RuntimeError("Model not set up. Call setup_model first.")
        print("Performing initial solve ...")
        # Call the Julia initial solve function
        jl.initial_solve_b(self.pm)
        print("Initial solve finished.")
        try:
            # Check status after solve
            status = jl.JuMP.termination_status(self.pm.model)
            if status != jl.MOI.OPTIMAL and status != jl.MOI.ALMOST_OPTIMAL:
                 print("Warning: Initial solve did not reach optimality.")
        except Exception as e:
            print(f"Could not get initial solve status: {e}")

    def resolve(self, new_u_cpu):
        """
        Updates the expected returns and re-solves the model.
        Measures the time for the fast, post-compilation solve.

        Args:
            new_u_cpu (np.ndarray): New expected returns vector.
        """
        if self.pm is None:
            raise RuntimeError("Model not set up. Call setup_model first.")

        # Update return ratio in the Julia model object
        jl.update_return_ratio_b(self.pm, new_u_cpu.astype(MyFloat))

        # Re-solve the model
        jl.resolve_b(self.pm)

    def get_risk(self):
        """Gets the calculated risk (standard deviation) from the solved model."""
        if self.pm is None:
            raise RuntimeError("Model not set up.")
        try:
            return jl.get_risk(self.pm)
        except Exception as e:
            print(f"Error getting risk: {e}")
            return np.nan

    def get_objective_value(self):
        """Gets the objective value from the solved model."""
        if self.pm is None:
            raise RuntimeError("Model not set up.")
        try:
            return jl.JuMP.objective_value(self.pm.model)
        except Exception as e:
            print(f"Error getting objective value: {e}")
            return np.nan
        

def load_data():
    cov = np.loadtxt('/nvtest/single_period_optimization/cov_41_41.csv', delimiter=',', dtype=MyFloat)
    cov *= 12.0
    expo = np.loadtxt('/nvtest/single_period_optimization/expo_4558_41.csv', delimiter=',', dtype=MyFloat)
    expo = np.nan_to_num(expo)
    bias = np.loadtxt('/nvtest/single_period_optimization/bias_4558.csv', dtype=MyFloat)
    bias = np.nan_to_num(bias)*12.0
    return_ratio = np.loadtxt('/nvtest/single_period_optimization/stock_return_2023.csv', dtype=MyFloat)
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
    #return_ratio_new = return_ratio * (1.0 + 0.05 * np.random.randn(*return_ratio.shape))
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
    for i in range(1,4):    
        return_ratio_new = return_ratio * (1.0 + 0.05 * np.random.randn(*return_ratio.shape))    
        optimizer.resolve(return_ratio_new)
    print(f'each optimizing time: {(time()-start)/i}')
    print(f"risk = {jl.get_risk(optimizer.pm)}")
    print(f'objective = {jl.JuMP.objective_value(optimizer.pm.model)}')
    
if __name__ == "__main__":
    test_4558()
