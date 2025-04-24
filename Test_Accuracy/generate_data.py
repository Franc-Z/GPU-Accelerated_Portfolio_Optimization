import numpy as np

# 固定随机种子
np.random.seed(42)

# 参数设置
k = 50
n = k * 100
T = 2

# 生成数据
D_diag = np.random.rand(n) * np.sqrt(k)
F = np.random.randn(n, k) * (np.random.rand(n, k) < 0.5)
Omega_temp = np.random.randn(k, k)
Omega_temp = (Omega_temp @ Omega_temp.T) / k
mu_matrix = (3.0 + 9.0 * np.random.rand(n, T)) / 100.0

# 保存数据
np.save("D_diag.npy", D_diag)
np.save("F.npy", F)
np.save("Omega.npy", Omega_temp)
np.save("mu_matrix.npy", mu_matrix)
