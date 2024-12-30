# 问题求解形式说明

目前我们提供了两种问题求解的形式，分别是 **Mean-Variance** 和 **Mean-Risk**（分别对应不同的文件夹）。

## Mean-Variance 形式
在 **Mean-Variance** 形式中，我们还考虑到了**交易成本**因素。

### 相关文件
- **mean_var_optimization.py**: 使用 `cuDF` 和 `CuPy` 从量价数据求平均对数收益率和协方差矩阵，然后优化求解。
- **launch_mean_var_optimization.py**: 直接从 CSV 文件（无表头纯数据文件）中读取平均对数收益率和协方差数据，然后进行优化。

## Mean-Risk 形式
在 **Mean-Risk** 形式中，我们考虑了**无风险收益项**的因素。

### 相关文件
- **mean_std_optimization.py**: 使用 `cuDF` 和 `CuPy` 从量价数据求平均对数收益率和协方差矩阵，然后优化求解。
- **launch_mean_std_optimization.py**: 直接从 CSV 文件（无表头纯数据文件）中读取平均对数收益率和协方差数据，然后进行优化。

## 目的与扩展
提供这两种求解形式的目的旨在为实际求解提供代码参考。更多实际的约束条件或更复杂的目标函数需要您自己搭建（也欢迎随时联系我们）。
