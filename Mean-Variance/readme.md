# 优化问题的数学建模与求解

## 目标函数

最大化以下目标函数：

$$
\text{Maximize} \quad \alpha^T h - \text{cost} \cdot (h_{\text{buy}} + h_{\text{sell}}) - h^T (\text{expo} \cdot \text{cov} \cdot \text{expo}^T) h
$$

为了便于使用优化求解器，将其转换为最小化问题：

$$
\text{Minimize} \quad -\alpha^T h + \text{cost} \cdot (h_{\text{buy}} + h_{\text{sell}}) + h^T (\text{expo} \cdot \text{cov} \cdot \text{expo}^T) h
$$

## 约束条件

1. **变量范围约束**：

   $$
   0 \leq h \leq 0.02
   $$

   $$
   0 \leq h_{\text{buy}}, \quad 0 \leq h_{\text{sell}}
   $$

2. **平衡约束**：

   $$
   h = h_0 + h_{\text{buy}} - h_{\text{sell}}
   $$

3. **总和约束**：

   $$
   \sum h \leq 1
   $$

## 变量定义

- $h$: 决策变量向量
- $h_{\text{buy}}$: 购买变量向量
- $h_{\text{sell}}$: 卖出变量向量
- $h_0$: 常数向量，表示上一个时间段的权重向量
- $\alpha$: 收益系数向量
- $\text{cost}$: 成本系数
- $\text{expo}$: 暴露因子矩阵
- $\text{cov}$: 协方差矩阵

## 求解

求解上述二次规划问题，即可得到最优解 $h, h_{\text{buy}}, h_{\text{sell}}$。
