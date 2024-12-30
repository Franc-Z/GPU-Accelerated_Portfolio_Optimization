### **问题概述**

**目标函数：**

您正在研究的投资组合优化问题的目标函数为：

\[
f(w, t) = -\left( (\mu - r^f)^\top w + r^f - \lambda t \right)
\]

其中：

- \( w \in \mathbb{R}^n \) 是投资组合权重的向量。
- \( \mu \in \mathbb{R}^n \) 是预期收益率向量。
- \( r^f \in \mathbb{R} \) 是无风险收益率。
- \( \lambda \in \mathbb{R} \) 是风险厌恶系数。
- \( t \in \mathbb{R} \) 是引入的辅助变量，用于将问题从非凸优化转化为凸优化。

**约束条件：**

为了将原本的非凸问题转化为凸优化问题，您引入了变量 \( t \)，并建立以下约束：

\[
t^2 \geq w^\top \Sigma w
\]

其中 \( \Sigma \in \mathbb{R}^{n \times n} \) 是协方差矩阵。这个约束可以被表示为：

\[
c_1(w, t) = w^\top \Sigma w - t^2 \leq 0
\]

另外，考虑资金总和约束（即权重之和为1）：

\[
c_2(w) = e^\top w - 1 = 0
\]

其中 \( e \) 是元素全为1的向量。

---

### **目标函数的梯度和Hessian矩阵**

**目标函数对 \( w \) 和 \( t \) 的梯度：**

由于目标函数 \( f(w, t) \) 对 \( w \) 和 \( t \) 是线性的，所以梯度计算相对简单。

- 对 \( w \) 的梯度：

  \[
  \nabla_w f = -(\mu - r^f)
  \]

- 对 \( t \) 的梯度：

  \[
  \frac{\partial f}{\partial t} = \lambda
  \]

**目标函数的Hessian矩阵：**

由于目标函数对 \( w \) 和 \( t \) 是线性的，因此二阶导数（Hessian矩阵）为零矩阵：

\[
H_f = \begin{bmatrix}
0 & 0 \\
0^\top & 0
\end{bmatrix}
\]

---

### **构造拉格朗日函数**

为了求解含有约束的优化问题，我们构造拉格朗日函数 \( \mathcal{L}(w, t, \nu, \eta) \)：

\[
\mathcal{L}(w, t, \nu, \eta) = f(w, t) + \nu \left( w^\top \Sigma w - t^2 \right) + \eta \left( e^\top w - 1 \right)
\]

其中：

- \( \nu \geq 0 \) 是与锥约束相关的拉格朗日乘数。
- \( \eta \) 是与资金总和约束相关的拉格朗日乘数。

---

### **拉格朗日函数的梯度和Hessian矩阵**

**拉格朗日函数对 \( w \) 和 \( t \) 的梯度：**

- 对 \( w \) 的梯度：

  \[
  \nabla_w \mathcal{L} = -(\mu - r^f) + 2\nu \Sigma w + \eta e
  \]

- 对 \( t \) 的梯度：

  \[
  \frac{\partial \mathcal{L}}{\partial t} = \lambda - 2\nu t
  \]

**拉格朗日函数的Hessian矩阵：**

Hessian矩阵是拉格朗日函数对决策变量的二阶导数矩阵。

- 对 \( w \) 的二阶导数：

  \[
  H_{ww} = \nabla_w^2 \mathcal{L} = 2\nu \Sigma
  \]

- 对 \( t \) 的二阶导数：

  \[
  H_{tt} = \frac{\partial^2 \mathcal{L}}{\partial t^2} = -2\nu
  \]

- 交叉项（\( w \) 和 \( t \) 之间）：

  \[
  H_{wt} = \frac{\partial^2 \mathcal{L}}{\partial w \partial t} = 0, \quad H_{tw} = H_{wt}^\top = 0
  \]

因此，完整的Hessian矩阵为：

\[
H_{\mathcal{L}} = \begin{bmatrix}
H_{ww} & H_{wt} \\
H_{tw} & H_{tt}
\end{bmatrix} = \begin{bmatrix}
2\nu \Sigma & 0 \\
0^\top & -2\nu
\end{bmatrix}
\]

---

### **在NLPModels.jl和MadNLP.jl中的实现**

#### **1. Hessian矩阵的实现**

在 **MadNLP.jl** 中，需要实现 `hess_dense!` 函数来计算拉格朗日函数的Hessian矩阵。

**Julia代码示例：**

```julia
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
```

**注意事项：**

- `H` 是要被填充的Hessian矩阵。
- `ν` 是当前迭代中的拉格朗日乘数向量。
- `Σ` 是协方差矩阵，在模型 `nlp` 中应作为属性提供。
- `λ` 和 `x` 也会被传入，但在计算Hessian时主要使用 `ν`。

#### **2. Jacobian矩阵的实现**

`jac_dense!` 函数用于计算约束函数对决策变量的Jacobian矩阵。在您的问题中，需要考虑两个约束：锥约束和资金总和约束。

**计算约束的梯度：**

- **锥约束 \( c_1(w, t) = w^\top \Sigma w - t^2 \leq 0 \)：**

  - 对 \( w \) 的导数：

    \[
    \nabla_w c_1 = 2 \Sigma w
    \]

  - 对 \( t \) 的导数：

    \[
    \frac{\partial c_1}{\partial t} = -2 t
    \]

- **资金总和约束 \( c_2(w) = e^\top w - 1 = 0 \)：**

  - 对 \( w \) 的导数：

    \[
    \nabla_w c_2 = e
    \]

  - 对 \( t \) 的导数：

    \[
    \frac{\partial c_2}{\partial t} = 0
    \]

**Julia代码示例：**

```julia
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
```

**注意事项：**

- `J` 是要被填充的Jacobian矩阵，行数等于约束数量（2个约束），列数等于变量数量（\( n + 1 \)）。
- 确保矩阵维度匹配，`J` 的尺寸为 \( 2 \times (n+1) \)。
- 变量 \( w \) 和 \( t \) 在决策变量向量 \( x \) 中的排列应保持一致，通常将 \( w \) 的元素放在前 \( n \) 个位置，\( t \) 放在第 \( n+1 \) 个位置。

---

### **总结**

**关键公式：**

- **目标函数的梯度：**

  - 对 \( w \)：

    \[
    \nabla_w f = -(\mu - r^f)
    \]

  - 对 \( t \)：

    \[
    \frac{\partial f}{\partial t} = \lambda
    \]

- **拉格朗日函数的Hessian矩阵：**

  \[
  H_{\mathcal{L}} = \begin{bmatrix}
  2\nu \Sigma & 0 \\
  0^\top & -2\nu
  \end{bmatrix}
  \]

- **约束的Jacobian矩阵：**

  - **锥约束：**

    \[
    \nabla_w c_1 = 2 \Sigma w, \quad \frac{\partial c_1}{\partial t} = -2 t
    \]

  - **资金总和约束：**

    \[
    \nabla_w c_2 = e, \quad \frac{\partial c_2}{\partial t} = 0
    \]

**在NLPModels.jl和MadNLP.jl中的实现要点：**

- `hess_dense!` 函数用于计算拉格朗日函数的Hessian矩阵，需要使用当前的拉格朗日乘数 \( \nu \)。
- `jac_dense!` 函数用于计算约束的Jacobian矩阵。
- 在实现中，要确保矩阵的维度和变量的顺序一致。
- 只需考虑锥约束和资金总和约束，无需处理其他约束。

---

**最后说明：**

- 确保您的模型 `PortfolioNLPModelCUDA` 包含所有必要的数据，如 \( \Sigma \)、\( \mu \)、\( r^f \) 和 \( \lambda \)。
- 在使用 **MadNLP.jl** 求解优化问题时，提供目标函数、梯度、Hessian、约束函数和Jacobian的实现，以便算法能够正确地计算并迭代求解。
- 由于只考虑锥约束和资金总和约束，因此模型中仅需实现相关的函数，无需处理其他约束。

