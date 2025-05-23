# Docker 环境构建与 CPU/GPU Benchmark 测试指南

请先参考 `How to manually setup Julia environment inside docker container.md` 文件，构建包含 Python 及 Julia 运行环境的 Docker 容器。

如果您希望自动构建docker image，在Dockerfile所在的文件夹下可以运行 
```bash
docker build -t my-julia:cuda-12.8 .
```
如果构建完的docker container中运行相关代码时出现“Error: CUDA.jl could not find an appropriate CUDA runtime to use.”的错误的话，请在julia REPL中执行如下代码：
```julia
julia> import CUDA
julia> CUDA.set_runtime_version!(v"12.8")
julia> CUDA.precompile_runtime()
julia> exit()
```
之后再执行相关代码即可正常运行。

## 运行环境准备

- 构建完成后，在容器中执行相关的 Python 和 Julia 脚本，进行 CPU 与 GPU 的性能基准测试（benchmark）。

## 脚本说明

| 脚本名称                          | 说明                                                         |
|----------------------------------|--------------------------------------------------------------|
| `multi_period_optimization.jl`   | 用于跑一到多周期任务的 Julia 脚本                            |
| `multi_period_optimization_by_mosek.py` | 对应 `multi_period_optimization.jl` 的 Python 脚本，使用 Mosek 进行测速 |

此外，single_period_optimization文件夹下面是使用python调用相关Julia代码的范例，所需数据文件都在文件夹下面，您可以尝试使用Mean-Variance或Mean-Risk方法来进行相关组合优化。
## 如何运行 Julia 脚本

1. 在命令行中执行 `julia` 命令，进入 Julia REPL 环境。
2. 在 Julia REPL 中输入以下命令并回车执行：
```julia
julia> include("./multi_period_optimization.jl")
```
3. 注意事项：
- Julia 编译器采用先编译后执行的机制，首次运行时包含编译时间。
- 为获得准确的 benchmark 计时数据，建议在不退出julia REPL情况下重复运行脚本（再次输入 `include("./multi_period_optimization.jl")` 并回车）。

---

祝您测试顺利，性能卓越！

