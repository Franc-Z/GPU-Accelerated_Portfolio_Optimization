## generate_data.py
本目录中的generate_data.py用于在Python环境下生成相关的随机数数据，并保存为npz文件。

## use_cuclarabel.jl
此文件用于读取上一步生成的数据文件，并使用CuClarabel求解器进行求解。

## use_mosek.jl
此文件同样以JuMP.jl进行建模，但使用mosek求解器进行求解。注意如果线程设置大于1的情况下，需要使用如：julia -t 12的启动方法。

我们可以通过生成相同的输入，来检验两者求解的数值精度差异和耗时差异。
