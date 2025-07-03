# 基础镜像，选择带GPU支持的镜像
FROM nvcr.io/nvidia/base/ubuntu:jammy-20250415.1

# 设置环境变量和避免交互式安装
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="$PATH:/usr/local/julia-1.11.5/bin"

# 安装必要的工具和依赖
RUN apt-get update && apt-get install -y \
    wget \
    vim  \
    git  \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 安装Julia
RUN wget https://mirrors.tuna.tsinghua.edu.cn/julia-releases/bin/linux/x64/1.11/julia-1.11.5-linux-x86_64.tar.gz && \
    tar -zxvf julia-1.11.5-linux-x86_64.tar.gz && \
    mv julia-1.11.5 /usr/local/ && \
    rm julia-1.11.5-linux-x86_64.tar.gz

# 设置环境变量
RUN echo "export PATH=\$PATH:/usr/local/julia-1.11.5/bin" >> ~/.bashrc

# 安装Python依赖
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN pip3 install numpy pandas juliacall

# 添加Julia组件库
RUN julia -e 'using Pkg; \
    Pkg.add(["LinearAlgebra", "PythonCall", "CUDA", "SparseArrays", "JuMP", "Random", "Printf", "NPZ", "MathOptInterface", "CuClarabel"]); \
    import CUDA; \
    CUDA.set_runtime_version!(v"12.8"); \
    CUDA.precompile_runtime(); \
    Pkg.add(["MosekTools", "Mosek", "Gurobi"]); \
    Pkg.precompile();'

# 默认启动bash
CMD ["/bin/bash"]
