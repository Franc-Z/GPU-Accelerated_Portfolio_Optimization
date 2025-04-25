### 如何在Linux容器中安装Julia并设置运行环境

以下是在Linux容器中安装Julia并设置运行环境的步骤指南：

#### 1. 进入容器

确保在容器中具有root权限，可以在使用`docker run`时加上`--user=root`选项。

```bash
docker run -it --rm --gpus 1 --user=root your_image_name /bin/bash
```

#### 2. 安装Julia （建议使用Julia-1.11.x版本）

下载并安装Julia：

```bash
wget https://mirrors.tuna.tsinghua.edu.cn/julia-releases/bin/linux/x64/1.11/julia-1.11-latest-linux-x86_64.tar.gz
tar -zxvf julia-1.11-latest-linux-x86_64.tar.gz
```

将解压后的文件夹移动到`/usr/local/`目录中（假设版本为`julia-1.11.4`）：

```bash
mv julia-1.11.4 /usr/local/
```

#### 3. 设置环境变量

编辑`~/.bashrc`文件：

```bash
vim ~/.bashrc
```

在文件末尾添加以下行（请根据实际路径调整）：

```bash
export PATH="$PATH:/usr/local/julia-1.11.4/bin"
```

或者直接执行以下命令：

```bash
echo "export PATH=\"$PATH:/usr/local/julia-1.11.4/bin\"" >> ~/.bashrc
```

使配置生效：

```bash
source ~/.bashrc
```

#### 4. 验证安装

在bash中输入以下命令，验证Julia是否安装成功：

```bash
julia
```

进入Julia命令行环境后，输入以下命令查看系统信息：

```julia
versioninfo()
```

#### 5. 安装Julia组件库

进入Julia环境后，按`]`键进入包管理模式：

```
(@v1.12) pkg>
```

安装所需的组件库：

```julia
add LinearAlgebra PythonCall CUDA SparseArrays JuMP Random Printf NPZ
add https://github.com/exanauts/CUDSS.jl/tree/cudss-0.5.0
add https://github.com/cvxgrp/CuClarabel/tree/450e0ed42b8f38df94b47c2c7df6b4e8c7f82d46
add MosekTools      #如果需要在julia侧进行结果对比或性能benchmark，可以安装MosekTools,具体请见https://github.com/jump-dev/MosekTools.jl
```

返回正常Julia命令行模式：

```julia
按backspace键即可
```

退出Julia环境：

```julia
exit()
```

#### 6. 安装Python依赖

在bash中安装`juliacall`包：

```bash
pip install juliacall
```

#### 7. 保存容器为镜像

在裸机的bash环境下，保存容器为镜像文件：

```bash
docker ps
```

找到正在运行的容器名称（假设为`container_name`），然后执行以下命令保存为镜像：

```bash
docker commit container_name docker_image_name:release_version
```

这样，你就将设置好环境的容器保存为名为`docker_image_name:release_version`的Docker镜像。
