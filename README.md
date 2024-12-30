# GPU-Accelerated_Portfolio_Optimization
GPU-Accelerated Portfolio Optimization
---
如何在容器中安装Julia并设置运行环境
1.	进入容器中
	注意在容器中要有root权限，可以使用在使用docker run时加上--user=root选项。
	
2.	安装Julia
		wget https://mirrors.tuna.tsinghua.edu.cn/julia-releases/bin/linux/x64/1.11/julia-1.11-latest-linux-x86_64.tar.gz
		tar -zxvf julia-1.11-latest-linux-x86_64.tar.gz
	如果所安装的Julia版本为julia-1.11.2，则将上面解压的文件夹移动到/usr/local/目录中。
		mv julia-1.11.2 /usr/local/

3.	设置环境变量
		vim ~/.bashrc
	在文件末尾添加以下行（请根据实际路径调整）
		export PATH="$PATH:/usr/local/julia-1.11.2/bin"
	注意，如果环境内没安装vim或namo之类的编辑器，可以直接执行下列命令
		echo "export PATH="$PATH:/usr/local/julia-1.11.2/bin"" >> ~/.bashrc
	使配置生效：
		source ~/.bashrc
	
4.	验证安装是否成功
	在bash中，输入
		julia
	看是否能进入环境。如果顺利进入Julia的命令行环境后，可以输入：
		versioninfo()
	从而查看系统信息。

5.	安装所需的julia组件库
	进入Julia环境后，按]键，进入如下组件安装模式
		(@v1.11) pkg>
	在如上的组件安装模式中，输入如下命令
		add NLPModels MadNLP MadNLPGPU LinearAlgebra PythonCall CUDA

6.	安装完毕后，按backspace键，变回正常Julia的命令行模式，然后输入
		exit()
	退出到bash命令后，在python中安装必要的安装包：
		pip install juliacall

7.	保存container为镜像文件，在裸机的bash环境下输入
		docker ps
	找到刚刚安装好julia环境的container名称（假设为container_name），然后执行如下保存container的命令：
		docker commit container_name docker_image_name:release_version
	这样，我们就将刚刚设置好环境的container保存为名为docker_image_name:release_version的docker image了。
