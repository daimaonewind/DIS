## 🖥️ 要求

建议使用 anaconda 来管理 python 环境，确保你安装了anaconda，并且配置好了环境变量

## 配置虚拟环境

在anaconda命令行执行以下命令来配置虚拟环境

```
conda create --name DIS python=3.8
conda activate DIS
```

## 下载仓库

```
git clone https://github.com/daimaonewind/DIS.git
```

## 配置所需的库

```
pip install pyqt5 -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn
```

```
pip install pyqt5-tools -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn 
```

```
pip install opencv-python 
```

## 运行

运行main.py即可

```
python main001.py 
```

