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

（出现HTTP或SSL报错请关闭代理服务器）

```
pip install pyqt5 -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn
```

```
pip install pyqt5-tools -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn 
```

```
pip install opencv-python 
```

## 配置QTdesigner和PyUic

### (想要修改项目UI才进行这一项)

#### ①配置QTdesigner

点击“设置-工具-外部工具”，点击+号，在名称输入qt designer

程序路径参考：D:\Anaconda\envs\DIS\Lib\site-packages\qt5_applications\Qt\bin\designer.exe

工作目录：$FileDir$

点击确认，退出设置，在上方工具栏的外部工具即可看到QT designer

#### ②配置PyUic

同样在外部工作区域，点击+号，在名称输入PyUic，

程序路径参考：D:\Anaconda\envs\DIS\Scripts\pyuic5.exe

实参直接输入：$FileName$ -o $FileNameWithoutExtension$.py

工作目录：$FileDir$

点击确认，退出设置，在上方工具栏的外部工具即可看到PyUic

## 运行

运行main.py即可

```
python main001.py 
```
