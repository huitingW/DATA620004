# DATA620004--神经网络和深度学习
**FDU  22210989976 王慧婷**
<br>
<br>
该仓库用来存储神经网络和深度学习课程的所有作业相关代码

## Homework1 构建两层神经网络分类器
该作业包含四个py文件：`model.py`、`params_search.py`、`train.py`、`test.py`
<br>
* `model.py`：用来实现该神经网络所用到的函数及类的定义--激活函数、反向传播，loss及梯度计算、学习率下降策略、L2正则化、优化器SGD
* `params_search.py`：用来进行参数查找，学习率、隐藏层大小、正则化强度
* `train.py`：用来进行训练、保存模型、绘图
* `test.py`：导入模型，用经过参数查找后的模型进行测试，输出分类精度

两个文件夹：`images`、`mnist`
* `images`：用来存放可视化训练和测试的loss曲线，测试的accuracy曲线，以及可视化每层的网络参数
* `mnist`：数据集

一个npy文件：`model_params.npy`
* `model_params.npy`：模型参数

### 训练与测试
运行下述代码，得到预训练模型在测试集上的分类精度
 ```
 python train.py     #保存模型参数，可视化训练和测试的loss曲线，测试的accuracy曲线，以及可视化每层的网络参数
 python test.py      #输出分类精度
 ```
其中`train.py`可能耗时较久，可以直接查看images文件夹内图片


