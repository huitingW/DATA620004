import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import os
import gzip
import random

class Relu:
    def __init__(self):
        self.mask = None
 
    def forward(self, x):
        self.mask = (x <= 0)  # Numpy, True/False
        out = x.copy()
        out[self.mask] = 0
        return out
 
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
class Affine:

# L = f(Y)
# Y = XW+b
# ∂L/∂Y---上一层传回来的梯度 dout
# ∂L/∂X = ∂L/∂Y * ∂Y/∂X = ∂L/∂Y * W
# ∂L/∂W = ∂L/∂Y * ∂Y/∂W = ∂L/∂Y * X
# ∂L/∂b = ∂L/∂Y * ∂Y/∂b = ∂L/∂Y * 1

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None # 权重和偏置参数的梯度
        self.db = None
 
    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b  # Numpy自动广播
        return out
 
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)  # dout 是 ∂L/∂Y 即上一层的梯度
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx  # 将当前层的梯度dx向前回传
    
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 
 
    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))
     
     
def cross_entropy_error(y_hat, y):
          
    if y_hat.ndim == 1:
        y = y.reshape(1, y.size)
        y_hat = y_hat.reshape(1, y_hat.size)
         
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if y.size == y_hat.size:
        y = y.argmax(axis=1)
              
    batch_size = y_hat.shape[0]
    return -np.sum(np.log(y_hat[np.arange(batch_size), y] + 1e-7)) / batch_size
     
 
class SoftmaxWithLoss:
    """
    计算交叉熵损失时，由于y只有一个值为1时计算才不为0，因此损失函数要做的就是找到
    训练集中的真实类别，然后试图使该类别相应的概率尽可能的高---最大似然
    """
    def __init__(self):
        self.loss = None
        self.y_hat = None # softmax的输出
        self.y = None 
 
    def forward(self, x, y):
        self.y = y
        self.y_hat = softmax(x) 
        self.loss = cross_entropy_error(self.y_hat, self.y)
        return self.loss
 
    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        if self.y.size == self.y_hat.size: # 监督数据是one-hot-vector的情况
            dx = (self.y_hat - self.y) / batch_size
        else:
            """
            softmax(x)函数对x的导数为softmax(x) - softmax(x)*softmax(x)
            """
            dx = self.y_hat.copy()
            dx[np.arange(batch_size), self.y] -= 1
            dx = dx / batch_size
         
        return dx
    
class SGD:
    """随机梯度下降法（Stochastic Gradient Descent）"""
    def __init__(self, lr=0.1):
        self.lr = lr
         
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]   # 不同参数W1, ..., Wk有相同学习率
             

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01, learning_rate = 1e-3, reg = 0.0):
        print("Build Net")  
        # 初始化权重
        self.params = {}
        
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.lr = learning_rate
        self.reg = reg

 
        # 生成层
        self.layers = OrderedDict() #创建一个有序字典
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
         
        self.lastLayer = SoftmaxWithLoss()
         
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
         
    def loss(self, x, y):
        y_hat = self.predict(x)
        cost = self.lastLayer.forward(y_hat, y)
        w1 = self.params['W1']
        w2 = self.params['W2']
        m = x.shape[0]
        cost += 0.5 * self.reg * np.sum(w1**2) / m 
        cost += 0.5 * self.reg * np.sum(w2**2) / m
        
        return cost
     
    def accuracy(self, x, y):
        y_hat = self.predict(x)
        y_hat = np.argmax(y_hat, axis=1)
        if y.ndim != 1 : y = np.argmax(y, axis=1)
         
        accuracy = np.sum(y_hat == y) / float(x.shape[0])
        return accuracy
         
    def gradient(self, x, y):
        self.loss(x, y)  # forward
 
        dout = 1  # backward
        dout = self.lastLayer.backward(dout)
         
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        w1 = self.params['W1']
        w2 = self.params['W2']
        m = x.shape[0]
 
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW - self.reg * w1 / m, self.layers['Affine1'].db 
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW - self.reg * w2 / m, self.layers['Affine2'].db
        
        return grads
    
    def train(self, x_batch, y_batch):
        optimizer = SGD(lr=self.lr)
        grad = self.gradient(x_batch, y_batch) 
        optimizer.update(self.params, grad)  # 更新参数
        loss = self.loss(x_batch, y_batch)

        return loss

        

    def savemodel(self, file):
        dic = {'W1': self.params['W1'], 'b1': self.params['b1'],
               'W2': self.params['W2'], 'b2': self.params['b2'],
               'lr': self.lr,
               'reg': self.reg}
        np.save(file, dic)
        return
    
    def loadmodel(self, file):
        dic = np.load(file, allow_pickle=True)[()]
        self.params['W1'] = dic['W1']
        self.params['W2'] = dic['W2']
        self.params['b1'] = dic['b1']
        self.params['b2'] = dic['b2']
        self.lr = dic['lr']
        self.reg = dic['reg']
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        return
    
class MyDataLoader:
    def __init__(self, data, label, batch_size, drop_last=False):
        self.data = data
        self.label = label
        self.batch_size = batch_size

        nums = data.shape[0]
        a = [i for i in range(nums)]
        random.shuffle(a)

        self.sampler = a
        self.drop_last = drop_last

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]

    def __iter__(self):
        batch_index = []
        for index in self.sampler:
            batch_index.append(index)
            if len(batch_index) == self.batch_size:
                yield self.data[batch_index], self.label[batch_index]
                batch_index = []
        if len(batch_index) > 0 and not self.drop_last:
            # 如果最后剩余的数据不够一个batch_size,根据参数决定是否 drop out
            yield self.data[batch_index], self.label[batch_index]
        # 每一个epoch后洗牌一次
        random.shuffle(self.sampler)

# 导入数据
def one_hot(y, n_classes):
    """
    将label转成one-hot形式：np.eye(10)[y]：生成一个10列的二维数组，第y行为1，其余为0
    """
    return np.eye(n_classes)[y]

def load_mnist(path, kind = 'train', onehot = True):
    """
    path：存放数据路径（一般使用相对路径）
    kind：读取的数据种类：train/t10k
    onehot：是否要做one-hot处理：True/False
    
    """
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
        
    if onehot:
        labels = one_hot(labels, 10)

    return images, labels