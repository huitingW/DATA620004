import model as mo
import numpy as np
import pandas as pd

batch_size = 256
hidden_size = [128,256,512] #隐藏层大小
reg = [0, 1e-1, 1e-2, 1e-3] #正则化强度
lrs = [1e-1, 1e-2, 1e-3, 1e-4] #学习率
lr_decay = 0.95
iters_num = 5000


def monotonicDecrease(x):
    dx = np.diff(x)
    return np.all(dx <= 0)

min_epoch = 5 
max_epoch = 40  
decay_num = 5  # 如果连续decay_num个epoch的准确率一直减小，则认为后续训练没有意义，停止训练

if __name__ == '__main__':

    np.random.seed(666)

    X, y = mo.load_mnist('./mnist/', kind='train', onehot= True)
    ratio = 0.2
    size = X.shape[0]
    valid_size = int(size * ratio)
    train_size = size - valid_size
    x_train = X[valid_size:]
    y_train = y[valid_size:]
    x_valid = X[:valid_size]
    y_valid = y[:valid_size]



    result = {}
    for hidden in hidden_size:
        for l2_reg in reg:
            for lr in lrs:
                network = mo.TwoLayerNet(input_size=784, hidden_size=hidden, output_size=10, 
                      learning_rate = lr, reg = l2_reg)
                network.lr = lr
                train_dataloader = mo.DataLoader(x_train, y_train, batch_size=256, drop_last=True)
                valid_acc_list = []
                for e in range(max_epoch):
                    for x_batch, y_batch in train_dataloader:
                        loss = network.train(x_batch, y_batch)
                    network.lr *= lr_decay
                    valid_acc = network.accuracy(x_valid, y_valid)
                    valid_acc_list.append(valid_acc)
                    if e >= min_epoch:
                        if valid_acc_list[-1] < 0.7:
                            break
                        if np.all(np.diff(valid_acc_list[decay_num:]) <= 0):
                            break
                result[(hidden, l2_reg, lr)] = np.mean(valid_acc_list[-decay_num:])
                print('epoch:',e+1)
                print((hidden, l2_reg, lr), ' ', result[(hidden, l2_reg, lr)])

    #             valid_acc_list = []
    #             for i in range(iters_num):
    #                 batch_mask = np.random.choice(train_size, batch_size)
    #                 x_batch = x_train[batch_mask]
    #                 y_batch = y_train[batch_mask]
    #                 loss = network.train(x_batch, y_batch)
    #                 network.lr *= lr_decay
    #                 valid_acc = network.accuracy(x_valid, y_valid)
    #                 valid_acc_list.append(valid_acc)
                
    #             result[(hidden, l2_reg, lr)] = np.mean(valid_acc_list)
    #             print((hidden, l2_reg, lr),':',result[(hidden, l2_reg, lr)])
    
    print('-'*50)
    print('best params:', max(zip(result.values(),result.keys())))
    df = pd.concat({k: pd.Series(v) for k, v in result.items()}).reset_index()
    df.columns = ['hidden','reg','lr','n','acc']
    df = df.drop(['n'], axis=1)
    df.to_csv('params.csv')
    



                    
                