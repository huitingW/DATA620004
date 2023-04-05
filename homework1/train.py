
import numpy as np
import matplotlib.pyplot as plt
import model as mo

if __name__ == '__main__':
    np.random.seed(666)
    file = 'model_params'
    
    x_train, y_train = mo.load_mnist('./mnist/', kind='train', onehot= True)
    x_test, y_test = mo.load_mnist('./mnist', kind='t10k', onehot= True)

    train_dataloader = mo.MyDataLoader(x_train, y_train, batch_size=256, drop_last=True)

    network = mo.TwoLayerNet(input_size=784, hidden_size=512, output_size=10, 
                             learning_rate = 1e-3, reg = 0.1)
    
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    epochs = 100
    batch_size = 256
    learning_rate_decay = 0.95
    train_size = x_train.shape[0]

    for e in range(epochs):
    # 234个batch 每个batch 256个数据
        print('epoch:', e + 1)
        for x_train_batch, y_train_batch in train_dataloader:
            loss = network.train(x_train_batch, y_train_batch)
            train_loss_list.append(loss)

        network.lr *= learning_rate_decay
        test_loss = network.train(x_test, y_test)
        test_loss_list.append(test_loss)

        train_acc = network.accuracy(x_train, y_train)
        train_acc_list.append(train_acc)
        test_acc = network.accuracy(x_test, y_test)
        test_acc_list.append(test_acc)

        print('train accuracy: ', train_acc)

    # 保存模型
    network.savemodel(file)

    # 绘制图形 (Train, test)---loss
    markers = {'train': 'o', 'valid': 's'}
    x = np.linspace(1, epochs, epochs)
    plt.plot(x, train_loss_list[10:len(train_loss_list):234], label='train loss')
    plt.plot(x, test_loss_list, label='test loss')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(loc='upper right')
    plt.show()

    # 绘制图形 (Train, test)---accuracy
    markers = {'train': 'o', 'valid': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(loc='lower right')
    plt.show()

    # 可视化每层网络参数
    w1 = network.params['W1']
    w2 = network.params['W2']
    b1 = network.params['b1']
    b1 = b1.reshape(b1.shape[0],-1)
    b1 = np.repeat(b1, b1.shape[0], axis = 1)
    b2 = network.params['b2']
    b2 = b2.reshape(b2.shape[0],-1)
    b2 = np.repeat(b2, b2.shape[0], axis = 1)

    #创建新的figure
    fig = plt.figure()
     
     #绘制2x2两行两列共四个图，编号从1开始
    ax1 = fig.add_subplot(221)
    ax1.imshow(w1)
    ax1.set_title('W1')
    plt.xticks([]), plt.yticks([])
    ax2 = fig.add_subplot(222)
    ax2.imshow(b1)
    ax2.set_title('b1')
    plt.xticks([]), plt.yticks([])
    ax3 = fig.add_subplot(223)
    ax3.imshow(w2)
    ax3.set_title('W2')
    plt.xticks([]), plt.yticks([])
    ax4 = fig.add_subplot(224)
    ax4.imshow(b2)
    ax4.set_title('b2')
    plt.xticks([]), plt.yticks([])
     
     #图片的显示
    plt.show()



    
    
