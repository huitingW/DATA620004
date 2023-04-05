

import model as mo

if __name__ == '__main__':

    
    x_test, y_test = mo.load_mnist('./mnist', kind='t10k', onehot= True)


    network = mo.TwoLayerNet(input_size=784, hidden_size=128, output_size=10, 
                             learning_rate = 1e-3, reg = 0.1)
    
    #加载模型
    network.loadmodel('model_params.npy')

    test_acc = network.accuracy(x_test, y_test)
    test_loss = network.train(x_test, y_test)
    print('the accuracy is:', test_acc)

