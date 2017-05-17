#!/bin/bash python
# coding:utf8
import numpy as np
import logging
import time
logging.basicConfig(level=logging.INFO)

class DataIter():
    def __init__(self,x,y,batch_size):
        if len(x)!=len(y):
            raise ValueError('The sizes of x and y are not identical!')
        self.x = x
        self.y = y
        self.size = len(y)
        if len(x.shape) == 1:
            self.x_dim = 1
        else:
            self.x_dim = x.shape[1]
        if len(y.shape) == 1:
            self.y_dim = 1
        else:
            self.y_dim = y.shape[1]
        self.batch_size = batch_size
        self.curr=0

    def next_batch(self):
        x = np.zeros([self.batch_size, self.x_dim])
        y = np.zeros([self.batch_size, self.y_dim])
        if self.curr + self.batch_size > self.size:
            left_size = self.size - self.curr
            x[:left_size] = self.x[self.curr:]
            y[:left_size] = self.y[self.curr:]
            random_index = np.random.randint(self.size,
                                             size= self.batch_size - left_size)
            x[left_size:] = self.x[random_index]
            y[left_size:] = self.y[random_index]
            self.curr = 0
        else:
            x = self.x[self.curr:self.curr+self.batch_size]
            y = self.y[self.curr:self.curr+self.batch_size]
            self.curr += self.batch_size
            if self.curr == self.size:
                self.curr = 0
        return x,y

    def shuffle(self):
        np.random.seed(int(time.time()))
        shuffle_indices = np.random.permutation(np.arange(self.size))
        self.x = self.x[shuffle_indices]
        self.y = self.y[shuffle_indices]
        self.curr = 0

# targets=['O','D'] or targets['O','X']
def data_read(path='data/letter-recognition.data',targets=['O','D'],feature_filter=[]):
    raw_data = []
    with open(path) as f:
        line = f.readline()
        while line != "":
            if line[0] in targets:
                raw_data.append(line.strip().split(','))
            line = f.readline()
    if len(feature_filter)==0:
        size = len(raw_data)
        dim = len(raw_data[0])-1
        data = np.zeros([size, dim])
        labels = np.zeros([size,1])
        for i in range(size):
            if raw_data[i][0]==targets[0]:
                labels[i]=0
            else:
                labels[i]=1
            for j in range(dim):
                data[i][j]=int(raw_data[i][j+1])
    else:
        size = len(raw_data)
        dim = len(feature_filter)
        data = np.zeros([size, dim])
        labels = np.zeros([size,1])
        for i in range(size):
            if raw_data[i][0]==targets[0]:
                labels[i]=0
            else:
                labels[i]=1
            for j in range(dim):
                data[i][j]=int(raw_data[i][feature_filter[j]])

    return data,labels,size,dim

# Perceptron layer
# All variable are row vectors
class Perceptron():
    def __init__(self, batch_size, input_size, output_size, isOutput=False,lr=1) :
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.isOutput = False
        self.weight = np.random.randn(input_size,output_size)
        self.bias = np.ones([1,output_size])
        self.grad_weight = np.zeros([input_size,output_size])
        self.grad_bias = np.zeros([1,output_size])
        self.input = np.zeros([batch_size,input_size])
        self.output = np.zeros([batch_size, output_size])
        self.error_signal = np.zeros([batch_size,output_size])
        self.labels = np.zeros([batch_size,output_size])
        self.lr=lr

    def feed(self,input):
        self.input = input

    def forward(self):
        self.output = self.input.dot(self.weight) + self.bias
        self.output = 1/(1+np.exp(-self.output))
        return self.output

    def evaluate_loss(self,y):
        return 0.5*np.average(np.square(self.output - y))

    def set_error_signal(self, error_signal):
        self.error_signal = error_signal

    # Return the previous layer error_signal
    def calculate_grad(self):
        self.grad_weight = self.input.T.dot(self.error_signal * self.output * (1 - self.output))
        self.grad_weight /= self.batch_size

        self.grad_bias = np.average(self.error_signal * self.output * (1 - self.output),axis=0)
        return  np.dot(self.error_signal * self.output * (1 - self.output), self.weight.T)

    def update(self):
        self.weight -= self.lr * self.grad_weight
        self.bias -= self.lr * self.grad_bias

class Network():
    def __init__(self, layers):
        self.layers = layers

    def forward(self,input):
        self.layers[0].feed(input)
        for i in range(len(self.layers)-1):
            self.layers[i+1].feed(self.layers[i].forward())
        if len(self.layers)>1:
            self.layers[-1].forward()
        return self.layers[-1].output

    def backpropagation(self, labels):
        self.layers[-1].set_error_signal(self.layers[-1].output - labels)
        for i in range(len(self.layers)-1, 0, -1):
            self.layers[i-1].set_error_signal(self.layers[i].calculate_grad())
        self.layers[0].calculate_grad()

        for layer in self.layers:
            layer.update()

    def evaluate_loss(self,y):
        return self.layers[-1].evaluate_loss(y)

def evaluate(nn, data_iter):
    data_iter.curr = 0
    iters = data_iter.size / data_iter.batch_size
    if data_iter.size % data_iter.batch_size >0:
        iters+=1
    output_list = []
    for i in range(iters):
        x, _ = data_iter.next_batch()
        output_list.append (nn.forward(x))
    predictions = np.concatenate(output_list)[:data_iter.size]
    loss = 0.5*np.average(np.square(predictions - data_iter.y))
    correct_rate = np.sum((predictions > 0.5) == (data_iter.y > 0.5))/float(data_iter.size)
    return predictions,loss,correct_rate

def main(feature_filter=[]):
    # You can set some parameters for training
    batch_size=32
    # layers shape [0,hidden1,hidden2,...]
    layer_shapes = [0,3]
    # output size
    output_size=1
    epoch = 20000
    evaluate_every_steps = 100

    np.random.seed(int(time.time()))
    x,y,size,dim=data_read(feature_filter=feature_filter)
    shuffle_indices = np.random.permutation(np.arange(size))
    x=x[shuffle_indices]
    y=y[shuffle_indices]
    dev_size = int(0.3*size)
    train_iter = DataIter(x[-dev_size:],y[-dev_size:],batch_size)
    dev_iter = DataIter(x[:-dev_size],y[:-dev_size],batch_size)
    layer_shapes[0]=dim
    layer_shapes.append(train_iter.y_dim)
    layers = []
    for i in range(1,len(layer_shapes)):
        layers.append(Perceptron(batch_size, layer_shapes[i-1], layer_shapes[i]))
    nn = Network(layers)
    for i in range(epoch):
        iters = train_iter.size / batch_size
        if train_iter.size % batch_size >0:
            iters+=1
        for j in range(iters):
            x_,y_ = train_iter.next_batch()
            result = nn.forward(x_)
            nn.backpropagation(y_)

        # Evaluate
        if i % evaluate_every_steps == 0:
            train_pre,train_loss,train_correct = evaluate(nn, train_iter)
            dev_pre,dev_loss,dev_correct = evaluate(nn, dev_iter)
            logging.info('[epoch %d]trainig data loss: %f, training data correct rate: %f' % (i,train_loss, train_correct))
            logging.info('[epoch %d]dev data loss: %f, dev data correct rate: %f' %(i,dev_loss,dev_correct))

        train_iter.shuffle()

    train_pre,train_loss,train_correct = evaluate(nn, train_iter)
    dev_pre,dev_loss,dev_correct = evaluate(nn, dev_iter)
    with open('result.txt','w') as f:
        for pre,label in zip(train_pre,train_iter.y):
            f.write('%f\t%d\n' % (pre,label))
    return train_correct,dev_correct
if __name__=="__main__":
    # if you want to using specfic feature, you should change feature_filter
    # e.g. feature_filter=[1,2] specfic the first 2 dimensions of original features are used for training
    feature_filter=[]
    main(feature_filter)
    '''
    max_dev_correct=0.0
    while True:
        candidate=-1
        for i in range(1,17):
            if i not in feature_filter:
                _,dev_correct = main(feature_filter+[i])
                if dev_correct > max_dev_correct:
                    max_dev_correct = dev_correct
                    candidate = i
        if candidate == -1:
            break
        else:
            feature_filter.append(candidate)
    print(feature_filter)
    print(max_dev_correct)
    '''
