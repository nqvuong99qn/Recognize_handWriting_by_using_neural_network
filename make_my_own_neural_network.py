"""
    @author: Vuong. NQ
=====================
=====================
"""
print(__doc__)

import numpy as np 
import scipy.special
import matplotlib.pyplot as plt 


class NeuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes
        self.lr = learningRate
        
        # set activate funtion is sidmoid function 
        self.activate_function = lambda x: scipy.special.expit(x)  
        # probability density function of the normal distribution ( Gauss, Laplace)
        # loc : micro, scale: xich ma, size: size of matrix.
        self.w_ih = np.random.normal(0.0, pow(self.hNodes, -0.5), (self.hNodes, self.iNodes))  # w_ih (w_input_hidden)
        self.w_ho = np.random.normal(0.0, pow(self.oNodes, -0.5), (self.oNodes, self.hNodes))   # w_ho (w_hidden_output)
        pass

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2) .T
        targets = np.array(targets_list, ndmin=2) .T
        hidden_inputs = np.dot(self.w_ih, inputs)
        hidden_outputs = self.activate_function(hidden_inputs)
        final_inputs = np.dot(self.w_ho, hidden_outputs)
        final_outputs = self.activate_function(final_inputs)
        error_outputs = targets - final_outputs
        hidden_errors = np.dot(self.w_ho.T, error_outputs)
        #backpropagating
        self.w_ho += self.lr * np.dot(error_outputs * final_outputs * (1.0 - final_outputs), np.transpose(hidden_errors))
        self.w_ih += self.lr * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), np.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2) .T
        #print(inputs, '+++')
        hidden_inputs = np.dot(self.w_ih, inputs)
        hidden_outpusts = self.activate_function(hidden_inputs)
        final_inputs = np.dot(self.w_ho, hidden_outpusts)
        final_outputs = self.activate_function(final_inputs)
        return final_outputs
        pass

# setting number of Input nodes, hidden nodes and output nodes, learning rate
inputNodes = 784
hiddenNodes = 100
outputNodes = 10
learning_rate = 0.2

# read training data 
training_data_file = open("MNIST Dataset/mnist_train_100.csv")
training_data_list = training_data_file.readlines()
training_data_file.close()

#create instance of neural network
noron_network = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learning_rate)

# train neural network
for x in training_data_list:
    all_values = x.split(',')
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    
    # create the target output values
    targets = np.zeros(outputNodes) + 0.01
    targets[int(all_values[0])] = 0.99      #int(all_values[0]) is value be labeled 
    
    #handle training neural
    noron_network.train(inputs, targets)
    pass

#test and determind performance

# load test data 

test_data_file = open("MNIST Dataset/mnist_test_10.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()

# set a Score card list to determind performance of model
scoreCard = []
for test in test_data_list:
    all_values = test.split(',')
    value_labeled = int(all_values[0])

    ### Plot handwriting if you want....

    # imageArray = np.asfarray(all_values[1:]).reshape((28, 28))
    # plt.title("HandWriting")
    # plt.imshow(imageArray)
    # plt.show()
    

    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputTest = noron_network.query(inputs) #query to test model
    value_label_trained = np.argmax(outputTest) # find the index of highest value in array.

    if value_label_trained == value_labeled:
        scoreCard.append(1)
    else:
        scoreCard.append(0)
            
scoreArray = np.array(scoreCard)
print('Performance: ', scoreArray.sum() / scoreArray.size * 100, '%')
