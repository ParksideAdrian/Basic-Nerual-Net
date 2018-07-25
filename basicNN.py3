import numpy as np
import math

'''
This Neural Network will predict the output of an exclusive or gate, given two input bits
'''

def sigmoid(x):                 #Defining the sigmoid function
    return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):           #Defining the derivative of the sigmoid function
    return sigmoid(x) * (1-sigmoid(x))

epochs = 50000          #epochs or amount of iterations this NN will train over
input_size = 2          #Amount of nodes of the input layer
hidden_size = 3         #Amount of nodes in the hidden layer
output_size = 1         #Amount of nodes in the output layer
learning_rate = 0.1     #"Tuning knob", adjust if neccessary

#Now to define our Data
#Note that the x array holds the inputs, which we'll think of as a pair of bits
#Note that the y array holds their outputs, which is what the exclusive or gate produces
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

#Now to initialize the weights of the NN to random numbers
w_hidden = np.random.uniform(size = (input_size, hidden_size))  #Initializing the weights of the input to the hidden layer
                                                                #np.random.uniform(size = (m, n)) returns an array of size m * n
                                                                #returns an array of size 6, the amount of connections from the
                                                                #   input layer to the hidden layer
w_output = np.random.uniform(size = (hidden_size, output_size)) #Initializing weights from the hidden layer to the output layer
                                                                #returns an array of size 3, the amount of connections from the
                                                                #   hidden layer to the output layer

#Implementation of the Backpropagation algorithm
for epoch in range(epochs):
    #Forward feeding
    act_hidden = sigmoid(np.dot(x, w_hidden)) #Compute the dot product of the input, with the weight matrix of the hidden layer
                                              #After this dot product is computed, feed this into the sigmoid function
                                              #This will "Squish" This dot product value into a number between 0 and 1

    output = np.dot(act_hidden, w_output)     #Compute the dot product of the previous layer's values, with the
                                              #weight matrix of the output layer
    #Calculate Error
    error = y - output                        #Compare the computed value, with the corresponding entry in the 'y' matrix

    if epoch % 5000 == 0:
        print(f'error sum{sum(error)}')

    #Backward propagation
    dZ = error * learning_rate                              #Multiply the error by the learning rate
    w_output += act_hidden.T.dot(dZ)                        #Redefine the weights of the output layer to be the dot product
                                                            #   of the act_hidden matrix with the scalar, dZ
    dH = dZ.dot(w_output.T) * sigmoid_prime(act_hidden)
    w_hidden += x.T.dot(dH)

x_test = x[3] #This will test the input pair [0, 1], using the now, trained model
#We will predict what this input will yield, using only the forward feeding step
#Note that if the NN predicts a value incredibly close to 0, it can be interpreted as a 0
act_hidden = sigmoid(np.dot(x_test, w_hidden))
prediction = np.dot(act_hidden, w_output)
print(prediction)
