#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 22:13:56 2018

@author: oscar
"""
# we shall create a tensor flow neural network with 784 input, 2 hidden layers with 50 nodes and 10 nodes for outputlayer for the 10 digits to be recorgnised
# Importing the libraries
import numpy as np 
import tensorflow as tf

# this code automatically downloads the MNIST dataset to the directory of the editor you are working on
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Otlining the model
input_size = 784
output_size = 10
hidden_layer_size = 50

#clearong the memory of all variables left rom previous runs(reset the computational graph)
tf.reset_default_graph()

#Declare the placeholders

inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.float32, [None, output_size])


# 1. Creating the first layer (input layer)

weights_1 = tf.get_variable("weights_1", [input_size, hidden_layer_size])
biases_1 = tf.get_variable("biases_1", [hidden_layer_size])

# inserting activation function in the first ouput layer, with activatoin function as ReLu
# we create a linear combination by multiplying weights with inputs plus biases (y = WX + b)

# tf.matmul(A,B) is the same as np.dot(A,B) dot product of A and B but generalised for tensors
outputs_1 = tf.nn.relu(tf.matmul(inputs,weights_1) + biases_1) 
# tf.nn is a model that contains neural network support with corresponding activation fucntions


# 2. Creating the second layer.(1st hidden layer)
weights_2 = tf.get_variable("weights_2", [hidden_layer_size, hidden_layer_size])
biases_2 = tf.get_variable("biases_2", [hidden_layer_size])
# outputs_1 becomes our input for the ouputs_2
outputs_2 = tf.nn.relu(tf.matmul(outputs_1,weights_2) + biases_2) 

# 3. Creating the thir layer, (2nd hidden layer)
weights_3 = tf.get_variable("weights_3", [hidden_layer_size, output_size])
biases_3 = tf.get_variable("biases_3", [output_size])

outputs = tf.matmul(outputs_2, weights_3) + biases_3

# calculating the loss function
loss = tf.nn.softmax_cross_entropy_with_logits(logits = outputs, labels = targets)
# finding the mean loss since it determines a bigger performance boost
mean_loss = tf.reduce_mean(loss)
#tf.reduce.mean() is a method which finds the mean of the elements of a tensor across a dimension

# selecting the optimisation method
optimize = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(mean_loss)
# Adam Optimiser is the bbest advanced optimiser

# Calculating the prediction accurancy

out_equals_target = tf.equal(tf.argmax(outputs,1), tf.argmax(targets,1))
accuracy = tf.reduce_mean(tf.cast(out_equals_target, tf.float32))

# Preparation for execution of the tensorflow model

sess = tf.InteractiveSession()

# Initialising the variables
initializer = tf.global_variables_initializer()
# this method initialises all the tensorflow objects marked as variables

# time to execute the tensorFlow model
sess.run(initializer)

# time for Batching
# batch size = 1 means Stochastic Gradient descent and batch size equalling to the number of samples means gradient descent
batch_size = 100

# batches = number of samples diveded by the batch size
batches_number = mnist.train._num_examples // batch_size

# Stoppping the model
max_epoch = 15
prev_validation_loss = 9999999.

# Loop for optimising the algorrtihm
for epoch_counter in range(max_epoch):
    
    curr_epoch_size = 0.
    
    for batch_counter in range(batches_number):
        
        input_batch, target_batch = mnist.train.next_batch(batch_size)
        _, batch_loss = sess.run([optimizer, mean_loss], 
                                 feed_dict = {inputs: input_batch, targets: target_batch})
        
        curr_epoch_loss += batch_loss
        
    curr_epoch_loss /= batches_number
    
    #Validation process
    input_batch, target_batch = mnist.validation.next_batch(mnist.validation._num_examples)
    
    validation_loss, validation_accuracy = sess.run([mean_loss, accuracy], 
                                                    feed_dic = {inputs: input_batch, targets: target_batch})
    
    print('Epoch '+str(epoch_counter+1)+
          '. Mean loss: '+'{0:.3f}'.format(curr_epoch_loss)+
          '. Validation loss: '+'{0:.3f}'.format(validation_loss)+
          '. Validation accuracy: '+'{0:.2f}'.format(validation_accuracy * 100.)+'%')
    
    if validation_loss > prev_validation_loss:
        break
    
    prev_validation_loss = validation_loss
    
print('End of training.')

# Testing the model
input_batch, target_batch = mnist.test.next_batch(mnist.test._num_examples)
test_accuracy = sess.run([accuracy], 
    feed_dict={inputs: input_batch, targets: target_batch})

# Test accuracy is a list with 1 value, so we want to extract the value from it, using x[0]
# Uncomment the print to see how it looks before the manipulation
# print (test_accuracy)
test_accuracy_percent = test_accuracy[0] * 100.

# Print the test accuracy formatted in percentages
print('Test accuracy: '+'{0:.2f}'.format(test_accuracy_percent)+'%')

