# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 13:34:57 2017

@author: snehal vartak
"""
from __future__ import division
import numpy as np
import math
import random

random.seed(2)
def read_file(filename):
    f = open(filename,'r')
    lines = f.readlines()
    # Since the each line represents a input data
    input_data = [line.strip().split() for line in lines]
    training = []
    names =[]
    #for each example in training data get apirs of features and output
    # Divide the features by the max value 255 to bring it to a range of 0 to 1
    for i in range(len(input_data)):
        
        sample_features = [float(a)/255 for a  in input_data[i][2:]]
        val = int(input_data[i][1])
        if val == 0:
            sample_class = [1,0,0,0]
        elif val == 90:
            sample_class = [0,1,0,0]
        elif val == 180:
            sample_class = [0,0,1,0]
        else:
            sample_class = [0,0,0,1]
        sample_features.append(1.) # add a bias variable
        training.append([sample_features, sample_class])
        names.append(input_data[i][0])
        
    return training, names

def sigmoid(data):
    return 1./(1 + math.exp(-data))

def derv(data):
    return data*(1- data)

def trainNeuralNet(train_data, input_nodes, hidden_nodes, output_nodes, learning_rate):
    ## Initialize the weights for the hidden layer and output layer with random values between 0 and 1
    input_nodes += 1 
    hidden_weights  = [[random.uniform(-0.1,0.1) for i in range(hidden_nodes)] for j in range(input_nodes)]
    output_weights = [[random.uniform(-0.1,0.1) for i in range(output_nodes)] for j in range(hidden_nodes)]
    
    train_data = train_data
    error_count  = 0
    #print len(hidden_weights)
    #print len(output_weights)
    # Iterate over each example and update the weights based on error
    # Here I have set the value to 24701 as I am taking approximately 2/3 of the train data to train the weights
    for k in range(int(round(len(train_data)*2/3))):
        features = train_data[k][0]
        label = train_data[k][1]
        hidden_layer = []
        # Calulcate the values for each node of hidden layer and pass them to the sigmoid function
        for i in range(hidden_nodes):
            sum_for_each_hidden_node = 0.
            for j in range(input_nodes):
                sum_for_each_hidden_node += features[j] * hidden_weights[j][i]
            hidden_layer.append(sigmoid(sum_for_each_hidden_node))
        #print len(hidden_layer)
        output_layer = []
        # Calulcate the values for each node of output layer
        for i in range(output_nodes):
            sum_for_each_output_node = 0.
            for j in range(hidden_nodes):
                sum_for_each_output_node += hidden_layer[j] * output_weights[j][i]
            output_layer.append(sigmoid(sum_for_each_output_node))
        #print len(output_layer)
        
        if (output_layer.index(max(output_layer)) != label.index(max(label))):
            error_count += 1
        #output_file.append(output_layer.index(max(output_layer))*90)


        ## Calulcate the error and backpropogate
        output_deltas = []
        hidden_deltas = []
        for i in range(output_nodes):
            error =  label[i] - output_layer[i]
            output_deltas.append(error * derv(output_layer[i]))  
        
        #Calculate hidden layer deltas
        for i in range(hidden_nodes):
            hidden_error = 0.
            for j in range(output_nodes):
                hidden_error += output_deltas[j]* output_weights[i][j]
            hidden_deltas.append(hidden_error *  derv(hidden_layer[i])) 
        #Update output weights
        for i in range(hidden_nodes):
            for j in range(output_nodes):
                output_weights[i][j]+= (learning_rate * output_deltas[j] * hidden_layer[i])
        #Update the hidden layer weights
        for i in range(input_nodes):
            for j in range(hidden_nodes):
                hidden_weights[i][j] += (learning_rate * hidden_deltas[j]* features[i])
    print error_count
    return hidden_weights, output_weights

def dumpModel(model_file_name,hidden_weights, output_weights,input_nodes, hidden_nodes, output_nodes):
    filename = model_file_name
    f = open(filename,'w')
    lines_of_text = []
    input_nodes += 1
    lines_of_text.append("Number of input nodes "+ str(input_nodes)+"\n")
    lines_of_text.append("Number of hidden nodes "+ str(hidden_nodes)+"\n")
    lines_of_text.append("Number of output nodes "+ str(output_nodes)+"\n")
    lines_of_text.append("=======Weights of Hidden Layer=======\n")
    hidden_weights_str =  "\n".join([ " ".join([str(hidden_weights[i][j]) for j in range(hidden_nodes)]) for i in range(input_nodes)])
    f.writelines(lines_of_text)
    f.writelines(hidden_weights_str)
    f.writelines("\n=======Weights of Output Layer=======\n")
    output_weights_str =  "\n".join([ " ".join([str(output_weights[i][j]) for j in range(output_nodes)]) for i in range(hidden_nodes)])
    f.writelines(output_weights_str)
    f.close()
    
def loadModel(filename):
    f = open(filename,'r')
    lines = f.readlines()
    model_hidden_weights = []
    model_output_weights = []
    for i in range(len(lines)):
        if i == 0:
            model_input_nodes = int(lines[i].split()[-1])
            #print model_input_nodes
        if i == 1:
            model_hidden_nodes = int(lines[i].split()[-1])
            #print model_hidden_nodes
        if i == 2:
            model_output_nodes = int(lines[i].split()[-1])
            #print model_output_nodes
        #get the line where hidden nodes end
        end_hidden = 4 + model_input_nodes
        if i in range(4, end_hidden):
            #Build the hidden layer weight matrix
            model_hidden_weights.append([float(line) for line in lines[i].strip().split()])
        #output weights start one line after hidden layer weights end
        if i >= (end_hidden+1):
            #Build the hidden layer weight matrix
            model_output_weights.append([float(line) for line in lines[i].strip().split()])
            
    return model_input_nodes, model_output_nodes, model_hidden_nodes, model_hidden_weights, model_output_weights


def testNeuralNet(model_file_name, test_data, test_data_names):
    
    model_input_nodes, model_output_nodes, model_hidden_nodes, model_hidden_weights, model_output_weights = loadModel(model_file_name)
    # Cretae a confusion matrix where each index from 0 to 3 corresponds to angles 0, 90, 180, 270 respectively
    confusion_matrix = [[0 for i in range(4)] for j in range(4)]
    output_file = "nnet_output.txt"
    f = open(output_file,'w')
    correct_count = 0
     # Iterate over each example and update the weights based on error
    for k in range(len(test_data)):
        features = test_data[k][0]
        label = test_data[k][1]
        hidden_layer = []
        # Calulcate the values for each node of hidden layer and pass them to the sigmoid function
        for i in range(model_hidden_nodes):
            sum_for_each_hidden_node = 0.
            for j in range(model_input_nodes):
                sum_for_each_hidden_node += features[j] * model_hidden_weights[j][i]
            hidden_layer.append(sigmoid(sum_for_each_hidden_node))
        #print len(hidden_layer)
        output_layer = []
        # Calulcate the values for each node of output layer
        for i in range(model_output_nodes):
            sum_for_each_output_node = 0.
            for j in range(model_hidden_nodes):
                sum_for_each_output_node += hidden_layer[j] * model_output_weights[j][i]
            output_layer.append(sigmoid(sum_for_each_output_node))
        #print len(output_layer)
        
        predicted = output_layer.index(max(output_layer))
        actual = label.index(max(label))
        if (predicted == actual):
            correct_count += 1
        confusion_matrix[actual][predicted] += 1
        if predicted == 0:
            predicted_orient ='0'
        elif predicted == 1:
            predicted_orient ='90'
        elif predicted == 2:
            predicted_orient ='180'
        else:
            predicted_orient ='270'
        output_line = test_data_names[k] + " " +  predicted_orient+"\n"
        f.write(output_line)
    f.close()
    print "Confusion Matrix: " 
    print confusion_matrix
    print "Classification Accuracy: " + str(correct_count*100/len(test_data))



