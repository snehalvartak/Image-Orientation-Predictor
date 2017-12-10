#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 17:14:01 2017

@author: snehal vartak

"""
import sys
import time
import neuralnet
import random 

random.seed(1)
if len(sys.argv) != 5:
    print ("Incorrect Number of Arguments")
else:
    # get whether to run train or test method
    train_test = str(sys.argv[1])
    
    # get the input data  file
    input_data_file = sys.argv[2]
    
    # get the model file name
    model_file_name = sys.argv[3]
    
    #get the algorithm to run
    algorithm = sys.argv[4]
    
    
    if algorithm == "nearest":
        if train_test== "train":
            print("Training nearest neighbor classifier")
        elif train_test == "test":
            print ("Running the nearest neighbor classifier on test data")
        
    elif algorithm == "adaboost":
        if train_test== "train":
            print("Training adaboost classifier")
            
        elif train_test == "test":
            print ("Running the adaboost classifier on test data")
            
    elif algorithm == "nnet":
        if train_test== "train":
            print("Reading the training data...")
            train_data, train_data_names = neuralnet.read_file(input_data_file)   
            #SET TRAINING PARAMETERS
            NUM_IN_NODES = 192
            NUM_HIDDEN_NODES = 5
            NUM_OUTPUT_NODES = 4
            LEARNING_RATE = 0.6
            print("Training neural network...")
            start_training = time.time()
            hidden_weights, output_weights = neuralnet.trainNeuralNet(train_data, NUM_IN_NODES, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, LEARNING_RATE)
            print( "Training time: -- " + str(time.time()-start_training))
            print("Output the model to file...")
            neuralnet.dumpModel(model_file_name,hidden_weights, output_weights,NUM_IN_NODES, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES)
            print("Done!")
        elif train_test == "test":
            print("Reading the test data...")
            test_data, test_data_names = neuralnet.read_file(input_data_file)
            print("Loading the model and testing it...")
            start_test = time.time()
            neuralnet.testNeuralNet(model_file_name,test_data,test_data_names)
            print( "Test time: -- " + str(time.time()-start_test))
            print("Done")
        else:
            print("Invalid input arguments")
            
    elif algorithm == "best":
        if train_test== "train":
            print("Train the best classifier")
            
        elif train_test == "test":
            print ("Testing the best classifier")
            
    else:
        print ("Invalid algorithm")
