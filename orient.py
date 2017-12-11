#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 17:14:01 2017

@author: snehal vartak

This is the main program that call the train and test functions for approipriate algorithms

"""
import os,sys,pickle
import time
import neuralnet
import random

random.seed(1)
if len(sys.argv) < 3:
    print("Incorrect Number of Arguments")
else:
    # get whether to run train or test method
    train_test = str(sys.argv[1])

    # get the input data  file
    input_data_file = sys.argv[2]

    # get the algorithm to run
    if sys.argv[3] != "nearest":
        algorithm = sys.argv[4]

    if sys.argv[3] == "nearest":
        # Get the standand parameters from the command line
        train_file = sys.argv[1]
        test_file = sys.argv[2]
        classifier = sys.argv[3]

        from nearest import *
        if train_test == "train":
            print('Reading Training Data...')
        elif train_test == "test":
            print('Reading Test Data...')

        try:
            trainVectors = readData("train-data.txt", 'train')
            testVectors = readData("test-data.txt", 'test')
        except:
            print()
        calcNeighbours(trainVectors, testVectors)

    elif algorithm == "adaboost":
        # get the model file name
        model_file = sys.argv[3]

        # get the algorithm to run
        algorithm = sys.argv[4]

        if train_test == "train":
            print("Training adaboost classifier")

        elif train_test == "test":
            print("Running the adaboost classifier on test data")

        try:
            stump_count = int(sys.argv[5])
        except:
            print
            "Please enter stump count."
            sys.exit()
        from adaboost import *
        try:
            fileExists = os.path.isfile(model_file)

            if fileExists:
                # Load the model from file
                with open(model_file, 'rb') as input:
                    stumpAndError = pickle.load(input)
                    # Classify based on model from file
                    classify("test-data.txt", stumpAndError, stump_count)
            else:
                stumpAndError = train(train_file, stump_count)
                with open(model_file, 'wb') as output:
                    pickle.dump(stumpAndError, output, pickle.HIGHEST_PROTOCOL)
                    # Classify based on model
                    classify("test-data.txt", stumpAndError, stump_count)
        except IndexError:
            stumpAndError = train(train_file, stump_count)
            # Classify based on model
            classify(test_file, stumpAndError, stump_count)


    elif algorithm == "nnet":
        # get the model file name
        model_file_name = sys.argv[3]

        # get the algorithm to run
        algorithm = sys.argv[4]

        if train_test == "train":
            print("Reading the training data...")
            train_data, train_data_names = neuralnet.read_file(input_data_file)
            # SET TRAINING PARAMETERS
            NUM_IN_NODES = 192
            NUM_HIDDEN_NODES = 35
            NUM_OUTPUT_NODES = 4
            LEARNING_RATE = 0.4
            print("Training neural network...")
            start_training = time.time()
            hidden_weights, output_weights = neuralnet.trainNeuralNet(train_data, NUM_IN_NODES, NUM_HIDDEN_NODES,
                                                                      NUM_OUTPUT_NODES, LEARNING_RATE)
            print("Training time: -- " + str(time.time() - start_training))
            print("Output the model to file...")
            neuralnet.dumpModel(model_file_name, hidden_weights, output_weights, NUM_IN_NODES, NUM_HIDDEN_NODES,
                                NUM_OUTPUT_NODES)
            print("Done!")
        elif train_test == "test":
            print("Reading the test data...")
            test_data, test_data_names = neuralnet.read_file(input_data_file)
            print("Loading the model and testing it...")
            start_test = time.time()
            neuralnet.testNeuralNet(model_file_name, algorithm, test_data, test_data_names)
            print("Test time: -- " + str(time.time() - start_test))
            print("Done")
        else:
            print("Invalid input arguments")

    elif algorithm == "best":
        # get the model file name
        model_file_name = sys.argv[3]

        # get the algorithm to run
        algorithm = sys.argv[4]


        if train_test == "train":
            print("Reading the training data...")
            train_data, train_data_names = neuralnet.read_file(input_data_file)
            # SET TRAINING PARAMETERS
            NUM_IN_NODES = 192
            NUM_HIDDEN_NODES = 35
            NUM_OUTPUT_NODES = 4
            LEARNING_RATE = 0.4
            print("Training neural network...")
            start_training = time.time()
            hidden_weights, output_weights = neuralnet.trainNeuralNet(train_data, NUM_IN_NODES, NUM_HIDDEN_NODES,
                                                                      NUM_OUTPUT_NODES, LEARNING_RATE)
            print("Training time: -- " + str(time.time() - start_training))
            print("Output the model to file...")
            neuralnet.dumpModel(model_file_name, hidden_weights, output_weights, NUM_IN_NODES, NUM_HIDDEN_NODES,
                                NUM_OUTPUT_NODES)
            print("Done!")
        elif train_test == "test":
            print("Reading the test data...")
            test_data, test_data_names = neuralnet.read_file(input_data_file)
            print("Loading the model and testing it...")
            start_test = time.time()
            neuralnet.testNeuralNet(model_file_name, algorithm, test_data, test_data_names)
            print("Test time: -- " + str(time.time() - start_test))
            print("Done")
        else:
            print("Invalid input arguments")
    else:
        print("Invalid algorithm")