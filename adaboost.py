
import sys, math, numpy, random

# function to update weights
def updateWeights(weights, labels_array, error):
    current_labels = numpy.array(labels_array, dtype=object)

    # Reassign labels
    current_labels[current_labels==True] = float(error)/float(1-error)
    current_labels[current_labels==False] = 1
    current_labels = current_labels.astype(float)

    new_weights = numpy.multiply(weights, current_labels)

    # normalize weights
    new_weights = new_weights / numpy.sum(new_weights)
    return new_weights

    return numpy.array(new_weights)

# read training data, create stumps and weights
def train(train_file, stump_count):
    #list of lists -> each row contains an image having 192 feature vectors
    training_data = []

    list_0 = []
    list_90 = []
    list_180 = []
    list_270 = []

    print ("Reading training data")
    def addToLabels(a, b, c, d):
        list_0.append(a)
        list_90.append(b)
        list_180.append(c)
        list_270.append(d)

    with open(train_file) as f:
        content = f.readlines()

    for line in content:
        if line.split()[1] == '0':
            addToLabels(True,False,False,False)
        elif line.split()[1] == '90':
            addToLabels(False,True,False,False)
        elif line.split()[1] == '180':
            addToLabels(False,False,True,False)
        elif line.split()[1] == '270':
            addToLabels(False,False,False,True)
        training_data.append([int(x) for x in line.split()[2:]])

    #converting all arrays to numpy array
    training_data_array =  numpy.array(training_data)
    array_0 = numpy.array(list_0)
    array_90 = numpy.array(list_90)
    array_180 = numpy.array(list_180)
    array_270 = numpy.array(list_270)

    print ("Done reading training data")
    print("\nStumps and errors:")

    # compute weights
    weights = numpy.array([float(1)/float(len(array_0)) for x in range(len(array_0))])
    (stumps_0,errors_0) = getStumpsandErrors(training_data_array, array_0, weights, stump_count)
    (stumps_90,errors_90) = getStumpsandErrors(training_data_array, array_90, weights, stump_count)
    (stumps_180,errors_180) = getStumpsandErrors(training_data_array, array_180, weights, stump_count)
    (stumps_270,errors_270) = getStumpsandErrors(training_data_array, array_270, weights, stump_count)

    error_0_weight = [math.log((1-x)/x) for x in errors_0]
    error_90_weight = [math.log((1-x)/x) for x in errors_90]
    error_180_weight = [math.log((1-x)/x) for x in errors_180]
    error_270_weight = [math.log((1-x)/x) for x in errors_270]

    # Create and return a set of stumps and corresponding weights
    stumpAndError = (stumps_0, stumps_90, stumps_180, stumps_270, error_0_weight,\
        error_90_weight, error_180_weight, error_270_weight)
    return stumpAndError

# gets the set of decision stumps and errors
def getStumpsandErrors(training_data_array, labels, weight_array, stump_count):

    weights = weight_array
    stumps = []
    errors = []

    for stump in range(stump_count):
        stumpMax = -sys.maxsize - 1
        maxFirst = -sys.maxsize - 1
        maxSecond = -sys.maxsize - 1
        bestLabels = -sys.maxsize - 1
        stumpMaxError = 0

        for first in range(1000):
            first = random.randint(0, 100)

            for second in range(20):
                second = random.randint(0, 100)

                # check randomly for every (i,j) pair of columns in the feature vectors.
                if first!=second and (first, second) not in stumps and (second, first) not in stumps:
                    currentCols = numpy.greater(training_data_array[:,first], training_data_array[:,second])

                    # compare with labels to check for misclassification
                    result = (currentCols == labels)
                    total = numpy.sum(numpy.multiply(weights, result))
                    if stumpMax < total:
                        # putting to array to check inverse comb
                        stumpMax = total
                        maxFirst = first
                        maxSecond = second
                        bestLabels = result.astype(bool)

                        # number of misclassified labels
                        # used later to generate the final vote
                        stumpMaxError = (float(len(labels))-float(numpy.sum(currentCols == labels)))/float(len(labels))

        weights = updateWeights(weights, bestLabels, stumpMaxError)
        stumps.append((maxFirst, maxSecond))
        errors.append(stumpMaxError)

    print("\n")
    print (stumps)
    print (errors)
    return (stumps, errors)

# function to get the final vote
def getVote(stumps_x,error_x_weight_list, image, stump_count):
    votes = []

    for i in range(stump_count):
        # prepare votes
        if image[stumps_x[i][0]] > image[stumps_x[i][1]]:
            votes.append(+1)
        else:
            votes.append(-1)

    final_vote = numpy.multiply(numpy.array(votes),numpy.array(error_x_weight_list[:stump_count]))
    return numpy.sum(final_vote)

def classify(test_file, stumpAndError, stump_count):
    adaboost_output = []
    confusion_matrix = [[0] * 4 for i in range(4)]
    orientations = [0, 90, 180, 270]

    (stumpsFor0, stumpsFor90, stumpsFor180, stumpsFor270, error_0_weight, error_90_weight,\
        error_180_weight, error_270_weight) = stumpAndError

    with open(test_file) as f:
        test_data = f.readlines()

    print ("Done reading test data.")
    print ("Start Classification")

    # Classify based on the max vote from the individual votes
    for line in test_data:
        actual = orientations.index(int(line.split()[1]))
        image = [int(x) for x in line.split()[2:]]

        voteFor0 = getVote(stumpsFor0, error_0_weight, image, stump_count)
        voteFor90 = getVote(stumpsFor90, error_90_weight, image, stump_count)
        voteFor180 = getVote(stumpsFor180, error_180_weight, image, stump_count)
        voteFor270 = getVote(stumpsFor270, error_270_weight, image, stump_count)
        votes = [voteFor0, voteFor90, voteFor180, voteFor270]

        predicted = votes.index(max(votes))
        adaboost_output.append(line.split()[0] + " " + str(orientations[predicted]))
        confusion_matrix[actual][predicted] += 1

    print ("Done with Classification")
    adaboost_output_file = open('adaboost_output.txt', 'w')
    for line in adaboost_output:
        adaboost_output_file.write("%s\n" % line)

    # Print the confusion matrix and the accuracy
    print ("\nPrinting Confusion Matrix:")
    print ("0           "+str(confusion_matrix[0][0])+"   "+str(confusion_matrix[0][1])+"   "+str(confusion_matrix[0][2])+"   "+str(confusion_matrix[0][3]))
    print ("90           "+str(confusion_matrix[1][0])+"  "+str(confusion_matrix[1][1])+"   "+str(confusion_matrix[1][2])+"   "+str(confusion_matrix[1][3]))
    print ("180          "+str(confusion_matrix[2][0])+"   "+str(confusion_matrix[2][1])+"  "+str(confusion_matrix[2][2])+"   "+str(confusion_matrix[2][3]))
    print ("270          "+str(confusion_matrix[3][0])+"   "+str(confusion_matrix[3][1])+"   "+str(confusion_matrix[3][2])+"  "+str(confusion_matrix[3][3]))
    correct = 0
    total = 0

    for i in range(len(confusion_matrix)):
        correct += confusion_matrix[i][i]
        total += sum(confusion_matrix[i])

    print ("Overall Accuracy is (" + str(correct) + "/" + str(total) + "): " + str(float(correct)*100.0/float(total)) + "%")