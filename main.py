import numpy as np
from sys import argv
from MER_Classifier import MER_Classifier

success_rates_training = list()
success_rates_validation = list()

# Read data from file
if(len(argv) > 1):
    datafile = open(argv[1],'r')
else:
    datafile = open('ds-1.txt','r')

num_objs, num_obj_classs = [int(n) for n in datafile.readline().split()]
num_classes = 0

trainingset = dict()
validationset = dict()

dataset = dict()
count = 0
for line in datafile:
    count += 1

    obj_class = int(line.split()[0])
    # store features as column vectors
    feature_vector = np.mat([float(x) for x in line.split()[1:]]).T

    if obj_class not in trainingset:
        trainingset[obj_class] = []
        validationset[obj_class] = []
        num_classes += 1

    if(count%2 == 1):
        validationset[obj_class].append(feature_vector)
    else:
        trainingset[obj_class].append(feature_vector)

classifier = MER_Classifier()
classifier.train(trainingset)

# Testing classifier on training data
correct = 0.0
N = 0.0
for obj_class in trainingset:
    N += len(trainingset[obj_class])
    for obj in trainingset[obj_class]:
        if classifier.classify(obj) == obj_class:
            correct += 1
success_rate = 100 * correct/N
print("Successfully classified {0:.2f} of objects in training set".format(success_rate))

# Testing classifier on validation data
correct = 0.0
N = 0.0
for obj_class in validationset:
    N += len(validationset[obj_class])
    for obj in validationset[obj_class]:
        if classifier.classify(obj) == obj_class:
            correct += 1
success_rate = 100 * correct/N
print("Successfully classified {0:.2f} of objects in validation set".format(success_rate))
