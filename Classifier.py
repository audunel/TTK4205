import numpy as np

class Classifier:
    def __init__(self, datafile):
        # Store datasets
        self.trainingset = dict()
        self.validationset = dict()
        self.classes = list()
        # Data features
        self.num_objs = 0
        self.num_features = 0

    def read_datafile(self, datafile):
        self.num_objs, self.num_features = [int(n) for n in datafile.readline().split()]

        count = 0
        for line in datafile:
            count += 1

            obj_class = int(line.split()[0])
            # store features as column vectors
            feature_vector = np.mat([float(x) for x in line.split()[1:]]).T

            if obj_class not in self.trainingset:
                self.trainingset[obj_class] = []
                self.validationset[obj_class] = []
                self.classes.append(obj_class)

            if(count%2 == 1):
                self.validationset[obj_class].append(feature_vector)
            else:
                self.trainingset[obj_class].append(feature_vector)

    def validate(self):
        # Testing classifier on training data
        correct = 0.0
        N = 0.0
        for obj_class in self.trainingset:
            N += len(self.trainingset[obj_class])
            for obj in self.trainingset[obj_class]:
                if self.classify(obj) == obj_class:
                    correct += 1
        success_rate = 100 * correct/N
        print("Successfully classified {0:.2f} of objects in training set".format(success_rate))

        # Testing classifier on validation data
        correct = 0.0
        N = 0.0
        for obj_class in self.validationset:
            N += len(self.validationset[obj_class])
            for obj in self.validationset[obj_class]:
                if self.classify(obj) == obj_class:
                    correct += 1
        success_rate = 100 * correct/N
        print("Successfully classified {0:.2f} of objects in validation set".format(success_rate))
