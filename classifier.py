import numpy as np

class Classifier:
    def __init__(self):
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

    def error_estimate(self, use_training_set=False):
        '''Returns error rate estimate. If use_training_set is True,
        the validation will be performed on the training set'''
        num_classes = len(self.classes)
        C = np.matrix([np.zeros(num_classes)]*num_classes)

        if(use_training_set):
            dataset = self.trainingset
        else:
            dataset = self.validationset

        for obj_class in dataset:
            for obj in dataset[obj_class]:
                # Data is 1-indexed, Python is 0-indexed
                C[obj_class - 1, self.classify(obj) - 1] += 1

        # Estimated error rate is given by the sum of nondiagonal components divided by the sum of all components
        numerator = 0.0
        denominator = 0.0
        for i in range(num_classes):
            for j in range(num_classes):
                if(i != j):
                    numerator += C[i,j]
                denominator += C[i,j]
        return numerator/denominator
