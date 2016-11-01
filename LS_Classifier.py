import numpy as np

class LS_Classifier:
    'A Least Squares classifier'
    def __init__(self, datafile):
        self.trainingset = dict()
        self.validationset = dict()
        self.classes = list()
        self.num_objs = 0
        self.num_features = 0

        self.__trained = False

        self.a = np.matrix([[]]).T
        self.read_data(datafile)
        self.train()

    def read_data(self, datafile):
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

    def train(self):
        Y = np.matrix([np.zeros(self.num_features + 1)]*self.num_objs)
        b = np.matrix([np.zeros(self.num_objs)]).T
        self.a = np.matrix([np.zeros(self.num_features + 1)])

        i = 0
        for obj_class in self.trainingset:
            for feature_vector in self.trainingset[obj_class]:
                if(obj_class == 1):
                    b[i] = 1
                else:
                    b[i] = -1
                Y[i] = np.vstack(([1], feature_vector)).T
                i += 1
        self.a = (Y.T * Y).I * Y.T * b
        self.__trained = True

    def classify(self, x):
        if not self.__trained:
            print("Classifier has not been trained yet!")
            return

        y = np.vstack(([1], x))
        return 1 if self.a.T*y > 0 else 2

    def validate(self):
        if not self.__trained:
            print("Classifier has not been trained yet!")
            return

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
