import numpy as np

class MER_Classifier:
    'A Minimum Error-rate classifier for normally distributed data'
    def __init__(self, datafile):
        # Store datasets
        self.trainingset = dict()
        self.validationset = dict()
        self.classes = list()
        # Data features
        self.num_objs = 0
        self.num_features = 0
        # Matrices for discriminant function
        self.A = dict()
        self.b = dict()
        self.c = dict()
        # Flag to indicate whether training has been performed
        self.__trained = False

        self.read_datafile(datafile)
        self.train()

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

    def train(self):
        for obj_class in self.trainingset:
            N = len(self.trainingset[obj_class])

            # Estimate statistical properties
            P = float(N)/float(self.num_objs)
            mu = 1.0/N * sum(self.trainingset[obj_class])
            sigma = 1.0/N * sum([(x-mu)*(x-mu).T for x in self.trainingset[obj_class]])
            
            # Calculate values for discriminant functions
            self.A[obj_class] = -0.5*sigma.I
            self.b[obj_class] = sigma.I*mu
            self.c[obj_class] = -0.5*mu.T*sigma.I*mu - 0.5*np.linalg.det(sigma) + np.log(P)

            self.__trained = True

    def classify(self, x):
        if not self.__trained:
            print("Classifier has not been trained yet!")
            return

        g = lambda i : x.T*self.A[i]*x + self.b[i].T*x + self.c[i]
        return max(self.classes, key=g)

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
