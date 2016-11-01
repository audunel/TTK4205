from classifier import Classifier
import numpy as np

class MER_Classifier(Classifier):
    'A Minimum Error-rate classifier for normally distributed data'
    def __init__(self, datafile):
        Classifier.__init__(self)
        # Matrices for discriminant function
        self.A = dict()
        self.b = dict()
        self.c = dict()

        self.read_datafile(datafile)
        self.train()

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
        g = lambda i : x.T*self.A[i]*x + self.b[i].T*x + self.c[i]
        return max(self.classes, key=g)
