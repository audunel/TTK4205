import numpy as np

class MER_Classifier:
    'A Minimum Error-rate classifier for normally distributed data'
    def __init__(self):
        # Store classes
        classes = list()
        # Matrices for discriminant function
        self.A = dict()
        self.b = dict()
        self.c = dict()

    def train(self, dataset):
        self.classes = dataset.keys()
        num_objs = len(dataset.values())

        for obj_class in dataset:
            N = len(dataset[obj_class])

            # Estimate statistical properties
            P = float(N)/float(num_objs)
            mu = 1.0/N * sum(dataset[obj_class])
            sigma = 1.0/N * sum([(x-mu)*(x-mu).T for x in dataset[obj_class]])
            
            # Calculate values for discriminant functions
            self.A[obj_class] = -0.5*sigma.I
            self.b[obj_class] = sigma.I*mu
            self.c[obj_class] = -0.5*mu.T*sigma.I*mu - 0.5*np.linalg.det(sigma) + np.log(P)

    def classify(self, x):
        g = lambda i : x.T*self.A[i]*x + self.b[i].T*x + self.c[i]
        return max(self.classes, key=g)
