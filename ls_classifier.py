from classifier import Classifier
import numpy as np

class LS_Classifier(Classifier):
    'A Least Squares classifier'
    def __init__(self, datafile):
        Classifier.__init__(self)

        self.a = None

        self.read_datafile(datafile)
        self.train()

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

    def classify(self, x):
        if self.a is None:
            print("a vector undefined. Classifier may not have been trained yet!")
            return

        y = np.vstack(([1], x))
        return 1 if self.a.T*y > 0 else 2
