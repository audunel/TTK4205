from classifier import Classifier
import numpy as np

class kNN_Classifier(Classifier):
    'A Least Squares classifier'
    def __init__(self, datafile):
        Classifier.__init__(self)

        self.read_datafile(datafile)

    def classify(self, x):
        #TODO: Implement for k > 1
        return min(self.classes, key = lambda y : np.linalg.norm(x - y))
