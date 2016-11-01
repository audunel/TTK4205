import numpy as np

class MER_Classifier:
    'A Least Squares classifier'
    def __init__(self, datafile):
        self.trainingset = dict()
        self.validationset = dict()
        self.num_objs = 0
        self.num_features = 0
        self.classes = 0

        self.a = np.matrix([[]]).T

    def train(self):

    def classify(self, x):
