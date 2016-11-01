from classifier import Classifier
import numpy as np

class kNN_Classifier(Classifier):
    'A Least Squares classifier'
    def __init__(self, datafile):
        Classifier.__init__(self)

        self.dim = 1

        self.read_datafile(datafile)

    def classify(self, x):
        #TODO: Make faster implementation
        dist = lambda y : np.linalg.norm(x[:self.dim] - y[:self.dim])

        lowest_distance = float('inf')
        current_class = None
        for obj_class in self.trainingset:
            for feature_vector in self.trainingset[obj_class]:
                distance = dist(feature_vector)
                if distance < lowest_distance:
                    lowest_distance = distance
                    current_class = obj_class

        return current_class

    def error_estimate_dimensions(self, use_training_set=False):
        self.dim = 1
        error_estimates = dict()
        while(self.dim <= self.num_features):
            print("P(e) = {0:.2f} for {1} dimension(s)".format(Classifier.error_estimate(self, use_training_set), self.dim))
            self.dim += 1
