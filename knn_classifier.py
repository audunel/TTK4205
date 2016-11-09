from classifier import Classifier
import numpy as np
import itertools

class kNN_Classifier(Classifier):
    'A Least Squares classifier'
    def __init__(self, datafile, mask=None):
        Classifier.__init__(self, datafile, mask)

        self.dim = self.num_features

    def classify(self, x):
        #TODO: Faster implementation and support of k > 1
        dist = lambda y : np.linalg.norm(x[self.mask,...] - y[self.mask,...])

        lowest_distance = float('inf')
        current_class = None
        for obj_class in self.trainingset:
            for feature_vector in self.trainingset[obj_class]:
                distance = dist(feature_vector)
                if distance < lowest_distance:
                    lowest_distance = distance
                    current_class = obj_class

        return current_class

    def error_estimate_all_dimensions(self, use_training_set=False):
        for dim in range(1,self.num_features+1):
            print('For {} dimension(s)'.format(dim))
            for bits in itertools.combinations(range(self.num_features),dim):
                self.mask = np.zeros(self.num_features, dtype=bool)
                for bit in bits:
                    self.mask[bit] = True
                print('Mask = {}'.format(self.mask))
                print('P(e) = {0:.2f}'.format(Classifier.error_estimate(self, use_training_set)))
        mask = np.ones(self.num_features, dtype=bool)
