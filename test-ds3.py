import numpy as np
from mer_classifier import MER_Classifier
from ls_classifier import LS_Classifier
from knn_classifier import kNN_Classifier

print("Dataset 3")
print("\nTesting all combinations of features")
datafile = open('data/ds-3.txt', 'r')
classifier = kNN_Classifier(datafile)
#classifier.error_estimate_all_dimensions()

masks = [np.array([True, False, False, False]),
        np.array([False, True, True, False]),
        np.array([True, True, True, False]),
        np.array([True, True, True, True])]

for dim, mask in enumerate(masks):
    print("----------------------------")
    print("{} Dimension(s)".format(dim+1))
    print("mask = {}".format(mask))

    # MER classifier
    datafile = open('data/ds-3.txt', 'r')
    classifier = MER_Classifier(datafile, mask)
    print("\nMER Classifier")
    print("P(e) = {0:.2f} (Training data)".format(classifier.error_estimate(use_training_set=True)))
    print("P(e) = {0:.2f} (Validation data)".format(classifier.error_estimate()))

    # LS classifier
    datafile = open('data/ds-3.txt', 'r')
    classifier = LS_Classifier(datafile, mask)
    print("LS Classifier")
    print("P(e) = {0:.2f} (Training data)".format(classifier.error_estimate(use_training_set=True)))
    print("P(e) = {0:.2f} (Validation data)".format(classifier.error_estimate()))
