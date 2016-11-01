from sys import argv
import argparse
from mer_classifier import MER_Classifier
from ls_classifier import LS_Classifier
from knn_classifier import kNN_Classifier

parser = argparse.ArgumentParser()
parser.add_argument('datafile', help='The file the data is to be read from')
parser.add_argument('classifier', type=str.lower, help='The type of classifier to be used (MER, LS or kNN)')
args = parser.parse_args()

# Read data from file
datafile = open(args.datafile,'r')

if(args.classifier == 'mer'):
    classifier = MER_Classifier(datafile)
elif(args.classifier == 'ls'):
    classifier = LS_Classifier(datafile)
elif(args.classifier == 'knn'):
    classifier = kNN_Classifier(datafile)
else:
    print('Classifier not recognized. Using Least Squares')
    classifier = LS_Classifier(datafile)

if(args.classifier == 'knn'):
    classifier.error_estimate_dimensions()
else:
    error_rate_training = classifier.error_estimate(use_training_set=True)
    print ('P(e) = {0:.2f} (Training set)'.format(error_rate_training))
    error_rate_validation = classifier.error_estimate()
    print('P(e) = {0:.2f} (Validation set)'.format(error_rate_validation))
