from sys import argv
from mer_classifier import MER_Classifier
from ls_classifier import LS_Classifier
from knn_classifier import kNN_Classifier

# Read data from file
if(len(argv) > 1):
    datafile = open(argv[1],'r')
else:
    # Default to dataset 1
    datafile = open('ds-1.txt','r')

if(len(argv) > 2):
    classifier_type = argv[2]
    if(classifier_type == 'MER'):
        classifier = MER_Classifier(datafile)
    elif(classifier_type == 'LS'):
        classifier = LS_Classifier(datafile)
    elif(classifier_type == 'kNN'):
        classifier = kNN_Classifier(datafile)
    else:
        print("Classifier not found. Using Least Squares")
        classifier = LS_Classifier(datafile)
else:
    classifier = LS_Classifier(datafile)

C = classifier.error_estimate()
