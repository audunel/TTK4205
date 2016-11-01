from sys import argv
from MER_Classifier import MER_Classifier
from LS_Classifier import LS_Classifier
from kNN_Classifier import kNN_Classifier

# Read data from file
if(len(argv) > 1):
    datafile = open(argv[1],'r')
else:
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
        print("Classifier not found")
else:
    # Default to MER classifier
    classifier = MER_Classifier(datafile)

classifier.validate()
