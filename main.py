import numpy as np
from sys import argv
from MER_Classifier import MER_Classifier

# Read data from file
if(len(argv) > 1):
    datafile = open(argv[1],'r')
else:
    datafile = open('ds-1.txt','r')

classifier = MER_Classifier(datafile)
classifier.validate()
