# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math
import samples

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
    


"this code takes one face image and one data image, breaks up the images into quadrants and calculates"
"the number of pixels in each quadrant, still have to generalize these to functions"

if __name__ == '__main__':

    "stores training data and appropriate labels for faces"
    items = samples.loadDataFile("facedata/facedatatrain",1,60,70)
    labels = samples.loadLabelsFile("facedata/facedatatrainlabels",1)
    print items[0]

    "break up face data into 42 10x10 pixel quadrants for feature extraction"
    x = 0; y = 0;
    feature_quadrants = [] #will be a list of lists
    temp_array = []
    i_start = 0;  i_end = 10;
    j_start = 0;  j_end = 10;

    #converts facedata into 42 10x10 pixel quadrants each
    while i_end <= 60 and j_end <= 70:
        #parse through image and store pixels in a temporary array
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                temp_array.append(items[0].getPixel(i,j))

        #add temp_array to feature_quadrant array and reassign temp_array
        feature_quadrants.append(temp_array)
        temp_array = []

        #update iterators for parsing through image
        if j_end != 70:
            j_start = j_end
            j_end = j_end + 10
        else:
            j_start = 0
            j_end = 10
            i_start = i_end
            i_end = i_end + 10

    print feature_quadrants
    print len(feature_quadrants)

    "determines the number of non-zero of pixels in each quadrant"
    pix_counter = 0; #keeps track of non-zero pixels in a quadrant
    pcounter_array = []

    for x in range(len(feature_quadrants)):
        temp = feature_quadrants[x];
        for y in range(len(temp)):
            if temp[y] != 0:
                pix_counter = pix_counter + 1
        pcounter_array.append(pix_counter)
        pix_counter = 0

    print pcounter_array
    print len(pcounter_array)

    "break up digit data into 49 4x4 pixel quadrants for feature extraction"
    items = samples.loadDataFile("digitdata/trainingimages", 1, 28, 28)
    labels = samples.loadLabelsFile("digitdata/traininglabels", 1)
    print items[0]
    x = 0;
    y = 0;
    feature_quadrants2 = []  # will be a list of lists
    temp_array = []
    i_start = 0; i_end = 4;
    j_start = 0; j_end = 4;

    while i_end <= 28 and j_end <= 28:
        #parse through image and store pixels in a temporary array
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                temp_array.append(items[0].getPixel(i,j))

        #add temp_array to feature_quadrant array and reassign temp_array
        feature_quadrants2.append(temp_array)
        temp_array = []

        #update iterators for parsing through image
        if j_end != 28:
            j_start = j_end
            j_end = j_end + 4
        else:
            j_start = 0
            j_end = 4
            i_start = i_end
            i_end = i_end + 4

    print feature_quadrants2
    print len(feature_quadrants2)

    "determines the number of non-zero of pixels in each quadrant"
    pix_counter2 = 0;  # keeps track of non-zero pixels in a quadrant
    pcounter_array2 = []

    for x in range(len(feature_quadrants2)):
        temp = feature_quadrants2[x];
        for y in range(len(temp)):
            if temp[y] != 0:
                pix_counter2 = pix_counter2 + 1
        pcounter_array2.append(pix_counter2)
        pix_counter2 = 0

    print pcounter_array2
    print len(pcounter_array2)