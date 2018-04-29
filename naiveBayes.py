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

    "*** YOUR CODE HERE ***" #still need to code for digit data
    # stores training data and appropriate labels for faces
    n = 10;
    items = samples.loadDataFile("facedata/facedatatrain", n, 60, 70)
    labels = samples.loadLabelsFile("facedata/facedatatrainlabels", n)
    all_feature_vectors = []  # stores all 42 quadrants of all sample images
    all_pcounter_vectors = [] # stores num of pixels of in all 42 quadrants in all sample images
    all_feature_vectors = self.faceDivideData(items,n)
    all_pcounter_vectors = self.featureExtractor(all_feature_vectors,n)
        
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

#do we have to implement this??
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

#wrote helper methods to aid with feature extraction for naiveBayes below
  def faceDivideData(self, items, num_samples):
      "break up face data into 42 10x10 pixel quadrants for feature extraction"
      x = 0; y = 0;
      feature_quadrants = []  # will be a list of lists
      temp_array = []
      all_arrays = []
      i_start = 0; i_end = 10;
      j_start = 0; j_end = 10;

      for m in range(num_samples):
          # converts facedata into 42 10x10 pixel quadrants each
          while i_end <= 60 and j_end <= 70:
              # parse through image and store pixels in a temporary array
              for i in range(i_start, i_end):
                  for j in range(j_start, j_end):
                      temp_array.append(items[m].getPixel(i, j))

              # add temp_array to feature_quadrant array and reassign temp_array
              feature_quadrants.append(temp_array)
              temp_array = []

              # update iterators for parsing through image
              if j_end != 70:
                  j_start = j_end
                  j_end = j_end + 10
              else:
                  j_start = 0
                  j_end = 10
                  i_start = i_end
                  i_end = i_end + 10
          all_arrays.append(feature_quadrants)
          feature_quadrants = []

      return all_arrays

  def digitDivideData(self, items, num_samples):
      "break up digit data into 49 4x4 pixel quadrants for feature extraction"
      x = 0; y = 0;
      feature_quadrants = []  # will be a list of lists
      temp_array = []
      all_arrays = []
      i_start = 0; i_end = 4;
      j_start = 0; j_end = 4;

      for m in range(num_samples):
          while i_end <= 28 and j_end <= 28:
              # parse through image and store pixels in a temporary array
              for i in range(i_start, i_end):
                  for j in range(j_start, j_end):
                      temp_array.append(items[m].getPixel(i, j))

              # add temp_array to feature_quadrant array and reassign temp_array
              feature_quadrants.append(temp_array)
              temp_array = []

              # update iterators for parsing through image
              if j_end != 28:
                  j_start = j_end
                  j_end = j_end + 4
              else:
                  j_start = 0
                  j_end = 4
                  i_start = i_end
                  i_end = i_end + 4
          all_arrays.append(feature_quadrants)
          feature_quadrants = []

      return all_arrays

  def featureExtractor(self, all_feature_vectors, num_samples):
      "determines the number of non-zero of pixels in each quadrant"
      pix_counter = 0;  # keeps track of non-zero pixels in a quadrant
      pcounter_array = []
      all_arrays = []

      for m in range(num_samples):
          feature_quadrants = all_feature_vectors[m]
          for x in range(len(feature_quadrants)):
              temp = feature_quadrants[x];
              for y in range(len(temp)):
                  if temp[y] != 0:
                      pix_counter = pix_counter + 1
              pcounter_array.append(pix_counter)
              pix_counter = 0
          all_arrays.append(pcounter_array)
          pcounter_array = []

      return all_arrays

#currently working on figuring out the logic for trainAndTune()
if __name__ == '__main__':
    print "Training Phase"
    #stores training data and appropriate labels for faces
    n = 100;
    items = samples.loadDataFile("facedata/facedatatrain",n,60,70)
    labels = samples.loadLabelsFile("facedata/facedatatrainlabels",n)
    all_feature_vectors = [] #stores all 42 quadrants of all sample images

    for k in range(n):
        #break up face data into 42 10x10 pixel quadrants for feature extraction
        feature_quadrants = [] #will be a list of lists
        temp_array = []
        i_start = 0;  i_end = 10;
        j_start = 0;  j_end = 10;

        #converts facedata into 42 10x10 pixel quadrants each
        while i_end <= 60 and j_end <= 70:
            #parse through image and store pixels in a temporary array
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    temp_array.append(items[k].getPixel(i,j))

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
        all_feature_vectors.append(feature_quadrants)

    #determines the number of non-zero of pixels in each quadrant"
    all_pcounter_vectors = [] #stores feature vectors of all samples

    for k in range(len(all_feature_vectors)):
        pix_counter = 0;  #keeps track of non-zero pixels in a quadrant
        pcounter_array = []
        feature_quadrants = all_feature_vectors[k]
        for x in range(len(feature_quadrants)):
            temp = feature_quadrants[x];
            for y in range(len(temp)):
                if temp[y] != 0:
                    pix_counter = pix_counter + 1
            pcounter_array.append(pix_counter)
            pix_counter = 0
        all_pcounter_vectors.append(pcounter_array)

    #calculating and storing log joint probabilities
    true_count = 0;
    for i in range(n):
        if labels[i] == 1:
            true_count = true_count + 1;
    false_count = n - true_count

    #calculating number of times a feature = value in the training sets
    #first finds max num of non-zero pixels ifrom all_pcounter_vectors
    max_num = 0;
    for m in range(n):
        temp = all_pcounter_vectors[m]
        max_temp = max(temp)
        if max_temp > max_num:
            max_num = max_temp

    #create num_features x (max_num+1) array for storing number of times a feature = value in the training sets
    #two tables for true and false training data
    true_data = [[0 for i in range(max_num+1)] for j in range(42)]
    false_data = [[0 for i in range(max_num+1)] for j in range(42)]

    for i in range(len(all_pcounter_vectors)):
        temp = all_pcounter_vectors[i]
        for k in range(len(temp)):
            if labels[i] == 1:
                true_data[k][temp[k]] = true_data[k][temp[k]] + 1
            else:
                false_data[k][temp[k]] = false_data[k][temp[k]] + 1


    print "Testing Phase, using k = 1 as smoothing param (haven't tuned using validation data yet)"
    #calculating likelihood of testing data
    smoothing_param = 1
    #steps to take
    #1. separate testing data into quadrants
    #2. count number of non-zero pixels in each quadrant
    #3. calculate the likelihood based on results and stored log-joint probabilities
    test_num = 100
    testData = samples.loadDataFile("facedata/facedatatest",test_num,60,70)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels",test_num)

    all_feature_vectors = []  # stores all 42 quadrants of all test images

    #Step 1
    for k in range(test_num):
        # break up face data into 42 10x10 pixel quadrants for feature extraction
        x = 0;
        y = 0;
        feature_quadrants = []  # will be a list of lists
        temp_array = []
        i_start = 0;
        i_end = 10;
        j_start = 0;
        j_end = 10;

        # converts facedata into 42 10x10 pixel quadrants each
        while i_end <= 60 and j_end <= 70:
            # parse through image and store pixels in a temporary array
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    temp_array.append(testData[k].getPixel(i, j))

            # add temp_array to feature_quadrant array and reassign temp_array
            feature_quadrants.append(temp_array)
            temp_array = []

            # update iterators for parsing through image
            if j_end != 70:
                j_start = j_end
                j_end = j_end + 10
            else:
                j_start = 0
                j_end = 10
                i_start = i_end
                i_end = i_end + 10
        all_feature_vectors.append(feature_quadrants)

    #Step 2
    for k in range(len(all_feature_vectors)):
        pix_counter = 0;  #keeps track of non-zero pixels in a quadrant
        pcounter_array = []
        feature_quadrants = all_feature_vectors[k]
        for x in range(len(feature_quadrants)):
            temp = feature_quadrants[x];
            for y in range(len(temp)):
                if temp[y] != 0:
                    pix_counter = pix_counter + 1
            pcounter_array.append(pix_counter)
            pix_counter = 0
        all_pcounter_vectors.append(pcounter_array)

    #Step 3 - only doing it for facedata images right now
    p_x_ytrue = 1; # p(x|y = true)
    p_x_yfalse = 1; # p(x|y = false)
    denom1 = float((float(true_count)/float(n)) + smoothing_param)
    denom2 = float((float(false_count)/float(n)) + smoothing_param)
    l_x_array = []
    predictions = []

    for q in range(test_num):
        pcounter = all_pcounter_vectors[q]
        for r in range(len(pcounter)):
            if pcounter[r] > max_num:
                t1 = t2 = 0;
            else:
                t1 = true_data[r][pcounter[r]]
                t2 = false_data[r][pcounter[r]]
            num1 = float(true_count * t1 + smoothing_param)
            num2 = float(false_count * t2 + smoothing_param)
            temp1 = num1/denom1
            temp2 = num2/denom2
            p_x_ytrue = float(p_x_ytrue * temp1)
            p_x_yfalse = float(p_x_yfalse* temp2)
            l_x = float(p_x_ytrue*(float(true_count)/float(n)))/float(p_x_yfalse*(float(false_count)/float(n)))
        l_x_array.append(l_x)
        p_x_ytrue = p_x_yfalse = 1

    for q in range(len(l_x_array)):
        if l_x_array[q] >= 1:
            predictions.append(1)
        else:
            predictions.append(0)

    print "l_x array"
    print l_x_array
    print "predictions"
    print predictions
    print "actual"
    print testLabels

    correct = 0
    for q in range(len(l_x_array)):
        if predictions[q] == testLabels[q]:
            correct = correct + 1

    accurate_rate = float(float(correct)/float(test_num))
    print "accuracy rate - faces only"
    print accurate_rate

    #DIGIT DATA STUFF, NEED TO IMPLEMENT AS WELL
    #break up digit data into 49 4x4 pixel quadrants for feature extraction
    #items = samples.loadDataFile("digitdata/trainingimages", 1, 28, 28)
    #labels = samples.loadLabelsFile("digitdata/traininglabels", 1)
    #x = 0; y = 0;
    #feature_quadrants2 = []  # will be a list of lists
    #temp_array = []
    #i_start = 0; i_end = 4;
    #j_start = 0; j_end = 4;

    #while i_end <= 28 and j_end <= 28:
        #parse through image and store pixels in a temporary array
        #for i in range(i_start, i_end):
            #for j in range(j_start, j_end):
                #temp_array.append(items[0].getPixel(i,j))

        #add temp_array to feature_quadrant array and reassign temp_array
        #feature_quadrants2.append(temp_array)
        #temp_array = []

        #update iterators for parsing through image
        #if j_end != 28:
            #j_start = j_end
            #j_end = j_end + 4
        #else:
            #j_start = 0
            #j_end = 4
            #i_start = i_end
            #i_end = i_end + 4

    #print feature_quadrants2
    #print len(feature_quadrants2)

    #determines the number of non-zero of pixels in each quadrant
    #pix_counter2 = 0;  # keeps track of non-zero pixels in a quadrant
    #pcounter_array2 = []

    #for x in range(len(feature_quadrants2)):
        #temp = feature_quadrants2[x];
        #for y in range(len(temp)):
            #if temp[y] != 0:
                #pix_counter2 = pix_counter2 + 1
        #pcounter_array2.append(pix_counter2)
        #pix_counter2 = 0

    #print pcounter_array2
    #print len(pcounter_array2)