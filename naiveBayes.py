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
    self.k = 2 # this is the smoothing parameter, ** use it in your train method **
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
    #calculating prior probability of labels
    label_counters = util.Counter()
    for i in range(len(trainingLabels)):
        label_counters[trainingLabels[i]] += 1.0

    #initializes feature counter vectors to 0
    feat_count_zero = util.Counter()
    feat_count_nonzero = util.Counter()
    for i in self.features:
        feat_count_zero[i] = util.Counter()
        feat_count_nonzero[i] = util.Counter()

    #initialize counts for each label as 0
    for i in self.features:
        for j in self.legalLabels:
            feat_count_zero[i][j] = 0
            feat_count_nonzero[i][j] = 0

    #counts number of times a pixel is non-zero for all labels
    for i in range(len(trainingData)):
        temp = trainingData[i]
        temp_label = trainingLabels[i]
        for j in self.features:
            if temp[j] == 0:
                feat_count_zero[j][temp_label] += 1.0
            else:
                feat_count_nonzero[j][temp_label] += 1.0

    #calculating conditional probability along with tuning k value
    best_cond_zero = {}
    best_cont_nonzero = {}
    best_num_guesses = 0
    for k in kgrid:
        num_correct = 0
        temp_cond_zero = {}
        temp_cond_nonzero = {}
        for i in self.features:
            temp_cond_zero[i] = util.Counter()
            temp_cond_nonzero[i] = util.Counter()

        #calculating laplace smoothed conditional probabilities
        for i in self.features:
            for j in self.legalLabels:
                temp_cond_zero[i][j] = ( feat_count_zero[i][j] + k ) / ( label_counters[j] + 2*k )
                temp_cond_nonzero[i][j] = ( feat_count_nonzero[i][j] + k ) / ( label_counters[j] + 2*k )

        temp_cond = {0: util.Counter(), 1: util.Counter()}
        temp_cond[0] = temp_cond_zero
        temp_cond[1] = temp_cond_nonzero
        self.cond = temp_cond
        for i in self.legalLabels:
            label_counters[i] = label_counters[i] / float(len(trainingLabels))
        self.prior = label_counters
        #running the conditionals on validation data to determine accuracy and update k as needed
        
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
    for i in self.legalLabels:
        logJoint[i] = math.log(self.prior[i])
        for j in self.features:
            if datum[j] != 0:
                prob = self.cond[0][j][i]
            else:
                prob = self.cond[1][j][i]
            if (prob > 0 and math.log(prob) != 0):
                logJoint[i] += math.log(prob)
            else:
                logJoint[i] += 0.0
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

#accuracy rates for digit data
if __name__ == '__main__':
    print "Training Phase"
    #stores training data and appropriate labels for faces
    n = 450;
    items = samples.loadDataFile("facedata/facedatatrain",n,60,70)
    labels = samples.loadLabelsFile("facedata/facedatatrainlabels",n)
    all_feature_vectors = [] #stores all 42 quadrants of all sample images

    for k in range(n):
        #break up face data into 100 6x7 pixel quadrants for feature extraction
        feature_quadrants = [] #will be a list of lists
        temp_array = []
        i_start = 0;  i_end = 6;
        j_start = 0;  j_end = 7;

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
                j_end = j_end + 7
            else:
                j_start = 0
                j_end = 7
                i_start = i_end
                i_end = i_end + 6
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
    true_data = [[0 for i in range(max_num+1)] for j in range(100)]
    false_data = [[0 for i in range(max_num+1)] for j in range(100)]

    for i in range(len(all_pcounter_vectors)):
        temp = all_pcounter_vectors[i]
        for k in range(len(temp)):
            if labels[i] == 1:
                true_data[k][temp[k]] = true_data[k][temp[k]] + 1
            else:
                false_data[k][temp[k]] = false_data[k][temp[k]] + 1

    print "Tuning phase, using validation data to fine tune the smoothing parameter"
    kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    valData = samples.loadDataFile("facedata/facedatavalidation", 100, 60, 70)
    valLabels = samples.loadLabelsFile("facedata/facedatavalidationlabels", 100)
    all_feature_vectors = []  # stores all 42 quadrants of all test images
    num_correct = []

    for b in range(len(kgrid)):
        all_feature_vectors = []
        smoothing_param = kgrid[b]
        # Step 1
        for k in range(100):
            # break up face data into 42 10x10 pixel quadrants for feature extraction
            feature_quadrants = []  # will be a list of lists
            temp_array = []
            i_start = 0;
            i_end = 6;
            j_start = 0;
            j_end = 7;

            # converts facedata into 28 15x10 pixel quadrants each
            while i_end <= 60 and j_end <= 70:
                # parse through image and store pixels in a temporary array
                for i in range(i_start, i_end):
                    for j in range(j_start, j_end):
                        temp_array.append(valData[k].getPixel(i, j))

                # add temp_array to feature_quadrant array and reassign temp_array
                feature_quadrants.append(temp_array)
                temp_array = []

                # update iterators for parsing through image
                if j_end != 70:
                    j_start = j_end
                    j_end = j_end + 7
                else:
                    j_start = 0
                    j_end = 7
                    i_start = i_end
                    i_end = i_end + 6
            all_feature_vectors.append(feature_quadrants)

        # Step 2
        for k in range(len(all_feature_vectors)):
            pix_counter = 0;  # keeps track of non-zero pixels in a quadrant
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

        # Step 3 - only doing it for facedata images right now
        p_x_ytrue = 1;  # p(x|y = true)
        p_x_yfalse = 1;  # p(x|y = false)
        denom1 = float((float(true_count) / float(n)) + 2*smoothing_param)
        denom2 = float((float(false_count) / float(n)) + 2*smoothing_param)
        l_x_array = []
        predictions = []

        for q in range(100):
            pcounter = all_pcounter_vectors[q]
            for r in range(len(pcounter)):
                if pcounter[r] > max_num:
                    t1 = t2 = 0;
                else:
                    t1 = true_data[r][pcounter[r]]
                    t2 = false_data[r][pcounter[r]]
                num1 = float(true_count * t1 + smoothing_param)
                num2 = float(false_count * t2 + smoothing_param)
                temp1 = num1 / denom1
                temp2 = num2 / denom2
                p_x_ytrue = float(p_x_ytrue * temp1)
                p_x_yfalse = float(p_x_yfalse * temp2)
                l_x = float(p_x_ytrue*(float(true_count)/float(n)))/float(p_x_yfalse * (float(false_count) / float(n)))
            l_x_array.append(l_x)
            p_x_ytrue = p_x_yfalse = 1

        for q in range(len(l_x_array)):
            if l_x_array[q] >= 1:
                predictions.append(1)
            else:
                predictions.append(0)

        correct = 0
        for q in range(len(l_x_array)):
            if predictions[q] == valLabels[q]:
                correct = correct + 1
        num_correct.append(correct)

    best_k_ind = 0;
    best_k = 0;
    for b in range(len(num_correct)):
        if num_correct[b] > best_k:
            best_k = num_correct[b]
            best_k_ind = b

    smoothing_param = kgrid[best_k_ind]
    print "smoothing_param is "
    print smoothing_param
    print "best num correct in validation set is "
    print best_k

    print "Testing Phase"
    #calculating likelihood of testing data
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
        # break up face data into 28 15x10 pixel quadrants for feature extraction
        x = 0;
        y = 0;
        feature_quadrants = []  # will be a list of lists
        temp_array = []
        i_start = 0;
        i_end = 6;
        j_start = 0;
        j_end = 7;

        # converts facedata into 28 15x10 pixel quadrants each
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
                j_end = j_end + 7
            else:
                j_start = 0
                j_end = 7
                i_start = i_end
                i_end = i_end + 6
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

    correct = 0
    for q in range(len(l_x_array)):
        if predictions[q] == testLabels[q]:
            correct = correct + 1

    accurate_rate = float(float(correct)/float(test_num))
    print "accuracy rate - faces only"
    print accurate_rate