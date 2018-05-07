# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Perceptron implementation
import util
import time
PRINT = True

class PerceptronClassifier:
  """
  Perceptron classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "perceptron"
    self.max_iterations = max_iterations
    self.weights = {}
    for label in legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use

  def setWeights(self, weights):
    assert len(weights) == len(self.legalLabels);
    self.weights == weights;

    """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details. 
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """

  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    subTrainingData = []
    subTrainingLabels = []

    n = 0
    for index in range(45):
        subTrainingData.append(trainingData[n + index])
        subTrainingLabels.append(trainingLabels[n + index])


    self.features = trainingData[0].keys() # could be useful later
    # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
    # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

    startTime = time.time()
    for iteration in range(self.max_iterations):
      print "Starting iteration ", iteration, "..."

      for i in range(len(subTrainingData)):
          # uses every pixel in the image of the training data as feature
          # Face: 60 x 70, Digit: 28 x 28
          scores = util.Counter()
          for label in self.legalLabels:
            # calculate the dot product of the two vectors
            scores[label] = subTrainingData[i] * self.weights[label]

          #get the label with the highest score
          highestScore = scores.argMax()

          actual = subTrainingLabels[i]

          #Increase the weights of the correct label with the value of currentFeature
          #Decrease the weights of the incorrect label with the value of currentFeature
          if highestScore != actual:
            self.weights[actual] += subTrainingData[i]
            self.weights[highestScore] -= subTrainingData[i]

    elapsedTime = time.time() - startTime
    print elapsedTime

  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  
  def findHighWeightFeatures(self, label):
    """
    Returns a list of the 100 features with the greatest weight for some label
    """
    featuresWeights = []

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresWeights

