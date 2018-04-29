import util
import classificationMethod
import math
import samples

#currently working on improving accuracy for face data, still have to do it for digit data
if __name__ == '__main__':
    print "Training Phase"
    #stores training data and appropriate labels for faces
    n = 100
    items = samples.loadDataFile("facedata/facedatatrain",n,60,70)
    labels = samples.loadLabelsFile("facedata/facedatatrainlabels",n)

    all_features_vector = []
    #feature extraction - trial 1 - number of zero and non-zero pixels
    for i in range(n):
        pix_count = 0
        for x in range(70):
            for y in range(60):
                if items[i].getPixel(y,x) != 0:
                    pix_count = pix_count + 1
        all_features_vector.append(pix_count)

    true_count = 0
    for i in range(n):
        if labels[i] == 1:
            true_count = true_count + 1
    false_count = n - true_count

    p_true = float(float(true_count)/float(n))
    p_false = float(float(false_count)/float(n))

    max_num = max(all_features_vector)
    ranges = round(max_num,-2)
    while ranges > 10:
        ranges = ranges/float(10) #in order to have ranges of 0-100 pixels, 100-200 pixels etc.
    true_data = [0 for i in range(int(ranges) + 1)]
    false_data = [0 for i in range(int(ranges) + 1)]

    for i in range(n):
        feature = all_features_vector[i]
        while feature > 10:
            feature = feature/float(10)

        #to find range of feature
        ind = int(ranges) + 1
        while ind > feature:
            ind = ind - 1

        if labels[i] == 1:
            true_data[ind] = true_data[ind] + 1
        else:
            false_data[ind] = false_data[ind] + 1

    #Testing phase, with smoothing parameter = 1
    k = 1 #smoothing parameter
    items = samples.loadDataFile("facedata/facedatatest", n, 60, 70)
    labels = samples.loadLabelsFile("facedata/facedatatestlabels", n)

    all_features_vector = []
    for i in range(n):
        pix_count = 0
        for x in range(70):
            for y in range(60):
                if items[i].getPixel(y,x) != 0:
                    pix_count = pix_count + 1
        all_features_vector.append(pix_count)

    p_x_ytrue = 1;  # p(x|y = true)
    p_x_yfalse = 1;  # p(x|y = false)
    denom1 = float((float(true_count) / float(n)) + k)
    denom2 = float((float(false_count) / float(n)) + k)
    l_x_array = []
    predictions = []

    for q in range(n):
        pcounter = all_features_vector[q]
        while pcounter > 10:
            pcounter = pcounter/float(10)
        ind = int(ranges) + 1
        while ind > feature:
            ind = ind - 1
        num1 = float(true_data[ind]*true_count + k)
        num2 = float(false_data[ind]*false_count + k)
        temp1 = num1/denom1
        temp2 = num2/denom2
        p_x_ytrue = float(p_x_ytrue)*temp1
        p_x_yfalse = float(p_x_yfalse)*temp2
        l_x = float(p_true*p_x_ytrue)/float(p_false*p_x_yfalse)
        l_x_array.append(l_x)
        p_x_ytrue = p_x_yfalse = 1

    for q in range(len(l_x_array)):
        if l_x_array[q] >= 1:
            predictions.append(1)
        else:
            predictions.append(0)

    correct = 0
    for q in range(len(l_x_array)):
        if predictions[q] == labels[q]:
            correct = correct + 1

    accurate_rate = float(float(correct) / float(n))
    print "accuracy rate - faces only"
    print accurate_rate