import math
import samples
import util

#Naive Bayes - Digit Data
if __name__ == '__main__':
    print "Training Phase"
    #stores training data and appropriate labels for faces
    n = 5000;
    items = samples.loadDataFile("digitdata/trainingimages",n,28,28)
    labels = samples.loadLabelsFile("digitdata/traininglabels",n)
    all_feature_vectors = [] #stores all quadrants of all sample images

    for k in range(n):
        #break up face data into 16 7x7 pixel quadrants for feature extraction
        feature_quadrants = [] #will be a list of lists
        temp_array = []
        i_start = 0;  i_end = 7;
        j_start = 0;  j_end = 7;

        #converts facedata into 16 7x7 pixel quadrants each
        while i_end <= 28 and j_end <= 28:
            #parse through image and store pixels in a temporary array
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    temp_array.append(items[k].getPixel(i,j))

            #add temp_array to feature_quadrant array and reassign temp_array
            feature_quadrants.append(temp_array)
            temp_array = []

            #update iterators for parsing through image
            if j_end != 28:
                j_start = j_end
                j_end = j_end + 7
            else:
                j_start = 0
                j_end = 7
                i_start = i_end
                i_end = i_end + 7
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

    #calculating and storing prior probabilities of all labels
    marg_prob = [0 for i in range(10)]
    for i in range(n):
        marg_prob[labels[i]] += 1

    #stores how many times a quadrant was empty or not across all samples for each label
    zero_data = [[0 for i in range(16)] for j in range(10)]
    nonzero_data = [[0 for i in range(16)] for j in range(10)]
    label_data = util.Counter()
    for i in range(len(all_pcounter_vectors)):
        temp = all_pcounter_vectors[i]
        for j in range(16):
            if temp[j] > 0:
                nonzero_data[labels[i]][j] += 1
            else:
                zero_data[labels[i]][j] += 1

    print "Tuning phase, using validation data to fine tune the smoothing parameter"
    kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    valData = samples.loadDataFile("digitdata/validationimages", 100, 28, 28)
    valLabels = samples.loadLabelsFile("digitdata/validationlabels", 100)
    all_feature_vectors = []  # stores all 42 quadrants of all validation images
    all_pcounter_vectors = []
    num_correct = []

    for b in range(len(kgrid)):
        all_feature_vectors = []
        all_pcounter_vectors = []
        smoothing_param = kgrid[b]
        # Step 1
        for k in range(100):
            # break up digit data into 16 7x7 pixel quadrants for feature extraction
            feature_quadrants = []  # will be a list of lists
            temp_array = []
            i_start = 0;
            i_end = 7;
            j_start = 0;
            j_end = 7;

            while i_end <= 28 and j_end <= 28:
                # parse through image and store pixels in a temporary array
                for i in range(i_start, i_end):
                    for j in range(j_start, j_end):
                        temp_array.append(valData[k].getPixel(i, j))

                # add temp_array to feature_quadrant array and reassign temp_array
                feature_quadrants.append(temp_array)
                temp_array = []

                # update iterators for parsing through image
                if j_end != 28:
                    j_start = j_end
                    j_end = j_end + 7
                else:
                    j_start = 0
                    j_end = 7
                    i_start = i_end
                    i_end = i_end + 7
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

        # Step 3 - make predictions based on validation data
        predictions = []

        for i in range(len(all_pcounter_vectors)):
            l_x_array = []
            temp = all_pcounter_vectors[i]  # num pixels in each quadrant info for each image
            for i in range(10):
                mprob = float(marg_prob[i]) / float(n)
                l_x_array.append(mprob)

            for j in range(16):
                if temp[j] == 0:
                    for m in range(len(l_x_array)):
                        l_x_array[m] = l_x_array[m] * float(float(zero_data[m][j] + smoothing_param) / float(marg_prob[m] + 2*smoothing_param))
                else:
                    for m in range(len(l_x_array)):
                        l_x_array[m] = l_x_array[m] * float(float(nonzero_data[m][j] + smoothing_param) / float(marg_prob[m] + 2*smoothing_param))

            predictions.append(l_x_array.index(max(l_x_array)))
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == valLabels[i]:
                correct += 1
        num_correct.append(correct)

    best_k = num_correct.index(max(num_correct))
    k = kgrid[best_k]
    print "Best K value is "
    print k
    print "Validation set accuracy rate is: "
    print float(max(num_correct)) / float(100)

    print "Testing Phase"
    test_num = 100
    testData = samples.loadDataFile("digitdata/testimages",test_num,28,28)
    testLabels = samples.loadLabelsFile("digitdata/testlabels",test_num)
    all_feature_vectors = []  # stores all 16 quadrants of all test images
    all_pcounter_vectors = []

    #Step 1
    for g in range(test_num):
        # break up face data into 16 7x7 pixel quadrants for feature extraction
        x = 0;
        y = 0;
        feature_quadrants = []  # will be a list of lists
        temp_array = []
        i_start = 0;
        i_end = 7;
        j_start = 0;
        j_end = 7;

        # converts digitdata into 16 7x7 pixel quadrants each
        while i_end <= 28 and j_end <= 28:
            # parse through image and store pixels in a temporary array
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    temp_array.append(testData[g].getPixel(i, j))

            # add temp_array to feature_quadrant array and reassign temp_array
            feature_quadrants.append(temp_array)
            temp_array = []

            # update iterators for parsing through image
            if j_end != 28:
                j_start = j_end
                j_end = j_end + 7
            else:
                j_start = 0
                j_end = 7
                i_start = i_end
                i_end = i_end + 7
        all_feature_vectors.append(feature_quadrants)

    #Step 2
    for g in range(len(all_feature_vectors)):
        pix_counter = 0;  #keeps track of non-zero pixels in a quadrant
        pcounter_array = []
        feature_quadrants = all_feature_vectors[g]
        for x in range(len(feature_quadrants)):
            temp = feature_quadrants[x];
            for y in range(len(temp)):
                if temp[y] != 0:
                    pix_counter = pix_counter + 1
            pcounter_array.append(pix_counter)
            pix_counter = 0
        all_pcounter_vectors.append(pcounter_array)

    #Step 3
    predictions = []

    for i in range(len(all_pcounter_vectors)):
        l_x_array = []
        temp = all_pcounter_vectors[i]  # num pixels in each quadrant info for each image
        for i in range(10):
            mprob = float(marg_prob[i]) / float(n)
            l_x_array.append(mprob)

        for j in range(16):
            if temp[j] == 0:
                for m in range(len(l_x_array)):
                    l_x_array[m] = l_x_array[m] * float(float(zero_data[m][j] + k) / float(marg_prob[m] + 2 * k))
            else:
                for m in range(len(l_x_array)):
                    l_x_array[m] = l_x_array[m] * float(float(nonzero_data[m][j] + k) / float(marg_prob[m] + 2 * k))
        predictions.append(l_x_array.index(max(l_x_array)))

    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == testLabels[i]:
            correct += 1

    accurate_rate = float(float(correct) / float(test_num))
    print "Testing Set accuracy rate is: "
    print accurate_rate






