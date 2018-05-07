import math
import samples

#Naive Bayes - Face Data
if __name__ == '__main__':
    print "Training Phase"
    #stores training data and appropriate labels for faces
    n = 450;
    items = samples.loadDataFile("facedata/facedatatrain",n,60,70)
    labels = samples.loadLabelsFile("facedata/facedatatrainlabels",n)
    all_feature_vectors = [] #stores all quadrants of all sample images

    for k in range(n):
        #break up face data into 100 6x7 pixel quadrants for feature extraction
        feature_quadrants = [] #will be a list of lists
        temp_array = []
        i_start = 0;  i_end = 6;
        j_start = 0;  j_end = 7;

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

    #stores how many times a quadrant had 0 pixels or not for each label across all training data images
    zero_data = [[0 for i in range(100)] for j in range(2)]
    nonzero_data = [[0 for i in range(100)] for j in range(2)]

    for i in range(len(all_pcounter_vectors)):
        temp = all_pcounter_vectors[i]
        for j in range(100):
            if temp[j] > 0:
                nonzero_data[labels[i]][j] += 1
            else:
                zero_data[labels[i]][j] += 1

    print "Tuning phase, using validation data to fine tune the smoothing parameter"
    kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    valData = samples.loadDataFile("facedata/facedatavalidation", 100, 60, 70)
    valLabels = samples.loadLabelsFile("facedata/facedatavalidationlabels", 100)
    all_feature_vectors = []  # stores all quadrants of all test images
    all_pcounter_vectors = []
    num_correct = []

    for b in range(len(kgrid)):
        all_feature_vectors = []
        all_pcounter_vectors = []
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

        # Step 3 - make predictions
        predictions = []

        for i in range(len(all_pcounter_vectors)):
            #reset l_x_array for each test image
            l_x_array = []
            temp = all_pcounter_vectors[i]  # num pixels in each quadrant info for each image
            mprob_false = float(false_count) / float(n)
            l_x_array.append(mprob_false)
            mprob_true = float(true_count) / float(n)
            l_x_array.append(mprob_true)

            for j in range(100):
                if temp[j] == 0:
                    l_x_array[0] = l_x_array[0] * float(float(zero_data[0][j] + smoothing_param) / float(false_count + 2 * smoothing_param))
                    l_x_array[1] = l_x_array[1] * float(float(zero_data[1][j] + smoothing_param) / float(true_count + 2 * smoothing_param))
                else:
                    l_x_array[0] = l_x_array[0] * float(float(nonzero_data[0][j] + smoothing_param) / float(false_count + 2 * smoothing_param))
                    l_x_array[1] = l_x_array[1] * float(float(nonzero_data[1][j] + smoothing_param) / float(true_count + 2 * smoothing_param))

            predictions.append(l_x_array.index(max(l_x_array)))

        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == valLabels[i]:
                correct += 1
        num_correct.append(correct)

    best_k_ind = num_correct.index(max(num_correct))
    k = kgrid[best_k_ind]
    print "Best K value is "
    print k
    print "Validation set accuracy rate is: "
    print float(max(num_correct)) / float(100)

    print "Testing Phase"
    #calculating likelihood of testing data
    #steps to take
    #1. separate testing data into quadrants
    #2. count number of non-zero pixels in each quadrant
    #3. calculate the likelihood based on results and stored log-joint probabilities
    test_num = 100
    testData = samples.loadDataFile("facedata/facedatatest",test_num,60,70)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels",test_num)

    all_feature_vectors = []  # stores all quadrants of all test images
    all_pcounter_vectors = []

    #Step 1
    for k in range(test_num):
        # break up face data into 100 6x7 pixel quadrants for feature extraction
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
    predictions = []

    for i in range(len(all_pcounter_vectors)):
        # reset l_x_array for each test image
        l_x_array = []
        temp = all_pcounter_vectors[i]  # num pixels in each quadrant info for each image
        mprob_false = float(false_count) / float(n)
        l_x_array.append(mprob_false)
        mprob_true = float(true_count) / float(n)
        l_x_array.append(mprob_true)

        for j in range(100):
            if temp[j] == 0:
                l_x_array[0] = l_x_array[0] * float(float(zero_data[0][j] + k) / float(false_count + 2 * k))
                l_x_array[1] = l_x_array[1] * float(float(zero_data[1][j] + k) / float(true_count + 2 * k))
            else:
                l_x_array[0] = l_x_array[0] * float(float(nonzero_data[0][j] + k) / float(false_count + 2 * k))
                l_x_array[1] = l_x_array[1] * float(float(nonzero_data[1][j] + k) / float(true_count + 2 * k))

        predictions.append(l_x_array.index(max(l_x_array)))

    correct = 0
    for q in range(len(predictions)):
        if predictions[q] == testLabels[q]:
            correct += 1

    accurate_rate = float(float(correct)/float(test_num))
    print "Testing Set accuracy rate is: "
    print accurate_rate