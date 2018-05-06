import math
import samples
import util

#accuracy rates for digit data - working on rn
if __name__ == '__main__':
    print "Training Phase"
    #stores training data and appropriate labels for faces
    n = 50;
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

    #stores how many times a quadrant was 0 or not across all samples for each label
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
    num_correct = []

    for b in range(len(kgrid)):
        all_feature_vectors = []
        smoothing_param = kgrid[b]
        # Step 1
        for k in range(100):
            # break up face data into 16 7x7 pixel quadrants for feature extraction
            feature_quadrants = []  # will be a list of lists
            temp_array = []
            i_start = 0;
            i_end = 7;
            j_start = 0;
            j_end = 7;

            # converts facedata into 28 15x10 pixel quadrants each
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


    #print "Testing Phase"
    #calculating likelihood of testing data
    #steps to take
    #1. separate testing data into quadrants
    #2. count number of non-zero pixels in each quadrant
    #3. calculate the likelihood based on results and stored log-joint probabilities
    #test_num = 100
    #testData = samples.loadDataFile("facedata/facedatatest",test_num,28,28)
    #testLabels = samples.loadLabelsFile("facedata/facedatatestlabels",test_num)

    #all_feature_vectors = []  # stores all 42 quadrants of all test images

    #Step 1
    #for k in range(test_num):
        # break up face data into 28 15x10 pixel quadrants for feature extraction
        #x = 0;
        #y = 0;
        #feature_quadrants = []  # will be a list of lists
        #temp_array = []
        #i_start = 0;
        #i_end = 7;
        #j_start = 0;
        #j_end = 7;

        # converts facedata into 28 15x10 pixel quadrants each
        #while i_end <= 28 and j_end <= 28:
            # parse through image and store pixels in a temporary array
            #for i in range(i_start, i_end):
                #for j in range(j_start, j_end):
                    #temp_array.append(testData[k].getPixel(i, j))

            # add temp_array to feature_quadrant array and reassign temp_array
            #feature_quadrants.append(temp_array)
            #temp_array = []

            # update iterators for parsing through image
            #if j_end != 28:
                #j_start = j_end
                #j_end = j_end + 7
            #else:
                #j_start = 0
                #j_end = 7
                #i_start = i_end
                #i_end = i_end + 7
        #all_feature_vectors.append(feature_quadrants)

    #Step 2
    #for k in range(len(all_feature_vectors)):
        #pix_counter = 0;  #keeps track of non-zero pixels in a quadrant
        #pcounter_array = []
        #feature_quadrants = all_feature_vectors[k]
        #for x in range(len(feature_quadrants)):
            #temp = feature_quadrants[x];
            #for y in range(len(temp)):
                #if temp[y] != 0:
                    #pix_counter = pix_counter + 1
            #pcounter_array.append(pix_counter)
            #pix_counter = 0
        #all_pcounter_vectors.append(pcounter_array)

    #Step 3 - only doing it for facedata images right now
    #p_x_ytrue = 1; # p(x|y = true)
    #p_x_yfalse = 1; # p(x|y = false)
    #denom1 = float((float(true_count)/float(n)) + smoothing_param)
    #denom2 = float((float(false_count)/float(n)) + smoothing_param)
    #l_x_array = []
    #predictions = []

    #for q in range(test_num):
        #pcounter = all_pcounter_vectors[q]
        #for r in range(len(pcounter)):
            #if pcounter[r] > max_num:
                #t1 = t2 = 0;
            #else:
                #t1 = true_data[r][pcounter[r]]
                #t2 = false_data[r][pcounter[r]]
            #num1 = float(true_count * t1 + smoothing_param)
            #num2 = float(false_count * t2 + smoothing_param)
            #temp1 = num1/denom1
            #temp2 = num2/denom2
            #p_x_ytrue = float(p_x_ytrue * temp1)
            #p_x_yfalse = float(p_x_yfalse* temp2)
            #l_x = float(p_x_ytrue*(float(true_count)/float(n)))/float(p_x_yfalse*(float(false_count)/float(n)))
        #l_x_array.append(l_x)
        #p_x_ytrue = p_x_yfalse = 1

    #for q in range(len(l_x_array)):
        #if l_x_array[q] >= 1:
            #predictions.append(1)
        #else:
            #predictions.append(0)

    #correct = 0
    #for q in range(len(l_x_array)):
        #if predictions[q] == testLabels[q]:
            #correct = correct + 1

    #accurate_rate = float(float(correct)/float(test_num))
    #print "accuracy rate - faces only"
    #print accurate_rate






