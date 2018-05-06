import util
import math
import samples

img_height = 70
img_width = 60

#returns number of holes in am image
def detectHoles(data):
    # mark walls as -1 and close in until only holes left
    new_data = list(data)
    emptyspace = False
    for x in range(img_height):
        for y in range(img_width):
            if new_data[x][y] == 0 and (x == 0 or y == 0 or x == img_height - 1 or y == img_height - 1):
                new_data[x][y] = -1
                emptyspace = True
    for x in range(img_height):
        for y in range(img_width):
            if new_data[x][y] == 0 and (
                    new_data[x + 1][y] == -1 or new_data[x][y + 1] == -1 or new_data[x][y - 1] == -1 or new_data[x - 1][
                y] == -1):
                new_data[x][y] = -1

    foundholes = True
    countholes = 0
    while foundholes:
        #  print(new_data)
        foundholes = False
        for a in range(img_height):
            for b in range(img_width):
                if (new_data[a][b] == 0):
                    foundholes = True
                    countholes += 1
                    new_data[a][b] = -1
                    for y in range(img_height):
                        for x in range(img_width):
                            if new_data[x][y] == 0 and (
                                    new_data[x + 1][y] == -1 or new_data[x][y + 1] == -1 or new_data[x][y - 1] == -1 or
                                    new_data[x - 1][y] == -1):
                                new_data[x][y] = -1
    return countholes


#accuracy rates for face data
if __name__ == '__main__':
    print "Training Phase"
    #stores training data and appropriate labels for faces
    n = 1;
    items = samples.loadDataFile("facedata/facedatatrain",n,60,70)
    labels = samples.loadLabelsFile("facedata/facedatatrainlabels",n)
    all_feature_vectors = [] #stores all 42 quadrants of all sample images

    for k in range(n):
        #break up face data into 100 6x7 pixel quadrants for feature extraction
        numHoles = 0 #will be a list of lists
        temp_array = []

        #converts counter data provided by loadDataFile into a 60x70 matrix
        for i in range(70):
            row_array = []
            for j in range(60):
                row_array.append(items[k].getPixel(j,i))
            temp_array.append(row_array)

        #determines numHoles in the given data
        numHoles = detectHoles(temp_array)
        all_feature_vectors.append(numHoles)

    #calculating and storing log joint probabilities
    true_count = 0;
    for i in range(n):
        if labels[i] == 1:
            true_count = true_count + 1;
    false_count = n - true_count


    #need to redo to make it work with numHoles
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