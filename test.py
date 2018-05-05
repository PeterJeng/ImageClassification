import math

#Naive Bayes on Digit Data with raw pixels as features
img_dim = 28  # images are 28x28 chars
num_labels = 10 # possible labels for each data type
num_img = 10  # num of sample images
num_pixels = img_dim * img_dim

train_data = []

#extract pixel information from training, validation, and testing data
with open("digitdata/trainingimages", "r") as f:
    for x in range(num_img):
        temp = []
        for y in range(img_dim):
            temp2 = []
            for z in range(img_dim):  # include newline
                c = f.read(1)
                if c == '#':
                    temp2.append(1)
                elif c == '+':
                    temp2.append(1)
                else:
                    temp2.append(0)
            temp.append(temp2)
        train_data.append(temp)

train_labels = []
with open("digitdata/traininglabels", "r") as f:
    for x in range(num_img):
        train_labels.append(int(f.readline()))

validation_data = []
with open("digitdata/validationimages", "r") as f:
    for x in range(num_img):
        temp = []
        for y in range(img_dim):
            temp2 = []
            for z in range(img_dim):
                c = f.read(1)
                if c == '#':
                    temp2.append(1)
                elif c == '+':
                    temp2.append(1)
                else:
                    temp2.append(0)
            temp.append(temp2)
        validation_data.append(temp)

validation_labels = []
with open("digitdata/validationlabels", "r") as f:
    for x in range(num_img):
        validation_labels.append(int(f.readline()))

test_data = []
with open("digitdata/testimages", "r") as f:
    for x in range(num_img):
        temp = []
        for y in range(img_dim):
            temp2 = []
            for z in range(img_dim):
                c = f.read(1)
                if c == '#':
                    temp2.append(1)
                elif c == '+':
                    temp2.append(1)
                else:
                    temp2.append(0)
            temp.append(temp2)
        test_data.append(temp)

test_labels = []
with open("digitdata/testlabels", "r") as f:
    for x in range(num_img):
        test_labels.append(int(f.readline()))

#Training Phase - collects counts from the training data
fcount_nonzero = [[0 for i in range(num_pixels)] for i in range(num_labels)]
fcount_zero = [[0 for i in range(num_pixels)] for i in range(num_labels)]
print "Training Phase..."
for i in range(len(train_data)):
    temp = train_data[i]
    pcount = 0
    for x in range(img_dim):
        for y in range(img_dim):
            if temp[x][y] != 0:
                fcount_nonzero[train_labels[i]-1][pcount] += 1
            else:
                fcount_zero[train_labels[i]-1][pcount] += 1
            pcount += 1

marg_prob_true = [0 for i in range(num_labels)]
marg_prob_false = [0 for i in range(num_labels)]
for i in range(num_img):
    marg_prob_true[train_labels[i]] += 1
for i in range(num_img):
    marg_prob_false[i] = num_img - marg_prob_true[i]

#Tuning Phase - finds value of k that yields highest accuracy rate (to be used for testing data)
print "Tuning Phase..."
kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
k = 1
cond_prob_nonzero = [[0 for i in range(num_pixels)] for i in range(num_labels)]
cond_prob_zero = [[0 for i in range(num_pixels)] for i in range(num_labels)]
for x in range(num_labels):
    label_counts_nonzero = fcount_nonzero[x]
    label_counts_zero = fcount_zero[x]
    for y in range(len(fcount_nonzero[0])):
        temp = float(float(label_counts_nonzero[y] + k) / float(marg_prob_true[x] + k))
        cond_prob_nonzero[x][y] = temp
        temp = float(float(label_counts_zero[y] + k) / float(marg_prob_true[x] + k))
        cond_prob_zero[x][y] = temp

#Testing Phase - NEEDS TO BE FIXED
print "Testing Phase..."
#Calculating likelihood of various images
predictions = []
for x in range(num_img):
    fval = [0 for i in range(num_pixels)]
    l_x_array = [0 for i in range(num_labels)]
    temp = test_data[x]
    #extracts if pixel is nonzero or not from test image
    for i in range(len(test_data)):
        pcount = 0
        for g in range(img_dim):
            for y in range(img_dim):
                if temp[g][y] != 0:
                    fval[pcount] += 1
    #calculates likelihood for each label for this test image - need to fix use of logs
    for l in range(num_labels):
        temp_zero = fcount_zero[l]
        temp_nonzero = fcount_nonzero[l]
        mprob_true = marg_prob_true[l]
        mprob_false = marg_prob_false[l]
        l_x_true = 0; #initial likelihood value of it being true for a label
        l_x_false = 0; #initial likelihood value of it being false for a label
        for i in range(len(fval)):
            if fval[i] != 0:
                prob_true = float(temp_nonzero[i] + k) / float(mprob_true + k)
                false_count = 0
                for h in range(num_labels):
                    if h != l:
                        false_count += fcount_nonzero[h][i]
                prob_false = float(false_count + k) / float(mprob_false + k)
                l_x_true = l_x_true + math.log10(prob_true)
                l_x_false = l_x_false + math.log10(prob_false)
            else:
                prob_true = float(temp_zero[i] + k) / float(mprob_true + k)
                false_count = 0
                for h in range(num_labels):
                    if h != l:
                        false_count += fcount_zero[h][i]
                prob_false = float(false_count + k) / float(mprob_false + k)
                l_x_true = l_x_true + math.log10(prob_true)
                l_x_false = l_x_false + math.log10(prob_false)
        l_x = (l_x_true/l_x_false) * (float(mprob_true)/float(mprob_false))
        l_x_array[l] = l_x
    predictions.append(l_x_array.index(max(l_x_array)))

print predictions






