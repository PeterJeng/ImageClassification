import math

#Naive Bayes on Digit Data with raw pixels as features
img_dim_col = 60
img_dim_row = 70
num_labels = 2 # possible labels for each data type
num_img = 10  # num of sample images
num_pixels = img_dim_row * img_dim_col

train_data = []

#extract pixel information from training, validation, and testing data
with open("digitdata/trainingimages", "r") as f:
    for x in range(num_img):
        temp = []
        for y in range(img_dim_row):
            temp2 = []
            for z in range(img_dim_col):  # include newline
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
        for y in range(img_dim_row):
            temp2 = []
            for z in range(img_dim_col):
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
        for y in range(img_dim_row):
            temp2 = []
            for z in range(img_dim_col):
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