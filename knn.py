from collections import Counter
import math
import statistics

img_dim = 28 #images are 28x28 chars

num_img = 1000

train_data = list()

for x in range(num_img):
    train_data.append(list())
    train_data[x].append(0)
    train_data[x].append(0)

#    for y in range(img_dim):
 #       train_data[x].append(list())
  #      for z in range(img_dim):
   #         train_data[x][y].append(0)

           

         
with open("C:\\Users\\Aviv\\Downloads\\data\\digitdata\\trainingimages", "r") as f:
    for x in range(num_img):
        for y in range(img_dim):
            for z in range(img_dim): #include newline
                c = f.read(1)
                if c == '#':
                    train_data[x][0] += 1
    #                train_data[x][y][z] = 1
                elif c == '+': 
                    train_data[x][1] += 1
     #               train_data[x][y][z] = 2
 
train_labels = list()  
with open("C:\\Users\\Aviv\\Downloads\\data\\digitdata\\traininglabels", "r") as f:
    for x in range(num_img):
        train_labels.append(int(f.readline()))
    
test_data = list()
for x in range(num_img):
    test_data.append(list())
    test_data[x].append(0)
    test_data[x].append(0)

#    for y in range(img_dim):
 #       test_data[x].append(list())
  #      for z in range(img_dim):
   #         test_data[x][y].append(0)

          
with open("C:\\Users\\Aviv\\Downloads\\data\\digitdata\\testimages", "r") as f:
    for x in range(num_img):
        for y in range(img_dim):
            for z in range(img_dim): 
                c = f.read(1)
                if c == '#':
                    test_data[x][0] += 1
     #               test_data[x][y][z] = 1
                elif c == '+': 
                    test_data[x][1] += 1
      #              test_data[x][y][z] = 2

test_labels = list()                   
with open("C:\\Users\\Aviv\\Downloads\\data\\digitdata\\testlabels", "r") as f:
    for x in range(num_img):
        test_labels.append(int(f.readline()))
        
k = 15

count = 0
for x in range(1000):
    mindists = list()
    nearest_neighbors = list()
    for tx in range(num_img):
       # dist = 0
      #  for y in range(img_dim):
     #       for z in range(img_dim):
     #           dist += math.pow((train_data[tx][y][z] - test_data[x][y][z]), 2)  
    #    dist = math.sqrt(dist)
        dist = math.sqrt(math.pow((train_data[tx][0] - test_data[x][0]), 2) + math.pow((train_data[tx][1] - test_data[x][1]), 2))
        if not mindists:
            mindists.append(dist)
            nearest_neighbors.append(tx)
        else:
            endoflist = True
            for d in mindists: 
                if dist < d: 
                    i = mindists.index(d)
                    mindists.insert(i, dist)
                    nearest_neighbors.insert(i, tx)
                    endoflist = False
                    break
            if endoflist and len(mindists) < k:
                mindists.append(dist)
                nearest_neighbors.append(tx)
            elif len(mindists) > k:
                mindists.pop()
                nearest_neighbors.pop()
    nearest_labels = list()
    for n in nearest_neighbors:
        nearest_labels.append(train_labels[n])
    mode = Counter(nearest_labels).most_common(1)
    #print(nearest_labels)
    print("Predicted number for image %d: %d"%(x, mode[0][0]))
    print("Actual number: %d"%test_labels[x])
    
    if (mode[0][0] == test_labels[x]):
        count += 1
        
print("Accuracy: %f"%(count/num_img))