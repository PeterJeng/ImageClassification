from collections import Counter
import math
import copy
import operator

img_width = 28 #images are 28x28 chars
img_height = 28

num_img = 100

def detectHoles(data):
    #mark walls as -1 and close in until only holes left
    new_data = copy.deepcopy(data)
  #  print(new_data)
    for x in range(img_height):
        for y in range(img_width):
            if (new_data[x][y] == 0 and (x == 0 or y == 0 or x == img_height - 1 or y == img_height - 1)):
                new_data[x][y] = -1
   # print(new_data)
    emptyspace = True
    while emptyspace:
        emptyspace = False
        for x in range(img_height):
            for y in range(img_width):
                if (new_data[x][y] == 0 and (new_data[x][y - 1] == -1 or new_data[x - 1][y] == -1 or new_data[x + 1][y] == -1 or new_data[x][y + 1] == -1)):
                    new_data[x][y] = -1
                    emptyspace = True
                    
    #print(new_data)
    foundholes = True
    countholes = 0
    while foundholes:
        
        foundholes = False
        for a in range(img_height):
            for b in range(img_width):
                if (new_data[a][b] == 0):
                    foundholes = True
                    countholes += 1   
 #                   print("HELLO", a, b)
                    new_data[a][b] = -1
                    emptyspace = True
                    while emptyspace:
                        emptyspace = False
                        for y in range(img_height):
                            for x in range(img_width):
                                if new_data[x][y] == 0 and (new_data[x][y - 1] == -1 or new_data[x - 1][y] == -1 or new_data[x + 1][y] == -1 or new_data[x][y + 1] == -1):
                                    new_data[x][y] = -1
                                    emptyspace = True
        #            print(new_data)
    return countholes

def collect(l, index):
    return map(operator.itemgetter(index), l)

def loadData(path):
    buf = list()
    for x in range(num_img):
        buf.append(list())
        
        for y in range(img_height):
            buf[x].append(list())
            for z in range(img_width):
                buf[x][y].append(0)
                
        for y in range(4):
            buf[x].append(0)
            
    with open(path, "r") as f:
        for x in range(num_img):
            for row in range(img_height):
                for col in range(img_width): 
                    c = f.read(1)
                    if c == '#':    
             #           buf[x][img_height] += 1
                        buf[x][row][col] = 2
                    elif c == '+': 
             #           buf[x][img_height + 1] += 1
                        buf[x][row][col] = 1
                c = f.read(1) #newline
        #    buf[x][img_height + 2] = img_height*img_width - buf[x][img_height] - buf[x][img_height + 1] # get number of empty pixels
            buf[x][img_height + 3] = detectHoles(buf[x])
      #      print(x, detectHoles(buf[x]))

    return buf
    
def loadLabels(path):
    buf = list()  
    with open(path, "r") as f:
        for x in range(num_img):
            buf.append(int(f.readline()))
    return buf


    
                        
print("Loading Data")

train_data = loadData("digitdata/trainingimages")   
train_labels = loadLabels("digitdata/traininglabels")
 
test_data = loadData("digitdata/testimages")
test_labels = loadLabels("digitdata/testlabels")

#training
print("Beginning Training Phase")
training_distance_lists = list() #each img has a list of its distances to all other images

for x in range(num_img):
    print("Making distance list for img %d"%x)
    training_distance_lists.append(list())
    for y in range(num_img):
        if (train_labels[x] != train_labels[y]):
            dist = 0
       #     for row in range(img_height):
        #        for col in range(img_width):
         #           dist += math.pow((train_data[x][row][col] - train_data[y][row][col]), 2) 
            for z in range(4):       
                dist += math.pow((train_data[x][img_height + z] - train_data[y][img_height + z]), 2) 
            dist = math.sqrt(dist)
            training_distance_lists[x].append((dist, y))
            
    training_distance_lists[x].sort()

ORDER = list()
for x in range(num_img):
    nearest_neighbor = training_distance_lists[x][0][1]
    nearest_neighbor_dist = training_distance_lists[x][0][0]
    rank_to_neighbor = collect(training_distance_lists[nearest_neighbor], 1).index(x)
    MNV = rank_to_neighbor + 2
    ORDER.append((MNV, nearest_neighbor_dist, x))


ORDER.sort()
STORE = list()
added_to_STORE = True

print("Condensing training data...")

while added_to_STORE:
    added_to_STORE = False
    for sample in ORDER:
        if not STORE:
            STORE.append(ORDER[0])
        else:
            nearest_neighbor_rank = -1
            nearest_neighbor = None
            for try_sample in STORE:
                if train_labels[sample[2]] != train_labels[try_sample[2]]: 
                    try_sample_rank = collect(training_distance_lists[sample[2]], 1).index(try_sample[2])
                    if try_sample_rank < nearest_neighbor_rank or nearest_neighbor_rank == -1:
                        nearest_neighbor_rank = try_sample_rank
                        nearest_neighbor = try_sample
            print(sample[2], train_labels[sample[2]], nearest_neighbor[2], train_labels[nearest_neighbor[2]])
            if train_labels[sample[2]] != train_labels[nearest_neighbor[2]]: #check if nearest neighbor classification is correct, if not, add to STORE
                if sample not in STORE:
                    STORE.append(sample)
                    added_to_STORE = True
print(len(STORE))
          
def classify_ORDER(ORDER, STORE):
    outcomes = list()
    for sample in ORDER:
        nearest_neighbor_rank = -1
        nearest_neighbor = None
        for try_sample in STORE:
            if train_labels[sample[2]] != train_labels[try_sample[2]]: 
                try_sample_rank = collect(training_distance_lists[sample[2]], 1).index(try_sample[2])
                if try_sample_rank < nearest_neighbor_rank or nearest_neighbor_rank == -1:
                    nearest_neighbor_rank = try_sample_rank
                    nearest_neighbor = try_sample
                if train_labels[sample[2]] == train_labels[nearest_neighbor[2]]: #check if nearest neighbor classification is correct, if not, add to STORE
                    outcomes.append(True)
                else:
                    outcomes.append(False)
    return outcomes

none_removed = classify_ORDER(ORDER, STORE)
for s in STORE: #further condense samples
    temp = s
    STORE.remove(s)
    if (classify_ORDER(ORDER, STORE) != none_removed):
        STORE.append(temp)

print(len(STORE))        
#testing phase
print("Beginning Testing Phase")

k = 15

count = 0
for x in range(num_img):
    dists = list()
    nearest_neighbors = list()
    for training_sample in STORE:
        dist = 0
   #     for row in range(img_height):
    #        for col in range(img_width):
     #           dist += math.pow((train_data[training_sample[2]][row][col] - test_data[x][row][col]), 2)  
        for y in range(4):       
            dist += math.pow((train_data[x][img_height + y] - train_data[training_sample[2]][img_height + y]), 2) 
        dist = math.sqrt(dist)
        dists.append((dist, training_sample[2]))
        dists.sort()
        
    nearest_labels = list()
    for n in dists[:k]:
        nearest_labels.append(train_labels[n[1]])
    mode = Counter(nearest_labels).most_common(1)
    print("Predicted number for image %d: %d"%(x, mode[0][0]))
    print("Actual number: %d"%test_labels[x])
    
    if (mode[0][0] == test_labels[x]):
        count += 1

accuracy = float(count)/num_img
print("Accuracy: %f"%accuracy)
        
        