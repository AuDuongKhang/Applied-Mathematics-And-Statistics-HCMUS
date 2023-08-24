import numpy as np
from PIL import Image
from math import sqrt


def init_centroid(image_1d, K, type):
    if type == 'random':
        return np.random.randint(0,256, size = (K,image_1d.shape[1]))
    elif type == 'in_pixel':
        return image_1d[np.random.choice(image_1d.shape[0], K, replace=False)]

def compute_euclidean_distance(point, centroid):
    point = np.array(point, dtype= float)
    centroid = np.array(centroid, dtype= float)
    sum = 0
    for index in range (0, len(point)):
        sum += (point[index] - centroid[index]) * (point[index] - centroid[index])
        
    return sqrt (sum)     
        
def assign_label(image_1d, centroid):
    k = len(centroid)
    dist = np.zeros((image_1d.shape[0], k))
    for index_point in range(0,image_1d.shape[0]):
        for index_centroid in range(0, k):
            distance = compute_euclidean_distance(image_1d[index_point], centroid[index_centroid])
            dist[index_point][index_centroid] = distance   
        
    return np.argmin(dist, axis=1)
            
def update_centroid(image_1d, K, centroid):
    label_temp = assign_label(image_1d,centroid)
    centroid_temp = np.zeros((K, image_1d.shape[1]))
    for k in range(K): 
        temp = image_1d[label_temp == k]
        if (temp.shape[0] > 0):
            centroid_temp[k] = np.mean(temp, axis = 0)
            
    return centroid_temp
    
def has_converged(centroid, new_centroid):
    return np.array_equal(centroid, new_centroid)

def set_pixel(image_1d, label, centroid):
    image_1d = np.zeros((image_1d.shape[0], image_1d.shape[1]))
    for index, item in enumerate(label):
        image_1d[index][0] += int(round(centroid[item][0]))
        image_1d[index][1] += int(round(centroid[item][1]))
        image_1d[index][2] += int(round(centroid[item][2]))
        
    return image_1d

def kmeans(image_1d, k_clusters, max_iter, init_centroids = 'random'):
    width = image_1d.shape[0]
    height = image_1d.shape[1]
    num_chanel = image_1d.shape[2]
    
    image_1d = image_1d.reshape(height * width, num_chanel)
    
    centroids =[init_centroid(image_1d,k_clusters,init_centroids)]
    label = []

    for _ in range (0,max_iter):
        new_centroid = update_centroid(image_1d, k_clusters, centroids[-1])
        if (has_converged(centroids[-1],new_centroid)):
            centroids.append(new_centroid)
            label.append(assign_label(image_1d,new_centroid))
            break
        centroids.append(new_centroid)

    image_1d = set_pixel(image_1d, label[-1], centroids[-1])
    image_1d = image_1d.reshape(width,height,num_chanel)
    return centroids, label,image_1d

if __name__ == '__main__':
    centroids = []
    labels = []
    print("Input name of image (with the extension, e.x: 1.jpg):")
    image_1d =np.array(Image.open(input()))
    image_1d.setflags(write=1)
    K= 7
    max_tier = 100
    centroids, labels,image_1d = kmeans(image_1d,K, max_tier)
    
    
    print("Centroid found by algorithm: ")
    print(centroids[-1])
    print("Label found by algorithm: ")
    print(labels[-1])
    
    
    print('Input type of output (png or pdf): ')
    type = input()
    if type == 'png':
        img = Image.fromarray(image_1d.astype(np.uint8))
        img.save('output1.png')
    elif type == 'pdf':
        img = Image.fromarray(image_1d.astype(np.uint8)).convert('RGB')
        img.save('output2.pdf', format='PDF')
    
   

    