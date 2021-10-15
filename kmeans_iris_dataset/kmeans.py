from random import randint
from math import sqrt
import sys
import operator

class KMeansClusterClassifier:
    def __init__(self, n_clusters): 
        self.root = None
        self.n_clusters = n_clusters
        self.centroids = None
        self.centroid_labels = None

    #Euclidean Distance between 2 data with 4 features
    def euclidean_distance(self, a, b):
        return sqrt(((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2) + ((b[2] - a[2]) ** 2) + ((b[3] - a[3]) ** 2))


    def fit(self, X, y):
        #Shape of X
        n_samples, n_features = len(X), len(X[0])
        #Controls if labels are well chosen
        control = True

        while control:
            #Centroids
            self.centroids = list()
            self.centroid_labels = [0 for _ in range(self.n_clusters)]
            #New Labels for train set after Clustering
            labels = [0 for _ in range(n_samples)]

            #Select Random K Points
            for _ in range(self.n_clusters):
                rand_point = randint(0, n_samples - 1)
                self.centroids.append(X[rand_point])

            #Calculate New Centroid Coordinates
            while True:
                #Clustering - Assign Points to Centroids
                for i in range(n_samples):
                    min_dist = sys.maxsize
                    for j in range(self.n_clusters):
                        dist = self.euclidean_distance(self.centroids[j], X[i])
                        if min_dist > dist:
                            min_dist = dist
                            labels[i] = j
                
                #New Centroid Points
                new_centroids = [[0 for _ in range(n_features)] for _ in range(self.n_clusters)]
                counts = [0 for _ in range(self.n_clusters)]
                for i in range(n_samples):
                    new_centroids[labels[i]] = tuple(map(operator.add, new_centroids[labels[i]], X[i]))
                    counts[labels[i]] += 1

                #Calculate new Centroid Coordinates
                for i in range(self.n_clusters):
                    if counts[i] != 0:
                        new_centroids[i] = [k / counts[i] for k in new_centroids[i]]

                #Compare new and old coordiantes, stop process if they are same
                stop = True
                for i in range(self.n_clusters):
                    if self.centroids[i] != new_centroids[i]:
                        stop = False
                self.centroids = new_centroids
                #Stop The Code Here or Update Centroids
                if stop:
                    #find the labels of centroids then exit
                    hist = [[0 for _ in range(3)] for _ in range(self.n_clusters)]
                    for i in range(n_samples):
                        (hist[labels[i]])[y[i]] += 1

                    for i in range(self.n_clusters):
                        self.centroid_labels[i] = hist[i].index(max(hist[i]))
    
                    #Controls if labels are good. If they are not, selects new initial centroids and run from start
                    if self.n_clusters >= 3 and ((0 not in self.centroid_labels) or (1 not in self.centroid_labels) or (2 not in self.centroid_labels)):
                        control = True
                    else:
                        control = False

                    break

    #predict and return list of class(species)
    def predict(self, X):
        n_samples, n_features = len(X), len(X[0])

        y_pred = [0 for _ in range(n_samples)]
        for i in range(n_samples):
            mindist = sys.maxsize
            for j in range(self.n_clusters):
                dist = self.euclidean_distance(X[i], self.centroids[j])
                if mindist > dist:
                    mindist = dist
                    y_pred[i] = self.centroid_labels[j]

        return y_pred