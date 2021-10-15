class Node:
    def __init__(
        self, feature=None, threshold=None, gini=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.gini = gini
        self.left = left
        self.right = right
        self.value = value

    def isLeaf(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, max_depth): 
        self.root = None
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self.buildTree(X, y)
        
    def buildTree(self, X, y, depth=0):
        labels = len(set(y))

        bestGini, bestFeature, bestThreshold = self.findBestGini(X, y)

        if bestFeature == None or labels == 1 or bestGini == 0 or depth > self.max_depth:
            leaf_value = self.decideClass(y)
            return Node(value=leaf_value, gini=bestGini)

        XColumn = [k[bestFeature] for k in X]
        
        #grow tree
        leftSplit, rightSplit = self.split(XColumn, bestThreshold)

        leftArr = [X[k] for k in leftSplit]
        rightArr = [X[k] for k in rightSplit]

        leftLabels = [y[k] for k in leftSplit]
        rightLabels = [y[k] for k in rightSplit]

        left = self.buildTree(leftArr, leftLabels, depth + 1)
        right = self.buildTree(rightArr, rightLabels, depth + 1)

        return Node(bestFeature, bestThreshold, bestGini, left, right)

    def findBestGini(self, X, y):
        #sample and feature sizes
        n_samples, n_features = len(X), len(X[0])

        #best values
        bestGini = 5
        bestFeature = None #0 sepal length - 1 sepal width - 2 petal length - 3 petal width
        bestThreshold = None

        #iterate features column by column
        for i in range(n_features):
            XColumn = [k[i] for k in X]
            thresholds = set(XColumn)

            for threshold in thresholds:
                lowerArr, greaterArr = self.split(XColumn, threshold)
                gini = self.calculateGini(lowerArr, greaterArr, y)

                if gini < bestGini:
                    bestGini = gini
                    bestFeature = i
                    bestThreshold = threshold

        return bestGini, bestFeature, bestThreshold

    #splits the given feature list into two lists with feature index(not value)
    def split(self, XColumn, threshold):
        lowerArr = list()
        greaterArr = list()

        for i in range(len(XColumn)):
            if XColumn[i] > threshold:
                greaterArr.append(i)
            else:
                lowerArr.append(i)

        return lowerArr, greaterArr

    #calculates gini impurity
    def calculateGini(self, lowerArr, greaterArr, y):
        #calculate label class counts on left (true) node
        lowerValues = [0,0,0]
        for val in lowerArr:
            if y[val] == 0:
                lowerValues[0] += 1
            elif y[val] == 1:
                lowerValues[1] += 1
            else:
                lowerValues[2] += 1

        #calculate label class counts on right (false) node
        greaterValues = [0,0,0]
        for val in greaterArr:
            if y[val] == 0:
                greaterValues[0] += 1
            elif y[val] == 1:
                greaterValues[1] += 1
            else:
                greaterValues[2] += 1

        #if spliting went wrong set gini something higher than 1
        if len(lowerArr) == 0 or len(greaterArr) == 0:
            return 5

        lowerZeroPerc = lowerValues[0] / len(lowerArr)
        lowerFirstPerc = lowerValues[1] / len(lowerArr)
        lowerSecondPerc = lowerValues[2] / len(lowerArr)

        greaterZeroPerc = greaterValues[0] / len(greaterArr)
        greaterFirstPerc = greaterValues[1] / len(greaterArr)
        greaterSecondPerc = greaterValues[2] / len(greaterArr)

        giniLower = 1 - (lowerZeroPerc ** 2 + lowerFirstPerc ** 2 + lowerSecondPerc ** 2)
        giniGreater = 1 - (greaterZeroPerc ** 2 + greaterFirstPerc ** 2 + greaterSecondPerc ** 2)

        #calculate gini
        sampleCount = len(lowerArr) + len(greaterArr)
        gini = (len(lowerArr) / sampleCount) * giniLower + (len(greaterArr) / sampleCount) * giniGreater

        return float(format(gini, '.3f'))

    #decides the class(species) of given list
    def decideClass(self, y):
        hist = [0,0,0]

        for i in y:
            hist[i] += 1

        maxValue = max(hist)
        mostCommon = hist.index(maxValue)
        
        return mostCommon

    #traverse decision tree
    def traverseTree(self, x, node):
        if node.isLeaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverseTree(x, node.left)
        return self.traverseTree(x, node.right)

    #predict and return list of class(species)
    def predict(self, X):
        return list([self.traverseTree(x, self.root) for x in X])