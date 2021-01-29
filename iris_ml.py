#Guessing by giving different labels whether the flower is virginica or not
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

# printing descriptions of iris
#print(iris.DESCR)

'''
:Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                0- Iris-Setosa
                1- Iris-Versicolour
                2- Iris-Virginica
'''
features = iris.data
labels = iris.target
print(features[0], labels[0])

clf = KNeighborsClassifier()
clf.fit(features, labels)

preds = clf.predict([[31, 1, 1, 1]])
print(preds)
