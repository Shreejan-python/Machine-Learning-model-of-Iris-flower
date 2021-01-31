#Guessing by giving different labels what type of flower(iris) is?
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
#print(features[0], labels[0])

clf = KNeighborsClassifier()
clf.fit(features, labels)

sl = int(input("Enter a sepal length(cm) : "))
sw = int(input("Enter a sepal width(cm) : "))
pl = int(input("Enter a petal length(cm) : "))
pw = int(input("Enter a petal width(cm) : "))

preds = clf.predict([[int(sl), int(sw), int(pl), int(pw)]])

if preds == ([[0]]):
    print("Iris Setosa")

elif preds == ([[1]]):
    print("Iris Versicolor")

elif preds == ([[2]]):
    print("Iris Virginica")

