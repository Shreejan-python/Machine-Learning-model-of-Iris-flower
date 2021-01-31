from sklearn import datasets
import pandas as pd


iris = datasets.load_iris()
#print(iris.DESCR)

y = pd.read_csv(iris)