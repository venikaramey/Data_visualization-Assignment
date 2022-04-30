import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import  datasets

iris = datasets.load_iris()

data =pd.DataFrame(data=iris['data'],columns=iris['feature_names'])
data['target'] = iris['target']
# print(data.head())
x = data.iloc[:,:4].values
y = data.iloc[:,4].values
# print(x)
# print(y)

pca = decomposition.PCA(n_components=3)
x =pca.fit_transform(x)

fig = plt.figure(figsize=(8, 7))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y,
           cmap=plt.cm.viridis)


plt.show()