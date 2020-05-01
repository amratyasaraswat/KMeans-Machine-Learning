# Import required packages here (after they are installed)
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as mp
from pylab import show
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

# Load data. csv file should be in the same folder as the notebook for this to work, otherwise
# give data path.
data = np.loadtxt("data.csv")

#shuffle the data and select training and test data
np.random.seed(100)
np.random.shuffle(data)

features = []
digits = []

for row in data:
    #import the data and select only the 1's and 5's
    digit = str(row[0])
    feature = row[1:]
    digits.append(digit)
    features.append(feature)

# Creating a K-Means model
n_clusters = list(range(2,21))
inertia = []
for i in (n_clusters):
    kmeans = KMeans(n_clusters = i, init = 'random').fit(features)
    i = kmeans.inertia_
    inertia.append(i)

mp.title("Figure 5.1 Inertia vs K-means value from 2 to 20")
mp.plot(n_clusters, inertia)
mp.show()

n_clusters = list(range(2,21))
inertia = []
for i in (n_clusters):
    kmeans = KMeans(n_clusters = i, n_init = 1, max_iter = 1, init = 'random').fit(features)
    i = kmeans.inertia_
    inertia.append(i)

mp.title("Figure 5.2 Inertia vs K-means with n_init and max_iter = 1")
mp.plot(n_clusters, inertia)
mp.show()


n_clusters = list(range(2,21))
n_iter = []
for i in (n_clusters):
    kmeans = KMeans(n_clusters = i, init = 'random').fit(features)
    i = kmeans.n_iter_
    n_iter.append(i)

mp.title("Figure 5.3 N_iter vs K-means value from 2 to 20")
mp.plot(n_clusters, n_iter)
mp.show()
