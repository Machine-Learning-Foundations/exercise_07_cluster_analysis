import matplotlib.pyplot as plt
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# create dataset
num_classes = 3
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=10, n_classes=num_classes)
# create scatter plot for samples from each class
for class_value in range(num_classes):
 # get row indexes for samples with this class
 row_ix = where(y == class_value)
 # create scatter of these samples
 plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show plot
plt.show()

# create K-means model
# TODO

# fit K-means model to data
# TODO

# get cluster labels
# TODO

# get cluster centers
# TODO

# visualize clustering results
# TODO

# initialize GaussianMixture with random_state=0
# TODO

# fit the GMM model to the data
# TODO

# get cluster assignments
# TODO

# get cluster centers (means)
# TODO

# visualize clustering results
# TODO
