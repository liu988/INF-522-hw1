from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

# Data sets
PHISHING_DATASET = "phishing_dataset.csv"

# try K=1 through K=25 and record testing accuracy
k_range = range(1, 26)

# We can create Python dictionary using [] or dict()
scores = []

# load data
phishing_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=PHISHING_DATASET,
        target_dtype=np.int,
        features_dtype=np.int)

# prep phishing_dataset inputs, test_size: (train=0.6,test=0.4)
X_training, X_test, Y_training, Y_test = train_test_split(
            phishing_set.data,
            phishing_set.target,
            test_size=0.4,
            random_state=4)

# DO THE TRAINING!
# We use a loop through the range 1 to 26
# We append the scores in the dictionary
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_training, Y_training)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(Y_test, y_pred))

print(scores)

# plot the relationship between K and testing accuracy
# plt.plot(x_axis, y_axis)
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

