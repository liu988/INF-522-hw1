from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split

# Data sets
PHISHING_DATASET = "phishing_dataset.csv"

# load data
phishing_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=PHISHING_DATASET,
        target_dtype=np.int,
        features_dtype=np.int)

# prep phishing_dataset inputs, test_size:(train=0.4,test=0.6)
X_training, X_test, Y_training, Y_test = train_test_split(
            phishing_set.data,
            phishing_set.target,
            test_size=0.6,
            random_state=4)

# increase test inputs by 1: (-1,0,1)->(0,1,2)
for i in range(len(Y_training)):
    Y_training[i] +=1
    
for i in range(len(Y_test)):
    Y_test[i] +=1
    

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=9)]

# build 3 layer DNN with 10, 20, 10 units respectively. hidden_units:(how many layers, number of neurons)
mdir = "/tmp/model"
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[10,20,10], 
                                              n_classes=3,
                                            model_dir=mdir)

def get_train_inputs():
    x = tf.constant(X_training)
    y = tf.constant(Y_training)
    return x, y

# DO THE TRAINING! steps:(how many times to repeat the whole trainign/testing process untill it converges)
classifier.fit(input_fn=get_train_inputs, steps=2000)

# prep test inputs
def get_test_inputs():
    x = tf.constant(X_test)
    y = tf.constant(Y_test)
    return x, y

# how good is the training?
accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                       steps=1)["accuracy"]
print("\nAccuracy: {0:f}\n".format(accuracy_score))
