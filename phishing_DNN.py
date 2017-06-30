from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec

import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
tf.logging.set_verbosity(tf.logging.INFO)
# Data sets
PHISHING_DATASET = "phishing_dataset.csv"

# load data
phishing_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=PHISHING_DATASET,
        target_dtype=np.int,
        features_dtype=np.int)

# prep phishing_dataset inputs
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

validation_metrics = {
      "accuracy": MetricSpec(
                          metric_fn=tf.contrib.metrics.streaming_accuracy,
                          prediction_key="classes"),
      "recall": MetricSpec(
                          metric_fn=tf.contrib.metrics.streaming_recall,
                          prediction_key="classes"),
      "precision": MetricSpec(
                          metric_fn=tf.contrib.metrics.streaming_precision,
                          prediction_key="classes")
                        }
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
      X_test,
      Y_test,
      every_n_steps=50,
      metrics=validation_metrics,
      early_stopping_metric="loss",
      early_stopping_metric_minimize=True,
      early_stopping_rounds=200)
  
# build 3 layer DNN with 10, 20, 10 units respectively.
mdir = "/tmp/phish_model"
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10,20,10],
                                            n_classes=3,
                                            model_dir=mdir,
                                            config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))

def get_train_inputs():
    x = tf.constant(X_training)
    y = tf.constant(Y_training)
    return x, y

# DO THE TRAINING!
classifier.fit(input_fn=get_train_inputs, steps=1000, monitors=[validation_monitor])

# prep test inputs
def get_test_inputs():
    x = tf.constant(X_test)
    y = tf.constant(Y_test)
    return x, y

# how good is the training?
accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                       steps=1)["accuracy"]
print("\nAccuracy: {0:f}\n".format(accuracy_score))
