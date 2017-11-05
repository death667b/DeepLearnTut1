"""
This is code from a tutorial by Shehzad Noor Taus Priyo
From
https://medium.com/@sntaus/image-classification-using-deep-learning-hello-world-tutorial-a47d02fd9db1
"""

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy

acc = Accuracy()
network = input_data(shape=[None, 28, 28, 1])

# Conv Layers
