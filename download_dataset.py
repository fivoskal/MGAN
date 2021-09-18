import pickle
from pathlib import Path
import numpy as np
from keras.datasets import cifar10
import os

if not os.isdir('./data'):
  os.mkdir('./data')

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
with open('./data/cifar10_train.pkl','wb') as f:
  pickle.dump({'data':x_train, 'labels':y_train}, f)
