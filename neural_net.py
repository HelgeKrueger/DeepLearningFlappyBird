import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

import os.path

class NeuralNet():
    def __init__(self):
        self.num_outputs = 2
        self.input_shape = (80, 80, 2);
        self.filename = 'flappy.h5'

        if os.path.isfile(self.filename):
            self.model = load_model(self.filename)
        else:
            self.model = self._build_net()


    def _build_net(self):
        model = Sequential()
        model.add(Convolution2D(16, 5, 5,
            border_mode='valid',
            input_shape=self.input_shape,
            activation='relu',
            subsample=(2,2), name='conv1'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', name='dense1'))
        model.add(Dense(self.num_outputs, activation='softmax', name='final'))

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        return model

    def predict(self, frames):
        return self.model.predict(frames.reshape(-1, 80, 80, 2))

    def train(self, x, y):
        self.model.train_on_batch(x, y)

    def save(self):
        self.model.save(self.filename)
