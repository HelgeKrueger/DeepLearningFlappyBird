import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2
from keras.utils import np_utils

import os.path

class NeuralNet():
    def __init__(self):
        self.num_outputs = 2
        self.input_shape = (80, 80, 2);
        self.filename = 'flappy.h5'
        self.loss = 0
        self.prediction = [0, 0]

        if os.path.isfile(self.filename):
            self.model = load_model(self.filename)
        else:
            self.model = self._build_net()


    def _build_net(self):
        model = Sequential()
        model.add(Convolution2D(16, 5, 5,
            border_mode='same',
            input_shape=self.input_shape,
            activation='relu',
            subsample=(2,2)))
        model.add(Convolution2D(32, 5, 5,
            border_mode='same',
            activation='relu',
            subsample=(2,2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
#        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.num_outputs))

        model.compile(loss='mse', optimizer='adam')

        return model

    def state(self):
        return "Loss {:e} prediction {:f} {:f} Difference {:f}".format(self.loss, self.prediction[0], self.prediction[1], self.prediction[0] - self.prediction[1])

    def predict(self, frames):
        prediction = self.model.predict(frames.reshape(-1, 80, 80, 2))
        self.prediction = prediction[0]
        return prediction

    def train(self, x, y):
        self.loss = self.model.train_on_batch(x, y)
        # print("Training loss: ", loss)

    def save(self):
        self.model.save(self.filename)

    def train_on_replay_memory(self, replay_memory):
        minibatch = replay_memory.get_training_sample()

        inputs = np.array([ele['state'] for ele in minibatch])
        inputs_next = np.array([ele['next_state'] for ele in minibatch])

        targets = self.predict(inputs)
        predictions_next = self.predict(inputs_next)

        gamma = 0.95
        q_max = 1/(1-gamma)

        predictions_next = np.minimum(predictions_next, q_max * np.ones((32, 2)))

        for i in range(0, len(minibatch)):
            sample = minibatch[i]
            targets[i, sample['action']] = sample['reward']
            if not sample['terminal']:
                targets[i, sample['action']] += gamma * np.max(predictions_next[i])

        self.train(inputs, targets)
