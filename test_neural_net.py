import numpy as np

from neural_net import NeuralNet

def test_neural_net_initialisation_and_saving():
    nn = NeuralNet()
    nn.save()

def test_predict():
    nn = NeuralNet() #loads nn again
    nn.predict(np.zeros((80, 80, 4)))

