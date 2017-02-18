from replay_memory import ReplayMemory
from neural_net import NeuralNet

replay_memory = ReplayMemory(50000, 32)
neural_net = NeuralNet()

for i in range(0, 100):
    neural_net.train_on_replay_memory(replay_memory)
    print(neural_net.state())

neural_net.save()
