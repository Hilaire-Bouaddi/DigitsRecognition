import mnist_loader
import network
import pickle

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])

nEpochs = 30
mini_batch_size = 10
eta = 3.0

net.sgd(training_data, nEpochs, mini_batch_size, eta, test_data=test_data)

with open('savednetwork', 'wb') as net_file:
    pickle.dump(net,net_file)
