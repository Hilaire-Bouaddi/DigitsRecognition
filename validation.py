import mnist_loader
import pickle

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

with open('savednetwork', 'rb') as net_file:
    net = pickle.load(net_file)

print("Validation : {0} / {1}".format(net.evaluate(validation_data), 10000))
