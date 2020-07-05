import abc


class Sub_network(abc.ABC):
    @abc.abstractmethod
    def build_subnetwork(self, inputs, weights, last_layer='conv4_1'):
        pass

    @abc.abstractmethod
    def subnetwork_layer_params(self, layer):
        pass
