"""interface for network model and dataset load, for dcnn_shift_cnn.py using"""
import utils_network as net
import utils_data as data


class Network(object):
    """
    network for 10boards datasets
    """
    def __init__(self, modelfunc, matdir):
        self.model = modelfunc()
        self.matdir = matdir

    def fit(self, shift=2):
        self.model.fit()
