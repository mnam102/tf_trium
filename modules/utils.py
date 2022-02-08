import yaml
from munch import DefaultMunch
import pickle


def load_yaml(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    dataObj = DefaultMunch.fromDict(data)
    return dataObj

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data