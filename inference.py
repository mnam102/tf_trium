from modules.utils import load_yaml
from models.model import create_model
from modules.inferencer import Inferencer
from modules.dataset import Dataset


def main():
    data_config = load_yaml('config/data.yml')
    dataset = Dataset(data_config, mode='test')
    dataset.preprocess()
    
    model_config = load_yaml('config/model.yml')
    model = create_model(model_config)

    inf_config = load_yaml('config/inference.yml')
    inferencer = Inferencer(dataset=dataset, model=model, config=inf_config)
    inferencer.inference()
    
if __name__ == '__main__':
    main()