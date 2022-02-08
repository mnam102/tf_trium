import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import numpy as np
import yaml
import argparse

from modules.metrics import MAPE
from modules.visualization import sample_to_scaleogram
from models.model import create_model
from modules.trainer import Trainer
from modules.dataset import Dataset
from modules.utils import load_yaml


mpl.rc('font', family = "NanumBarunGothic") #맑은 고딕 설정 

#그래프에서 음수값이 나올때, 깨지는 현상 방지 
mpl.rc('axes', unicode_minus = False)

parser = argparse.ArgumentParser(description='Train soil contamination.')
parser.add_argument('--visualization', type=bool, default=False, 
                    help='visualize? or not')


def main(args):
    data_config = load_yaml('config/data.yml')
    dataset = Dataset(data_config)
    dataset.preprocess()
    
    if args.visualization:
        sample_to_scaleogram(dataset.df, signal_length=64, wavelet='mexh')
    
    model_config = load_yaml('config/model.yml')
    model = create_model(model_config)

    train_config = load_yaml('config/train.yml')
    trainer = Trainer(dataset=dataset, model=model, config=train_config)
    trainer.train()
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)