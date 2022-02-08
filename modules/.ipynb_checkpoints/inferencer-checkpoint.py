import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Model
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from modules.metrics import MAPE
from modules.visualization import get_scatter_result
from modules.dataset import Dataset


class Inferencer():
    def __init__(self, dataset:Dataset, model:Model, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.__load_model()
        
    def __load_model(self):
        self.model.load_weights(self.config.WEIGHT_PATH)
    
    def inference(self):
        pred_test = self.model.predict(self.dataset.test_X, batch_size=self.config.BATCH_SIZE ,verbose=1)
        self.pred_test_inverse = np.expm1(self.dataset.scaler.inverse_transform(pred_test))
        true_test = self.dataset.scaler.inverse_transform(self.dataset.test_Y_scaled)
        self.true_test = np.expm1(true_test)
        self.__get_each_result(0, self.config.TARGET[0])
        self.__get_each_result(1, self.config.TARGET[1])
        self.__get_each_result(2, self.config.TARGET[2])
        
    def __get_each_result(self, idx, name='F1'):
        pred_test_df = pd.DataFrame(self.pred_test_inverse).iloc[:,idx]
        true_test_df = pd.DataFrame(self.true_test).iloc[:, idx]

        plt.figure(figsize=(10, 10))
        plt.title(name)
        plt.scatter(true_test_df.index, true_test_df, label='target')
        plt.scatter(pred_test_df.index, pred_test_df, label='pred')
        plt.legend()
        
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/{name}_scatter_result.png')

        print(f"R2 Score : {r2_score(true_test_df, pred_test_df)}")
        print(f"RMSE Score : {np.sqrt(mean_squared_error(true_test_df, pred_test_df))}")
        print(f"MAE Score : {mean_absolute_error(true_test_df, pred_test_df)}")
        print(f"MAPE Score : {MAPE(true_test_df, pred_test_df)/100}")
        get_scatter_result(true_test_df, pred_test_df, name)
        