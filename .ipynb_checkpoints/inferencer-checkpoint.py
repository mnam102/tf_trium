from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from metrics import MAPE
from config import InferenceConfig
from keras.models import Model
from dataset import Dataset
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Inferencer():
    def __init__(self, dataset:Dataset, model:Model, config:InferenceConfig):
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
        plt.show()

        print(f"R2 Score : {r2_score(true_test_df, pred_test_df)}")
        print(f"RMSE Score : {np.sqrt(mean_squared_error(true_test_df, pred_test_df))}")
        print(f"MAE Score : {mean_absolute_error(true_test_df, pred_test_df)}")
        print(f"MAPE Score : {MAPE(true_test_df, pred_test_df)/100}")


# f, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(12,5))

# # Plot results
# ax0.scatter(y1_ans_f1, y1_pred_f1)
# ax0.plot([0, 70000], [0, 70000], "--k")
# ax0.set_ylabel("Target predicted")
# ax0.set_xlabel("True Target")
# ax0.set_title("F1")
# ax0.text(
#     4500,
#     70500,
#     r"$R^2$=%.2f, MAE=%.2f, MAPE=%.4f"
#     % (r2_score(y1_ans_f1, y1_pred_f1), mean_absolute_error(y1_ans_f1, y1_pred_f1)
#        ,MAPE(y1_ans_f1, y1_pred_f1)/100),
# )
# ax0.set_xlim([0, 80000])
# ax0.set_ylim([0, 80000])

# ################
# ax1.scatter(y1_ans_f2, y1_pred_f2)
# ax1.plot([0, 70000], [0, 70000], "--k")
# ax1.set_ylabel("Target predicted")
# ax1.set_xlabel("True Target")
# ax1.set_title("F2")
# ax1.text(
#     4500,
#     70500,
#     r"$R^2$=%.2f, MAE=%.2f, MAPE=%.4f"
#     % (r2_score(y1_ans_f2, y1_pred_f2), mean_absolute_error(y1_ans_f2, y1_pred_f2)
#        ,MAPE(y1_ans_f2, y1_pred_f2)/100),
# )
# ax1.set_xlim([0, 80000])
# ax1.set_ylim([0, 80000])

# f.suptitle("Hydrocarbon Dataset", y=0.035)
# f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])