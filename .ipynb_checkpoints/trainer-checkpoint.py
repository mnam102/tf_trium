import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.models import Model

from dataset import Dataset
from config import TrainConfig


class Trainer():
    def __init__(self, dataset:Dataset, model:Model, config:TrainConfig):
        self.dataset = dataset
        self.model = model
        self.config = config  
        
        self.make_callback()

    def make_callback(self):
        self.callbacks = [ModelCheckpoint(self.config.MODEL_CHECKPOINT_PATH, save_best_only=True), 
                          EarlyStopping(patience=80),
                          ReduceLROnPlateau(patience=30,factor=0.1, min_lr=1e-6, verbose=1)]
        
    def train(self):

        # Store training stats
        history = self.model.fit(self.dataset.train_X, 
                            self.dataset.train_Y_scaled, 
                            epochs=self.config.EPOCH,
                            validation_data=(self.dataset.valid_X, self.dataset.valid_Y_scaled),
                            batch_size=self.config.BATCH_SIZE ,
                            verbose=1,
                            callbacks=self.callbacks)

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
