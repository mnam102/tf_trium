import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pywt
import scaleogram as scg 
from skimage.transform import resize
from tqdm import tqdm
import pickle
from modules.utils import load_pickle


class Dataset():
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        
    def preprocess(self, test_val_size=0.2):
        if self.config.LOAD_PICKLE or (self.mode=='test'):
            print('load_pickles from files...')
            try:
                self.__load_pickles()
            except Exception as e:
                print(e)  
                
        elif not self.config.LOAD_PICKLE:     
            self.__resampling()
            self.__split_data(test_val_size=test_val_size)    
            self.__make_data()

                
        self.__scaling()
    
    def __load_pickles(self):
        if self.mode == 'train':
            self.train_X = load_pickle(self.config.TRAIN_PICKLE_PATH)
            self.valid_X = load_pickle(self.config.VALID_PICKLE_PATH) 
            self.train_Y = load_pickle(self.config.TRAIN_Y_PICKLE_PATH)
            self.valid_Y = load_pickle(self.config.VALID_Y_PICKLE_PATH)

        else:
            self.test_X = load_pickle(self.config.TEST_PICKLE_PATH)
            self.test_Y = load_pickle(self.config.TEST_Y_PICKLE_PATH)

    def __resampling(self):
        sample_column = self.config.GROUPBY_COLUMN_NAME
        df=pd.read_feather(self.config.RAW_PATH)
        if 'Unnamed: 10' in df.columns:
            df=df.drop(['Unnamed: 10'], axis=1)
        self.df = df.groupby([sample_column])[df.columns[7:]].mean().reset_index()

    def __split_data(self, test_val_size=0.2, val_size=0.5):
        df = self.df.copy()
        self.train, test = train_test_split(df, random_state=21, shuffle=True, test_size=test_val_size)
        self.test, self.valid = train_test_split(test, random_state=21, shuffle=True, test_size=val_size)
    
    def __make_data(self):
        self.train_X = self.__make_cwt_data(self.train, pickle_path=self.config.TRAIN_PICKLE_PATH)[..., np.newaxis]
        self.valid_X = self.__make_cwt_data(self.valid, pickle_path=self.config.VALID_PICKLE_PATH)[..., np.newaxis]
        self.test_X = self.__make_cwt_data(self.test, pickle_path=self.config.TEST_PICKLE_PATH)[..., np.newaxis]
    
        self.train_Y = self.__make_target_data(self.train, pickle_path=self.config.TRAIN_Y_PICKLE_PATH)
        self.valid_Y = self.__make_target_data(self.valid, pickle_path=self.config.VALID_Y_PICKLE_PATH)
        self.test_Y = self.__make_target_data(self.test, pickle_path=self.config.TEST_Y_PICKLE_PATH)

    def __scaling(self):
        self.scaler = RobustScaler()
        
        if self.mode == 'train':
            self.train_Y_scaled=self.scaler.fit_transform(self.train_Y)
            self.valid_Y_scaled=self.scaler.transform(self.valid_Y)   
            with open('data/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
        else:
            try:
                with open('data/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                self.test_Y_scaled=self.scaler.transform(self.test_Y)
            except Exception as e:
                print(e)

    def __make_cwt_data(self, df, pickle_path='') -> np.array:
        signal_length= self.config.SIGNAL_LENGTH
        scales =scg.periods2scales( np.arange(1, signal_length+1) ) # range of scales 

        scalo_df=[]
        for i in tqdm(range(len(df))):
            coeffs, freqs= pywt.cwt(df.iloc[i,4:], scales, wavelet=self.config.WAVELET) 
            rescale_coeffs = resize(coeffs, (signal_length, signal_length), mode = 'constant')
            scalo_df.append(rescale_coeffs)

        scalo_df = np.array(scalo_df)

        if pickle_path:
            try:
                with open(pickle_path, 'wb') as f:
                    pickle.dump(scalo_df, f)
            except Exception as e:
                print(e)

        return scalo_df

    def __make_target_data(self, df, pickle_path=''):
        log_Y_df=np.log1p(df[self.config.TARGET])
        if pickle_path:
            try:
                with open(pickle_path, 'wb') as f:
                    pickle.dump(log_Y_df, f)
            except Exception as e:
                print(e)

        return log_Y_df