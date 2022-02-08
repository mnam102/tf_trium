class TrainConfig(object):
    BATCH_SIZE = 64
    SHAPE = (64, 64, 1)
    EPOCH = 500
    MODEL_CHECKPOINT_PATH = 'weights/best_all.h5'
    
class DatasetConfig(object):
    RAW_PATH = 'data/nir_all_resampled_df.ftr'
    WAVELET = 'mexh'
    SIGNAL_LENGTH = 64
    TARGET = ['F1','F2','F3']
    GROUPBY_COLUMN_NAME = 'Sample Name'
    TRAIN_PICKLE_PATH = 'data/train_scalo_df_mexh_3.pkl'
    VALID_PICKLE_PATH = 'data/valid_scalo_df_mexh_3.pkl'
    TEST_PICKLE_PATH = 'data/test_scalo_df_mexh_3.pkl'
    
    TRAIN_Y_PICKLE_PATH = 'data/log_Y_train.pkl'
    VALID_Y_PICKLE_PATH = 'data/log_Y_valid.pkl'
    TEST_Y_PICKLE_PATH = 'data/log_Y_test.pkl'
    
class ModelConfig(object):
    INPUT_SHAPE = [64, 64, 1]
    FILTERS = [16, 32, 64, 64, 128]
    
class InferenceConfig(object):
    WEIGHT_PATH = 'weights/best_all.h5'
    BATCH_SIZE = 64
    TARGET = ['F1','F2','F3']
    