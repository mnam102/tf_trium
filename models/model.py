import keras
from keras.models import *
from keras.layers import *


def create_model(config) -> Model:
    model_input=Input(shape=(config.INPUT_SHAPE))
    
    x=Conv2D(filters=config.FILTERS[0],kernel_size=(3,3),padding='same')((model_input))
    x=LeakyReLU(alpha=0.1)(x)
    x=BatchNormalization()(x)
    x=MaxPooling2D(pool_size=(3, 3))(x)

    x=Conv2D(filters=config.FILTERS[1],kernel_size=(3,3),padding='same')(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=BatchNormalization()(x)
    x=MaxPooling2D(pool_size=(3, 3))(x)

    x=Conv2D(filters=config.FILTERS[2],kernel_size=(3,3),padding='same')(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=BatchNormalization()(x)
    x=MaxPooling2D(pool_size=(3, 3))(x)
    
    x=Conv2D(filters=config.FILTERS[3],kernel_size=(3,3),padding='same')(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=BatchNormalization()(x)
    
    x=Conv2D(filters=config.FILTERS[4],kernel_size=(3,3),padding='same')(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=BatchNormalization()(x)

    x=GlobalAveragePooling2D()(x)
    
    x = Dense(3, activation = "linear")(x)
    
    model = Model(inputs = model_input, outputs = x)
    
    model.compile(loss='mae', optimizer='adam',metrics=['mae'])


    return model